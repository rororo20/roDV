#include "launch_plan.h"

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace rodv {
namespace inference {
namespace {

int parse_cache_size(const std::string &text) {
  // sysfs writes "36608K" or "35M".
  if (text.empty()) {
    return 0;
  }
  char *end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || value <= 0) {
    return 0;
  }
  if (*end == 'K' || *end == 'k') {
    return static_cast<int>(value) * 1024;
  }
  if (*end == 'M' || *end == 'm') {
    return static_cast<int>(value) * 1024 * 1024;
  }
  return static_cast<int>(value);
}

int count_numa_nodes() {
  DIR *dir = opendir("/sys/devices/system/node");
  if (dir == nullptr) {
    return 0;
  }
  int nodes = 0;
  while (dirent *ent = readdir(dir)) {
    const std::string name = ent->d_name;
    if (name.size() > 4 && name.compare(0, 4, "node") == 0 &&
        name[4] >= '0' && name[4] <= '9') {
      ++nodes;
    }
  }
  closedir(dir);
  return nodes;
}

int count_cpu_list(const std::string &list) {
  // "0,48" or "0-1,48-49". A bare range is inclusive.
  int count = 0;
  std::stringstream ss(list);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const auto dash = token.find('-');
    if (dash == std::string::npos) {
      ++count;
      continue;
    }
    const int begin = std::atoi(token.substr(0, dash).c_str());
    const int end = std::atoi(token.substr(dash + 1).c_str());
    if (end >= begin) {
      count += end - begin + 1;
    }
  }
  return count;
}

int threads_per_core() {
  std::ifstream in("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
  if (!in) {
    return 1;
  }
  std::string line;
  std::getline(in, line);
  return std::max(count_cpu_list(line), 1);
}

uint64_t l3_bytes() {
  for (int index = 0; index < 6; ++index) {
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/index" +
                             std::to_string(index) + "/";
    std::ifstream level_in(base + "level");
    if (!level_in) {
      continue;
    }
    int level = 0;
    level_in >> level;
    if (level != 3) {
      continue;
    }
    std::ifstream size_in(base + "size");
    std::string size;
    size_in >> size;
    const int bytes = parse_cache_size(size);
    if (bytes > 0) {
      return static_cast<uint64_t>(bytes);
    }
  }
  return 0;
}

} // namespace

InferenceLaunchPlan plan_cpu_inference(const CpuBudget &cpu,
                                       const ImageShape &image) {
  InferenceLaunchPlan plan;
  const int nodes = std::max(cpu.numa_nodes, 1);
  const int physical = std::max(cpu.physical_cores_per_node, 1);

  // Prefer groups of ~6 physical cores. AVX-512 conv throughput flattens
  // after that, and a second stream on the same node hides oneDNN barriers.
  // On a waste/distance tie, keep the larger group (fewer streams).
  int best_cores = std::min(physical, 6);
  int best_streams = 1;
  int best_waste = physical;
  int best_dist = 100;
  const int core_lo = physical < 4 ? physical : 4;
  const int core_hi = std::min(physical, 8);
  for (int cores = core_lo; cores <= core_hi; ++cores) {
    const int streams = physical / cores;
    if (streams < 1) {
      continue;
    }
    const int waste = physical - streams * cores;
    const int dist = std::abs(cores - 6);
    const bool better = waste < best_waste ||
                        (waste == best_waste && dist < best_dist) ||
                        (waste == best_waste && dist == best_dist &&
                         streams < best_streams);
    if (better) {
      best_waste = waste;
      best_dist = dist;
      best_streams = streams;
      best_cores = cores;
    }
  }

  plan.streams_per_node = best_streams;
  plan.threads_per_process = best_cores;
  plan.processes = nodes * best_streams;

  const int height = std::max(image.height, 1);
  const int width = std::max(image.width, 1);
  const int channels = std::max(image.channels, 1);
  const uint64_t bytes_per_example =
      static_cast<uint64_t>(height) * static_cast<uint64_t>(width) *
      static_cast<uint64_t>(channels) * sizeof(float);

  const uint64_t l3 =
      cpu.l3_bytes_per_node == 0 ? (32ull << 20) : cpu.l3_bytes_per_node;
  // Input batch gets a quarter of this stream's L3 slice. The rest stays
  // available for the first conv activations and whatever weights fit.
  const uint64_t budget = std::max(l3 / static_cast<uint64_t>(best_streams) / 4,
                                   bytes_per_example * 8);
  int batch = static_cast<int>(budget / bytes_per_example);
  batch = std::clamp(batch, 8, 32);
  batch &= ~7;
  if (batch < 8) {
    batch = 8;
  }
  plan.batch_size = batch;
  return plan;
}

CpuBudget detect_cpu_budget() {
  CpuBudget cpu;
  const long logical = sysconf(_SC_NPROCESSORS_ONLN);
  const int logical_cpus = logical > 0 ? static_cast<int>(logical) : 1;
  const int nodes = std::max(count_numa_nodes(), 1);
  const int tpc = std::max(threads_per_core(), 1);
  const int physical_total = std::max(logical_cpus / tpc, 1);

  cpu.numa_nodes = nodes;
  cpu.physical_cores_per_node = std::max(physical_total / nodes, 1);
  cpu.logical_cpus_per_node = std::max(logical_cpus / nodes, 1);
  const uint64_t l3 = l3_bytes();
  cpu.l3_bytes_per_node = l3 == 0 ? (32ull << 20) : l3;
  return cpu;
}

std::string format_launch_plan(const CpuBudget &cpu, const ImageShape &image,
                               const InferenceLaunchPlan &plan) {
  std::ostringstream out;
  out << "call_variants CPU plan (DeepVariant r1.6.1 InceptionV3)\n"
      << "  machine: " << cpu.numa_nodes << " NUMA nodes, "
      << cpu.physical_cores_per_node << " physical cores/node, "
      << cpu.logical_cpus_per_node << " logical cpus/node, L3 "
      << (cpu.l3_bytes_per_node >> 20) << " MiB/node\n"
      << "  image: " << image.height << "x" << image.width << "x"
      << image.channels << " uint8 NHWC\n"
      << "  processes: " << plan.processes << " (" << plan.streams_per_node
      << " streams/node)\n"
      << "  threads per process: " << plan.threads_per_process
      << " physical cores, do not add hyperthreads\n"
      << "  batch size: " << plan.batch_size
      << "  (upstream call_variants default is " << plan.upstream_batch_size
      << ")\n"
      << "  bind each process to one node (numactl --cpunodebind)\n"
      << "  env: OMP_NUM_THREADS=" << plan.threads_per_process
      << " KMP_BLOCKTIME=0 OMP_WAIT_POLICY=PASSIVE\n"
      << "  backend: OpenVINO CPU plugin, FP32 first; INT8 only after a "
         "hap.py gate. Not linked in this binary yet.\n";
  return out.str();
}

} // namespace inference
} // namespace rodv
