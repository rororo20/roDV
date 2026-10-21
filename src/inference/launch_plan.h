#ifndef RODV_INFERENCE_LAUNCH_PLAN_H_
#define RODV_INFERENCE_LAUNCH_PLAN_H_

#include <cstdint>
#include <string>

namespace rodv {
namespace inference {

// One pileup tensor. r1.6.1 WGS is 100 x 221 x 7 (six base channels + insert
// size). alt_aligned_pileup defaults to none in that release.
struct ImageShape {
  int height = 100;
  int width = 221;
  int channels = 7;
};

// Physical cores and L3 of a single NUMA node (one socket on Xeon 8260).
struct CpuBudget {
  int numa_nodes = 1;
  int physical_cores_per_node = 1;
  int logical_cpus_per_node = 1;
  uint64_t l3_bytes_per_node = 32ull << 20;
};

// How to run call_variants on this CPU.
//
// Upstream r1.6.1 uses one TensorFlow process and --batch_size=1024. On a
// dual-socket Cascade Lake that oversubscribes one thread pool across NUMA
// and blows L3: a 1024-batch of 100x221x7 fp32 is ~600MB of input before
// Inception activations. AVX-512 conv also stops scaling well past ~4-8
// physical cores, and hyperthreads share the same FP units.
//
// The plan is therefore many small processes, each pinned to one node, each
// with a batch whose input slice fits in a fraction of that node's L3.
struct InferenceLaunchPlan {
  int processes = 1;
  int streams_per_node = 1;
  int threads_per_process = 1;
  int batch_size = 8;
  // Upstream call_variants default, kept so callers can see the contrast.
  int upstream_batch_size = 1024;
};

// Pure function. Does not read the machine.
InferenceLaunchPlan plan_cpu_inference(const CpuBudget &cpu,
                                       const ImageShape &image);

// Reads /sys topology. Missing sysfs falls back to one node, no HT, 32MB L3.
CpuBudget detect_cpu_budget();

// Multi-line description of the plan and the runtime knobs that go with it.
std::string format_launch_plan(const CpuBudget &cpu, const ImageShape &image,
                               const InferenceLaunchPlan &plan);

} // namespace inference
} // namespace rodv

#endif
