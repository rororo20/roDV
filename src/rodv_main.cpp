#include "launch_plan.h"
#include "preprocess.h"
#include "runtime/example_writer.h"
#include "runtime/hts_read_source.h"
#include "runtime/pipeline.h"
#include "runtime/reference_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  rodv_main call-variants-plan [--height N --width N --channels N]\n"
      << "  rodv_main preprocess-bench\n"
      << "  rodv_main make-examples --ref FASTA --reads BAM --out TSV"
      << " [--nthreads N] [--decode-threads N] [--region-bp N]\n";
}

int plan_main(int argc, char **argv) {
  rodv::inference::ImageShape image;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--height") {
      image.height = std::atoi(need("--height"));
    } else if (arg == "--width") {
      image.width = std::atoi(need("--width"));
    } else if (arg == "--channels") {
      image.channels = std::atoi(need("--channels"));
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage();
      return 2;
    }
  }
  const auto cpu = rodv::inference::detect_cpu_budget();
  const auto plan = rodv::inference::plan_cpu_inference(cpu, image);
  std::cout << rodv::inference::format_launch_plan(cpu, image, plan);
  return 0;
}

int bench_main() {
  constexpr int kHeight = 100;
  constexpr int kWidth = 221;
  constexpr int kChannels = 7;
  constexpr int kBatch = 16;
  constexpr int kIters = 200;
  const size_t n =
      static_cast<size_t>(kBatch) * kHeight * kWidth * kChannels;
  std::vector<uint8_t> src(n);
  std::vector<float> dst(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = static_cast<uint8_t>(i);
  }
  rodv::inference::preprocess_inception(src.data(), dst.data(), n);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    rodv::inference::preprocess_inception(src.data(), dst.data(), n);
  }
  const double sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  const double examples = static_cast<double>(kIters) * kBatch;
  volatile float sink = dst[0] + dst[n - 1];
  std::cout << "preprocess " << kHeight << "x" << kWidth << "x" << kChannels
            << " batch " << kBatch << ": " << (examples / sec)
            << " examples/s (sink " << sink << ")\n";
  return 0;
}

int make_examples_main(int argc, char **argv) {
  std::string ref;
  std::string reads;
  std::string out;
  int nthreads = 1;
  int decode_threads = 2;
  int64_t region_bp = 1000;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--ref") {
      ref = need("--ref");
    } else if (arg == "--reads") {
      reads = need("--reads");
    } else if (arg == "--out") {
      out = need("--out");
    } else if (arg == "--nthreads") {
      nthreads = std::atoi(need("--nthreads"));
    } else if (arg == "--decode-threads") {
      decode_threads = std::atoi(need("--decode-threads"));
    } else if (arg == "--region-bp") {
      region_bp = std::atoll(need("--region-bp"));
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage();
      return 2;
    }
  }
  if (ref.empty() || reads.empty() || out.empty()) {
    usage();
    return 2;
  }

  rodv::make_examples::ReferenceCache cache;
  cache.start_fasta(ref);
  rodv::make_examples::HtsReadSource source(reads, decode_threads, ref);
  rodv::make_examples::TsvExampleSink sink(out);
  rodv::make_examples::PipelineConfig config;
  config.nthreads = nthreads;
  config.region_bp = region_bp;
  rodv::make_examples::MakeExamplesPipeline pipeline(config, nullptr);
  pipeline.run(source, cache, sink);
  cache.join();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  if (std::strcmp(argv[1], "call-variants-plan") == 0) {
    return plan_main(argc, argv);
  }
  if (std::strcmp(argv[1], "preprocess-bench") == 0) {
    return bench_main();
  }
  if (std::strcmp(argv[1], "make-examples") == 0) {
    return make_examples_main(argc, argv);
  }
  usage();
  return 2;
}
