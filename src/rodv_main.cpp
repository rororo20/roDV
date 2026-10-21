#include "launch_plan.h"
#include "preprocess.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage:\n"
      << "  rodv_main call-variants-plan [--height N --width N --channels N]\n"
      << "  rodv_main preprocess-bench\n";
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
  usage();
  return 2;
}
