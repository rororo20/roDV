#include "assemble_log_parser.h"
#include "test_case_common.h"

#include "../pairhmm/common/cpu_features.h"
#include "../pairhmm/intra/pairhmm_api.h"
#include "../pairhmm_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace pairhmm::test;
using namespace pairhmm::schedule;
using namespace pairhmm::intra;
using namespace pairhmm::common;
using pairhmm::common::CpuFeatures;

/**
 * @brief 高精度时间统计
 */
class HighPrecisionTimer {
public:
  void start() { start_time_ = std::chrono::high_resolution_clock::now(); }

  void stop() { end_time_ = std::chrono::high_resolution_clock::now(); }

  double elapsedSeconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time_ - start_time_);
    return duration.count() / 1e9;
  }

  double elapsedMilliseconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time_ - start_time_);
    return duration.count() / 1e6;
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
};

/**
 * @brief 使用 intra::computeLikelihoods 计算并统计时间
 * 包括数据解析时间
 */
double computeWithIntra(const ParsedRegion &region,
                        std::vector<std::vector<double>> &results) {
  HighPrecisionTimer timer;
  timer.start();

  const size_t M = region.haplotypes.size();
  const size_t N = region.reads.size();

  results.resize(M);

  for (size_t h = 0; h < M; ++h) {
    results[h].resize(N);
    for (size_t r = 0; r < N; ++r) {
      // 构建 TestCaseData（数据解析）
      TestCaseData data;
      data.hap_bases = region.haplotypes[h];
      data.read_bases = region.reads[r].sequence;
      data.read_qual = region.reads[r].read_qual;
      data.read_ins_qual = region.reads[r].read_ins_qual;
      data.read_del_qual = region.reads[r].read_del_qual;
      data.gcp = region.reads[r].gcp;

      // 验证质量值长度匹配
      size_t read_len = data.read_bases.length();
      if (data.read_qual.size() != read_len ||
          data.read_ins_qual.size() != read_len ||
          data.read_del_qual.size() != read_len ||
          data.gcp.size() != read_len) {
        results[h][r] = 0.0;
        continue;
      }

      // 使用 TestCaseWrapper 构建 TestCase
      TestCaseWrapper<64> wrapper(data);
      const TestCase &tc = wrapper.getTestCase();

      // 根据CPU特性选择API并计算
      if (CpuFeatures::hasAVX512Support()) {
        results[h][r] = computeLikelihoodsAVX512(tc, false);
      } else if (CpuFeatures::hasAVX2Support()) {
        results[h][r] = computeLikelihoodsAVX2(tc, false);
      } else {
        std::cerr << "Error: CPU does not support AVX2 or AVX512" << std::endl;
        results[h][r] = 0.0;
      }
    }
  }

  timer.stop();
  return timer.elapsedSeconds();
}

/**
 * @brief 使用 schedule_pairhmm 计算并统计时间
 * 包括数据解析时间
 * @param region 解析的区域
 * @param results 输出结果
 * @param max_idle_ratio_float float 精度最大 idle 比例
 * @param max_idle_ratio_double double 精度最大 idle 比例
 * @param verbose 是否输出详细信息
 * @param data_prep_time 输出参数：数据准备时间（秒）
 * @return 总执行时间（秒），失败返回 -1.0
 */
double computeWithSchedule(const ParsedRegion &region,
                           std::vector<std::vector<double>> &results,
                           double max_idle_ratio_float,
                           double max_idle_ratio_double, bool verbose,
                           double &data_prep_time) {
  HighPrecisionTimer total_timer;
  HighPrecisionTimer prep_timer;

  total_timer.start();
  prep_timer.start();

  const size_t M = region.haplotypes.size();
  const size_t N = region.reads.size();

  // 数据解析：转换为 schedule_pairhmm 的输入格式
  std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
  std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;

  hap_vecs.reserve(M);
  for (const auto &hap : region.haplotypes) {
    hap_vecs.emplace_back(hap.begin(), hap.end());
  }

  read_vecs.reserve(N);
  qual_vecs.reserve(N);
  ins_vecs.reserve(N);
  del_vecs.reserve(N);
  gcp_vecs.reserve(N);

  for (const auto &read : region.reads) {
    read_vecs.emplace_back(read.sequence.begin(), read.sequence.end());
    qual_vecs.push_back(read.read_qual);
    ins_vecs.push_back(read.read_ins_qual);
    del_vecs.push_back(read.read_del_qual);
    gcp_vecs.push_back(read.gcp);
  }

  prep_timer.stop();
  data_prep_time = prep_timer.elapsedSeconds();

  // 调用 schedule_pairhmm
  bool success = schedule_pairhmm(hap_vecs, read_vecs, results, qual_vecs,
                                  ins_vecs, del_vecs, gcp_vecs,
                                  false, // use_double
                                  max_idle_ratio_float, max_idle_ratio_double,
                                  verbose // verbose
  );

  total_timer.stop();

  if (!success) {
    std::cerr << "Error: schedule_pairhmm failed" << std::endl;
    return -1.0;
  }

  return total_timer.elapsedSeconds();
}

/**
 * @brief 检查序列是否包含 N 碱基
 */
bool containsN(const std::string &sequence) {
  for (char c : sequence) {
    if (c == 'N' || c == 'n') {
      return true;
    }
  }
  return false;
}

/**
 * @brief 比较两个结果矩阵，检查误差
 * @param results1 第一个结果矩阵
 * @param results2 第二个结果矩阵
 * @param region 区域数据（用于检查 reads 是否包含 N）
 * @param tolerance 误差容忍度
 * @param ignore_n_base 如果为 true，跳过包含 N 碱基的 reads 的比较
 */
bool compareResults(const std::vector<std::vector<double>> &results1,
                    const std::vector<std::vector<double>> &results2,
                    const ParsedRegion &region, double tolerance,
                    bool ignore_n_base) {
  if (results1.size() != results2.size()) {
    std::cerr << "Error: Result size mismatch (M)" << std::endl;
    return false;
  }

  for (size_t h = 0; h < results1.size(); ++h) {
    if (results1[h].size() != results2[h].size()) {
      std::cerr << "Error: Result size mismatch (N) for hap " << h << std::endl;
      return false;
    }
    if(ignore_n_base && containsN(region.haplotypes[h])) {
      continue;
    }

    for (size_t r = 0; r < results1[h].size(); ++r) {
      // 如果启用 ignore_n_base，且 read 包含 N 碱基，则跳过
      if (ignore_n_base && r < region.reads.size()) {
        if (containsN(region.reads[r].sequence)) {
          continue;
        }
      }

      // 跳过无效值
      if (std::isnan(results1[h][r]) || std::isnan(results2[h][r]) ||
          std::isinf(results1[h][r]) || std::isinf(results2[h][r])) {
        continue;
      }

      // 如果期望值为0，跳过比较
      if (std::abs(results1[h][r]) < 1e-10) {
        continue;
      }

      double diff = std::abs(results1[h][r] - results2[h][r]);
      if (diff > tolerance) {
        std::cerr << "Error: Result mismatch for H" << h << "_R" << r
                  << "\n  Intra result: " << results1[h][r]
                  << "\n  Schedule result: " << results2[h][r]
                  << "\n  Difference: " << diff
                  << "\n  Tolerance: " << tolerance << std::endl;
        return false;
      }
    }
  }

  return true;
}

/**
 * @brief 打印使用说明
 */
void printUsage(const char *program_name) {
  std::cout
      << "Usage: " << program_name
      << " <log_file> <max_idle_ratio_float> <max_idle_ratio_double> [OPTIONS]"
      << std::endl;
  std::cout << "Arguments:" << std::endl;
  std::cout << "  log_file: Path to the log file containing regions"
            << std::endl;
  std::cout << "  max_idle_ratio_float: Maximum idle ratio for float precision "
               "(e.g., 0.5)"
            << std::endl;
  std::cout << "  max_idle_ratio_double: Maximum idle ratio for double "
               "precision (e.g., 0.7)"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -v, --verbose: Enable verbose output for schedule_pairhmm"
            << std::endl;
  std::cout
      << "  -i, --ignore-N-base: Skip comparison for reads containing N bases"
      << std::endl;
  std::cout << "  -h, --help: Show this help message" << std::endl;
}

int main(int argc, char *argv[]) {
  // 定义长选项
  static struct option long_options[] = {{"verbose", no_argument, 0, 'v'},
                                         {"ignore-N-base", no_argument, 0, 'i'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  // 1. 解析命令行参数
  bool verbose = false;
  bool ignore_n_base = false;
  std::string log_file;
  double max_idle_ratio_float = 0.0;
  double max_idle_ratio_double = 0.0;

  int opt;
  int option_index = 0;

  // 先解析选项
  while ((opt = getopt_long(argc, argv, "vih", long_options, &option_index)) !=
         -1) {
    switch (opt) {
    case 'v':
      verbose = true;
      break;
    case 'i':
      ignore_n_base = true;
      break;
    case 'h':
      printUsage(argv[0]);
      return 0;
    case '?':
      // getopt 已经打印了错误信息
      printUsage(argv[0]);
      return 1;
    default:
      printUsage(argv[0]);
      return 1;
    }
  }

  // 检查必需参数
  if (optind + 3 > argc) {
    std::cerr << "Error: Missing required arguments" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // 解析必需参数
  log_file = argv[optind];
  try {
    max_idle_ratio_float = std::stod(argv[optind + 1]);
    max_idle_ratio_double = std::stod(argv[optind + 2]);
  } catch (const std::exception &e) {
    std::cerr << "Error: Invalid number format: " << e.what() << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  // 检查是否有额外的未知参数
  if (optind + 3 < argc) {
    std::cerr << "Error: Unexpected argument: " << argv[optind + 3]
              << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  init_native();

  std::cout << "==========================================" << std::endl;
  std::cout << "PairHMM Performance Test" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Log file: " << log_file << std::endl;
  std::cout << "Max idle ratio (float): " << max_idle_ratio_float << std::endl;
  std::cout << "Max idle ratio (double): " << max_idle_ratio_double
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  Verbose: " << (verbose ? "Yes" : "No") << std::endl;
  std::cout << "  Ignore N-base: " << (ignore_n_base ? "Yes" : "No")
            << std::endl;
  std::cout << "CPU Features:" << std::endl;
  std::cout << "  AVX2: " << (CpuFeatures::hasAVX2Support() ? "Yes" : "No")
            << std::endl;
  std::cout << "  AVX512: " << (CpuFeatures::hasAVX512Support() ? "Yes" : "No")
            << std::endl;
  std::cout << "==========================================" << std::endl;

  // 检查CPU支持
  if (!CpuFeatures::hasAVX2Support() && !CpuFeatures::hasAVX512Support()) {
    std::cerr << "Error: CPU does not support AVX2 or AVX512" << std::endl;
    return 1;
  }

  // 2. 解析日志文件
  std::vector<ParsedRegion> regions;
  try {
    regions = AssembleLogParser::parseLogFile(log_file);
  } catch (const std::exception &e) {
    std::cerr << "Error: Cannot parse log file: " << e.what() << std::endl;
    return 1;
  }

  if (regions.empty()) {
    std::cerr << "Error: No regions found in log file" << std::endl;
    return 1;
  }

  std::cout << "Parsed " << regions.size() << " region(s)" << std::endl;
  std::cout << "==========================================" << std::endl;

  // 3. 统计总时间
  double total_intra_time = 0.0;
  double total_schedule_time = 0.0;
  double total_schedule_data_prep_time = 0.0; // 数据准备总时间
  size_t total_pairs = 0;
  bool all_passed = true;

  // 4. 循环执行测试
  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    const auto &region = regions[region_idx];
    std::cout << "\nRegion " << (region_idx + 1) << "/" << regions.size()
              << ": " << region.region_str << std::endl;
    std::cout << "  Haplotypes: " << region.haplotypes.size() << std::endl;
    std::cout << "  Reads: " << region.reads.size() << std::endl;

    const size_t M = region.haplotypes.size();
    const size_t N = region.reads.size();
    total_pairs += M * N;

    // 使用 intra::computeLikelihoods
    std::vector<std::vector<double>> intra_results;
    double intra_time = computeWithIntra(region, intra_results);
    total_intra_time += intra_time;

    std::cout << "  Intra time: " << std::fixed << std::setprecision(6)
              << intra_time << " seconds" << std::endl;

    // 使用 schedule_pairhmm
    std::vector<std::vector<double>> schedule_results;
    double schedule_data_prep_time = 0.0;
    double schedule_time = computeWithSchedule(
        region, schedule_results, max_idle_ratio_float, max_idle_ratio_double,
        verbose, schedule_data_prep_time);

    if (schedule_time < 0.0) {
      std::cerr << "  Error: schedule_pairhmm failed" << std::endl;
      all_passed = false;
      continue;
    }

    total_schedule_time += schedule_time;
    total_schedule_data_prep_time += schedule_data_prep_time;

    std::cout << "  Schedule time: " << std::fixed << std::setprecision(6)
              << schedule_time << " seconds" << std::endl;
    std::cout << "    - Data prep time: " << std::fixed << std::setprecision(6)
              << schedule_data_prep_time << " seconds" << std::endl;
    std::cout << "    - Compute time: " << std::fixed << std::setprecision(6)
              << (schedule_time - schedule_data_prep_time) << " seconds"
              << std::endl;

    // 5. 比较结果（误差不超过 1e-5）
    const double tolerance = 1e-5;
    bool passed = compareResults(intra_results, schedule_results, region,
                                 tolerance, ignore_n_base);

    if (passed) {
      std::cout << "  Result check: PASSED (tolerance: " << tolerance << ")"
                << std::endl;
    } else {
      std::cout << "  Result check: FAILED (tolerance: " << tolerance << ")"
                << std::endl;
      all_passed = false;
      break;
    }

    // 计算加速比
    if (schedule_time > 0.0) {
      double speedup = intra_time / schedule_time;
      std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                << speedup << "x" << std::endl;
    }
  }

  // 6. 打印总结
  std::cout << "\n==========================================" << std::endl;
  std::cout << "Summary" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Total regions: " << regions.size() << std::endl;
  std::cout << "Total pairs (M*N): " << total_pairs << std::endl;
  std::cout << "Total intra time: " << std::fixed << std::setprecision(6)
            << total_intra_time << " seconds" << std::endl;
  std::cout << "Total schedule time: " << std::fixed << std::setprecision(6)
            << total_schedule_time << " seconds" << std::endl;
  std::cout << "  - Total data prep time: " << std::fixed
            << std::setprecision(6) << total_schedule_data_prep_time
            << " seconds" << std::endl;
  std::cout << "  - Total compute time: " << std::fixed << std::setprecision(6)
            << (total_schedule_time - total_schedule_data_prep_time)
            << " seconds" << std::endl;

  if (total_schedule_time > 0.0) {
    double overall_speedup = total_intra_time / total_schedule_time;
    std::cout << "Overall speedup: " << std::fixed << std::setprecision(2)
              << overall_speedup << "x" << std::endl;

    // 计算数据准备时间占比
    double data_prep_ratio =
        (total_schedule_data_prep_time / total_schedule_time) * 100.0;
    std::cout << "Data prep time ratio: " << std::fixed << std::setprecision(2)
              << data_prep_ratio << "%" << std::endl;
  }

  std::cout << "Result check: " << (all_passed ? "PASSED" : "FAILED")
            << std::endl;
  std::cout << "==========================================" << std::endl;

  return all_passed ? 0 : 1;
}
