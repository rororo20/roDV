#include "pairhmm_unittest.cpp" // 复用 TestCaseData 结构体
#include "assemble_log_parser.h" // 使用独立的解析器
#include "../pairhmm_schedule.h"
#include "../pairhmm/common/cpu_features.h"
#include "../pairhmm/intra/pairhmm_api.h"
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

using namespace pairhmm::test;
using namespace pairhmm::schedule;
using namespace pairhmm::intra;
using namespace pairhmm::common;
using pairhmm::common::CpuFeatures;

/**
 * @brief 测试从assemble日志文件读取并验证schedule_pairhmm
 */
TEST(AssembleLogParserTest, ParseAndTestSchedule) {
  using namespace pairhmm::test;
  
  // 1. 解析日志文件
  // 尝试多个可能的路径
  std::vector<std::string> possible_paths = {
    "resouces/pairhmm_debug.txt",
    "../resouces/pairhmm_debug.txt",
    "../../resouces/pairhmm_debug.txt",
    "resouces/pairhmm_debug.txt"
  };
  
  std::string log_file;
  std::ifstream test_file;
  for (const auto& path : possible_paths) {
    test_file.open(path);
    if (test_file.is_open()) {
      log_file = path;
      test_file.close();
      break;
    }
  }
  
  if (log_file.empty()) {
    GTEST_SKIP() << "Cannot find log file in any of the expected locations";
    return;
  }
  
  std::vector<ParsedRegion> regions;
  
  try {
    regions = AssembleLogParser::parseLogFile(log_file);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Cannot parse log file: " << e.what();
    return;
  }
  
  ASSERT_GT(regions.size(), 0) << "No regions found in log file";
  
  // 2. 对每个region进行测试
  for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx) {
    const auto& region = regions[region_idx];
    
    SCOPED_TRACE("Region: " + region.region_str + " (index: " + 
                 std::to_string(region_idx) + ")");
    
    const size_t M = region.haplotypes.size();
    const size_t N = region.reads.size();
    
    ASSERT_GT(M, 0) << "No haplotypes in region";
    ASSERT_GT(N, 0) << "No reads in region";
    
    // 3. 使用 intra::computeLikelihoods 计算期望值
    std::vector<std::vector<double>> expected_results(M);
    
    for (size_t h = 0; h < M; ++h) {
      expected_results[h].resize(N);
      for (size_t r = 0; r < N; ++r) {
        // 构建 TestCaseData
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
          // 如果质量值长度不匹配，跳过这个read
          expected_results[h][r] = 0.0;
          continue;
        }
        
        // 使用 TestCaseWrapper 构建 TestCase
        TestCaseWrapper<64> wrapper(data);
        const TestCase& tc = wrapper.getTestCase();
        
        // 根据CPU特性选择API
        if (CpuFeatures::hasAVX512Support()) {
          expected_results[h][r] = computeLikelihoodsAVX512(tc, false);
        } else if (CpuFeatures::hasAVX2Support()) {
          expected_results[h][r] = computeLikelihoodsAVX2(tc, false);
        } else {
          GTEST_SKIP() << "CPU does not support AVX2 or AVX512";
          return;
        }
        
        // 验证结果有效性
        ASSERT_FALSE(std::isnan(expected_results[h][r])) 
            << "Expected result is NaN for H" << h << "_R" << r;
        ASSERT_FALSE(std::isinf(expected_results[h][r])) 
            << "Expected result is Inf for H" << h << "_R" << r;
      }
    }
    
    // 4. 准备 schedule_pairhmm 的输入数据
    std::vector<std::vector<uint8_t>> hap_vecs, read_vecs;
    std::vector<std::vector<uint8_t>> qual_vecs, ins_vecs, del_vecs, gcp_vecs;
    
    // 转换单倍型
    hap_vecs.reserve(M);
    for (const auto& hap : region.haplotypes) {
      hap_vecs.emplace_back(hap.begin(), hap.end());
    }
    
    // 转换reads和质量值
    read_vecs.reserve(N);
    qual_vecs.reserve(N);
    ins_vecs.reserve(N);
    del_vecs.reserve(N);
    gcp_vecs.reserve(N);
    
    for (const auto& read : region.reads) {
      read_vecs.emplace_back(read.sequence.begin(), read.sequence.end());
      qual_vecs.push_back(read.read_qual);
      ins_vecs.push_back(read.read_ins_qual);
      del_vecs.push_back(read.read_del_qual);
      gcp_vecs.push_back(read.gcp);
    }
    
    // 5. 调用 schedule_pairhmm
    std::vector<std::vector<double>> schedule_results;
    bool success = schedule_pairhmm(
        hap_vecs, read_vecs, schedule_results,
        qual_vecs, ins_vecs, del_vecs, gcp_vecs,
        false,  // use_double
        0.5,    // max_idle_ratio_float
        0.7,    // max_idle_ratio_double
        false   // verbose
    );
    
    ASSERT_TRUE(success) << "schedule_pairhmm failed for region " << region_idx;
    ASSERT_EQ(schedule_results.size(), M) 
        << "Result size mismatch (M) for region " << region_idx;
    
    // 6. 比较结果，误差 1e-5
    for (size_t h = 0; h < M; ++h) {
      ASSERT_EQ(schedule_results[h].size(), N) 
          << "Result size mismatch (N) for hap " << h << " in region " << region_idx;
      
      for (size_t r = 0; r < N; ++r) {
        // 验证结果有效性
        ASSERT_FALSE(std::isnan(schedule_results[h][r])) 
            << "Schedule result is NaN for H" << h << "_R" << r 
            << " in region " << region_idx;
        ASSERT_FALSE(std::isinf(schedule_results[h][r])) 
            << "Schedule result is Inf for H" << h << "_R" << r 
            << " in region " << region_idx;
        
        // 如果期望值为0（可能是质量值长度不匹配导致的），跳过比较
        if (expected_results[h][r] == 0.0) {
          continue;
        }
        
        // 使用 ASSERT_NEAR 比较结果，误差 1e-5
        ASSERT_NEAR(schedule_results[h][r], expected_results[h][r], 1e-5)
            << "Mismatch for H" << h << "_R" << r 
            << " in region " << region_idx 
            << " (" << region.region_str << ")"
            << "\nExpected: " << expected_results[h][r]
            << "\nGot: " << schedule_results[h][r]
            << "\nDifference: " << std::abs(schedule_results[h][r] - expected_results[h][r]);
      }
    }
  }
}


