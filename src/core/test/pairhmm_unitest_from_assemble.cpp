#include "pairhmm_unittest.cpp" // 复用 TestCaseData 结构体
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

namespace pairhmm {
namespace test {

/**
 * @brief 解析日志文件中的单个Read信息
 */
struct ParsedRead {
  std::string sequence;
  std::vector<uint8_t> read_qual;
  std::vector<uint8_t> read_ins_qual;
  std::vector<uint8_t> read_del_qual;
  std::vector<uint8_t> gcp;
};

/**
 * @brief 解析日志文件中的单个区域信息
 */
struct ParsedRegion {
  std::string region_str;           // 区域字符串，如 "chr1:1000-2000"
  std::vector<std::string> haplotypes;  // 单倍型序列列表
  std::vector<ParsedRead> reads;     // Reads列表
};

/**
 * @brief 从日志文件解析器
 * 
 * 解析格式：
 * === Region: chr1:1000-2000 ===
 * Haplotypes: 3
 * Reads: 5
 * 
 * H0: ACGTACGT...
 * H1: TGCATGCAT...
 * 
 * R0: ACGTACGT...
 *   read-qual: 30 31 32 33 ...
 *   read-ins-qual: 20 21 22 23 ...
 *   read-del-qual: 25 26 27 28 ...
 *   gcp: 15 16 17 18 ...
 */
class AssembleLogParser {
public:
  /**
   * @brief 从单个日志文件解析所有区域
   * @param filename 日志文件名
   * @return 解析出的区域列表
   */
  static std::vector<ParsedRegion> parseLogFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open log file: " + filename);
    }

    std::vector<ParsedRegion> regions;
    std::string line;
    bool line_available = false;
    
    while (line_available || std::getline(file, line)) {
      line_available = false;
      
      // 查找区域开始标记
      if (line.find("=== Region:") != std::string::npos) {
        ParsedRegion region;
        
        // 解析区域字符串
        std::regex region_regex(R"(=== Region: (.+) ==)");
        std::smatch match;
        if (std::regex_search(line, match, region_regex)) {
          region.region_str = match[1].str();
        }
        
        // 读取 Haplotypes 和 Reads 数量
        int haplotype_count = 0;
        int read_count = 0;
        
        if (std::getline(file, line)) {
          std::regex hap_regex(R"(Haplotypes: (\d+))");
          if (std::regex_search(line, match, hap_regex)) {
            haplotype_count = std::stoi(match[1].str());
          }
        }
        
        if (std::getline(file, line)) {
          std::regex read_regex(R"(Reads: (\d+))");
          if (std::regex_search(line, match, read_regex)) {
            read_count = std::stoi(match[1].str());
          }
        }
        
        // 跳过空行
        std::getline(file, line);
        
        // 解析单倍型
        for (int i = 0; i < haplotype_count; ++i) {
          if (std::getline(file, line)) {
            // 格式: H0: ACGTACGT...
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
              std::string hap_seq = line.substr(colon_pos + 1);
              // 去除前导空格
              hap_seq.erase(0, hap_seq.find_first_not_of(" \t"));
              region.haplotypes.push_back(hap_seq);
            }
          }
        }
        
        // 跳过空行
        std::getline(file, line);
        
        // 解析Reads
        for (int i = 0; i < read_count; ++i) {
          ParsedRead read;
          
          // 读取序列行: R0: ACGTACGT...
          if (!std::getline(file, line)) {
            break; // 文件结束
          }
          
          size_t colon_pos = line.find(':');
          if (colon_pos != std::string::npos) {
            read.sequence = line.substr(colon_pos + 1);
            read.sequence.erase(0, read.sequence.find_first_not_of(" \t"));
          }
          
          // 读取质量值行（每个read有4行质量值：read-qual, read-ins-qual, read-del-qual, gcp）
          int quality_lines_read = 0;
          while (quality_lines_read < 4 && std::getline(file, line)) {
            // 如果遇到下一个区域标记，需要停止解析（但外层循环会继续）
            if (line.find("=== Region:") != std::string::npos) {
              // 遇到下一个区域，当前read可能不完整，但仍然保存已解析的部分
              break;
            }
            
            // 跳过空行
            if (line.empty()) {
              continue;
            }
            
            // 解析 read-qual
            if (line.find("read-qual:") != std::string::npos) {
              read.read_qual = parseQualityLine(line);
              quality_lines_read++;
            }
            // 解析 read-ins-qual
            else if (line.find("read-ins-qual:") != std::string::npos) {
              read.read_ins_qual = parseQualityLine(line);
              quality_lines_read++;
            }
            // 解析 read-del-qual
            else if (line.find("read-del-qual:") != std::string::npos) {
              read.read_del_qual = parseQualityLine(line);
              quality_lines_read++;
            }
            // 解析 gcp
            else if (line.find("gcp:") != std::string::npos) {
              read.gcp = parseQualityLine(line);
              quality_lines_read++;
              // gcp是最后一个质量值，读取完成后应该跳到下一个read或区域
              // 不需要额外读取，让外层循环继续
              break;
            }
          }
          
          region.reads.push_back(std::move(read));
          
          // 检查是否遇到下一个区域标记（在读取gcp后可能已经读取了下一行）
          if (line.find("=== Region:") != std::string::npos) {
            // 标记这一行可用，让外层循环继续处理
            line_available = true;
            break;
          }
        }
        
        regions.push_back(std::move(region));
        
        // 如果遇到下一个区域标记，继续外层循环处理
        if (line_available) {
          continue;
        }
      }
    }
    
    return regions;
  }
  
  /**
   * @brief 查找所有匹配的日志文件
   * @param directory 目录路径
   * @param pattern 文件名模式（默认: pairhmm_debug_t*.log）
   * @return 文件路径列表
   */
  static std::vector<std::string> findLogFiles(
      const std::string& directory = ".",
      const std::string& pattern = "pairhmm_debug_t") {
    std::vector<std::string> files;
    
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
      throw std::runtime_error("Error opening directory: " + directory);
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      // 跳过 . 和 ..
      if (entry->d_name[0] == '.') {
        continue;
      }
      
      std::string filename(entry->d_name);
      // 检查文件名是否匹配模式
      if (filename.find(pattern) != std::string::npos && 
          filename.find(".log") != std::string::npos) {
        // 构建完整路径
        std::string full_path = directory;
        if (!directory.empty() && directory.back() != '/') {
          full_path += "/";
        }
        full_path += filename;
        
        // 检查是否为普通文件
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
          files.push_back(full_path);
        }
      }
    }
    
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
  }
  
  /**
   * @brief 将解析的区域转换为 TestCaseData 列表
   * 为每个 (haplotype, read) 对生成一个 TestCaseData
   * @param region 解析的区域
   * @param expected_results 可选的期望结果矩阵 [haplotype_idx][read_idx]
   * @return TestCaseData 列表
   */
  static std::vector<TestCaseData> convertToTestCaseData(
      const ParsedRegion& region,
      const std::vector<std::vector<double>>* expected_results = nullptr) {
    std::vector<TestCaseData> test_cases;
    
    for (size_t h = 0; h < region.haplotypes.size(); ++h) {
      for (size_t r = 0; r < region.reads.size(); ++r) {
        TestCaseData data;
        data.hap_bases = region.haplotypes[h];
        data.read_bases = region.reads[r].sequence;
        data.read_qual = region.reads[r].read_qual;
        data.read_ins_qual = region.reads[r].read_ins_qual;
        data.read_del_qual = region.reads[r].read_del_qual;
        data.gcp = region.reads[r].gcp;
        
        // 设置期望结果（如果有提供）
        if (expected_results && 
            h < expected_results->size() && 
            r < (*expected_results)[h].size()) {
          data.expected_result = (*expected_results)[h][r];
        } else {
          data.expected_result = 0.0; // 默认值，后续可以通过计算得到
        }
        
        data.line_number = 0; // 从日志文件解析，行号不太有意义
        
        test_cases.push_back(data);
      }
    }
    
    return test_cases;
  }

  /**
   * @brief 解析质量值行（公有方法，供测试使用）
   * 格式: "  read-qual: 30 31 32 33 ..."
   */
  static std::vector<uint8_t> parseQualityLine(const std::string& line);
};

} // namespace test

// 实现 parseQualityLine（需要在类外定义）
std::vector<uint8_t> pairhmm::test::AssembleLogParser::parseQualityLine(const std::string& line) {
  std::vector<uint8_t> qualities;
  
  // 找到冒号位置
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos) {
    return qualities;
  }
  
  // 提取冒号后的内容
  std::string qual_str = line.substr(colon_pos + 1);
  
  // 使用字符串流解析数字
  std::istringstream iss(qual_str);
  int qual_value;
  while (iss >> qual_value) {
    if (qual_value >= 0 && qual_value <= 255) {
      qualities.push_back(static_cast<uint8_t>(qual_value));
    }
  }
  
  return qualities;
}
} // namespace pairhmm

// 测试用例：从assemble日志文件读取并测试
#include "../pairhmm_schedule.h"
#include "../pairhmm/common/cpu_features.h"
#include "../pairhmm/intra/pairhmm_api.h"
#include <cmath>

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


