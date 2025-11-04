#include "assemble_log_parser.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

namespace pairhmm {
namespace test {

std::vector<ParsedRegion> AssembleLogParser::parseLogFile(const std::string& filename) {
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

std::vector<std::string> AssembleLogParser::findLogFiles(
    const std::string& directory,
    const std::string& pattern) {
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

std::vector<uint8_t> AssembleLogParser::parseQualityLine(const std::string& line) {
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

} // namespace test
} // namespace pairhmm

