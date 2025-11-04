#ifndef ASSEMBLE_LOG_PARSER_H_
#define ASSEMBLE_LOG_PARSER_H_

#include <string>
#include <vector>
#include <cstdint>

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
  std::string region_str;                // 区域字符串，如 "chr1:1000-2000"
  std::vector<std::string> haplotypes;  // 单倍型序列列表
  std::vector<ParsedRead> reads;        // Reads列表
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
  static std::vector<ParsedRegion> parseLogFile(const std::string& filename);

  /**
   * @brief 查找所有匹配的日志文件
   * @param directory 目录路径
   * @param pattern 文件名模式（默认: pairhmm_debug_t*.log）
   * @return 文件路径列表
   */
  static std::vector<std::string> findLogFiles(
      const std::string& directory = ".",
      const std::string& pattern = "pairhmm_debug_t");

  /**
   * @brief 解析质量值行
   * 格式: "  read-qual: 30 31 32 33 ..."
   */
  static std::vector<uint8_t> parseQualityLine(const std::string& line);
};

} // namespace test
} // namespace pairhmm

#endif // ASSEMBLE_LOG_PARSER_H_

