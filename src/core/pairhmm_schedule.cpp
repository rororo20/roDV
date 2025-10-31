#include "pairhmm_schedule.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace pairhmm {
namespace schedule {

// 辅助结构：表示一个单倍型-read对
struct HapReadPair {
  size_t hap_idx;
  size_t read_idx;
  uint32_t hap_len;
  uint32_t read_len;
  bool used;

  HapReadPair(size_t h_idx, size_t r_idx, uint32_t h_len, uint32_t r_len)
      : hap_idx(h_idx), read_idx(r_idx), hap_len(h_len), read_len(r_len),
        used(false) {}
};

// 辅助结构：表示一组
struct Group {
  std::vector<size_t> pair_indices; // 指向pairs数组的索引
  uint32_t max_hap_len = 0;
  uint32_t max_read_len = 0;
  double idle_ratio = 0.0;
};

// 贪心分组算法
template <uint32_t SimdWidth>
std::vector<Group> greedy_grouping(std::vector<HapReadPair> &pairs,
                                   double max_idle_ratio) {
  std::vector<Group> groups;
  if (SimdWidth == 0) {
    return groups;
  }

  const size_t total = pairs.size();
  for (size_t i = 0; i < total; ++i) {
    if (pairs[i].used)
      continue;

    Group group;
    group.pair_indices.reserve(SimdWidth);

    size_t idx = i;
    while (idx < total && group.pair_indices.size() < SimdWidth) {
      if (!pairs[idx].used) {
        group.pair_indices.push_back(idx);
      }
      ++idx;
    }

    if (group.pair_indices.size() < SimdWidth) {
      break;
    }

    uint32_t max_hap_len = 0;
    uint32_t max_read_len = 0;
    uint64_t sum_elements = 0;
    for (size_t pair_idx : group.pair_indices) {
      const auto &pair = pairs[pair_idx];
      max_hap_len = std::max(max_hap_len, pair.hap_len);
      max_read_len = std::max(max_read_len, pair.read_len);
      sum_elements += static_cast<uint64_t>(pair.hap_len) * pair.read_len;
    }

    uint64_t total_capacity =
        static_cast<uint64_t>(max_hap_len) * max_read_len * SimdWidth;
    uint64_t idle =
        total_capacity > sum_elements ? total_capacity - sum_elements : 0;
    double idle_ratio =
        total_capacity == 0
            ? 0.0
            : static_cast<double>(idle) / static_cast<double>(total_capacity);

    group.max_hap_len = max_hap_len;
    group.max_read_len = max_read_len;
    group.idle_ratio = idle_ratio;

    if (idle_ratio <= max_idle_ratio) {
      for (size_t pair_idx : group.pair_indices) {
        pairs[pair_idx].used = true;
      }
      groups.push_back(std::move(group));
    }
  }

  return groups;
}

bool schedule_pairhmm(
    const std::vector<std::vector<uint8_t>> &haplotypes,
    const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result,
    const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities,
    const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities,
    bool use_double) {

  const size_t M = haplotypes.size();
  const size_t N = reads.size();

  if (M == 0 || N == 0) {
    return false;
  }

  // 初始化结果矩阵
  result.resize(M);
  for (size_t i = 0; i < M; ++i) {
    result[i].resize(N, 0.0);
  }

  // 生成所有单倍型-read对
  std::vector<HapReadPair> pairs;
  pairs.reserve(M * N);
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      pairs.emplace_back(h, r, static_cast<uint32_t>(haplotypes[h].size()),
                         static_cast<uint32_t>(reads[r].size()));
    }
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const HapReadPair &lhs, const HapReadPair &rhs) {
              if (lhs.hap_len == rhs.hap_len) {
                return lhs.read_len < rhs.read_len;
              }
              return lhs.hap_len < rhs.hap_len;
            });

  // 定义阈值
  const double max_idle_ratio_float = 0.5; // float类型允许的最大idle比例（50%）
  const double max_idle_ratio_double =
      0.7; // double类型允许的最大idle比例（70%）

  // 根据编译选项确定SIMD宽度
#if defined(__AVX512F__)
  constexpr uint32_t float_simd_width = 16;
  constexpr uint32_t double_simd_width = 8;
#elif defined(__AVX2__)
  constexpr uint32_t float_simd_width = 8;
  constexpr uint32_t double_simd_width = 4;
#else
  // 不支持SIMD，全部用intra
  constexpr uint32_t float_simd_width = 0;
  constexpr uint32_t double_simd_width = 0;
#endif

  // 标记哪些对已经被处理
  std::vector<bool> processed_float(M * N, false);
  std::vector<bool> processed_double(M * N, false);

  // 第一步：尝试float类型分组（如果未强制使用double且支持SIMD）
  if (!use_double && float_simd_width > 0) {
    // 确保所有 Pair 的 used 标记为 false，以便重新分组
    for (auto &pair : pairs) {
      pair.used = false;
    }

    auto float_groups =
        greedy_grouping<float_simd_width>(pairs, max_idle_ratio_float);

    // 处理所有满足条件的float组
    for (const auto &group : float_groups) {
      // TODO: 调用inter float计算
      // 说明：group.pair_indices是pairs中的索引，可以直接使用
      // 1. 生成TestCase数组：
      //    -
      //    遍历group.pair_indices，对每个idx，使用pairs[idx]获取hap_idx和read_idx
      //    - 从haplotypes[hap_idx], reads[read_idx],
      //    quality[read_idx]等构建TestCase
      //    - 需要对齐内存分配（使用_mm_malloc）
      // 2. 调用inter API（根据编译选项选择）：
      //    - #if defined(__AVX512F__) -> compute_inter_pairhmm_AVX512_float
      //    - #elif defined(__AVX2__) -> compute_inter_pairhmm_AVX2_float

      for (size_t idx : group.pair_indices) {
        // TODO:
        //  将结果写入result矩阵更新：
        //    -
        //    遍历results数组，写入result[pairs[group.pair_indices[i]].hap_idx][pairs[group.pair_indices[i]].read_idx]
        const auto &pair = pairs[idx];
        if (0) {
          // 满足条件则写入result矩阵更新，并且标记对已被处理
          processed_double[pair.hap_idx * N + pair.read_idx] = true;
        }
      }
    }
  }

  // 第二步：尝试double类型分组（如果支持SIMD）
  if (double_simd_width > 0) {
    for (auto &pair : pairs) {
      // float没处理,则double也不处理
      if (pair.used)
        pair.used = processed_double[pair.hap_idx * N + pair.read_idx];
      else
        pair.used = true;
    }

    auto double_groups =
        greedy_grouping<double_simd_width>(pairs, max_idle_ratio_double);

    // 处理所有满足条件的double组
    for (const auto &group : double_groups) {
      // TODO: 调用inter double计算
      // 说明：group.pair_indices是pairs中的索引，需要通过pairs[idx]获取hap/read
      // 1. 生成TestCase数组：
      //    - 遍历group.pair_indices，对每个idx，取出pairs[idx]
      //      的hap_idx/read_idx，并从输入数据构建TestCase
      //    - 需要对齐内存分配（使用_mm_malloc）
      // 2. 调用inter API（根据编译选项选择）：
      //    - #if defined(__AVX512F__) -> compute_inter_pairhmm_AVX512_double
      //    - #elif defined(__AVX2__) -> compute_inter_pairhmm_AVX2_double
      for (size_t idx : group.pair_indices) {
        const auto &pair = pairs[idx];
        // TODO:
        //  将结果写入result矩阵更新：
        //    -
        //    遍历results数组，写入result[pairs[group.pair_indices[i]].hap_idx][pairs[group.pair_indices[i]].read_idx]
        if (0) {
          // 满足条件则写入result矩阵更新，并且标记对已被处理
          processed_double[pair.hap_idx * N + pair.read_idx] = true;
        }
      }
    }
  }

  // 第三步：处理剩余未分组的对，使用intra策略
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      if (!processed_double[h * N + r]) {
       
        if (processed_float[h * N + r]) {
           // 只处理double没处理的Pair对
           // TODO: 调用intra计算
           // 1. 生成单个TestCase：
           //    - 从haplotypes[h], reads[r], quality[r]等构建TestCase
           //    - 需要对齐内存分配（使用_mm_malloc）
           // 2. 调用intra API computedoubleLikelihoodsAVX2/computedoubleLikelihoodsAVX512，需要添加新接口
           // 3. 将结果写入result[h][r]
        }else{
          // 处理float和double都没处理的对
          // TODO: 调用intra计算
          // 1. 生成单个TestCase：
          //    - 从haplotypes[h], reads[r], quality[r]等构建TestCase
          //    - 需要对齐内存分配（使用_mm_malloc）
          // 2. 调用intra API（根据编译选项选择）：
          //    - #if defined(__AVX512F__) -> computeLikelihoodsAVX512(tc,
          //    use_double)
          //    - #elif defined(__AVX2__) -> computeLikelihoodsAVX2(tc,
          //    use_double)
          // 3. 将结果写入result[h][r]
        }
      }
    }
  }

  return true;
}

} // namespace schedule
} // namespace pairhmm
