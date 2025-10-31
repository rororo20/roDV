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

  HapReadPair(size_t h_idx, size_t r_idx, uint32_t h_len, uint32_t r_len)
      : hap_idx(h_idx), read_idx(r_idx), hap_len(h_len), read_len(r_len) {}
};

// 辅助结构：表示一组
struct Group {
  std::vector<size_t> pair_indices;  // 指向原始pairs数组的索引
  uint32_t max_hap_len;
  uint32_t max_read_len;
  uint64_t total_elements;  // Hmax * Rmax
  uint64_t total_idle;      // 总idle元素数
  double idle_ratio;        // idle比例 = total_idle / (total_elements * num_pairs)

  void calculate_idle(const std::vector<HapReadPair> &all_pairs) {
    if (pair_indices.empty()) {
      max_hap_len = max_read_len = 0;
      total_elements = total_idle = 0;
      idle_ratio = 0.0;
      return;
    }

    // 找到组内最大的单倍型长度和read长度
    max_hap_len = all_pairs[pair_indices[0]].hap_len;
    max_read_len = all_pairs[pair_indices[0]].read_len;

    for (size_t idx : pair_indices) {
      max_hap_len = std::max(max_hap_len, all_pairs[idx].hap_len);
      max_read_len = std::max(max_read_len, all_pairs[idx].read_len);
    }

    total_elements = static_cast<uint64_t>(max_hap_len) * max_read_len;
    total_idle = 0;

    // 计算每个对的idle元素
    for (size_t idx : pair_indices) {
      const auto &pair = all_pairs[idx];
      uint64_t pair_elements =
          static_cast<uint64_t>(pair.hap_len) * pair.read_len;
      uint64_t idle = total_elements - pair_elements;
      total_idle += idle;
    }

    // idle比例 = 总idle元素 / (总元素数 * 对数)
    if (total_elements > 0 && !pair_indices.empty()) {
      idle_ratio =
          static_cast<double>(total_idle) /
          static_cast<double>(total_elements * pair_indices.size());
    } else {
      idle_ratio = 0.0;
    }
  }
};

// 贪心分组算法
template <uint32_t SimdWidth>
std::vector<Group> greedy_grouping(
    const std::vector<HapReadPair> &all_pairs, double max_idle_ratio) {
  std::vector<Group> groups;
  std::vector<bool> used(all_pairs.size(), false);

  // 按长度乘积降序排序，优先处理大的对
  std::vector<size_t> indices(all_pairs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&all_pairs](size_t a, size_t b) {
              uint64_t prod_a = static_cast<uint64_t>(all_pairs[a].hap_len) *
                                all_pairs[a].read_len;
              uint64_t prod_b = static_cast<uint64_t>(all_pairs[b].hap_len) *
                                all_pairs[b].read_len;
              return prod_a > prod_b;
            });

  for (size_t start_idx : indices) {
    if (used[start_idx])
      continue;

    // 创建一个新组，从当前对开始
    Group group;
    group.pair_indices.push_back(start_idx);
    used[start_idx] = true;
    group.calculate_idle(all_pairs);

    // 贪心地添加相似的对，直到组满或找不到合适的对
    while (group.pair_indices.size() < SimdWidth) {
      size_t best_idx = SIZE_MAX;
      double best_idle_ratio = std::numeric_limits<double>::max();

      // 尝试添加每个未使用的对，找到能使idle比例最小的
      for (size_t i = 0; i < all_pairs.size(); ++i) {
        if (used[i])
          continue;

        // 创建一个测试组，包含当前组的所有对加上新对
        Group test_group = group;
        test_group.pair_indices.push_back(i);
        test_group.calculate_idle(all_pairs);

        // 找到idle比例最小的
        if (test_group.idle_ratio < best_idle_ratio) {
          best_idle_ratio = test_group.idle_ratio;
          best_idx = i;
        }
      }

      // 如果添加最佳对后idle比例仍可接受，则添加
      if (best_idx != SIZE_MAX && best_idle_ratio <= max_idle_ratio) {
        group.pair_indices.push_back(best_idx);
        used[best_idx] = true;
        group.calculate_idle(all_pairs);
      } else {
        // 无法找到合适的对，停止添加
        break;
      }
    }

    // 只有当组达到SimdWidth大小且idle比例可接受时，才添加到结果中
    if (group.pair_indices.size() == SimdWidth &&
        group.idle_ratio <= max_idle_ratio) {
      groups.push_back(group);
    } else {
      // 如果组不完整，取消标记这些对，让它们在后续处理中被考虑
      for (size_t idx : group.pair_indices) {
        used[idx] = false;
      }
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
  std::vector<HapReadPair> all_pairs;
  all_pairs.reserve(M * N);
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      all_pairs.emplace_back(h, r, static_cast<uint32_t>(haplotypes[h].size()),
                             static_cast<uint32_t>(reads[r].size()));
    }
  }

  // 定义阈值
  const double max_idle_ratio_float =
      0.5; // float类型允许的最大idle比例（50%）
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
  std::vector<bool> processed(M * N, false);

  // 第一步：尝试float类型分组（如果未强制使用double且支持SIMD）
  if (!use_double && float_simd_width > 0) {
    auto float_groups = greedy_grouping<float_simd_width>(
        all_pairs, max_idle_ratio_float);

    // 处理所有满足条件的float组
    for (const auto &group : float_groups) {
      // 标记这些对已被处理
      for (size_t idx : group.pair_indices) {
        processed[all_pairs[idx].hap_idx * N +
                  all_pairs[idx].read_idx] = true;
      }

      // TODO: 调用inter float计算
      // 说明：group.pair_indices是all_pairs中的索引，可以直接使用
      // 1. 生成TestCase数组：
      //    - 遍历group.pair_indices，对每个idx，使用all_pairs[idx]获取hap_idx和read_idx
      //    - 从haplotypes[hap_idx], reads[read_idx], quality[read_idx]等构建TestCase
      //    - 需要对齐内存分配（使用_mm_malloc）
      // 2. 调用inter API（根据编译选项选择）：
      //    - #if defined(__AVX512F__) -> compute_inter_pairhmm_AVX512_float
      //    - #elif defined(__AVX2__) -> compute_inter_pairhmm_AVX2_float
      // 3. 将结果写入result矩阵：
      //    - 遍历results数组，写入result[all_pairs[group.pair_indices[i]].hap_idx][all_pairs[group.pair_indices[i]].read_idx]
    }
  }

  // 第二步：尝试double类型分组（如果支持SIMD）
  if (double_simd_width > 0) {
    // 收集未处理的对的索引（在all_pairs中的索引）
    std::vector<size_t> remaining_indices;
    for (size_t i = 0; i < all_pairs.size(); ++i) {
      const auto &pair = all_pairs[i];
      if (!processed[pair.hap_idx * N + pair.read_idx]) {
        remaining_indices.push_back(i);
      }
    }

    if (!remaining_indices.empty()) {
      // 创建remaining_pairs用于分组算法（保持索引对应关系）
      std::vector<HapReadPair> remaining_pairs;
      remaining_pairs.reserve(remaining_indices.size());
      for (size_t idx : remaining_indices) {
        remaining_pairs.push_back(all_pairs[idx]);
      }

      auto double_groups = greedy_grouping<double_simd_width>(
          remaining_pairs, max_idle_ratio_double);

      // 处理所有满足条件的double组
      for (const auto &group : double_groups) {
        // 标记这些对已被处理（将remaining_pairs中的索引映射回all_pairs的索引）
        for (size_t local_idx : group.pair_indices) {
          size_t original_idx = remaining_indices[local_idx];
          processed[all_pairs[original_idx].hap_idx * N +
                    all_pairs[original_idx].read_idx] = true;
        }

        // TODO: 调用inter double计算
        // 说明：group.pair_indices是remaining_pairs中的索引，需要通过remaining_indices映射回all_pairs
        // 1. 生成TestCase数组：
        //    - 遍历group.pair_indices，对每个local_idx：
        //      size_t original_idx = remaining_indices[local_idx];
        //      const auto &pair = all_pairs[original_idx];
        //    - 使用pair.hap_idx和pair.read_idx获取数据，构建TestCase
        //    - 需要对齐内存分配（使用_mm_malloc）
        // 2. 调用inter API（根据编译选项选择）：
        //    - #if defined(__AVX512F__) -> compute_inter_pairhmm_AVX512_double
        //    - #elif defined(__AVX2__) -> compute_inter_pairhmm_AVX2_double
        // 3. 将结果写入result矩阵：
        //    - 遍历results数组，写入result[pair.hap_idx][pair.read_idx]
      }
    }
  }

  // 第三步：处理剩余未分组的对，使用intra策略
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      if (!processed[h * N + r]) {
        // TODO: 调用intra计算
        // 1. 生成单个TestCase：
        //    - 从haplotypes[h], reads[r], quality[r]等构建TestCase
        //    - 需要对齐内存分配（使用_mm_malloc）
        // 2. 调用intra API（根据编译选项选择）：
        //    - #if defined(__AVX512F__) -> computeLikelihoodsAVX512(tc, use_double)
        //    - #elif defined(__AVX2__) -> computeLikelihoodsAVX2(tc, use_double)
        // 3. 将结果写入result[h][r]
      }
    }
  }

  return true;
}

} // namespace schedule
} // namespace pairhmm
