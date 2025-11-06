#include "pairhmm_schedule.h"
#include "pairhmm/common/cpu_features.h"
#include "pairhmm/inter/pairhmm_inter.h"
#include "pairhmm/inter/pairhmm_inter_api.h"
#include "pairhmm/intra/pairhmm_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <net/if.h>
#include <numeric>
#include <vector>

using namespace pairhmm::common;

namespace pairhmm {
namespace schedule {

#define MIN_ACCEPTED 1e-28f
// 辅助结构：表示一个单倍型-read对
struct HapReadPair {
  size_t hap_idx;
  size_t read_idx;
  uint32_t hap_len;
  uint32_t read_len;
  bool contains_n;
  bool used;

  HapReadPair(size_t h_idx, size_t r_idx, uint32_t h_len, uint32_t r_len,
              bool contains_n)
      : hap_idx(h_idx), read_idx(r_idx), hap_len(h_len), read_len(r_len),
        contains_n(contains_n), used(false) {}
};

// 辅助结构：表示一组
struct Group {
  std::vector<size_t> pair_indices; // 指向pairs数组的索引
  uint32_t max_hap_len = 0;
  uint32_t max_read_len = 0;
  double idle_ratio = 0.0;
};

// 贪心分组算法
std::vector<Group> greedy_grouping(std::vector<HapReadPair> &pairs,
                                   double max_idle_ratio, uint32_t SimdWidth) {
  std::vector<Group> groups;
  if (SimdWidth == 0) {
    return groups;
  }

  const size_t total = pairs.size();
  for (size_t i = 0; i < total; ++i) {
    if (pairs[i].used || pairs[i].contains_n)
      continue;

    Group group;
    group.pair_indices.reserve(SimdWidth);

    size_t idx = i;
    while (idx < total && group.pair_indices.size() < SimdWidth) {
      if (!pairs[idx].used && !pairs[idx].contains_n) {
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

bool containsN(const std::vector<uint8_t> &sequence) {
  for (uint8_t c : sequence) {
    if (c == 'N' || c == 'n') {
      return true;
    }
  }
  return false;
}

bool schedule_pairhmm(
    const std::vector<std::vector<uint8_t>> &haplotypes,
    const std::vector<std::vector<uint8_t>> &reads,
    std::vector<std::vector<double>> &result,
    const std::vector<std::vector<uint8_t>> &quality,
    const std::vector<std::vector<uint8_t>> &insertion_qualities,
    const std::vector<std::vector<uint8_t>> &deletion_qualities,
    const std::vector<std::vector<uint8_t>> &gap_contiguous_qualities,
    bool use_double, double max_idle_ratio_float, double max_idle_ratio_double,
    bool verbose) {

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
  std::vector<bool> contains_n_hap_vec(M, false);
  std::vector<bool> contains_n_read_vec(N, false);

  for (size_t h = 0; h < M; ++h) {
    contains_n_hap_vec[h] = containsN(haplotypes[h]);
  }
  for (size_t r = 0; r < N; ++r) {
    contains_n_read_vec[r] = containsN(reads[r]);
  }
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      pairs.emplace_back(h, r, static_cast<uint32_t>(haplotypes[h].size()),
                         static_cast<uint32_t>(reads[r].size()),
                         contains_n_hap_vec[h] || contains_n_read_vec[r]);
    }
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const HapReadPair &lhs, const HapReadPair &rhs) {
              if (lhs.hap_len == rhs.hap_len) {
                return lhs.read_len > rhs.read_len;
              }
              return lhs.hap_len > rhs.hap_len;
            });
  if (verbose) {
    std::cerr << "Pairs: " << pairs.size() << std::endl;
    for (const auto &pair : pairs) {
      std::cerr << "Pair: " << pair.hap_idx << "," << pair.read_idx
                << " hap_len: " << pair.hap_len
                << " read_len: " << pair.read_len << std::endl;
    }
  }

  uint32_t float_simd_width = 0;
  uint32_t double_simd_width = 0;

  if (CpuFeatures::hasAVX512Support()) {
    float_simd_width = 16;
    double_simd_width = 8;
  } else if (CpuFeatures::hasAVX2Support()) {
    float_simd_width = 8;
    double_simd_width = 4;
  }

  // 标记哪些对已经被处理
  std::vector<bool> processed_float(M * N, false);
  std::vector<bool> processed_double(M * N, false);

  int total_num_pairs_float = 0;
  // 第一步：尝试float类型分组（如果未强制使用double且支持SIMD）
  if (!use_double && float_simd_width > 0) {
    auto float_groups =
        greedy_grouping(pairs, max_idle_ratio_float, float_simd_width);
    TestCase *tc = new TestCase[float_simd_width];
    double *results = new double[float_simd_width];
    // 处理所有满足条件的float组
    for (const auto &group : float_groups) {

      if (verbose) {
        std::cerr << "Processing float group: ";
        for (uint32_t i = 0; i < float_simd_width; i++) {
          std::cerr << pairs[group.pair_indices[i]].hap_idx << ","
                    << pairs[group.pair_indices[i]].read_idx << " ";
        }
        std::cerr << std::endl;
      }

      for (uint32_t i = 0; i < float_simd_width; i++) {
        const auto &pair = pairs[group.pair_indices[i]];
        tc[i].hap = haplotypes[pair.hap_idx].data();
        tc[i].rs = reads[pair.read_idx].data();
        tc[i].q = quality[pair.read_idx].data();
        tc[i].i = insertion_qualities[pair.read_idx].data();
        tc[i].d = deletion_qualities[pair.read_idx].data();
        tc[i].c = gap_contiguous_qualities[pair.read_idx].data();
        tc[i].haplen = pair.hap_len;
        tc[i].rslen = pair.read_len;
        total_num_pairs_float++;
      }
      if (CpuFeatures::hasAVX512Support()) {
        inter::compute_inter_pairhmm_AVX512_float(tc, float_simd_width, results,
                                                  false);
      } else if (CpuFeatures::hasAVX2Support()) {
        inter::compute_inter_pairhmm_AVX2_float(tc, float_simd_width, results,
                                                false);
      }
      for (uint32_t i = 0; i < float_simd_width; i++) {
        if (results[i] < MIN_ACCEPTED) {
          pairs[group.pair_indices[i]].used = false;
          if (verbose) {
            std::cerr << "Float pair: " << pairs[group.pair_indices[i]].hap_idx
                      << "," << pairs[group.pair_indices[i]].read_idx
                      << " is needed to be processed by double."
                      << " result: " << results[i] << std::endl;
          }
        } else {
          pairs[group.pair_indices[i]].used = true;
          result[pairs[group.pair_indices[i]].hap_idx]
                [pairs[group.pair_indices[i]].read_idx] =
                    inter::loglikelihoodfloat(results[i]);
          processed_double[pairs[group.pair_indices[i]].hap_idx * N +
                           pairs[group.pair_indices[i]].read_idx] = true;
        }
        processed_float[pairs[group.pair_indices[i]].hap_idx * N +
                        pairs[group.pair_indices[i]].read_idx] = true;
      }
    }
    delete[] tc;
    delete[] results;
  }

  // 第二步：尝试double类型分组（如果支持SIMD）
  if (double_simd_width > 0) {
    TestCase *tc = new TestCase[double_simd_width];
    double *results = new double[double_simd_width];
    // 将float没处理的pair标记为used
    for (auto &pair : pairs) {
      if (!processed_float[pair.hap_idx * N + pair.read_idx])
        pair.used = true;
    }
    auto double_groups =
        greedy_grouping(pairs, max_idle_ratio_double, double_simd_width);

    // 处理所有满足条件的double组
    for (const auto &group : double_groups) {

      if (verbose) {
        std::cerr << "Processing double group: ";
        for (uint32_t i = 0; i < double_simd_width; i++) {
          std::cerr << pairs[group.pair_indices[i]].hap_idx << ","
                    << pairs[group.pair_indices[i]].read_idx << " ";
        }
        std::cerr << std::endl;
      }
      for (uint32_t i = 0; i < double_simd_width; i++) {
        const auto &pair = pairs[group.pair_indices[i]];
        tc[i].hap = haplotypes[pair.hap_idx].data();
        tc[i].rs = reads[pair.read_idx].data();
        tc[i].q = quality[pair.read_idx].data();
        tc[i].i = insertion_qualities[pair.read_idx].data();
        tc[i].d = deletion_qualities[pair.read_idx].data();
        tc[i].c = gap_contiguous_qualities[pair.read_idx].data();
        tc[i].haplen = pair.hap_len;
        tc[i].rslen = pair.read_len;
      }
      if (CpuFeatures::hasAVX512Support()) {
        inter::compute_inter_pairhmm_AVX512_double(tc, double_simd_width,
                                                   results);
      } else if (CpuFeatures::hasAVX2Support()) {
        inter::compute_inter_pairhmm_AVX2_double(tc, double_simd_width,
                                                 results);
      }
      for (uint32_t i = 0; i < double_simd_width; i++) {
        result[pairs[group.pair_indices[i]].hap_idx]
              [pairs[group.pair_indices[i]].read_idx] = results[i];
        processed_double[pairs[group.pair_indices[i]].hap_idx * N +
                         pairs[group.pair_indices[i]].read_idx] = true;
      }
    }
    delete[] tc;
    delete[] results;
  }
  int total_num_pairs_intra = 0;
  // 第三步：处理剩余未分组的对，使用intra策略
  for (size_t h = 0; h < M; ++h) {
    for (size_t r = 0; r < N; ++r) {
      if (!processed_double[h * N + r]) {
        TestCase tc;
        bool use_double = processed_float[h * N + r];
        double results = 0.0;
        tc.hap = haplotypes[h].data();
        tc.rs = reads[r].data();
        tc.q = quality[r].data();
        tc.i = insertion_qualities[r].data();
        tc.d = deletion_qualities[r].data();
        tc.c = gap_contiguous_qualities[r].data();
        tc.haplen = haplotypes[h].size();
        tc.rslen = reads[r].size();
        if (CpuFeatures::hasAVX512Support()) {
          results = intra::computeLikelihoodsAVX512(tc, use_double);
        } else if (CpuFeatures::hasAVX2Support()) {
          results = intra::computeLikelihoodsAVX2(tc, use_double);
        }
        result[h][r] = results;
        if (verbose) {
          std::cerr << "Intra pair: " << h << "," << r << " use "
                    << (use_double ? "double" : "all") << " is computed"
                    << std::endl;
        }
        total_num_pairs_intra++;
      }
    }
  }
  std::cerr << "Total num pairs float: " << total_num_pairs_float << std::endl;
  std::cerr << "Total num pairs intra: " << total_num_pairs_intra << std::endl;

  return true;
}

} // namespace schedule
} // namespace pairhmm
