#ifndef PAIRHMM_API_H_
#define PAIRHMM_API_H_

#include "common.h"

namespace pairhmm {
namespace intra {

/**
 * @brief PairHMM API - 对外接口
 * 
 * 提供简洁的函数接口，内部使用模板实现
 */

// AVX2 版本
float compute_pairhmm_avx2_float(const TestCase& tc);
double compute_pairhmm_avx2_double(const TestCase& tc);

// AVX512 版本
float compute_pairhmm_avx512_float(const TestCase& tc);
double compute_pairhmm_avx512_double(const TestCase& tc);

}  // namespace intra
}  // namespace pairhmm

#endif  // PAIRHMM_API_H_

