#ifndef PAIRHMM_API_H_
#define PAIRHMM_API_H_

#include "../common/common.h"

namespace pairhmm {
namespace intra {

/**
 * @brief PairHMM API - 对外接口
 *
 * 提供简洁的函数接口，内部使用模板实现
 */

// AVX2 版本
double computeLikelihoodsAVX2(const TestCase &tc);

// AVX512 版本
double computeLikelihoodsAVX512(const TestCase &tc);

} // namespace intra
} // namespace pairhmm

#endif // PAIRHMM_API_H_
