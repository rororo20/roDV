#ifndef PAIRHMM_INTER_API_H_
#define PAIRHMM_INTER_API_H_

#include "pairhmm_inter.h"

namespace pairhmm {
namespace inter {

bool compute_inter_pairhmm_AVX512_float(TestCase *tc,uint32_t num);
bool compute_inter_pairhmm_AVX512_double(TestCase *tc,uint32_t num);
bool compute_inter_pairhmm_AVX2_float(TestCase *tc,uint32_t num);
bool compute_inter_pairhmm_AVX2_double(TestCase *tc,uint32_t num);

}  // namespace inter
}  // namespace pairhmm

#endif  // PAIRHMM_INTER_API_H_
