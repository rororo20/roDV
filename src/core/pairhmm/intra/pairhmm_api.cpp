#include "pairhmm_api.h"
#include "pairhmm_impl.h"

#define MIN_ACCEPTED 1e-28f

namespace pairhmm {
namespace intra {

constexpr float LOG10_INITIAL_CONSTANT_F = 36.1236000061;
constexpr double LOG10_INITIAL_CONSTANT_D = 307.050595577260822;

double computeLikelihoodsAVX2(const TestCase &tc) {

  double result_final = PairHMMComputer<AVX2FloatTraits>::compute(tc);

  if (result_final < MIN_ACCEPTED) {
    result_final = PairHMMComputer<AVX2DoubleTraits>::compute(tc);
    result_final = log10(result_final) - LOG10_INITIAL_CONSTANT_D;
  } else {
    result_final = log10f(result_final) - LOG10_INITIAL_CONSTANT_F;
  }
  return result_final;
}

double computeLikelihoodsAVX512(const TestCase &tc) {
  double result_final = PairHMMComputer<AVX512FloatTraits>::compute(tc);
  if (result_final < MIN_ACCEPTED) {
    result_final = PairHMMComputer<AVX512DoubleTraits>::compute(tc);
    result_final = log10(result_final) - LOG10_INITIAL_CONSTANT_D;
  } else {
    result_final = log10f(result_final) - LOG10_INITIAL_CONSTANT_F;
  }
  return result_final;
}

} // namespace intra
} // namespace pairhmm