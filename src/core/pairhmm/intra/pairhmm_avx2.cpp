#include "pairhmm_api.h"
#include "context.h"
#include "common.h"
#include "pairhmm_impl.h"

namespace pairhmm {
namespace intra {

// AVX2 Float 实现
float compute_pairhmm_avx2_float(const TestCase& tc) {
    ConvertChar::init();
    Context<float> ctx;
    
    // 使用模板化实现
    return PairHMMComputer<AVX2FloatTraits>::compute(tc);
}

// AVX2 Double 实现
double compute_pairhmm_avx2_double(const TestCase& tc) {
    ConvertChar::init();
    Context<double> ctx;
    
    // 使用模板化实现
    return PairHMMComputer<AVX2DoubleTraits>::compute(tc);
}

}  // namespace intra
}  // namespace pairhmm