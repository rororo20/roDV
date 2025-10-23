#include "pairhmm_inter.h"
#include "../common/context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace pairhmm::common; // 使用 common 命名空间的 Context

namespace pairhmm {
namespace inter {

template <typename Traits>
void InterPairHMMComputer<Traits>::compute(const MultiTestCase<Traits> &tc) {

  uint32_t hap_lens[Traits::simd_width];
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    hap_lens[i] = tc.test_cases[i].haplen;
  }
  SimdType mm[tc.max_haplen + 1], ii[tc.max_haplen + 1], dd[tc.max_haplen + 1];
  SimdType distm, _1_distm;
  SimdType p_gapm, p_mm, p_mx, p_xx, p_my, p_yy;
  SimdType M, M_i1, M_j1, M_i1j1, I, I_i1, I_j1, I_i1j1, D, D_i1, D_j1, D_i1j1;

  initialize_matrices(tc, mm, ii, dd, hap_lens);

  for (int i = 0; i < tc.min_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    

    for (int j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx, p_my, p_yy,
                          M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1, D_j1, M_i1j1, I_i1j1, D_i1j1,
                          mm, ii, dd, j);
    }
    // MASK Haplotypes
    for (int j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx, p_my, p_yy,
                          M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1, D_j1, M_i1j1, I_i1j1, D_i1j1,
                          mm, ii, dd, j);
    }
  }
  for (int i = tc.min_rslen; i < tc.max_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    // MASK Reads
    for (int j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx, p_my, p_yy,
                          M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1, D_j1, M_i1j1, I_i1j1, D_i1j1,
                          mm, ii, dd, j);
    }
    // MASK Reads and Haplotypes
    for (int j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx, p_my, p_yy,
                          M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1, D_j1, M_i1j1, I_i1j1, D_i1j1,
                          mm, ii, dd, j);
    }
  }
  return;
}
template <typename Traits>
void InterPairHMMComputer<Traits>::initialize_matrices(
    const MultiTestCase<Traits> &tc, SimdType *mm, SimdType *ii, SimdType *dd,
    uint32_t *hap_lens) {
  for (int i = 0; i <= tc.max_haplen; ++i) {
    mm[i] = Traits::setzero();
    ii[i] = Traits::setzero();
    dd[i] = Traits::set_init_d(hap_lens);
  }
}

template <typename Traits>
void InterPairHMMComputer<Traits>::process_matrix_cell(
    const SimdIntType &rbase, const SimdIntType &h,
    const SimdType &distm, const SimdType &_1_distm,
    const SimdType &p_mm, const SimdType &p_gapm,
    const SimdType &p_mx, const SimdType &p_xx,
    const SimdType &p_my, const SimdType &p_yy,
    SimdType &M, SimdType &I, SimdType &D,
    SimdType &M_i1, SimdType &I_i1, SimdType &D_i1,
    SimdType &M_j1, SimdType &I_j1, SimdType &D_j1,
    SimdType &M_i1j1, SimdType &I_i1j1, SimdType &D_i1j1,
    SimdType *mm, SimdType *ii, SimdType *dd, int j) {
  
  MaskType mask = Traits::test_cmpeq(rbase, h);
  SimdType distm_chosen = Traits::mask_blend(mask, distm, _1_distm);
  
  // 计算新的矩阵值
  M = Traits::mul(Traits::add(Traits::add(Traits::mul(M_i1j1, p_mm),
                                          Traits::mul(I_i1j1, p_gapm)),
                              Traits::mul(D_i1j1, p_gapm)),
                  distm_chosen);
  I = Traits::add(Traits::mul(M_i1, p_mx), Traits::mul(I_i1, p_xx));
  D = Traits::add(Traits::mul(M_j1, p_my), Traits::mul(D_j1, p_yy));
  
  // 更新状态变量
  M_i1j1 = M_i1;
  I_i1j1 = I_i1;
  D_i1j1 = D_i1;
  M_j1 = M;
  I_j1 = I;
  D_j1 = D;
  
  // 存储到矩阵中
  mm[j] = M;
  ii[j] = I;
  dd[j] = D;
  
  // 准备下一次迭代
  M_i1 = mm[j + 1];
  I_i1 = ii[j + 1];
  D_i1 = dd[j + 1];
}
// 显式实例化
#if defined(__AVX512F__)
template class InterPairHMMComputer<AVX512FloatTraits>;
template class InterPairHMMComputer<AVX512DoubleTraits>;
#elif defined(__AVX2__)
template class InterPairHMMComputer<AVX2FloatTraits>;
template class InterPairHMMComputer<AVX2DoubleTraits>;
#endif

} // namespace inter
} // namespace pairhmm
