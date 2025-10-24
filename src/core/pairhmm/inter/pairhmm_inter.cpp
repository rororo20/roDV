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
template <typename ALLOCATOR>
void InterPairHMMComputer<Traits>::precompute(MultiTestCase<Traits> &tc,
                                              ALLOCATOR &allocator) {

  int alloc_bytes =
      tc.max_rslen * Traits::simd_width * sizeof(Traits::MainType);
  tc.distm = allocator.allocate(alloc_bytes, Traits::alignment);
  tc._1_distm = allocator.allocate(alloc_bytes, Traits::alignment);
  tc.gapm = allocator.allocate(alloc_bytes, Traits::alignment);
  tc.mm = allocator.allocate(alloc_bytes, Traits::alignment);
  tc.mi = allocator.allocate(alloc_bytes, Traits::alignment);
  tc.ii = allocator.allocate(alloc_bytes, Traits::alignment);
  tc.md = allocator.allocate(alloc_bytes, Traits::alignment);
  // TODO: initialize haps and reads seqs
  // TODO: initialize distm and _1_distm  
  // TODO: initialize gapm
  // TODO: initialize mm
  // TODO: initialize mi
  // TODO: initialize ii
  // TODO: initialize md
  // TODO: initialize dd

}

template <typename Traits>
template <typename ALLOCATOR>
void InterPairHMMComputer<Traits>::finalize(MultiTestCase<Traits> &tc,
                                            ALLOCATOR &allocator) {
  int alloc_bytes =
      tc.max_rslen * Traits::simd_width * sizeof(Traits::MainType);
  allocator.deallocate(tc.distm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc._1_distm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.gapm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.mm, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.mi, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.ii, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.md, alloc_bytes, Traits::alignment);
  allocator.deallocate(tc.dd, alloc_bytes, Traits::alignment);
}

template <typename Traits>
void InterPairHMMComputer<Traits>::load_parameters_for_read(
    const MultiTestCase<Traits> &tc, int i, SimdType &distm, SimdType &_1_distm,
    SimdType &p_gapm, SimdType &p_mm, SimdType &p_mx, SimdType &p_xx,
    SimdType &p_my, SimdType &p_yy) {
  distm = Traits::load(tc.distm + i * Traits::simd_width);
  _1_distm = Traits::load(tc._1_distm + i * Traits::simd_width);
  p_gapm = Traits::load(tc.gapm + i * Traits::simd_width);
  p_mm = Traits::load(tc.mm + i * Traits::simd_width);
  p_mx = Traits::load(tc.mi + i * Traits::simd_width);
  p_xx = Traits::load(tc.ii + i * Traits::simd_width);
  p_my = Traits::load(tc.md + i * Traits::simd_width);
  p_yy = Traits::load(tc.dd + i * Traits::simd_width);
}

template <typename Traits>
void InterPairHMMComputer<Traits>::compute(MultiTestCase<Traits> &tc) {

  uint32_t hap_lens[Traits::simd_width];
  uint32_t rs_lens[Traits::simd_width];

  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    hap_lens[i] = tc.test_cases[i].haplen;
    rs_lens[i] = tc.test_cases[i].rslen;
  }
  SimdType mm[tc.max_haplen + 1], ii[tc.max_haplen + 1], dd[tc.max_haplen + 1];
  SimdType distm, _1_distm;
  SimdType p_gapm, p_mm, p_mx, p_xx, p_my, p_yy;
  SimdType M, M_i1, M_j1, M_i1j1, I, I_i1, I_j1, I_i1j1, D, D_i1, D_j1, D_i1j1;

  initialize_matrices(tc, mm, ii, dd, hap_lens);
  MaskType all_mask = Traits::set1_all_mask();

  for (int i = 0; i < tc.min_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    load_parameters_for_read(tc, i, distm, _1_distm, p_gapm, p_mm, p_mx, p_xx,
                             p_my, p_yy);

    for (int j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j, false,
                          all_mask);
    }
    // MASK Haplotypes
    for (int j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j, true,
                          Traits::generate_length_mask(j, hap_lens));
    }
  }
  for (int i = tc.min_rslen; i < tc.max_rslen; i++) {
    SimdIntType rbase = Traits::load_seqs(tc.rs_seqs + i * Traits::simd_width);
    // MASK Reads
    load_parameters_for_read(tc, i, distm, _1_distm, p_gapm, p_mm, p_mx, p_xx,
                             p_my, p_yy);
    for (int j = 0; j < tc.min_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j, true,
                          Traits::generate_length_mask(j, rs_lens));
    }
    // MASK Reads and Haplotypes
    MaskType reads_mask = Traits::generate_length_mask(i, rs_lens);
    for (int j = tc.min_haplen; j < tc.max_haplen; j++) {
      SimdIntType h = Traits::load_seqs(tc.hap_seqs + j * Traits::simd_width);
      MaskType hap_mask = Traits::generate_length_mask(j, hap_lens);
      process_matrix_cell(rbase, h, distm, _1_distm, p_mm, p_gapm, p_mx, p_xx,
                          p_my, p_yy, M, I, D, M_i1, I_i1, D_i1, M_j1, I_j1,
                          D_j1, M_i1j1, I_i1j1, D_i1j1, mm, ii, dd, j, true,
                          Traits::mask_and(reads_mask, hap_mask));
    }
  }
  SimdType sum_m = Traits::setzero();
  SimdType sum_i = Traits::setzero();
  for (int i = 0; i < tc.max_haplen; i++) {
    sum_m = Traits::add(sum_m, mm[i]);
    sum_i = Traits::add(sum_i, ii[i]);
  }
  MainType m_result_temp[Traits::simd_width];
  MainType i_result_temp[Traits::simd_width];
  Traits::store(m_result_temp, sum_m);
  Traits::store(i_result_temp, sum_i);
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    tc.results[i] = m_result_temp[i] + i_result_temp[i];
  }
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
    const SimdIntType &rbase, const SimdIntType &h, const SimdType &distm,
    const SimdType &_1_distm, const SimdType &p_mm, const SimdType &p_gapm,
    const SimdType &p_mx, const SimdType &p_xx, const SimdType &p_my,
    const SimdType &p_yy, SimdType &M, SimdType &I, SimdType &D, SimdType &M_i1,
    SimdType &I_i1, SimdType &D_i1, SimdType &M_j1, SimdType &I_j1,
    SimdType &D_j1, SimdType &M_i1j1, SimdType &I_i1j1, SimdType &D_i1j1,
    SimdType *mm, SimdType *ii, SimdType *dd, int j, bool is_masked,
    MaskType len_mask) {

  MaskType mask = Traits::test_cmpeq(rbase, h);
  SimdType distm_chosen = Traits::mask_blend(mask, distm, _1_distm);

  // 计算新的矩阵值
  M = Traits::mul(Traits::add(Traits::add(Traits::mul(M_i1j1, p_mm),
                                          Traits::mul(I_i1j1, p_gapm)),
                              Traits::mul(D_i1j1, p_gapm)),
                  distm_chosen);
  I = Traits::add(Traits::mul(M_i1, p_mx), Traits::mul(I_i1, p_xx));
  D = Traits::add(Traits::mul(M_j1, p_my), Traits::mul(D_j1, p_yy));

  if (is_masked) {
    M = Traits::mask_blend(len_mask, M, Traits::setzero());
    I = Traits::mask_blend(len_mask, I, Traits::setzero());
    D = Traits::mask_blend(len_mask, D, Traits::setzero());
  }
  // 更新状态变量
  M_i1j1 = M_i1;
  I_i1j1 = I_i1;
  D_i1j1 = D_i1;
  M_j1 = M;
  I_j1 = I;
  D_j1 = D;
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
