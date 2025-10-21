#ifndef SIMD_TRAITS_H_
#define SIMD_TRAITS_H_

#include <x86intrin.h>
#include <cstdint>

namespace pairhmm {
namespace intra {

/**
 * @brief SIMD Traits 封装不同指令集和精度的类型和操作
 * 
 * 使用编译时多态（模板）实现零开销抽象
 */

// ============================================================================
// AVX2 Float (256-bit, 8 个 float)
// ============================================================================
struct AVX2FloatTraits {
    using MainType = float;
    using SimdType = __m256;
    using SimdIntType = __m256i;
    using VecIntType = __m256i;
    using MaskType = uint32_t;
    
    static constexpr uint32_t simd_width = 8;
    static constexpr uint32_t simd_bits = 256;
    static constexpr uint32_t alignment = 32;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFF;
    
    // SIMD 操作 - 全部内联，编译后零开销
    static inline SimdType set1(MainType v) { return _mm256_set1_ps(v); }
    static inline SimdIntType setzero_int() { return _mm256_set1_epi32(0); }
    static inline SimdType setzero() { return _mm256_setzero_ps(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_ps(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_ps(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_ps(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_ps(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_ps(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_ps(ptr, v); }
    
    // 向量移位操作
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        SimdType reversed_x = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
        shift_out = _mm256_cvtss_f32(reversed_x);
        x = _mm256_blend_ps(reversed_x, _mm256_set1_ps(shift_in), 0b00000001);
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        SimdType reversed_x = _mm256_permutevar8x32_ps(x, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
        x = _mm256_blend_ps(reversed_x, _mm256_set1_ps(shift_in), 0b00000001);
    }
    

    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, v);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_ps(a, b, mask);
    }
    
    static inline SimdType castsi(SimdIntType v) {
        return _mm256_castsi256_ps(v);
    }
    
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi32(
            static_cast<int32_t>(arr[rs_arr[7]]), static_cast<int32_t>(arr[rs_arr[6]]), 
            static_cast<int32_t>(arr[rs_arr[5]]), static_cast<int32_t>(arr[rs_arr[4]]),
            static_cast<int32_t>(arr[rs_arr[3]]), static_cast<int32_t>(arr[rs_arr[2]]),
            static_cast<int32_t>( arr[rs_arr[1]]), static_cast<int32_t>(arr[rs_arr[0]])
        );
    }
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi32(0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm256_set_epi32(25, 26, 27, 28, 29, 30, 31, 32);
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi32(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi32(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, int imm8) {
        return _mm256_slli_epi32(a, imm8);
    }
    
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
     
};

// ============================================================================
// AVX2 Double (256-bit, 4 个 double)
// ============================================================================
struct AVX2DoubleTraits {
    using MainType = double;
    using SimdType = __m256d;
    using SimdIntType = __m256i;
    using VecIntType = __m256i;
    using MaskType = uint64_t;
    
    static constexpr uint32_t simd_width = 4;
    static constexpr uint32_t simd_bits = 256;
    static constexpr uint32_t alignment = 32;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFFFFFFFFFF;
    
    static inline SimdType set1(MainType v) { return _mm256_set1_pd(v); }
    static inline SimdIntType setzero_int() { return _mm256_set1_epi64x(0); }
    static inline SimdType setzero() { return _mm256_setzero_pd(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_pd(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_pd(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_pd(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_pd(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_pd(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_pd(ptr, v); }
    
    
    // 整数向量操作
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi64x(
            static_cast<int64_t>(arr[rs_arr[3]]),static_cast<int64_t>(arr[rs_arr[2]]),
            static_cast<int64_t>(arr[rs_arr[1]]),static_cast<int64_t>(arr[rs_arr[0]])
        );
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi64(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi64(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, int imm) {
        return _mm256_slli_epi64(a, imm);
    }
    
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_pd(a, b, mask);
    }
    
    static inline SimdType castsi(SimdIntType v) {
        return _mm256_castsi256_pd(v);
    }
    
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        SimdType reversed_x = _mm256_permute4x64_pd(x, 0b10010011);
        shift_out = _mm256_cvtsd_f64(reversed_x);
        x = _mm256_blend_pd(reversed_x, _mm256_set1_pd(shift_in), 0b0001);
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        SimdType reversed_x = _mm256_permute4x64_pd(x, 0b10010011);
        x = _mm256_blend_pd(reversed_x, _mm256_set1_pd(shift_in), 0b0001);
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_pd(0, 0, 0, v);
    }
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm256_set_epi64x(3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi64x(0b111, 0b11, 0b10, 0b0);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm256_set_epi64x(29, 30, 31, 32);
    }
};

// ============================================================================
// AVX512 Float (512-bit, 16 个 float)
// ============================================================================
struct AVX512FloatTraits {
    using MainType = float;
    using SimdType = __m512;
    using SimdIntType = __m512i;
    using VecIntType = __m512i;
    using MaskType = uint32_t;
    
    static constexpr uint32_t simd_width = 16;
    static constexpr uint32_t simd_bits = 512;
    static constexpr uint32_t alignment = 64;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFF;
    
    static inline SimdType set1(MainType v) { return _mm512_set1_ps(v); }
    static inline SimdIntType setzero_int() { return _mm512_set1_epi32(0); }
    static inline SimdType setzero() { return _mm512_setzero_ps(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm512_add_ps(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm512_sub_ps(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm512_mul_ps(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm512_div_ps(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm512_load_ps(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm512_store_ps(ptr, v); }
    
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        shift_out = _mm512_cvtss_f32(x);
        
        union { int i; float f; } shift_in_h;
        shift_in_h.f = shift_in;
        x = _mm512_castsi512_ps(_mm512_alignr_epi32(
            _mm512_set1_epi32(shift_in_h.i), 
            _mm512_castps_si512(x), 0x1));
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        union { int i; float f; } shift_in_h;
        shift_in_h.f = shift_in;
        x = _mm512_castsi512_ps(_mm512_alignr_epi32(
            _mm512_set1_epi32(shift_in_h.i), 
            _mm512_castps_si512(x), 0x1));
    }
    
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm512_set_ps(v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    

    
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm512_set_epi32(
            arr[rs_arr[0]], arr[rs_arr[1]], arr[rs_arr[2]], arr[rs_arr[3]],
            arr[rs_arr[4]], arr[rs_arr[5]], arr[rs_arr[6]], arr[rs_arr[7]],
            arr[rs_arr[8]], arr[rs_arr[9]], arr[rs_arr[10]], arr[rs_arr[11]],
            arr[rs_arr[12]], arr[rs_arr[13]], arr[rs_arr[14]], arr[rs_arr[15]]
        );
    }
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm512_set_epi32(0, 1 , 2, 3 , 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm512_set_epi32(0b0, 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111,
                               0b11111111, 0b111111111, 0b1111111111, 0b11111111111, 0b111111111111, 0b1111111111111, 0b11111111111111, 0b111111111111111);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm512_set_epi32(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17);
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm512_srlv_epi32(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm512_sllv_epi32(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, int imm8) {
        return _mm512_slli_epi32(a, imm8);
    }
    
    
    // 别名方法，用于兼容性
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        __m512i val = _mm512_set1_epi32(static_cast<int32_t>(0x80000000));
        __mmask16 mask_val = _mm512_test_epi32_mask(val, mask);
        return _mm512_mask_blend_ps(mask_val, a, b);
    }

    static inline SimdType castsi(SimdIntType v) {
        return _mm512_castsi512_ps(v);
    }
    
};

// ============================================================================
// AVX512 Double (512-bit, 8 个 double)
// ============================================================================
struct AVX512DoubleTraits {
    using MainType = double;
    using SimdType = __m512d;
    using SimdIntType = __m512i;
    using VecIntType = __m512i;
    using MaskType = uint64_t;
    
    static constexpr uint32_t simd_width = 8;
    static constexpr uint32_t simd_bits = 512;
    static constexpr uint32_t alignment = 64;
    static constexpr MaskType mask_all_ones = 0xFFFFFFFFFFFFFFFF;
    
    static inline SimdType set1(MainType v) { return _mm512_set1_pd(v); }
    static inline SimdIntType setzero_int() { return _mm512_set1_epi64(0); }
    static inline SimdType setzero() { return _mm512_setzero_pd(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm512_add_pd(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm512_sub_pd(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm512_mul_pd(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm512_div_pd(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm512_load_pd(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm512_store_pd(ptr, v); }
        
    static inline void vector_shift(SimdType& x, MainType shift_in, MainType& shift_out) {
        shift_out = _mm512_cvtsd_f64(x);
        
        union { int64_t i; double f; } shift_in_h;
        shift_in_h.f = shift_in;
        x = _mm512_castsi512_pd(_mm512_alignr_epi64(
            _mm512_set1_epi64(shift_in_h.i), 
            _mm512_castpd_si512(x), 0x1));
    }
    
    static inline void vector_shift_last(SimdType& x, MainType shift_in) {
        union { int64_t i; double f; } shift_in_h;
        shift_in_h.f = shift_in;
        x = _mm512_castsi512_pd(_mm512_alignr_epi64(
            _mm512_set1_epi64(shift_in_h.i), 
            _mm512_castpd_si512(x), 0x1));
    }
    
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm512_set_pd(v, 0, 0, 0, 0, 0, 0, 0);
    }
    
    
    static inline SimdIntType get_forward_shift_vector() {
        return _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm512_set_epi64( 0b0, 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111);
    }
    
    static inline SimdIntType get_backward_shift_vector() {
        return _mm512_set_epi64(64, 63, 62, 61, 64, 63, 62, 61);
    }
    
    // 整数向量操作
    static inline SimdIntType set_epi_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm512_set_epi64(
            static_cast<int64_t>(arr[rs_arr[0]]), static_cast<int64_t>(arr[rs_arr[1]]), 
            static_cast<int64_t>(arr[rs_arr[2]]), static_cast<int64_t>(arr[rs_arr[3]]), 
            static_cast<int64_t>(arr[rs_arr[3]]), static_cast<int64_t>(arr[rs_arr[4]]), 
            static_cast<int64_t>(arr[rs_arr[5]]), static_cast<int64_t>(arr[rs_arr[6]])
        );
    }
    
    static inline SimdIntType forward_shift(SimdIntType a, SimdIntType b) {
        return _mm512_srlv_epi64(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, SimdIntType b) {
        return _mm512_sllv_epi64(a, b);
    }
    
    static inline SimdIntType backward_shift(SimdIntType a, int imm) {
        return _mm512_slli_epi64(a, imm);
    }
    
    static inline SimdIntType or_si(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        __m512i val = _mm512_set1_epi64(static_cast<int64_t>(0x8000000000000000));
        __mmask8 mask_val = _mm512_test_epi64_mask(val, mask);
        return _mm512_mask_blend_pd(mask_val, a, b);
    }
    
    static inline SimdType castsi(SimdIntType v) {
        return _mm512_castsi512_pd(v);
    }
    
};

}  // namespace intra
}  // namespace pairhmm

#endif  // SIMD_TRAITS_H_

