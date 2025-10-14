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
    static inline SimdType setzero() { return _mm256_setzero_ps(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_ps(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_ps(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_ps(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_ps(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_ps(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_ps(ptr, v); }
    
    static inline SimdIntType set1_epi32(int32_t v) { return _mm256_set1_epi32(v); }
    static inline SimdIntType setzero_si256() { return _mm256_setzero_si256(); }
    
    // 特殊操作：设置最低元素
    static inline SimdType set_lse(MainType v, MainType zero) {
        return _mm256_set_ps(zero, zero, zero, zero, zero, zero, zero, v);
    }
    
    // Blend 操作（条件选择）
    static inline SimdType blendv_ps(SimdType a, SimdType b, SimdType mask) {
        return _mm256_blendv_ps(a, b, mask);
    }
    
    // Cast 操作
    static inline SimdType castsi256_ps(SimdIntType v) {
        return _mm256_castsi256_ps(v);
    }
    
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
    
    // 提取结果
    static inline MainType extract_result(SimdType v, uint32_t index) {
        alignas(alignment) MainType arr[simd_width];
        store(arr, v);
        return arr[index];
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, v);
    }
    
    static inline SimdType mask_blend_ps(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_ps(a, b, mask);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_ps(a, b, mask);
    }
    
    static inline SimdType castsi256(SimdIntType v) {
        return _mm256_castsi256_ps(v);
    }
    
    static inline SimdIntType set_epi32_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi32(
            arr[rs_arr[7]], arr[rs_arr[6]], arr[rs_arr[5]], arr[rs_arr[4]],
            arr[rs_arr[3]], arr[rs_arr[2]], arr[rs_arr[1]], arr[rs_arr[0]]
        );
    }
    
    static inline SimdIntType get_right_shift_vector() {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi32(0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00);
    }
    
    static inline SimdIntType get_left_shift_vector() {
        return _mm256_set_epi32(25, 26, 27, 28, 29, 30, 31, 32);
    }
    
    static inline SimdIntType srlv_epi32(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi32(a, b);
    }
    
    static inline SimdIntType sllv_epi32(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi32(a, b);
    }
    
    static inline SimdIntType slli_epi32(SimdIntType a, int imm8) {
        return _mm256_slli_epi32(a, imm8);
    }
    
    static inline SimdIntType or_si256(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si256(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
    
    static inline SimdType permutevar8x32_ps(SimdType a, SimdIntType b) {
        return _mm256_permutevar8x32_ps(a, b);
    }
    
    static inline SimdIntType get_reverse_permute() {
        return _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    }
    
    template<int imm8>
    static inline SimdType blend_ps(SimdType a, SimdType b) {
        return _mm256_blend_ps(a, b, imm8);
    }
    
    static inline MainType cvtss_f32(SimdType a) {
        return _mm256_cvtss_f32(a);
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
    static inline SimdType setzero() { return _mm256_setzero_pd(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm256_add_pd(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm256_sub_pd(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm256_mul_pd(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm256_div_pd(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm256_load_pd(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm256_store_pd(ptr, v); }
    
    static inline SimdIntType set1_epi64x(int64_t v) { return _mm256_set1_epi64x(v); }
    static inline SimdIntType set1_epi32(int32_t v) { return _mm256_set1_epi32(v); }
    static inline SimdIntType setzero_si256() { return _mm256_setzero_si256(); }
    
    static inline SimdType set_lse(MainType v, MainType zero) {
        return _mm256_set_pd(zero, zero, zero, v);
    }
    
    static inline SimdType blendv_pd(SimdType a, SimdType b, SimdType mask) {
        return _mm256_blendv_pd(a, b, mask);
    }
    
    static inline SimdType castsi256_pd(SimdIntType v) {
        return _mm256_castsi256_pd(v);
    }
    
    // 整数向量操作
    static inline SimdIntType set_epi32_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi32(
            static_cast<int32_t>(arr[7]), static_cast<int32_t>(arr[6]), 
            static_cast<int32_t>(arr[5]), static_cast<int32_t>(arr[4]), 
            static_cast<int32_t>(arr[3]), static_cast<int32_t>(arr[2]), 
            static_cast<int32_t>(arr[1]), static_cast<int32_t>(arr[0])
        );
    }
    
    static inline SimdIntType srlv_epi32(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi32(a, b);
    }
    
    static inline SimdIntType sllv_epi32(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi32(a, b);
    }
    
    static inline SimdIntType slli_epi32(SimdIntType a, int imm) {
        return _mm256_slli_epi32(a, imm);
    }
    
    static inline SimdIntType or_si256(SimdIntType a, SimdIntType b) {
        return _mm256_or_si256(a, b);
    }
    
    static inline SimdIntType and_si256(SimdIntType a, SimdIntType b) {
        return _mm256_and_si256(a, b);
    }
    
    // Blend 操作（条件选择）
    static inline SimdType mask_blend_pd(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_pd(a, b, mask);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        return _mm256_blendv_pd(a, b, mask);
    }
    
    static inline SimdType castsi256(SimdIntType v) {
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
    
    static inline MainType extract_result(SimdType v, uint32_t index) {
        alignas(alignment) MainType arr[simd_width];
        store(arr, v);
        return arr[index];
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm256_set_pd(0, 0, 0, v);
    }
    
    static inline SimdIntType set_epi64_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm256_set_epi64x(
            arr[rs_arr[3]], arr[rs_arr[2]], arr[rs_arr[1]], arr[rs_arr[0]]
        );
    }
    
    static inline SimdIntType get_right_shift_vector() {
        return _mm256_set_epi64x(3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm256_set_epi64x(0x1F, 0x0F, 0x07, 0x03);
    }
    
    static inline SimdIntType get_left_shift_vector() {
        return _mm256_set_epi64x(61, 62, 63, 64);
    }
    
    static inline SimdIntType srlv_epi64(SimdIntType a, SimdIntType b) {
        return _mm256_srlv_epi64(a, b);
    }
    
    static inline SimdIntType sllv_epi64(SimdIntType a, SimdIntType b) {
        return _mm256_sllv_epi64(a, b);
    }
    
    static inline SimdIntType slli_epi64(SimdIntType a, int imm8) {
        return _mm256_slli_epi64(a, imm8);
    }
    
    template<int imm8>
    static inline SimdType permute4x64_pd(SimdType a) {
        return _mm256_permute4x64_pd(a, imm8);
    }
    
    template<int imm8>
    static inline SimdType blend_pd(SimdType a, SimdType b) {
        return _mm256_blend_pd(a, b, imm8);
    }
    
    static inline MainType cvtsd_f64(SimdType a) {
        return _mm256_cvtsd_f64(a);
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
    static inline SimdType setzero() { return _mm512_setzero_ps(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm512_add_ps(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm512_sub_ps(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm512_mul_ps(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm512_div_ps(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm512_load_ps(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm512_store_ps(ptr, v); }
    
    static inline SimdIntType set1_epi32(int32_t v) { return _mm512_set1_epi32(v); }
    static inline SimdIntType setzero_si512() { return _mm512_setzero_si512(); }
    
    static inline SimdType set_high(MainType v, MainType zero) {
        return _mm512_set_ps(v, zero, zero, zero, zero, zero, zero, zero,
                            zero, zero, zero, zero, zero, zero, zero, zero);
    }
    
    // AVX512 使用 mask 寄存器
    static inline SimdType mask_blend_ps(__mmask16 mask, SimdType a, SimdType b) {
        return _mm512_mask_blend_ps(mask, a, b);
    }
    
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
    
    static inline MainType extract_result(SimdType v, uint32_t index) {
        alignas(alignment) MainType arr[simd_width];
        store(arr, v);
        return arr[index];
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm512_set_ps(v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    
    static inline SimdType mask_blend_ps(SimdType mask, SimdType a, SimdType b) {
        __mmask16 mask_val = _mm512_cmp_ps_mask(mask, _mm512_setzero_ps(), _CMP_NEQ_OQ);
        return _mm512_mask_blend_ps(mask_val, a, b);
    }
    
    static inline SimdIntType set_epi32_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm512_set_epi32(
            arr[rs_arr[15]], arr[rs_arr[14]], arr[rs_arr[13]], arr[rs_arr[12]],
            arr[rs_arr[11]], arr[rs_arr[10]], arr[rs_arr[9]], arr[rs_arr[8]],
            arr[rs_arr[7]], arr[rs_arr[6]], arr[rs_arr[5]], arr[rs_arr[4]],
            arr[rs_arr[3]], arr[rs_arr[2]], arr[rs_arr[1]], arr[rs_arr[0]]
        );
    }
    
    static inline SimdIntType get_right_shift_vector() {
        return _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm512_set_epi32(0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00,
                               0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01, 0x00);
    }
    
    static inline SimdIntType get_left_shift_vector() {
        return _mm512_set_epi32(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
    }
    
    static inline SimdIntType srlv_epi32(SimdIntType a, SimdIntType b) {
        return _mm512_srlv_epi32(a, b);
    }
    
    static inline SimdIntType sllv_epi32(SimdIntType a, SimdIntType b) {
        return _mm512_sllv_epi32(a, b);
    }
    
    static inline SimdIntType slli_epi32(SimdIntType a, int imm8) {
        return _mm512_slli_epi32(a, imm8);
    }
    
    static inline SimdIntType or_si512(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si512(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    // 别名方法，用于兼容性
    static inline SimdIntType or_si256(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si256(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    static inline SimdType permutevar16x32_ps(SimdType a, SimdIntType b) {
        return _mm512_permutevar_ps(a, b);
    }
    
    static inline SimdIntType get_reverse_permute() {
        return _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
    }
    
    static inline SimdType blend_ps(SimdType a, SimdType b, int imm8) {
        return _mm512_mask_blend_ps(imm8, a, b);
    }
    
    static inline MainType cvtss_f32(SimdType a) {
        return _mm512_cvtss_f32(a);
    }
    
    // Cast 操作
    static inline SimdType castsi256_ps(SimdIntType v) {
        return _mm512_castsi512_ps(v);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        __mmask16 mask_val = _mm512_cmp_ps_mask(mask, _mm512_setzero_ps(), _CMP_NEQ_OQ);
        return _mm512_mask_blend_ps(mask_val, a, b);
    }
    
    static inline SimdType castsi256(SimdIntType v) {
        return _mm512_castsi512_ps(v);
    }
    
    // 模板方法
    template<int imm8>
    static inline SimdType blend_ps(SimdType a, SimdType b) {
        return _mm512_mask_blend_ps(imm8, a, b);
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
    static inline SimdType setzero() { return _mm512_setzero_pd(); }
    static inline SimdType add(SimdType a, SimdType b) { return _mm512_add_pd(a, b); }
    static inline SimdType sub(SimdType a, SimdType b) { return _mm512_sub_pd(a, b); }
    static inline SimdType mul(SimdType a, SimdType b) { return _mm512_mul_pd(a, b); }
    static inline SimdType div(SimdType a, SimdType b) { return _mm512_div_pd(a, b); }
    static inline SimdType load(const MainType* ptr) { return _mm512_load_pd(ptr); }
    static inline void store(MainType* ptr, SimdType v) { _mm512_store_pd(ptr, v); }
    
    static inline SimdIntType set1_epi64(int64_t v) { return _mm512_set1_epi64(v); }
    static inline SimdIntType set1_epi32(int32_t v) { return _mm512_set1_epi32(v); }
    static inline SimdIntType setzero_si512() { return _mm512_setzero_si512(); }
    
    static inline SimdType set_high(MainType v, MainType zero) {
        return _mm512_set_pd(v, zero, zero, zero, zero, zero, zero, zero);
    }
    
    static inline SimdType mask_blend_pd(__mmask8 mask, SimdType a, SimdType b) {
        return _mm512_mask_blend_pd(mask, a, b);
    }
    
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
    
    static inline MainType extract_result(SimdType v, uint32_t index) {
        alignas(alignment) MainType arr[simd_width];
        store(arr, v);
        return arr[index];
    }
    
    // 新增方法：用于模板实现
    static inline SimdType set_lse(MainType v) {
        return _mm512_set_pd(v, 0, 0, 0, 0, 0, 0, 0);
    }
    
    static inline SimdType mask_blend_pd(SimdType mask, SimdType a, SimdType b) {
        __mmask8 mask_val = _mm512_cmp_pd_mask(mask, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        return _mm512_mask_blend_pd(mask_val, a, b);
    }
    
    static inline SimdIntType set_epi64_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm512_set_epi64(
            arr[rs_arr[7]], arr[rs_arr[6]], arr[rs_arr[5]], arr[rs_arr[4]],
            arr[rs_arr[3]], arr[rs_arr[2]], arr[rs_arr[1]], arr[rs_arr[0]]
        );
    }
    
    static inline SimdIntType get_right_shift_vector() {
        return _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    }
    
    static inline SimdIntType get_reserved_mask() {
        return _mm512_set_epi64(0x1F, 0x0F, 0x07, 0x03, 0x1F, 0x0F, 0x07, 0x03);
    }
    
    static inline SimdIntType get_left_shift_vector() {
        return _mm512_set_epi64(61, 62, 63, 64, 61, 62, 63, 64);
    }
    
    static inline SimdIntType srlv_epi64(SimdIntType a, SimdIntType b) {
        return _mm512_srlv_epi64(a, b);
    }
    
    static inline SimdIntType sllv_epi64(SimdIntType a, SimdIntType b) {
        return _mm512_sllv_epi64(a, b);
    }
    
    static inline SimdIntType slli_epi64(SimdIntType a, int imm8) {
        return _mm512_slli_epi64(a, imm8);
    }
    
    static inline SimdIntType or_si512(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si512(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    static inline SimdType permutevar8x64_pd(SimdType a, SimdIntType b) {
        return _mm512_permutevar_pd(a, b);
    }
    
    static inline SimdIntType get_reverse_permute() {
        return _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 7);
    }
    
    static inline SimdType blend_pd(SimdType a, SimdType b, int imm8) {
        return _mm512_mask_blend_pd(imm8, a, b);
    }
    
    // 整数向量操作
    static inline SimdIntType set_epi32_from_array(const MaskType* arr, const uint8_t* rs_arr) {
        return _mm512_set_epi32(
            static_cast<int32_t>(arr[15]), static_cast<int32_t>(arr[14]), 
            static_cast<int32_t>(arr[13]), static_cast<int32_t>(arr[12]), 
            static_cast<int32_t>(arr[11]), static_cast<int32_t>(arr[10]), 
            static_cast<int32_t>(arr[9]), static_cast<int32_t>(arr[8]),
            static_cast<int32_t>(arr[7]), static_cast<int32_t>(arr[6]), 
            static_cast<int32_t>(arr[5]), static_cast<int32_t>(arr[4]), 
            static_cast<int32_t>(arr[3]), static_cast<int32_t>(arr[2]), 
            static_cast<int32_t>(arr[1]), static_cast<int32_t>(arr[0])
        );
    }
    
    static inline SimdIntType srlv_epi32(SimdIntType a, SimdIntType b) {
        return _mm512_srlv_epi32(a, b);
    }
    
    static inline SimdIntType sllv_epi32(SimdIntType a, SimdIntType b) {
        return _mm512_sllv_epi32(a, b);
    }
    
    static inline SimdIntType slli_epi32(SimdIntType a, int imm) {
        return _mm512_slli_epi32(a, imm);
    }
    
    static inline SimdIntType or_si256(SimdIntType a, SimdIntType b) {
        return _mm512_or_si512(a, b);
    }
    
    static inline SimdIntType and_si256(SimdIntType a, SimdIntType b) {
        return _mm512_and_si512(a, b);
    }
    
    // Cast 操作
    static inline SimdType castsi256_pd(SimdIntType v) {
        return _mm512_castsi512_pd(v);
    }
    
    // 统一接口
    static inline SimdType mask_blend(SimdType mask, SimdType a, SimdType b) {
        __mmask8 mask_val = _mm512_cmp_pd_mask(mask, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        return _mm512_mask_blend_pd(mask_val, a, b);
    }
    
    static inline SimdType castsi256(SimdIntType v) {
        return _mm512_castsi512_pd(v);
    }
    
    static inline MainType cvtsd_f64(SimdType a) {
        return _mm512_cvtsd_f64(a);
    }
    
    // 模板方法
    template<int imm8>
    static inline SimdType blend_pd(SimdType a, SimdType b) {
        return _mm512_mask_blend_pd(imm8, a, b);
    }
    
};

}  // namespace intra
}  // namespace pairhmm

#endif  // SIMD_TRAITS_H_

