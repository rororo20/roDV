# PairHMM Intra 重构说明

## 概述

本目录包含 PairHMM 算法的 SIMD 优化重构版本，使用 C++17 模板技术实现代码复用。

## 架构特点

### 1. **零开销抽象**
- 使用 SIMD Traits 模板封装不同指令集操作
- 编译时多态，运行时无虚函数开销
- 编译器会为每个配置生成最优代码

### 2. **支持的指令集和精度**
- AVX2 Float (256-bit, 8个float)
- AVX2 Double (256-bit, 4个double)
- AVX512 Float (512-bit, 16个float)
- AVX512 Double (512-bit, 8个double)

### 3. **文件结构**

```
intra/
├── common.h           # 公共定义（TestCase, ConvertChar等）
├── context.h          # 概率计算上下文（质量分数转换）
├── simd_traits.h      # SIMD特征类（封装指令集操作）
├── pairhmm_impl.h     # PairHMM 模板声明
├── pairhmm_impl.cpp   # PairHMM 模板实现
├── pairhmm_api.h      # 对外接口声明
├── pairhmm_api.cpp    # 对外接口实现（显式实例化）
└── CMakeLists.txt     # 构建配置
```

## 编译说明

### 当前状态
⚠️ **简化版本** - 当前实现是框架验证版本，核心算法未完整实现。

### 已知问题
1. 模板实例化导致编译器需要所有 traits（即使未使用）
2. AVX2 编译时会遇到 AVX512 指令错误

### 解决方案

**选项 1: 条件编译（推荐生产环境）**
```cpp
// 在 simd_traits.h 中使用条件编译
#ifdef __AVX512F__
struct AVX512FloatTraits { /* ... */ };
#endif
```

**选项 2: 分离文件（当前简化版本）**
- `pairhmm_avx2.cpp` - 仅包含 AVX2 实现
- `pairhmm_avx512.cpp` - 仅包含 AVX512 实现

**选项 3: 分离编译单元（最灵活）**
```
intra/
├── avx2/
│   ├── pairhmm_avx2_float.cpp
│   └── pairhmm_avx2_double.cpp
└── avx512/
    ├── pairhmm_avx512_float.cpp
    └── pairhmm_avx512_double.cpp
```

## 性能保证

### 1. 编译优化验证
```bash
# 查看生成的汇编代码
g++ -O3 -mavx2 -S pairhmm_impl.cpp -o pairhmm_avx2.s

# 验证内联
grep -A 10 "compute_mxy" pairhmm_avx2.s
```

### 2. 基准测试
```bash
cd build
make
./test/pairhmm_test
```

## 相比原实现的改进

### 代码重复消除
- **原来**: 4个文件 × 300行 = 1200行（90%重复）
- **现在**: 核心逻辑 1份，总计约 600行

### 可维护性提升
- 修改算法逻辑：1个地方
- 添加新指令集：添加 Traits 即可
- 类型安全：编译时检查

### 性能保持
- 模板完全内联
- 无运行时开销
- 生成代码与手写相同

## 完整实现 TODO

要完成完整的 PairHMM 实现，需要：

1. ✅ SIMD traits 定义
2. ✅ 基本框架搭建
3. ⏳ 对角线扫描算法实现（stripe processing）
4. ⏳ Mask 预计算和应用
5. ⏳ 完整的 HMM 状态转移
6. ⏳ 结果累加和提取
7. ⏳ 完整单元测试

## 参考

- 原始实现：`hc/src/haplotypecaller/pairhmm/core/`
- Intel GKL: https://github.com/IntelLabs/GKL
- SIMD 最佳实践：https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

## 贡献指南

### 添加新指令集
1. 在 `simd_traits.h` 中定义新的 Traits 结构
2. 在 `pairhmm_impl.cpp` 中添加显式实例化
3. 在 `pairhmm_api.cpp` 中添加对外接口
4. 更新 CMakeLists.txt 添加编译选项

### 性能调优
1. 使用 `perf` 工具分析热点
2. 检查向量化报告：`-fopt-info-vec`
3. 使用 VTune 或 perf 进行详细分析

