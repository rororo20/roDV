# roDV

面向 CPU 的 C++17 工具库，实现 DeepVariant 风格变异检测流程里的几段计算：读段与单倍型的 PairHMM、区域级样例生成（等位基因计数与局部重比对），以及 Inception 输入预处理和 CPU 推理启动规划。

主程序目前只提供规划与预处理基准两个入口，完整的 `make_examples` / `call_variants` 流水线还没有串成一个可执行文件。各模块以静态库和单元测试的形式提供。

## 模块

```
src/
├── core/            PairHMM 计算与调度
│   └── pairhmm/
│       ├── intra/   单条单倍型–读段对内的 SIMD（AVX2 / AVX-512，float / double）
│       └── inter/   多对之间的 SIMD
├── make_examples/   等位基因计数、候选窗口、局部重比对、按输入顺序汇合的区域流水线
├── inference/       pileup 图像预处理，以及按 NUMA / L3 规划 CPU 推理
├── iostream/        基于 HTSlib 的 BAM / BED / VCF 读写
└── rodv_main.cpp    命令行入口
```

- **PairHMM**：用 C++17 traits 模板把 AVX2 与 AVX-512、float 与 double 收成一套实现，不同指令集分开编译。调度器按空闲率把单倍型–读段对分组，含 `N` 的对不进入 SIMD 路径。
- **make_examples**：按区域统计 SNP / indel 支持度，并用 de Bruijn 图做局部重比对。默认参数对齐 DeepVariant r1.6.1。比对本身是全局仿射空位，不是上游的条带 Smith–Waterman。区域任务用有界队列分发给工作线程，结果按输入顺序收回。
- **inference**：把 uint8 pileup 转成 Inception 输入（`uint8 / 128 - 1`，与标量路径逐位一致），并按单 NUMA 节点的物理核数和 L3 给出进程数、线程数和 batch size。默认图像形状是 r1.6.1 WGS 的 `100 x 221 x 7`。

## 依赖

- CMake ≥ 3.16
- 支持 C++17 的编译器
- [HTSlib](https://github.com/samtools/htslib)
- GoogleTest
- 运行 PairHMM 与预处理的 AVX2 / AVX-512 路径需要对应指令集；标量预处理不需要

## 构建

CMake 工程在 `src/`：

```bash
cmake -S src -B src/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/build -j
```

产物：

| 目标 | 作用 |
| --- | --- |
| `rodv_main` | 命令行入口 |
| `pairhmm_test` / `pairhmm_test_avx2` | PairHMM 单元测试 |
| `pairhmm_perftest` | PairHMM 性能测试 |
| `src/make_examples/test/`、`src/inference/test/` | 样例生成与预处理测试 |

运行测试：

```bash
cd src/build
ctest --output-on-failure
```

## 命令行

```bash
# 按本机 CPU 拓扑打印 call_variants 的进程 / 线程 / batch 规划
./src/build/rodv_main call-variants-plan
./src/build/rodv_main call-variants-plan --height 100 --width 221 --channels 7

# Inception 预处理吞吐（固定 100x221x7，batch 16）
./src/build/rodv_main preprocess-bench
```

`resouces/` 里有 PairHMM 的调试与测试数据。

## 许可

[MIT](LICENSE)
