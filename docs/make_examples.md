# make_examples 并发框架

面向 DeepVariant r1.6.1 calling 路径的 C++ 流水线。计算对齐上游的读过滤、局部重比对和 SNP/indel 候选；线程所有权对齐 `hc`：只有派发线程碰读流，worker 独占一份区间资源，写线程按 `source_id` 输出。

代码在 `src/make_examples/runtime/`。不使用 Boost。`nthreads` 限制在 1..128，资源槽和队列容量都是 `nthreads * 2`。

## 线程，从低到高

```text
L0  BlockingQueue
L1  RegionPool          nthreads * 2 份 RegionResource
L2  ReferenceCache      一个线程按染色体发布参考序列
L3  ReadDispatch        唯一调用 ReadSource::next 的线程
L4  RegionWorker × N    重比对 → 计数 → PileupStage → 打包
L5  ExampleWriter       按 source_id 顺序写入 ExampleSink
```

`MakeExamplesPipeline::run` 的启动顺序是：写线程、N 个 worker、派发线程。加入顺序相反：先等派发结束，再等 worker，最后等写线程。写线程最后退出，避免还在用参考序列视图时缓存被拆掉。

一个 worker 内部不再开线程。区间宽度默认 1000 bp，在派发线程按 `pos / region_bp` 切开；`tid` 变化时也切开。只发出里面至少有一条保留 read 的区间。

结束标记：派发线程先推一条 `finish=true` 且 `source_id` 等于区间个数的 `RegionSource`，再为其余 worker 各推一条 `source_id = -1` 的结束标记。只有第一条会通知写线程。写线程在 `expected_id == finish_id` 时退出，结束记录本身不写出。

## 模块关系

```text
MakeExamplesPipeline
    |-- ReferenceCache.get(tid) ---------> RegionSource.ref
    |-- ReadDispatch
    |       ReadSource::next()              内存读流或 HtsReadSource
    |       keep_read()                     过滤
    |       bam_copy1                      拷进 RegionResource
    |       RegionPool.acquire/release
    |       推出 RegionSource{source_id, tid, start, end, ref, resource}
    |                    |
    |                    v
    |-- RegionWorker
    |       realign_reads                  realigner.h
    |       with_alignment                 新 CIGAR 的 query 长度必须一致
    |       accumulate_snps + candidates   snp_counter.h
    |       PileupStage::encode            默认返回空字节
    |                    |
    |                    v
    |-- ExampleRecord{source_id, start, candidates, tensors}
    |                    |
    +-- ExampleWriter --> ExampleSink
            对不上 expected_id 的记录进 map，凑齐再写
```

| 模块 | 文件 | 职责 | 不做的事 |
|---|---|---|---|
| `BlockingQueue` | `blocking_queue.h` | 有界队列，满等、空等 | 不超时返回空指针 |
| `RegionResource` / `RegionSource` / `ExampleRecord` | `runtime/region_types.h` | 区间资源、作业、结果 | worker 之间不共享同一份 resource |
| `RegionPool` | `runtime/region_pool.h` | 预分配并循环归还。`release` 会 `bam_destroy1` 拷贝 | 不读文件 |
| `ReferenceCache` | `runtime/reference_cache.h` | 后台按 tid 发布碱基。`start` 接收内存序列，`start_fasta` 用 faidx 读 FASTA。`get` 阻塞到该染色体已发布 | 不读 BAM |
| `ReadSource` / `HtsReadSource` | `runtime/read_dispatch.h`, `runtime/hts_read_source.h` | `next(tid, read)`。文件实现读坐标排序的 BAM/CRAM，解压用单独的 `hts_tpool` | 只有派发线程调用 `next`。返回的 `bam1_t` 在下一次 `next` 后失效 |
| `ReadDispatch` | `runtime/read_dispatch.cpp` | 切窗口、过滤、拷贝、编号 | worker 不调用 `next` |
| `PileupStage` / `WgsPileup` | `runtime/pileup_stage.h`, `pileup.h` | 每个候选画成 100×221×7。WGS 默认不做 alt 图，和 r1.6.1 `alt_aligned_pileup=none` 一致 | 不是上游逐像素回归 |
| `RegionWorker` | `runtime/region_worker.cpp` | 一个区间的四步计算，然后还资源 | 不碰读流 |
| `ExampleWriter` / `MemorySink` / `TsvExampleSink` | `runtime/example_writer.h` | 按序号写。内存 sink 给测试，TSV sink 给 `rodv_main make-examples` | 不计算 |
| `MakeExamplesPipeline` | `runtime/pipeline.h` | 把上面几层串起来 | 不含 `call_variants` |

`count_snp_regions`（`snp_region.cpp`）仍是内存批处理接口，内部用 `run_ordered`。它和这条文件流水线并行存在，算法回归继续走它。两条路径都调用 `with_alignment`（`apply_alignment.h`）。

## 所有权

1. `ReadSource` 交出来的 `bam1_t` 属于调用方。派发线程只 `bam_copy1`，不销毁源 read。
2. 拷贝归 `RegionResource`。`reset()` 和析构时销毁。worker 结束后 `RegionPool::release` 清空再放回空闲队列。
3. 参考序列字符串在 `RegionSource` 里，是 `ReferenceCache::get` 的切片拷贝。
4. 重比对产生的临时 `bam1_t` 只活在 worker 这一次调用里，计数完就销毁。
5. 写出顺序只由 `ExampleWriter` 保证。worker 可以乱序完成。

## 和 r1.6.1 已对齐的计算

这些发生在 worker 里，默认参数与 r1.6.1 一致：

- 读过滤：MAPQ &lt; 5、duplicate、secondary、supplementary、QC fail 丢掉。
- 重比对：支持数 2..300 才开窗，窗距 80 bp，k 从 10 到 101，非参考边权重至少 2。比对是全局仿射（匹配 +4、错配 −6、开 gap −8、延伸 −2），不是上游的 Smith–Waterman。
- 计数在重比对之后。SNP 阈值 `count >= 2` 且比例 `>= 0.12`，indel 是 `>= 2` 且 `>= 0.06`。同一条 read 上 indel 盖掉锚点的 SNP。

## 单元测试

```bash
cmake -S src -B src/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/build --target make_examples_unittest make_examples_runtime_unittest -j
./src/build/make_examples/test/make_examples_unittest
./src/build/make_examples/test/make_examples_runtime_unittest
```

| 二进制 | 覆盖 |
|---|---|
| `make_examples_unittest` | 过滤、SNP/indel、窗口合并、de Bruijn、仿射比对、CIGAR 合成、`count_snp_regions` 的顺序和重比对后计数 |
| `make_examples_runtime_unittest` | 资源池、参考序列阻塞、派发拷贝、过滤、乱序写出、worker 候选，以及 FASTA+BAM 端到端 SNP |

`HtsPipelineTest` 在临时目录写一个小 FASTA 和 BAM，不依赖外部数据。`VectorReadSource` 和 `SlowPileup` 在 `src/make_examples/test/runtime_test.cpp`。

文件入口：

```bash
./src/build/rodv_main make-examples \
  --ref ref.fa --reads sample.bam --out candidates.tsv \
  --nthreads 8 --decode-threads 2 --region-bp 1000
```

`--decode-threads` 只给 htslib 解压，不占用 `--nthreads` 个计算 worker。输入必须按坐标排序。TSV 每行一个候选。

## 还没有

- 计数前会把重复序列上的 indel CIGAR 左移（`left_align_cigar`）。
- pileup 固定 14 通道：0–6 是参考对齐，7–13 是 alt 对齐。indel 会把 alt 等位写进后 7 个通道。
- `PipelineConfig::truth` 给候选打 0/1/2 标签，写进 TSV 的 `label` 列。
- `gvcf=true` 时，没有候选但有深度的连续位点合成 `REF` 块。
- `phase=true` 时，支持 alt 的 read 排在支持 ref 的 read 上面。
- 第二个 `ReadSource` 是 DeepTrio 的亲本，画在子样本下方。

## 还没有

- 和上游 `make_examples` 的逐像素 diff。分相不是 `direct_phasing` 那张图，DeepTrio 也不是三样本官方模型的高度切分。gVCF 块按深度合并，没有 GQ banding。
