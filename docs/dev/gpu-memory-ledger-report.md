# GPU semaphore 内存账本（memory ledger）—— 设计、实现与实验

2026-11-09，本地 SF1000 TPC-H（GPU1，RTX PRO 4500 32 GB；数据 /disk3/tpch-sf1000；本地 parquet），GOLD 本地配置。
分支均为本地分支，未 push：

- spark-rapids-jni：`/disk2/haoyangl/plugin_perf/ledger-20261109/jni`，分支 `claude-ledger`，commit `fb26620c`（基于 `b43e6989`）
- spark-rapids：`/disk2/haoyangl/plugin_perf/claude-semaphore-20261108/src`，分支 `claude-ledger`（`527766e2` 账本、`76405cd0`/`6adcd76b` 首租地板、**`b5a13c9e` 扫描显式声明解码需求**、`b986e8dc` cloud reader 声明 + IO 等待释放、`25d6207a` reserveEstimates 开关，基于 claude-main `37d0501e`）；`claude-ledger-b2` = 上述 + Codex B2（`bca4b7d8`）
- 实验 jar：`jars/ledger4-352.jar`（p5/p6）、`ledger5`（p7/p8，含 cloud reader）、`ledger6`（p9–p14）、`ledger6b2`（p10 B2）；native：`native/ledger`（`native/base` 为同 commit 未改动的对照）

## 1. 结论先行（以全 GPU workload 为准）

1. **账本加上"扫描显式声明解码需求"，可以在准入阶段预防小 pool 下的扫描冷启动 OOM，而且代价很小。**
   - 30%/50% pool 下：main 的 6 个 JVM 失败 3 个；`leddecl` 第一遍 6 个 JVM 全部通过，30% pool 补跑 4 遍的结果见第 6 节。
   - 代价：30% pool 下 Q8 63.1 s（main 成功时约 59–60 s），Q9 86.2 对 83.2 s；50% pool 与 main 持平。
   - 100% pool 无回退：Q8 52.4 对 52.1 s，Q9 79.5 对 78.5 s。
2. **为什么只靠账本不够：** 账本只在拿 semaphore 时做一次准入判断，能否防 OOM 取决于对未来需求的预留准不准。
   - 冷启动的扫描 task 足迹未知。原来的首租地板只维持 100 ms，扫描还在读文件、拼约 1 GiB 合并缓冲时地板就过期了，账本于是多放进任务。两个任务同时解码（各约 4.4 GiB）就超过 9.4 GiB 的 pool。
   - 读路径的合并 batch 走 `withRetryNoSplit`，切不开，最终报 `GpuSplitAndRetryOOM`，任务失败。
   - 把地板一直保持（p4 `ledhold`）也能做到 0 失败，但慢 28–38%：它对所有 task 收费，join/agg 也被压低了并发。
   - **显式声明只在扫描解码那一段、只对扫描 task 收费**，所以既防住了 OOM，又基本不影响并发。
3. **不带声明的账本**（`led`/`ledfl`）单独用在 GPU 扫描上过于保守：30% pool 下慢 13%，而且防不住 OOM（第 5 节）。在这台 503 GB 内存的机器上 spill 几乎免费，靠等待来避免 spill 不划算。
4. **满额 pool、16 slot（默认最优配置）下没有加速，也没有回退。** 本地 21 个可跑查询（Q15 是三段式视图查询，未测）逐个对比都在噪声内，合计 380.8 对 380.0 s（第 7 节 p11/p14）。
5. **唯一测到的速度收益出现在 slot 开得过多时：`local[48]` 下 leddecl 比 main 快约 6%**（7 个重查询 292 对 311 s，各 3 遍，区间不重叠）。main 在 Q8/Q9/Q5/Q21 上会溢出 3–21 GiB 并出现 OOM 重试阻塞，账本版本 0 溢出、0 重试。但 `local[48]` 仍比 `local[16]`（约 268 s）慢，所以账本的价值是**让性能对 slot 配置不敏感**，而不是打破当前最优记录（第 7 节 p9/p13）。
6. **S3（SF100）上没有收益。** semaphore 等待只占 slot 时间约 3%，瓶颈是 S3 IO 和 slot 数（main 从 `local[16]` 到 `local[48]`，Q17 22.9→8.2 s）。"IO 等待时释放 semaphore + 账本"在 S3 上反而更慢、更不稳定（第 8 节），因此没有做 S3 全查询挑战。
7. **Codex 的 B2（拿 semaphore 前先在 host 上做 shuffle concat）没有收益**，有没有账本都一样（第 7 节 p10）。
8. 次要结论（CPU 解码型 workload，第 4 节）：PR #16719 的 release 加上账本，在 100/50/30% pool 下比 main 快 1.14x/1.29x/1.46x；只做 release 会在小 pool 下 spill 暴涨。

## 2. 为什么需要账本

现状：PrioritySemaphore 用 32 MiB permit 计数，每个 task 的 permit 来自 dynamic estimator（stage 级 p80，冷启动用 pool / concurrentGpuTasks）。它有两个问题：

- **不知道真实占用。** permit 是对未来峰值的估计，和当前驻留量无关。持有者在做 CPU 工作（解码、shuffle 写）时，permit 仍然"占着"。一旦释放（rel），被释放者的驻留显存仍在 GPU 上，新进入者再按估计申请，就会超额，结果是 spill/retry。
- **JNI 里的 `gpu_memory_allocated_bytes` 不准。** 它只统计已注册线程的分配，不能作为准入依据。

**账本准入条件**（holders 为当前持有 semaphore 的 task，t 为申请者）：

```
allocated_total + Σ_holders max(0, reserved_h − resident_h) + max(0, reserve_t − resident_t)
    ≤ capacityFraction × pool
```

- `allocated_total`：所有 RMM 分配的真实总量，包括已释放 semaphore 但数据仍驻留的 task，以及无法归属到 task 的分配。
- `resident_h`：每个 task 当前真实驻留的字节数（JNI 新增的 per-task 账本）。
- `reserved_h`：持有者的 permit 估计。只计"尚未兑现"的部分，所以已经分配的部分不会被重复计数。
- 没有持有者时总是放行，保证不会死锁。等待者中，队首每 5 ms 重新评估一次，因为驻留显存可能在没有 release 事件的情况下被 free 或 spill。
- **首租地板（first-lease floor，ledfl）：** 第一次拿 semaphore 的 task 的足迹未知，按 `max(reserve, 4 × batchSizeBytes)` 计费，持续到它拿到 100 ms 以上或驻留超过地板为止。这修复了 Q9 宽扫描冷启动超额准入导致的 OOM：默认估计 pool/6 ≈ 1.6 GiB，真实扫描 ≈ 4.4 GiB。
  - 之前试过的严格 slow start（ledss，首租者只允许一个）在显存充裕时也会限流：Q21 55.4 s vs 44.7 s，已废弃。
- 可选 reclaim：队首等待时主动 spill 赤字部分。本次矩阵未启用。

## 3. 实现

### JNI（spark-rapids-jni，+178 行）

- `SparkResourceAdaptorJni.cpp`：
  - 新增 `live_gpu_allocations: ptr -> (owner task, size)`、`task_resident_gpu_bytes: task -> bytes` 和 `ledger_total_gpu_bytes`。
  - 在 `post_alloc_success(…, ptr)` 中，在已持有的锁内记账；在 `deallocate` 中，于 `dealloc_core` 之前销账。
  - 归属规则：线程绑定的 task_id；pool 线程只服务一个 task 时归属到该 task；否则记为 -1，只计入总量。
  - 新接口 `get_gpu_memory_ledger(task_ids, resident_out, n)`，一次加锁返回总量和各 task 的驻留量。
- `RmmSpark.getGpuMemoryLedger(long[] taskIds, long[] residentOut)`：未初始化时返回 -1；对应的 JNI 入口在 `SparkResourceAdaptor`。
- 测试 `RmmSparkTest.testGpuMemoryLedgerFollowsAllocations`。

### Plugin（spark-rapids，+384/−33 行）

- `PrioritySemaphore`：
  - 新增 `SemaphoreAdmissionPolicy` 接口和 `holders`/`firstLeaseSince` 表。
  - 设置了 policy 时，用 policy 替代 permit 计数（并发 task 上限仍然生效），等待者轮询队首。
  - 新增 `release(permits, taskId)`。
- `GpuSemaphore.GpuMemoryLedger`：实现上述准入公式；带计数器，每 10 s 打一次 `GPU_MEMORY_LEDGER` 日志；可选 reclaim。
- `HostColumnarToGpu`：PR #16719 的释放逻辑改为由开关 `spark.rapids.sql.hostToGpu.releaseSemaphore` 控制。
- 配置（均为 internal，默认关闭）：
  - `spark.rapids.sql.gpuSemaphore.memoryLedger.enabled`
  - `.capacityFraction`（1.0）
  - `.pollMs`（5）
  - `.reclaim`（false）
  - `.minFirstLeaseMs`
  - `.firstLeaseFloorBatches`（4）
- 测试：`PrioritySemaphoreSuite` 新增 3 个 case：policy 替代 permit；轮询能在没有 release 的情况下放行；task 上限仍然生效。
  - 2026-11-09 在 `claude-ledger`（`25d6207a`）上运行，10/10 通过。

## 4. CPU 扫描矩阵（p1b，Q21+Q6+Q1，n=3，36 个 JVM 全部成功）

pool 用 `allocFraction`/`maxAllocFraction` 限制为 100% / 50% / 30%（30% ≈ 9.4 GiB）。

### 总时间（s）

| pool | off | rel | ledfl | **relledfl** | relledfl vs off |
|---|---|---|---|---|---|
| 100% | 100.1 ±0.7 | 88.5 ±0.8 | 98.4 ±1.8 | **87.9 ±0.2** | 1.14x |
| 50%  | 115.6 ±0.9 | 94.7 ±1.1 | 123.2 ±1.4 | **89.9 ±1.0** | 1.29x |
| 30%  | 143.7 ±0.8 | 103.5 ±1.4 | 172.5 ±1.0 | **98.7 ±1.6** | 1.46x |

### 单查询（s）

| query/pool | off | rel | ledfl | relledfl |
|---|---|---|---|---|
| Q1 100% | 44.6 | 32.9 | 43.4 | **32.6** |
| Q1 50%  | 55.1 | 32.8 | 65.9 | **32.5** |
| Q1 30%  | 80.8 | 34.5 | 105.2 | **32.0** |
| Q21 100% | 44.7 | 44.6 | 44.7 | 44.6 |
| Q21 50%  | 49.1 | 50.7 | **46.3** | 46.6 |
| Q21 30%  | **51.6** | 57.9 | 55.8 | 55.5 |
| Q6 | ~10.4–11.5，各 cell 无差别 | | | |

### 机制指标（每查询窗口，3 次均值）

| | Q1 sem wait (task·s) | Q1 spill GiB | Q1 GPU 峰值 GiB | Q21 spill GiB | Q21 retry |
|---|---|---|---|---|---|
| p50 off | 379 | 0 | 13.3 | 1.5 | 0.67 |
| p50 rel | 7 | 30.4 | 14.9 | 18.0 | 1.0 |
| p50 ledfl | 541 | 0 | 9.4 | 0 | 0 |
| p50 relledfl | 9 | **8.9** | 14.9 | **7.3** | 0 |
| p30 off | 761 | 5.7 | 8.7 | 0.3 | 0 |
| p30 rel | 30 | 69.8 | 8.7 | 39.0 | 2.67 |
| p30 ledfl | 1134 | 0 | 5.6 | 0 | 0 |
| p30 relledfl | 9 | **47.7** | 8.7 | **9.2** | 0.33 |

### 解读

- Q1 的瓶颈是持有 semaphore 时做 CPU 解码。释放后 sem wait 从数百 task·s 降到个位数，Q1 与 pool 大小无关。
- 账本把释放带来的 spill 压回去了：Q21 在 50% pool 下从 18 → 7 GiB，30% pool 下从 39 → 9 GiB，retry 也基本消失。Q1 在 30% pool 下 spill 仍有 48 GiB，但 spill 时间只有 1.4 s（rel 为 10.8 s）。
- 只开账本不释放，会因为未兑现的预留而限流，GPU 显存利用率低（30% pool 下峰值只有 5.6 GiB）。这说明在当前插件里，"持有者"并不等于"在用 GPU"。所以账本必须和释放一起上，这也正是 lease 应该跟着显存走、而不是跟着计算走的直接证据。
- 残留问题：Q21 在 30% pool 下 relledfl 为 55.5 s，off 为 51.6 s，账本并没有完全追回释放带来的损失。原因是 Q21 的 shuffle-heavy 阶段在小 pool 下仍有少量 spill，加上准入更保守、sem wait 更高（311 vs 247 task·s）。

## 5. GPU 扫描矩阵（p2b，heavy-6：Q21/Q18/Q17 | Q5/Q1 | Q8 | Q9）

heavy-6 拆成 4 个 JVM：A=Q21,Q18,Q17｜B=Q5,Q1｜C=Q8｜D=Q9。GPU 扫描配置为 maxPartitionBytes 2g、batch 1 GiB。每个 cell 跑 n=3，失败的 JVM 记录后跳过。共 88 个 JVM，82 个成功。

### 总时间（7 个查询都成功的 run）与单查询（s）

| | p100 off | p100 ledfl | p100 led | p50 off | p50 ledfl | p30 off | p30 ledfl |
|---|---|---|---|---|---|---|---|
| 总计 | 267.5 | 267.5 | 266.4 | 274.8 (n=1) | 279.6 | 305.3 (n=2) | 346.3 (n=1) |
| Q21 | 32.2 | 32.7 | 32.1 | 35.8 | **32.8** | **37.8** | 41.3 |
| Q18 | 24.5 | 23.6 | 24.9 | 26.2 | 26.0 | **28.7** | 31.5 |
| Q17 | 33.4 | 32.8 | 33.7 | **34.1** | 34.8 | **35.2** | 44.3 |
| Q5 | 38.1 | 37.7 | 37.8 | **38.0** | 41.6 | **47.0** | 56.1 |
| Q1 | 9.4 | 9.4 | 9.5 | 10.4 | **8.8** | 11.6 | 11.3 |
| Q8 | 52.4 | 53.0 | 52.3 | **53.1** | 56.0 | **60.0** | 70.2 |
| Q9 | 77.5 | 78.3 | 76.2 | 77.3 | 79.7 | **84.8** | 91.7 |

另外，`gpu-p100-off-base`（未改动的 native 库，n=1）总计 268.9 s，说明 JNI 账本在 alloc/free 路径上的开销测不出来。

### 失败（GPU OOM）

| cell | 失败的 JVM |
|---|---|
| p50 off | Q9 ×2/3 |
| p30 off | Q9 ×1/3 |
| p50 ledfl | 0 |
| p30 ledfl | Q8 ×2/3、Q9 ×1/3（其中 1 次在 OOM 之后 shutdown 时 SIGABRT，没有产生 crash 文件） |
| p100 全部 | 0 |

所有失败都是同一条路径。查询开始后 3–4 秒，lineitem 扫描的第一波任务出现 `GpuRetryOOM`。重试升级为 split 后，`MultiFileCoalescingPartitionReaderBase.readBatchData` 用的是 `withRetryNoSplit`，输入是 2 个文件合并成的约 1 GiB 读缓冲，切不开，于是报 `GpuSplitAndRetryOOM`，任务失败。`decodeNext` 会把异常包成 IOException，但重试框架能沿 cause 链识别出来，所以包装本身不是原因。

**结论：** 冷启动 OOM 的根源是读路径。它一次解码一个约 1 GiB、不可切分的合并 batch，峰值约 4.4 GiB，而且不可重试。账本的首租地板（4×batch）只能降低概率：p50 下 0/3 对 2/3；p30 下没有改善。真正的修复应该在读路径上：让合并读支持 split-retry，或者按 pool 大小限制合并 batch。

### 机制：小 pool 下账本把 spill 换成了等待，而在这台机器上 spill 几乎是免费的

| 30% pool | off sem wait | off spill GiB / s | ledfl sem wait | ledfl spill GiB | ledfl GPU 峰值 |
|---|---|---|---|---|---|
| Q17 | 191 | 1.1 / 0.1 | 410 | 0 | 5.8 / 9.4 GiB |
| Q5 | 314 | 9.6 / 1.0 | 512 | 0 | 7.4 |
| Q8 | 335 | 106.5 / 6.8 | 603 | 1.5 | 8.6 |
| Q9 | 464 | 88.1 / 5.2 | 699 | 1.0 | 8.5 |

sem wait 和 spill_s 都是 task·s 累计。

- 503 GB 内存加 PCIe 5，spill 100 GiB 的累计代价只有约 7 task·s；而账本让 sem wait 多出 200–300 task·s，SM 利用率从 58% 降到 47%。
- 账本保守的来源，是持有者那部分尚未兑现的预留（p80 估计）。Q17 在 30% pool 下峰值只用到 5.8/9.4 GiB。
- CPU 扫描时开 release 不同：持有者只在真正做 GPU 工作时才持有 lease，预留很快就兑现，保守性就小得多。这就是为什么账本只有和 release 配套才有正收益。
- 唯一的例外是 Q21/Q1 在 50% pool：ledfl 消除了 retry，反而更快（32.8 vs 35.8，8.8 vs 10.4）。

### p3（已中止）

计划测试三种"把等待换回 spill"的参数：主动 reclaim、capacityFraction=1.3、地板 1 batch。跑了 2/42 后中止，改为排查 OOM 根因（第 6 节）。

## 6. 根因定位与显式声明预留（p4、p5、p6）

### p4：验证"地板过期"假设（30% pool，Q8/Q9，n=3，ledger3）

| | Q8 | Q9 | 失败 |
|---|---|---|---|
| main | 59.2 s | 84.2 s | **3/6** |
| ledhold（地板 4 GiB 保持到驻留超过地板，`minFirstLeaseMs=600000`） | 81.9 | 107.8 | 0/6 |
| ledhold5（地板 5 GiB） | 99.6 | 132.9 | 0/6 |

假设成立，但地板会让整个查询的并发变低（SM 利用率 58% → 40%，sem wait 翻倍）。

### 实现：扫描显式声明（`b5a13c9e`）

- `GpuSemaphore.declareGpuDemand(ctx, bytes)` / `clearGpuDemand(ctx)`。账本记录 `target = 声明时驻留量 + decodeReserveFraction × bytes`。在准入公式里，这个 task 不论是持有者还是候选者，都按 `max(原收费, target − resident)` 计。task 结束时自动清除。
- Parquet 合并读 `readBufferToTablesAndClose`：在 `acquireIfNecessary` 之前声明 `dataSize + max(chunked reader 内存上限, 目标 batch)`。默认配置下就是压缩数据大小 + 4 GiB。
- `MultiFileCoalescingPartitionReaderBase.readBatchData`：`withRetryNoSplit` 返回时 producer 已被 `CachedGpuBatchIterator` 读空，在 finally 里清除声明。
- 新配置 `spark.rapids.sql.gpuSemaphore.memoryLedger.decodeReserveFraction`（internal，默认 1.0，设为 0 关闭）。只在账本开启时生效。
- 目前接了 parquet 合并读（本地文件）和 parquet cloud reader（`b986e8dc`，S3）；单文件 reader 和 ORC 尚未接入。

### p5：声明预留（`leddecl`，地板关闭）对比 main，n=1

| | main | leddecl |
|---|---|---|
| 30% Q5 / Q1 | **OOM 失败** | 46.3 / 11.1 |
| 30% Q8 | **OOM 失败** | 63.1 |
| 30% Q9 | 83.2 | 86.2 |
| 50% Q5 / Q1 | 38.2 / 10.4 | 38.1 / 8.6 |
| 50% Q8 | **OOM 失败** | 52.5 |
| 50% Q9 | 78.0 | 78.1 |
| 100% Q5 / Q1 | 37.5 / 9.5 | 37.3 / 9.0 |
| 100% Q8 | 52.1 | 52.4 |
| 100% Q9 | 78.5 | 79.5 |

- 每个 JVM 约有 230 次声明，首租地板拒绝为 0，说明拦住多余任务的是声明。
- 30% pool Q9：spill 从 88.5 降到 17.5 GiB，retry 从 2 降到 0。sem wait 从 448 增加到 537 task·s，涨幅远小于 ledhold。

### p6：30% pool 下 leddecl 补跑 4 遍（OOM 率）

12 个 JVM 全部成功。加上 p5 的 3 个，leddecl 在 30% pool 下为 **0/15 失败**。同条件下 main 在 p2b/p4/p5 中合计约一半失败（Q5、Q8、Q9 均出现过）。

时间（4 次均值）：Q5 46.27 s，Q1 11.06 s，Q8 63.22 ±0.60 s，Q9 85.29 ±0.50 s。Q8 和 main 成功时的 59–60 s 相比慢约 5%，Q9 和 83–85 s 相比基本持平到慢 2%。

## 7. 满额 pool 下的性能（p9–p11、p13、p14，本地 SF1000，jar `ledger6`）

`leddecl` = 账本 + 扫描声明，保留估计预留（`reserveEstimates=true`）；`off` = main 行为。下面的总时间都是 7 个重查询（Q21/Q18/Q17/Q5/Q1/Q8/Q9）之和，单位 s。

### slot 数（p9 n=1；`local[48]` 由 p13 补跑 2 遍，合计 n=3）

| slot | off | leddecl | leddecl 相对 off |
|---|---|---|---|
| `local[16]` | 271.4 | 267.7 | 快 1.4%（噪声内） |
| `local[32]` | 281.9 | 278.0 | 快 1.4%（噪声内） |
| `local[48]` | 311.0（317.9 / 306.7 / 308.5） | 292.1（291.3 / 287.7 / 297.3） | **快 6.1%** |

- p13 `local[48]` 单查询（2 遍均值）：Q8 61.4 对 55.4 s，Q5 48.2 对 43.8 s，Q9 89.1 对 85.1 s，其余持平。
- 机制：
  - main 在 Q8 上溢出 16.5 GiB，并有 13.9 s 重试阻塞；Q9 溢出 7.0 GiB，Q5 溢出 3.1 GiB。p9 中 Q21 溢出 20.7 GiB，另有 24 s 重试阻塞。
  - leddecl 在所有查询上 0 溢出、0 重试。
  - 两边 sem wait 相近，说明账本并不是靠多等待换来的。
- 解读：slot 多时，main 的 permit 估计会让太多 task 同时持有 GPU，真实驻留超出 pool；账本按真实驻留加声明挡住了这部分。16 slot 本来就没有超额，所以那里没有收益。

### B2 shuffle（p10，`local[16]`，n=1）

off 266.2 s，leddecl 265.3 s，offb2 266.8 s，leddeclb2 266.6 s。B2 和账本都没有影响，放弃 B2 方向。

### 全部查询（p11 + p14，`local[16]`，n=1）

| 查询 | off | leddecl | 查询 | off | leddecl |
|---|---|---|---|---|---|
| Q1 | 9.34 | 9.10 | Q12 | 9.48 | 8.44 |
| Q2 | 8.93 | 9.12 | Q13 | 8.36 | 8.23 |
| Q3 | 10.94 | 11.02 | Q14 | 4.44 | 4.22 |
| Q4 | 7.19 | 7.01 | Q16 | 3.49 | 3.52 |
| Q5 | 38.03 | 37.30 | Q17 | 32.44 | 33.34 |
| Q6 | 2.93 | 2.61 | Q18 | 25.05 | 26.03 |
| Q7 | 12.86 | 12.62 | Q19 | 10.01 | 9.38 |
| Q8 | 52.26 | 52.48 | Q20 | 9.30 | 9.04 |
| Q9 | 78.15 | 80.29 | Q21 | 31.79 | 30.85 |
| Q10 | 11.91 | 11.61 | Q22 | 2.83 | 2.80 |
| Q11 | 11.09 | 10.95 | **合计（21 个）** | **380.8** | **380.0** |

Q15 在测试框架里是 `query15_part1/2/3` 三段式，未测。其余查询全部成功，0 失败。

## 8. S3 SF100（p7、p8、p12，`s3a://ndsh/parquet_float_sf100`，us-west-2）

S3 读走 cloud reader，所以这里额外测了两个变体：
- `io`：开启 `releaseSemaphoreOnIoWait`，即下一个 buffer 还没就绪时，先释放 semaphore 再阻塞等待。
- `ioled`：`io` 加账本。

7 个重查询总时间（s）：

| plan | slot | off | io | led | ioled |
|---|---|---|---|---|---|
| p7（n=1） | 16 | 147.9 | 163.6 | 164.0 | 481.6（卡顿：Q1 247 s，Q5 112 s） |
| p8（n=1） | 16 | 154.2 | | | 159.9 |
| p8（n=1） | 48 | 220.6（卡顿：Q5 71 s，Q8 90 s） | | | 226.4（卡顿：Q21 110 s） |
| p12（n=2，缩短 s3a 超时） | 48 | 86.5 ± 8.2 | | | 115.7 ± 22.3 |

- **S3 有随机的 1–4 分钟读卡顿**，main 也会遇到，卡顿期间 CPU 和 GPU 都空闲。p12 缩短了 s3a 超时（connection 15 s，establish 5 s）后，没再出现分钟级卡顿，`local[48]` 下 main 的 7 个查询只用 81–92 s。
- **杠杆是 slot 数，不是 semaphore。**
  - 不卡顿时，main 从 16 slot 到 48 slot：Q21 22.8→12.2 s，Q18 13.2→5.6 s，Q17 22.9→8.2 s。
  - `local[16]` 在 S3 上受 IO 并发限制。即使开到 48 slot，sem wait 也只占 slot 时间约 3%。
- **ioled 没有更好，在 p12 中明显更慢**（Q5 28.5 对 15.9 s，Q1 9.8 对 5.4 s）。
  - 可能的原因：释放后要重新排队拿 semaphore；更多并发读放大了 S3 抖动。
  - 没有进一步定位：S3 噪声大，而且 semaphore 不是瓶颈，这个方向的上限很低。
- S3 上 Q17/Q5/Q8/Q9 的输出 hash 在各次运行之间也会变，off 和 ioled 都有。数据集是 float 列，求和顺序不确定。

## 9. 注意事项 / 局限

- 只在单 GPU、单 executor 下测过，没有测多 executor 共享 GPU。S3 只测了 SF100 的 7 个重查询。
- JNI 账本在每次 alloc/free 时多一次 hash map 操作，而且是在已持有的锁内。微基准结果见第 5 节附录。
- Q1 输出 hash 在 main（off）上同样会变化：浮点求和顺序不确定，不是正确性问题。
- 为了加载自建的 JNI，实验使用了 `spark.rapids.cudfVersionOverride=true`、`-Dai.rapids.cudf.lib-native-dir`，以及 `-XX:-CreateCoredumpOnCrash`。
