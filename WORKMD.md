# UF3 诊断与改进工作进度 — 2026-06-10

## 诊断概览 (Fable-5 审计)

对 `src/force/uf3.cu` (MD推理) + `src/main_uf3/` (训练器) 全部代码审阅，
对照论文 (Xie et al., npj Comput. Mater. 2023)、reference uf3 (develop 分支)、
及 GPUMD 成熟实现 (NEP/LJ/EAM) 后得出。

### A. 正确性问题 (P0 — 已完成修复)

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| A1 | 3B kernel 完全不累加 virial (压强/应力/NPT/热流全错) | 硬伤 | ✅ 已修复 |
| A2 | r < r_min 范围无排斥保护 (三次外推可塌缩) | 硬伤 | ✅ 已修复 |
| A3 | 2B ordered-pair 不对称: Si-Ge ≠ Ge-Si (违反牛顿第三定律, sumF≠0) | 硬伤 | ✅ 已修复 |
| A4 | default trim_3b=0: 3B 在 rc 处不连续 (非保守力) | 中等 | ✅ 已修复 (默认改为3) |
| A5 | 与 reference uf3 的 knot 约定不同 (trainer 自洽, ref-lammps 3B 拒载) | 中低 | 待处理 |

### B. 训练质量问题 (P1 — 已完成修复)

| # | 问题 | 说明 | 状态 |
|---|------|------|------|
| B1 | lstsq 硬编码正则化, `lambda_1/lambda_2` 未接入 | 用户参数只进 SNES/Adam loss, 不进 lstsq (最关键的训练步) | ✅ 已接入 |
| B2 | 能量行 feature 用 float atomicAdd 累加 | 大帧舍入误差 ~1e-4, 污染能量拟合下限 | ✅ 升 double |
| B3 | Host 单线程 Cholesky O(n^3) | 3B时 nparam~10^4, 单线程 Cholesky min级 | ✅ 换 cusolver potrf/potrs |
| B4 | 无 virial/stress 训练项 | `lambda_v` 已解析但未用 | ✅ 已接入+已测试 (2026-06-11) |
| B5 | SNES/ES/Adam/LBFGS 对强凸线性问题冗余 | 论文就是一次线性求解 | 文档建议 |

### C. MD 推理性能 (P2 — ✅ 已验证, 3.1× 加速)

| # | 问题 | 根因 |
|---|------|------|
| C1 | 3B kernel 并行粒度错 | thread-per-atom, 串行 O(NN²/2) 三元组, 全 warp-divergent |
| C2 | 3B 在 2B 大邻居表上跑 | rc_3b=4.25 vs rc_2b=5.5 → 候选三元组多 ~4.7× |
| C3 | 3B tensor 全走 L2 | 单元素 ~8.8KB, 完全可进 shared memory |
| C4 | double atomicAdd 风暴 | 每三元组 6次 double atomics, 切换 float 局部累加收益大 |

## 当前进度

### P0 (正确性) — 全部完成 ✅

1. **3B virial 累加** (`src/force/uf3.cu`)
   - `uf3_eval_triplet` / `uf3_eval_triplet_hoisted` 增加 `float vir[6]` 参数
   - 三元组 virial W = -Σ_edges t_e (r_e ⊗ r_e), 对称6分量
   - `find_force_uf3_3b`: 每原子累加 virial 到 `g_virial`, 与 2B 同布局
   - `compute()`: 3B kernel 传入 `virial_per_atom.data()`

2. **短程排斥保护** (`src/force/uf3.cu`)
   - 2B kernel: r < knot_min 时 clamp u=0, 用 `V(0) + V'(0)*ext` 线性外延
   - 力项 `deriv` = `V'(0)*inv_h`, 保持连续可导

3. **2B ordered-pair 对称化** (`src/main_uf3/uf3.cu`, `uf3.cuh`, `main.cu`, `lstsq.cu`)
   - `d_type_map` 改为 canonical unordered-pair map: `tmap[ti*nt+tj] = sorted(ti,tj)`
   - 非 canonical slot 的系数冻结 (整个 frozen mask)
   - `write_uf3_file`: 每个 ordered block 从 canonical slot 镜像
   - `lstsq_energy_rows` / `lstsq_force_rows`: 通过 `tmap` 路由到 canonical 列
   - **验证**: cluster virial identity PASS, FD pressure PASS (2B & 2B+3B)
   - **验证**: `max|c_SiGe - c_GeSi| = 0.0`

4. **trim_3b 默认值** (`src/main_uf3/parameters.cuh`)
   - 从 0 改为 3 (与论文 leading_trim=3, trailing_trim=3 一致)
   - 验证: FD pressure 从 FAIL → PASS (trim=3 给了正确的零边界导数)

### P1 (训练质量) — 全部完成 ✅

5. **lambda_1/lambda_2 接入 lstsq** (`src/main_uf3/lstsq.cu`)
   - `ridge_rel = para.lambda_1` (默认 1e-8)
   - `lam2b = para.lambda_2 * dm` (默认 1e-4*dm)
   - `lam3b = 10 * para.lambda_2 * dm` (默认 1e-3*dm)
   - 保留正的 env var 回退 (兼容旧 workflow)
   - 删除了 `UF3_C2/UF3_C3` env hack

6. **能量行 feature 升 double** (`src/main_uf3/lstsq.cu`)
   - 去掉 float shared-memory 中间缓冲, 直接 atomicAdd 到 double A 矩阵行
   - 语义简单, 精度提升, 无大帧舍入

7. **cusolver Cholesky** (`src/main_uf3/lstsq.cu`)
   - 用 `cusolverDnDpotrf` + `cusolverDnDpotrs` 替换 host 单线程 `cholesky()`
   - 3B (10702 params, 36 frames): 178.6s → 5.6s (**~32× 加速**)
   - falleck 到 host Cholesky (非正定时)

8. **梯度核 3B 对称化** (`src/main_uf3/fitness.cu`)
   - Adam/SNES 梯度计算后调用 `model_->symmetrize_3b_gradient(grad)`
   - 保持梯度在对称子空间内, 不自破坏模型的邻接顺序不变性

### P2 (推理加速) — ✅ 已测试 (2026-06-11, RTX 5090D + CUDA 13.1 + sm_120)

9. **P2-1: 分离 3B 紧凑邻居表** (`src/force/uf3.cu`, `uf3.cuh`)
   - 新 kernel `filter_neighbor_3b`: 每步把活动邻居表 (全局 rc, 通常 = rc_2b)
     过滤到 rc_keep = max(rc_ij, rc_ik), 保留 image-shift code
     (MIC ↔ 显式 shift 约定原样传递: g_shift_out 非空 ⇔ g_shift_in 非空)
   - 仅当 rc_3b < 全局 rc - 1e-4 时启用 (`use_3b_list_`, init 时判定);
     rc_2b=5.5 vs rc_3b=4.25 → 候选三元组 ~(5.5/4.25)^6 ≈ 4.7× 削减
   - dual-order 遗留路径也跑在过滤后的表上

10. **P2-2: warp-per-atom 3B kernel** (`find_force_uf3_3b_warp`)
    - 并行粒度: 1 warp / 中心原子; (j,k) 三角对循环扁平化为线性索引 t,
      32 lane 跨步 — 同 warp 全部 lane 处理同一原子, 消除 NN 差异分歧
    - unranking: C(j)=j(2NN-1-j)/2, float sqrt + 2 个整数修正循环;
      已用 float32 模拟穷举验证 NN≤400 全部 t 零误差
    - 3B tensor 全表 ≤48KB 时 staged 进 dynamic shared memory
      (单元素 13³≈8.8KB 必中; 2 元素 70KB 回退 L2/__ldg; 无需 opt-in attr)
    - float 累加全程: 中心原子 pe/f1/vir 在寄存器, 邻居力 f2/f3 用原生
      float atomicAdd 写入 per-atom scratch `d_scratch_3b` [10N]
      (布局 fx fy fz | pe | vir6), 越界 cutoff 的三元组跳过原子操作
    - 不用 warp shuffle (GPUMD 代码库无先例, 顾及 HIP 兼容);
      lane 末尾 10 次 float atomics 归并, 代价可忽略
    - 新 kernel `uf3_3b_collect_scratch`: 每步一次把 float scratch 折叠进
      double 全局数组 (全 N 原子, 邻居力可落在 [N1,N2) 外)
    - 旧 templated kernel 简化为 `find_force_uf3_3b_dual` (仅非对称遗留模型),
      删除 `uf3_eval_triplet_hoisted`
    - 顺手修复: 3B-only 模型 (无 2B block) 时 1B e0 之前被丢弃,
      现在由 collect kernel / dual kernel 补加 (`e0_for_3b`)
- P2-3: ncu profiling 验证 — **待做** (需 ncu 工具)

### P2-Bugfix: __ldg on shared memory pointer (2026-06-11)

12. **Bug: `uf3_eval_triplet` line 588 `__ldg()` on shared-memory tensor** (`src/force/uf3.cu`)
    - 当 3B tensor 被 stage 进 shared memory 时 (`smem_count > 0`)，`Crow` 指向
      shared memory，`__ldg(&Crow[...])` 尝试对 shared memory 执行
      `ld.global.nc` 指令，触发 `cudaErrorInvalidAddressSpace` (717)
    - **修复**: 第 588 行 `__ldg(&Crow[...])` → `Crow[...]`，普通 load 对
      shared/global 均兼容
    - 修复后 sym_3b 模型全部通过测试

### B4: virial 训练项 — ✅ 已测试 (2026-06-11)

11. **B4: lstsq virial 行** (`src/main_uf3/`)
    - `dataset.cuh/.cu`: `Uf3Frame` 增加 `virial[6]` (xx yy zz xy xz yz, eV) +
      `has_virial`; extxyz 解析 `virial="..."` (9 分量行主序, 对称化) 或
      `stress="..."` (eV/Å³, virial = -stress·V, 需有 Lattice)
    - `dataset_gpu.cuh/.cu`: `d_virial_ref` [6N_frames 帧主序] + `d_has_virial` +
      CPU 镜像 `h_virial/h_has_virial` + `num_virial_frames` 计数与打印
    - `lstsq.cu`: 新 kernel `lstsq_virial_rows` — 每帧 6 行
      (布局: 能量 [0,ncf) | 力 [ncf,ncf+3·creal) | virial 末尾 6·ncf 行);
      无参考 virial 的帧整行保持 0 (对 AtA/Atb 零贡献, 无需特判)
    - 特征与 MD virial 定义严格一致:
      2B: dW_ab/dC = -0.5·(dB/dr)/r·r_a·r_b (有序对, 每物理对两次)
      3B: dW_ab/dC = -Σ_edges (G_e/r_e)·r_e,a·r_e,b (per-centre 三元组)
    - 权重: 三路方差归一 share_e:share_f:share_v = λe:λf:λv,
      w_v = sqrt(share_v/(6·n_vf·Var(W))); 数据集无 virial 时 λv 自动退出,
      与旧的双路权重完全一致 (不破坏现有 workflow)
    - 限制: 仅 lstsq 路径; SNES/Adam/LBFGS 的 loss 仍只有 E+F
      (B5: 论文本来就是一次线性求解, lstsq 是主路径)

### P2 待验证清单 (换回 GPU 机器后)

1. 编译: `make -C src` (或项目正常构建流程), 关注新 kernel 语法/类型
2. 正确性: 重跑 `virial_check/` FD pressure + cluster virial (2B+3B);
   能量/力与 P0/P1 版本 bit 级不要求一致 (float 累加顺序变了),
   但 |ΔE| 应 <1e-5 eV/atom, |ΔF| <1e-4 eV/A
3. 小盒子 (multi-image) 路径: mini-test/uf3_md (303 atoms) 跑通且能量守恒
4. 性能: full-test/uf3_md (66990 atoms) 对比基线 6.3-9.7 M atom*step/s;
   3B vs 2B-only 的 12.5× 差距应显著缩小
5. ncu: 看 3b_warp kernel 的 SM 占用率 / L2 命中率 / atomic 吞吐
6. 邻居缓存两条路径一致性: 正常体系 (NN≤64, cache 命中) 与人为高密度
   或 max_neighbor 调大的体系 (NN>64, global fallback) 力/能量应一致;
   可用 UF3_3B_NB_CACHE 临时改小 (如 4) 重编译强制走 fallback 对照
7. 小盒子 multi-image + use_3b_list_ 组合: shift code 过滤传递是否正确
   (mini-test 303 atoms 若 cell < 2*rc 会触发)

### 第二轮审查 + 进一步优化 (2026-06-11, 未编译/未测试)

12. **审查修复** (`src/force/uf3.cu`)
    - 修复: 多 GPU 空 partition (N2==N1) 时 warp kernel grid 会算成 0
      (CUDA 非法 launch); 现 clamp 到 ≥1
    - 确认: stress= 转换约定与 main_nep 一致 (virial = -stress·|det(box)|;
      NEP 另除 num_atom 因为它存 per-atom, UF3 用帧总量与能量行一致)
    - 确认: lambda_v 在 parameters.cu 已有解析入口
    - 审查通过: filter/eval 距离表达式一致 (边界 float 舍入自洽);
      dual 路径在过滤表上每腿仍独立查 cutoff; collect 覆盖全 N
      (邻居力可落在 [N1,N2) 外); 3B-only 无 header 时 e0 索引安全 (type 全 0)

13. **P2-2b: warp kernel 邻居 shared 缓存** (`find_force_uf3_3b_warp`)
    - 每 warp 把中心原子的过滤后邻居预载进 shared memory:
      image-resolved float4 位置 (type bit-cast 进 .w) + 原子索引
    - O(NN²) 对循环每条腿从 shared 读 20B, 不再每三元组重走
      NL→pos→type 的 global 依赖链 (~2×NN 次/邻居)
    - 容量 `UF3_3B_NB_CACHE=64` (rc_3b~4-5Å 固体 NN~20-50);
      NN>64 回退 global 直读 (warp 内分支一致, 无 divergence)
    - 同步: 单次 `__syncthreads()` 覆盖 tensor staging + 全部 warp 缓存,
      之前无任何 return (inactive warp 之后才退出) — 无死锁;
      不用 __syncwarp/__shfl (HIP 兼容, GPUMD 代码库无先例)
    - smem 布局: [float4 caches | int caches | tensor], float4 区在 16B
      对齐基址; tensor 预算改为 48KB - cache (5KB), 单元素模型仍必中

### B4 待验证清单 (换回 GPU 机器后)

1. 含 `virial=` 的 train.xyz: 确认 "GPU dataset: ... virial=N/M" 计数正确,
   lstsq 打印 "(+virial rows)"
2. FD 校验: 训练后的模型在 MD 里 FD pressure 仍 PASS (特征定义与 MD 一致,
   若 virial 行引入后 FD 失败 → 检查 2B 的 0.5 因子或 3B 边符号)
3. 对照实验: lambda_v=0 vs 默认 0.1, 看 E/F loss 是否未明显劣化、
   MD 压强/应力是否更贴近 DFT 参考
4. 无 virial 数据集回归: 权重路径退化为旧双路 (E/F loss 应与上一版完全一致)
5. stress= 输入路径: 单位约定 eV/Å³ (extxyz 常见也有 GPa — 如遇 GPa 数据,
   先换算; 解析代码假定 eV/Å³)

## 验证结果
### P0+P1 基线验证 (回归, 2026-06-11 re-verified)
| 测试 | 模型 | 结果 |
|------|------|------|
| FD pressure | 2B | PASS (|err|=0.005%) |
| FD pressure | 2B+3B symmetric (warp) | PASS (|err|=0.004%) |
| FD pressure | 2B+3B stripped (dual) | PASS (|err|=0.001%) |
| Cluster virial | 2B | PASS (|err|=2.1e-9 GPa) |
| Cluster virial | 2B+3B symmetric | PASS (|err|=1.3e-8 GPa) |
| Pair symmetry | 2B (SiGe) | PASS (max|c_delta|=0) |
| Training time | 2B+3B (10702 params, 36 frames) | 7.15s (old: 5.6s; +virial rows) |

### P2 推理加速验证 (2026-06-11, RTX 5090D + CUDA 13.1 + sm_120)
| 测试 | 模型/条件 | 结果 |
|------|-----------|------|
| 编译 | sm_120, C++17 | PASS (仅 dead-code warning) |
| 小盒子 MD (303 atoms) | NVE 2000 steps, 500K | PASS, energy drift=8.5e-4 eV/atom |
| 邻居缓存一致性 | CACHE=64 vs CACHE=4 fallback | PASS, ΔU=0.0, max|ΔF|=2.4e-7 eV/Å |
| 性能 (2B+3B, 66990 atoms) | Stage 1-4 平均 | **22.9 M atom·step/s** (基线 7.4 M, **3.1x**) |
| 3B cost fraction | 2B-only vs 2B+3B | **7.7x** (基线 12.5x, **-38%**) |

### B4 virial 训练验证 (2026-06-11)
| 测试 | 条件 | 结果 |
|------|------|------|
| stress= 解析 + virial 计数 | 36 frames, train.xyz | PASS, "virial=36/36" |
| (+virial rows) 打印 | lstsq 输出 | PASS |
| 2B lstsq + virial | 54 params, 36 frames | PASS, 0.17s |
| 2B+3B lstsq + virial | 10702 params, 36 frames | PASS, 7.15s |
| FD pressure (训练后) | B4-trained 2B+3B model | PASS (|err|=0.004%) |
| lambda_v=0 回归 | 2B, no virial rows | PASS, E/F loss 退化正常 |


## MD 推理性能 (RTX 5090D, 66990 atoms, 2B+3B)

| 区间 | Speed 基线 (atom*step/s) | Speed P2 (atom*step/s) | 加速比 |
|------|--------------------------|-------------------------|--------|
| Stage 1 (200→650K) | 9.67 M | **27.44 M** | **2.84×** |
| Stage 2 (650K hold) | 7.02 M | **22.99 M** | **3.28×** |
| Stage 3 (650K hold) | 6.65 M | **19.07 M** | **2.87×** |
| Stage 4 (650→200K) | 6.30 M | **21.82 M** | **3.46×** |
| **Total time** | 74.44 s | **23.92 s** | **3.11×** |

3B cost fraction: 旧基线 2B+3B = 2.53s vs 2B-only = 0.20s → **12.5× slower**.
P2 基线: 2B+3B = 30.3M vs 2B-only = 234M atom·step/s → **7.7× slower (-38%).**

## 测试文件位置

- 源: `/mnt/c/Users/Zemeng Feng/Desktop/test-uf3/`
- 工作拷贝: `/home/zemengfeng/test-uf3-work/`
  - `mini-test/uf3_train/` — 小型训练 (36帧 SiGe)
  - `mini-test/uf3_md/` — 小型 MD (303 atoms)
  - `full-test/uf3_train/` — 全量训练 (36帧)
  - `full-test/uf3_md/` — 全量 MD (66990 atoms)
  - `virial_check/` — FD pressure & cluster virial 验证脚本

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/force/uf3.cu` | P0修复+P2重写 | 3B virial, r_min 排斥保护, 2B virial 0.5 修正; P2: filter_neighbor_3b + find_force_uf3_3b_warp + uf3_3b_collect_scratch, 旧 kernel 简化为 dual-only, e0 3B-only 修复 |
| `src/force/uf3.cuh` | 恢复+P2 | sym_3b 成员 + 声明; P2: d_NN_3b/d_NL_3b/d_NL_shift_3b, d_scratch_3b, use_3b_list_, smem_floats_3b_ |
| `src/main_uf3/uf3.cu` | P0+P1修复 | canonical pair map, 非 canonical slot frozen, 3B 对称化 |
| `src/main_uf3/uf3.cuh` | P0+P1修复 | type_map_host_ 成员, sym_3b 公开, symmetrize_3b_gradient |
| `src/main_uf3/lstsq.cu` | P1修复 | double 能量行, cusolver, lambda_1/lambda_2 接入, tmap 传导 |
| `src/main_uf3/main.cu` | P0修复 | write_uf3_file 从 canonical slot 镜像 |
| `src/main_uf3/fitness.cu` | P1修复 | 梯度 3B 对称化 (从 stash@{0} 恢复) |
| `src/main_uf3/parameters.cuh` | P0修复 | trim_3b 默认值 0→3 |

---

# 第三轮：大改动前置验证任务 (2026-06-12 下发, 电脑B执行)

> 背景: 下一阶段计划开启大工程 (方向: NEP4 推理 kernel 融合 + 局部基/小 readout
> 模型, 详见电脑A讨论记录)。开工前必须: ① 确认第二轮未测代码; ② 拿到 UF3 与
> NEP4 的 ncu/nsys 剖析数据作为决策依据。本轮**只验证, 不改代码**。
> 执行顺序: V1 → V2 → V3。V1 不过则停, 回传日志。

## V1 — 确认当前树状态 + 全量回归 (必做, 最高优先)

第二轮改动 (条目12: 多GPU空分区 clamp; 条目13: P2-2b warp 邻居 shared 缓存)
在 workmd 中标记"未编译/未测试", 但验证表中已出现 CACHE=64 vs 4 测试,
时间线有歧义。需要先确认电脑B当前文件就是最新版本。

1. 确认版本:
   ```
   cd <GPUMD仓库目录>
   git log -1 --oneline; git status; git diff --stat
   md5sum src/force/uf3.cu src/force/uf3.cuh
   grep -n "UF3_3B_NB_CACHE" src/force/uf3.cu | head -5
   ```
   - 预期: `UF3_3B_NB_CACHE` 存在 (= 含 P2-2b); 回传 md5 与 grep 输出。
   - 若 grep 无结果 → 电脑B不是最新树, 停止, 回传 git log/status, 等 A 同步。
2. 编译: 项目正常构建流程 (sm_120, 同 P2 验证时配置)。
   - 预期: 编译通过, 仅 dead-code warning。
   - 失败 → 回传完整 nvcc 命令行 + 完整错误日志 + nvcc --version。
3. 回归 (全部用当前二进制重跑, 即使之前 PASS 过):
   - `virial_check/`: FD pressure (2B / 2B+3B warp / 2B+3B dual) + cluster virial
     → 全部 PASS, 误差量级与上表一致 (<0.01%)
   - `mini-test/uf3_md` (303 atoms, 小盒子 multi-image): NVE 2000 steps
     → 跑通, energy drift ≤ 1e-3 eV/atom
   - `full-test/uf3_md` (66990 atoms): 记录 4 个 Stage 的 atom·step/s 与总时间
     → 预期 ≥ 22.9 M atom·step/s (P2-2b 缓存若此前未计入, 可能更快; 记录新数字)
   - 2B-only 同体系: 记录 atom·step/s, 更新 3B cost fraction
4. 回传: 上述每项的 PASS/FAIL + 数字, 失败项附完整 stdout/stderr。

## V2 — ncu 剖析 UF3 kernel (P2-3, 必做)

在 `full-test/uf3_md` 目录:
```
ncu --set full --launch-count 3 -k "regex:uf3" -o uf3_ncu <gpumd可执行文件>
ncu --import uf3_ncu.ncu-rep --page details > uf3_ncu.txt
```
- 注意: WSL 下 ncu 需要 GPU performance counter 权限
  (驱动设置 NVIDIA Control Panel → Developer → Allow access to GPU
  performance counters, 或以管理员运行)。若 ncu 在 WSL 不可用,
  回退方案: 在 Windows 侧原生 ncu attach, 或改用
  `nsys profile -o uf3_nsys <gpumd>` 仅拿 kernel 时间占比。
- 需要回传的指标 (对 `find_force_uf3_3b_warp` / `find_force_uf3_2b` /
  `filter_neighbor_3b` / `uf3_3b_collect_scratch` 各一份):
  1. Duration 与各 kernel 占总步时间比例
  2. Achieved occupancy / registers per thread / shared memory per block
  3. SM throughput (%) vs DRAM throughput (%) — 判断 compute-bound 还是 memory-bound
  4. L2 hit rate
  5. Warp stall 原因 Top-3 (Stall LG Throttle / Long Scoreboard / ...)
- 预期用途: 确认 3B warp kernel 当前瓶颈 (假设: tensor/坐标读取与 atomic),
  决定大工程里 moment 形式重写的收益上限。

## V3 — NEP4 同机基线 + kernel 时间分解 (大工程决策依据, 必做)

目的: 在同一台 RTX 5090D 上量化 (a) UF2/UF3 vs NEP4 的真实速度比;
(b) NEP4 各 kernel 时间占比 → 推断 kernel 融合的提速上限;
(c) 径向部分占比 → 判断"B-spline 换基"值不值得做。

1. 准备 NEP 模型: 任选其一, 优先①
   - ① UNEP-v1 (16元素): Zenodo https://doi.org/10.5281/zenodo.11533864 下载
     nep.txt; 体系用 bcc W 或等摩尔 MoTaVW, 原子数与 full-test 同量级 (~6-7万)
   - ② 手头任何现成 nep.txt + 对应体系 (注明元素与截断)
2. MD 基准: 与 full-test 相同的 run.in 结构 (NVE 或 NVT, ≥2000 steps,
   排除前 200 步预热), 记录 atom·step/s。
3. kernel 时间分解:
   ```
   nsys profile -o nep_nsys <gpumd可执行文件>
   nsys stats --report cuda_gpu_kern_sum nep_nsys.nsys-rep > nep_kern_sum.txt
   ```
   回传 `find_descriptor` / `find_force_radial` / `find_partial_force_angular` /
   `find_force_ZBL` / 邻居表 各自的时间占比表。
4. (可选, 若 V2 ncu 可用) ncu 对 `find_partial_force_angular` 与
   `find_descriptor` 各跑一份, 指标同 V2。
5. 回传汇总表: 同机同量级体系下
   | 模型 | atom·step/s | 备注 |
   |------|------------|------|
   | UF2 (2B-only) | | |
   | UF2+UF3 (warp) | | |
   | NEP4 | | |

## 失败时统一回传清单

完整命令行、完整错误日志、`nvcc --version`、`nvidia-smi` 头部、
相关文件 `git diff`、ncu/nsys 版本号。不要只回传结论。

## 本轮明确不做

- 任何源代码修改 (包括 A5 knot 约定 — 这是设计决策, 由电脑A定方案)
- NEP 径向制表实验、kernel 融合原型 — 属于大工程, 等本轮数据回来再立项
