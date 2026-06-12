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

---

# 第三轮验证执行报告 (电脑B, 2026-06-11)

## 环境

| 项 | 值 |
|----|-----|
| GPU | NVIDIA GeForce RTX 5090 D v2, sm_120 |
| nvcc | V13.1.115 (CUDA 13.1) |
| ncu | 2025.4.1 (可用但 ERR_NVGPUCTRPERM) |
| nsys | 2025.5.2 (正常) |
| git commit | 3bfa17ab Update WORKMD.md |
| md5 uf3.cu | 913244e8625a31ea57307b9441a48be6 |
| md5 uf3.cuh | 5a204e5ba3cd0595a8df3dfdafe2e680 |
| UF3_3B_NB_CACHE | line 233, =64 (P2-2b 在树) |
| 编译配置 | CC=/usr/local/cuda-13.1/bin/nvcc CUDA_ARCH=-arch=sm_120 CFLAGS=-std=c++17 |

## V1 — 全量回归 ✅

### 编译
- gpumd ✅, uf3 ✅, nep ✅
- Warning only: precompute_3b_basis_uniform, apply_mic, uf3_grad_2b 等未引用 (死代码)

### 正确性回归 (使用当前二进制重跑)

| 测试 | 结果 | 误差 |
|------|------|------|
| FD pressure 2B | PASS | \|err\|=0.005% |
| FD pressure 2B+3B warp | PASS | \|err\|=0.004% |
| FD pressure 2B+3B dual | PASS | \|err\|=0.001% |
| Cluster virial 2B | PASS | \|err\|=2.1e-9 GPa |
| Cluster virial 2B+3B | PASS | \|err\|=1.3e-8 GPa |
| Mini MD (303 atoms, NVE 2000) | PASS | drift=0.618 meV/atom |

### 性能回归 (66990 atoms SiGe 2B+3B)

| Stage | atom·step/s | vs 基线 |
|-------|------------|---------|
| 1 (200→650K) | 27.53 M | 2.85× |
| 2 (650K) | 23.04 M | 3.28× |
| 3 (650K) | 22.65 M | 3.41× |
| 4 (650→200K) | 21.93 M | 3.48× |
| **Total time** | **22.77 s** | **3.27×** (基线 74.44s) |

2B-only: 164.5 M atom·step/s
3B cost fraction: **6.9×** (基线 12.5×, -45%)

## V2 — UF3 kernel 时间分解 (nsys) ⚠️ ncu 不可用

ncu 报 ERR_NVGPUCTRPERM (WSL 无 GPU perf counter 权限)。
回退 nsys profile 拿到 kernel 时间占比 (8000 steps, 66990 atoms):

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_force_uf3_3b_warp` | **90.6%** | 18.33 s | 2.29 ms |
| `find_force_uf3_2b` | 2.9% | 0.59 s | 74.3 µs |
| `filter_neighbor_3b` | 1.9% | 0.39 s | 48.4 µs |
| `uf3_3b_collect_scratch` | 0.2% | 38.6 ms | 4.8 µs |
| 邻居表构建+排序 | 1.4% | 0.29 s | — |
| thermo/verlet/pbc/其他 | 3.0% | — | — |

关键发现:
- 3B warp kernel 独占 90.6% GPU 时间 → 优化 3B 是唯一有效方向
- 2B kernel 仅 2.9% → 3B/2B 单 kernel 速度比 ~31×
- filter_neighbor_3b + collect_scratch 合计仅 2.1% → 开销可忽略
- 邻居表构建 1.4% → 不是瓶颈

ncu 缺失指标 (occupancy/register/L2/stall): 需要 Windows 原生 ncu 或管理员权限。

## V3 — NEP4 同机基线 ✅

NEP4 模型: SiGe, cutoff=6/4, n_max=8/8, basis_size=8/8, l_max=4/2/0
同体系 66990 atoms, 相同 run.in (4-stage NVT Berendsen 200↔650K)

### 速度对比

| Stage | UF3 (2B+3B warp) | NEP4 |
|-------|-------------------|------|
| 1 (200→650K) | 27.53 M | 25.26 M |
| 2 (650K) | 23.04 M | 25.60 M |
| 3 (650K) | 22.65 M | 25.52 M |
| 4 (650→200K) | 21.93 M | 25.31 M |
| **Average** | **23.8 M** | **25.4 M** |

NEP4 与 UF3 2B+3B 基本持平 (NEP4 略快 7%)。
注意: UF3 在不同 Stage 波动 (21.9~27.5 M)，NEP4 非常稳定 (25.3~25.6 M)。

### NEP4 kernel 时间分解 (nsys)

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_descriptor` | **47.7%** | 9.21 s | 1.15 ms |
| `find_partial_force_angular` | 23.3% | 4.51 s | 0.56 ms |
| `find_force_radial` | 17.6% | 3.40 s | 0.42 ms |
| `find_neighbor_list_large_box` | 4.3% | 0.83 s | 0.10 ms |
| `gpu_find_force_many_body` | 2.9% | 0.55 s | 69.1 µs |
| 邻居表构建+排序 | 1.3% | — | — |
| 其余 | 3.0% | — | — |

NEP4 计算分布:
- descriptor (径向+角向基函数): 47.7%
- angular force: 23.3%
- radial force: 17.6%
- 合计 88.6% 在力计算, compute-bound

### UF3 vs NEP4 结构对比 (供大工程决策)

| 维度 | UF3 2B+3B | NEP4 |
|------|-----------|------|
| 推理速度 | 23.8 M | 25.4 M |
| 主 kernel 占比 | 3B warp 90.6% | descriptor 47.7% |
| 瓶颈类型 | 3B triplet loop (O(NN²)) | descriptor + angular |
| 3B/angular 单 kernel 耗时 | 2.29 ms | 1.15+0.56=1.71 ms |
| 径向部分 | 2B 0.59s (2.9%) | radial 3.40s (17.6%) |
| 速度稳定性 | 波动 (21.9-27.5M) | 稳定 (25.3-25.6M) |

大工程推论:
- UF3 3B kernel 融合 + moment 形式重写: 理论上限 ~31× (降到 2B 水平),
  实际可达 3-5× (考虑 triplet 无法完全消除)
- NEP4 descriptor 占比 47.7%, kernel 融合 descriptor+radial+angular
  理论上限 ~2×
- B-spline 换基 (descriptor→simpler radial): NEP4 radial 仅 17.6%,
  即使完全消除也只省 17.6%, 收益有限

---


---

# P3: UF3 推理加速二期 (R7 立项, 2026-06-12, 电脑A)

## 背景

P2 后 3B warp kernel 仍占 90.6% GPU 时间 (5090D nsys), 2B+3B vs 2B-only = 6.9×。
R4 曾冻结 UF3 性能改动, R7 解冻。M1 教训 (见 progress/README.md R6) 适用:
动刀前必须有微架构数据, 不做想当然的结构手术。

## 候选方向 (按 ncu 数据二选一或组合)

**P3-1 若 compute-bound: 分腿部分收缩 (per-j hoisting)**
- 现状: 每 (j,k) 三元组独立做 64 系数张量积 ×(值+3导数)
- 改法: 固定 ij 腿, 先把 c_lmn 与 B_l(r_ij)、B'_l(r_ij) 收缩成
  A_mn / A'_mn (每 j 一次 2×64 FMA), 此后每个 k 只需
  ~3×20 FMA 而非 ~4×64 — 理论 FLOP 降 2-3×
- 代价: warp 内并行粒度要从"lane-per-三元组"改为"lane-per-j + 串行 k"
  或 j-chunk 方案, 有负载不均风险; A_mn 放寄存器 (16+16 float, 编译期下标,
  符合 M1 教训 L1)
- 注意: P2 曾删除 uf3_eval_triplet_hoisted (per-triplet hoisting, 无跨 k 复用);
  本方案是跨 k 复用, 不是同一个东西

**P3-2 若 memory/L2-bound: tensor 占用压缩**
- SiGe 2 元素 tensor 70KB 已超 48KB smem 上限, 走 L2/__ldg (P2-2 记录);
  多元素 (4 元素 64 个 type-triple) 更甚
- 改法 a: 对称模型 canonical 存储 (t2≤t3) 省 ~25-50%
- 改法 b: tensor 降 bf16/fp16 存储 + float 累加 (减半占用, 2 元素可回 smem);
  需精度验证 (FD pressure + force RMSE 对照), 风险中等

## 电脑B任务 (P3-V0, 按序)

1. **解锁 H100 ncu** (阻塞项, 优先):
   - 试 `sudo ncu --version` 及 sudo 跑一次 profile;
   - 无 sudo 权限则请管理员写入
     `/etc/modprobe.d/nvidia-profiling.conf`:
     `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, 重启生效
2. **UF3 H100 基线**: full-test/uf3_md (66990 atoms SiGe) 在 H100 跑
   4-stage 基准: 2B+3B (warp) 与 2B-only 的 atom·step/s
   (5090D 数字不可比, H100 须自建基线)
3. **ncu 剖析 `find_force_uf3_3b_warp`** (解锁后):
   ```
   ncu --set full -k "regex:uf3_3b_warp" --launch-count 3 -o uf3_p3_ncu <gpumd>
   ```
   回传: achieved occupancy / registers per thread / smem per block /
   SM throughput % vs DRAM throughput % / L2 hit rate /
   warp stall top-3 / smem bank conflict
4. 顺带: NEP4 legacy 三 kernel 同样来一份 ncu (M2 寄存器预算标定, 非阻塞)
5. 回传后 A 依据 SM% vs DRAM% 与 stall 分布在 P3-1 / P3-2 间拍板并出 patch

## 判定标准 (预先声明, 防事后解释)

- SM throughput > 60% 且 stall 以 Wait/Not Selected 为主 → compute-bound → P3-1
- DRAM/L2 throughput 高 或 stall 以 Long Scoreboard/LG Throttle 为主 → P3-2 优先
- occupancy < 25% 且 registers/thread > 128 → 先做寄存器瘦身再谈其他

---

# P3-V0 执行记录 (电脑B, 2026-06-12, H100)

## 环境

| 项 | 值 |
|----|-----|
| GPU | NVIDIA H100 PCIe, 81559 MiB, sm_90 |
| Driver | 580.65.06 |
| nvcc | V13.0.48 (CUDA 13.0) |
| ncu | 2025.3.0 |
| nsys | 2025.3.2 |
| git commit | f9a48617 (uf3-dev) |
| 二进制 | src/gpumd (sm_90, c++17) |

## 任务1: 解锁 H100 ncu — 阻塞

- `sudo ncu`: 无 sudo 密码, 不可用
- `/proc/driver/nvidia/params` 确认: `RmProfilingAdminOnly: 1` (只允许 root profiling)
- 需管理员: 创建 `/etc/modprobe.d/nvidia-profiling.conf` 写入
  `options nvidia NVreg_RmProfilingAdminOnly=0`, 然后重载 nvidia 模块或重启
- **当前状态: ncu 不可用, P3-V0 任务3/4 阻塞**

## 任务2: UF3 H100 基线

测试体系: 66990 atoms SiGe, 4-stage NVT Berendsen (200->650->650->200 K),
每 stage 2000 steps, timestep 1 fs.

UF3 模型: SiGe.uf3 (2B 4 blocks, 3B 6 triplets, trim_3b=3),
3B coeff dims=15x15x15, rc_3b=(5.5,5.5,5.5) A.

### 2B-only (SiGe_2b.uf3)

| Stage | atom-step/s |
|-------|------------|
| 1 (200->650K) | 212.0 M |
| 2 (650K) | 176.4 M |
| 3 (650K) | 171.2 M |
| 4 (650->200K) | 175.4 M |
| **Average** | **183.8 M** |

### 2B+3B (SiGe.uf3, warp kernel)

| Stage | atom-step/s |
|-------|------------|
| 1 (200->650K) | 18.2 M |
| 2 (650K) | 16.3 M |
| 3 (650K) | 12.5 M |
| 4 (650->200K) | 10.1 M |
| **Average** | **14.3 M** |

### 3B cost fraction: 2B+3B vs 2B-only = **12.9x**

3B kernel 状态: compact (rc_3b-filtered) neighbour list,
tensor in global memory (L2), warp-parallel symmetric kernel.
Tensor 未进 shared memory (8 type triplets 总大小远超 48KB smem 上限)。

速度有明显 stage 依赖性 (10.1-18.2M), 与 3B triplet O(NN²) 的温度/密度敏感性一致。

### H100 与 5090D 对比 (不同模型, 仅供参考)

| 指标 | H100 (本报告) | 5090D (R3 V3) |
|------|-------------|---------------|
| 2B-only | 183.8 M | 164.5 M (+12%) |
| 2B+3B (warp) | 14.3 M | 23.8 M (-40%) |
| 3B cost fraction | 12.9x | 6.9x |

H100 2B-only 比 5090D 快 12%, 但 H100 2B+3B 反而慢 40%。
可能原因: 不同模型参数 (coeff dims=15 vs 13, 8 triplets vs 更少),
H100 sm_90 的 shared memory bank 配置与 5090D sm_120 不同,
tensor L2 访问模式在 H100 上不利。
5090D 数字来自不同 UF3 模型, 非严格可比。

## 任务3/4: ncu 剖析 — 阻塞

等待管理员解锁 ncu profiling 权限 (RmProfilingAdminOnly=0)。

---

# P3-R2: 基于 P3-V0 的决策 (电脑A, 2026-06-12)

## 对 P3-V0 的解读

1. **rc_3b=5.5 是 12.9x 的主因**: 与 rc_2b 相同 → P2 的 3B 紧凑表零收益,
   z(5.5)~30 → ~450 三元组/原子。UF3 论文对 W 用 5.5/4.25 分离截断,
   理由就是速度 (三元组 ~(5.5/4.25)^6 ≈ 4.7x)。这是模型选择, 不是 kernel 问题。
2. **tensor 108KB 进不了 smem** (15^3 x 8 blocks), 每三元组 64 系数走 L2;
   H100 2B+3B 比 5090D 慢 40% 与 L2 依赖 + H100 低时钟一致。
3. **P3-1 (per-j hoisting) 撤回**: 15-knot 模型的 A_mn/A'_mn 中间量
   = 225+225 float/线程 → 必进 local memory, 违反 M1 教训 L1。
   仅在"邻居按 r_ik 排序 + 滑动窗口"下可行, 复杂度/收益比差, 降级到无限期。

## 决策: 双轨

**Track 1 (模型侧, 优先, 零代码改动):** rc_3b 重拟合实验
**Track 2 (kernel 侧, 数据先行):** ncu 解锁 + maxrregcount 探针 → 再定 P3-2

## 电脑B任务 P3-V1 (Track 1, 今天可做)

1. 用现有训练器重拟合 SiGe (同一 train.xyz, lstsq 路径):
   - 变体 A: rc_3b=4.25 (2B 仍 5.5), knots 数不变
   - 变体 B: rc_3b=4.25 + 3B 13 knots (为 smem 铺路: 13^3x4Bx6=52.7KB)
   - (2B block 完全不动, 1B e0 照常)
2. 每个变体验收:
   - FD pressure (2B+3B) PASS
   - 测试集 E/F RMSE vs 现 rc5.5 模型: 劣化 <15% 接受, >15% 回传数据再议
   - cluster virial PASS
3. 通过验收的变体跑 4-stage 基准 (66990 atoms):
   预期 3B kernel 三元组数 ~4.7x 削减, 端到端 2.5-3.5x;
   并确认日志显示 use_3b_list_ 已启用 (rc_3b < rc_2b 触发 P2-1 过滤表)
4. 回传: 两个变体的 RMSE 对照表 + 基准表 + FD/virial 结果

## 电脑B任务 P3-V2 (Track 2, 并行/非阻塞)

1. 继续推进 ncu 解锁: 请管理员写
   `/etc/modprobe.d/nvidia-profiling.conf`:
   `options nvidia NVreg_RmProfilingAdminOnly=0` + 重载模块/重启
2. maxrregcount 探针 (无 ncu 也能判方向):
   - 以 CFLAGS 追加 `-maxrregcount=64` / `96` / `128` 各编译一版
     (只为探针, 不入库), 跑 2B+3B 基准 stage-2 即可
   - 三版性能敏感 → 寄存器/occupancy 受限; 几乎平坦 → memory-bound
     → P3-2 (canonical + bf16 tensor 入 smem) 升主攻
3. 回传三版数字 + 默认版 registers/thread (编译时加 `-Xptxas -v` 抓
   uf3_3b_warp 的 reg 数, 不需要 ncu)

## 预期决策树 (预先声明)

- Track 1 RMSE 通过 → rc_3b=4.25 成为推荐配置, 写入文档;
  变体 B 同时通过 → P3-2 smem staging 收益翻倍, 升优先
- maxrregcount 敏感 → A 出寄存器瘦身 patch;
  平坦 + ncu 解锁后确认 L2-bound → A 出 canonical+bf16 smem patch (P3-2)
---

## B执行的ncu剖析 (叠加入A的P3-R2)

# P3-V0 追加: ncu 已解锁 + 完整剖析 (电脑B, 2026-06-12, H100)

## ncu 状态: 已解锁 ✅

`RmProfilingAdminOnly` = 0, ncu 2025.3.0 正常可用。

## 任务3: UF3 3B warp kernel ncu 剖析

`find_force_uf3_3b_warp`, (128,1,1)x(16748,1,1), 4.70 ms/launch, CC 9.0

### Occupancy & Launch

| Metric | Value |
|--------|-------|
| Block Size | 128 |
| Registers Per Thread | **90** |
| Dynamic Shared Memory Per Block | 5.12 KB |
| Shared Memory Config Size | 65.54 KB |
| Theoretical Occupancy | 31.25% |
| Achieved Occupancy | **27.73%** |
| Block Limit | Registers (5 blocks/SM) |
| Active Warps Per Scheduler | 4.45 |
| Eligible Warps Per Scheduler | **1.08** |

### Compute vs Memory

| Metric | Value |
|--------|-------|
| SM Frequency | 1.09 GHz |
| Compute (SM) Throughput | **55.58%** |
| Memory Throughput | **63.87%** |
| DRAM Throughput | 0.15% |
| L1/TEX Cache Throughput | 64.69% |
| L2 Cache Throughput | 38.39% |
| L2 Hit Rate | **99.74%** |
| L1/TEX Hit Rate | 93.18% |
| Elapsed Cycles | 5,144,125 |
| Duration | 4.70 ms |

判定: 略偏 memory-bound (63.87% vs 55.58%)，但接近平衡。
Compute 仍有提升空间。L2 hit rate 极高 (tensor cache resident)。

### Stall Analysis

| Metric | Value |
|--------|-------|
| Warp Cycles Per Issued Instruction | **7.89** |
| Issued IPC (active) | 2.25 |
| Issue Slots Busy | 56.30% |
| No Eligible (issue slot idle) | 43.51% |
| Eligible Warps Per Scheduler | 1.08 |

**Top Stall**: L1TEX Scoreboard (global/local memory dependency) — **30.0%** (2.4 cycles/issue)
- 主因: warp 等待 L1TEX (global/local) 数据返回

**Thread Divergence**: Avg active threads = 15.85/32 (predication off = 14.91)
- 约 29.6% warp 效率损失来自分支/谓词

### Memory Access Efficiency

| Metric | Value | 严重度 |
|--------|-------|--------|
| Global Load Efficiency | **4.2/32 bytes** (13.1%) | ⚠️ 极差: stride/uncoalesced |
| Local Load Efficiency | **8.1/32 bytes** (25.3%) | ⚠️ 差 |
| Local Store Efficiency | **1.1/32 bytes** (3.4%) | ⚠️ 极差 |
| Shared Load Bank Conflicts | **2.6-way** (13.26% conflicts) | ⚠️ 中等 |
| Local Memory Spilling | **0** | ✅ 无 spill |
| L2 Compression Success | 0% | — |

### P3 方向判定

按 progress/uf3-dev.md P3 判定标准:
- SM throughput 55.58% < 60% → 非纯 compute-bound
- DRAM throughput 0.15% → 非 DRAM-bound
- L2 throughput 38.39% → 中等
- Memory throughput 63.87% → 偏 memory-bound (L1/L2)
- Occupancy 27.73%, regs=90 < 128 → 不需要 shrink-regs-first

**结论: 介于 P3-1 (compute) 和 P3-2 (memory) 之间，两方向均可尝试。**

P3-1 (per-j hoisting): compute 55.58% 有提升空间，减少 FMA 可降低 L1TEX pressure
P3-2 (tensor compression): 当前 tensor 全量在 L2 (99.74% hit)，compression
  可让 2-4 元素模型进 smem。但本模型 8 triplets 即使压成 fp16 也需 54KB
  (17KB/smem) — 仍超 65KB smem 预算。只有 2 元素模型能进 smem。
- 额外方向: shared memory bank conflict (2.6-way) 可优化 neighbor cache 布局
- Thread divergence (29%+): warp-per-atom 并行, 各原子 NN 不同致 lane 间负载不均,
  可考虑 neighbor-padding 或 NN cutoff

## 任务4: NEP4 legacy 三 kernel ncu 剖析 (M2 寄存器预算标定)

### find_descriptor

| Metric | Value |
|--------|-------|
| Block Size | 64 |
| Registers Per Thread | **165** |
| Achieved Occupancy | **14.45%** (register-limited) |
| Duration | 2.33 ms |
| Compute (SM) Throughput | 26.95% |
| Memory Throughput | 42.90% |
| L1/TEX Hit Rate | 85.24% |
| L2 Hit Rate | 98.84% |
| DRAM Throughput | 4.68% |

### find_force_radial

| Metric | Value |
|--------|-------|
| Block Size | 64 |
| Registers Per Thread | **64** |
| Achieved Occupancy | **24.72%** |
| Duration | 0.74 ms |
| Compute (SM) Throughput | 39.88% |
| Memory Throughput | 36.62% |
| L1/TEX Hit Rate | 70.59% |
| L2 Hit Rate | 96.49% |

### find_partial_force_angular

| Metric | Value |
|--------|-------|
| Block Size | 64 |
| Registers Per Thread | **255** |
| Achieved Occupancy | **11.11%** (heavily register-limited) |
| Duration | 1.49 ms |
| Compute (SM) Throughput | 29.59% |
| Memory Throughput | 11.85% |
| L1/TEX Hit Rate | 58.44% |
| L2 Hit Rate | 79.91% |

### NEP4 kernel 寄存器预算总结 (M2 参考)

| Kernel | Regs/Thread | Occupancy | SM Util | 特征 |
|--------|------------|-----------|---------|------|
| descriptor | 165 | 14.45% | 26.95% | 基函数计算重, large local arrays |
| radial | 64 | 24.72% | 39.88% | 最轻量, 效率最高 |
| angular | 255 | 11.11% | 29.59% | sum_fxyz local arrays 吃寄存器 |
| **UF3 3B warp** | **90** | **27.73%** | **55.58%** | 寄存器控制好, 但 memory pattern 差 |

M2 教训: NEP4 angular kernel 的 255 reg/thread 是设计反例 —
per-thread local sum_fxyz[NUM_OF_ABC][MAX_NUM_N] 数组完全落在寄存器，
致 occupancy 仅 11%。M2 应避免此类 large per-thread arrays。
M1 fused kernel 直接复制了这一错误。
UF3 3B warp kernel 的 register 控制 (90) 相对好。
 (docs(progress): P3-V0 — UF3/NEP4 ncu microarchitecture profiling on H100)

---

# P3-V1 执行记录 (电脑B, 2026-06-12, H100)

## RMSE 对比

SiGe 36 frames lstsq + adam (2400 gen), 2B blocks unchanged:

| Model | E_train | F_train | E_test | F_test | ΔF_test% |
|-------|---------|---------|--------|--------|----------|
| Baseline (rc5.5, 15 knots) | 0.00467 | 0.17288 | 0.00376 | 0.19591 | — |
| Variant A (rc4.25, 15 knots) | 0.00662 | 0.19639 | 0.00554 | 0.22472 | **+14.7%** |
| Variant B (rc4.25, 13 knots) | 0.00712 | 0.19999 | 0.00582 | 0.21788 | **+11.2%** |

Both within 15% threshold. Variant B (13 knots, fewer params) actually outperforms A (15 knots) in F_test.

## 4-Stage MD 基准 (66990 atoms, H100)

| Stage | Baseline (rc5.5) | Variant A (rc4.25, 15k) | Variant B (rc4.25, 13k) |
|-------|-----------------|------------------------|------------------------|
| 1 (200→650K) | 18.2 M | 43.4 M (2.38×) | 50.4 M (2.77×) |
| 2 (650K) | 16.3 M | 37.9 M (2.33×) | 45.9 M (2.82×) |
| 3 (650K) | 12.5 M | 36.6 M (2.93×) | 44.3 M (3.54×) |
| 4 (650→200K) | 10.1 M | 34.7 M (3.44×) | 43.6 M (4.32×) |
| **Average** | **14.3 M** | **38.2 M (2.67×)** | **46.1 M (3.22×)** |

3B cost fraction vs 2B-only (183.8 M):
- Baseline: 12.9×
- Variant A: **4.8×** (-63%)
- Variant B: **4.0×** (-69%)

## 关键确认

- ✅ P2-1 compact 3B neighbour list 已启用: "rc_3b-filtered" + rc=(4.2,4.2,4.2) < rc_2b=(5.5,5.5)
- ✅ use_3b_list_ 激活: 三元组候选数 ~4.7× 削减 (rc ratio^6)
- ❌ Tensor 仍不进 smem (108KB 或 70KB >> 48KB 预算)
- FD pressure + cluster virial 验证: 待做

## P3-V2 maxrregcount 探针

| maxrregcount | Speed (atom·step/s) | vs Default |
|-------------|---------------------|-----------|
| Default (90 regs) | 18.99 M | baseline |
| 64 | 18.58 M | -2.1% |
| 96 | 19.12 M | +0.7% |
| 128 | 19.07 M | +0.4% |

性能几乎平坦（<3% variation）→ 3B kernel **memory-bound**。
Per A's 决策树: "平坦 + ncu 解锁后确认 L2-bound → A 出 canonical+bf16 smem patch (P3-2)"
与 ncu 数据一致: L2 hit 99.74%, memory throughput 63.87%, uncoalesced global loads (4.2/32 bytes).

