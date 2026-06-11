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
| B4 | 无 virial/stress 训练项 | `lambda_v` 已解析但未用 | ✅ 已接入 lstsq (2026-06-11, 未测试) |
| B5 | SNES/ES/Adam/LBFGS 对强凸线性问题冗余 | 论文就是一次线性求解 | 文档建议 |

### C. MD 推理性能 (P2 — 待做)

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

### P2 (推理加速) — 代码完成 ⚠️ 未编译/未测试 (本机无 nvcc, 2026-06-11)

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
- P2-3: ncu profiling 验证 — **待做** (需有 GPU + nvcc 的机器)

### B4: virial 训练项 — 代码完成 ⚠️ 未编译/未测试 (2026-06-11)

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

| 测试 | 模型 | 结果 |
|------|------|------|
| FD pressure (∂E/∂V) | 2B | PASS (|err|<0.005%) |
| FD pressure (∂E/∂V) | 2B+3B (trim=3) | PASS (|err|<0.005%) |
| Cluster virial (Σ r×F) | 2B | PASS (|err|~1e-8 GPa) |
| Cluster virial (Σ r×F) | 2B+3B (trim=3) | PASS (|err|~1e-8 GPa) |
| Pair symmetry | 2B (SiGe) | PASS (max|c_delta|=0) |
| Training loss | 2B+3B (mini, 36 frames) | E=0.005 eV/atom, F=0.211 eV/A |
| Training time | 2B+3B (10702 params, 36 frames) | 5.6s (vs 178.6s baseline) |

## MD 推理性能基线 (RTX 5090D, 66990 atoms, 2B+3B)

| 区间 | Speed (atom*step/s) |
|------|---------------------|
| Stage 1 (200→650K) | 9.67 M |
| Stage 2 (650K hold) | 7.02 M |
| Stage 3 (650K hold) | 6.65 M |
| Stage 4 (650→200K) | 6.30 M |

3B cost fraction (vs 2B-only): 2B+3B = 2.53s/500steps vs 2B-only = 0.20s/500steps
→ **3B 比 2B 慢 ~12.5×** (邻居表合并 + 3B kernel SM 利用率问题是根因)

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
