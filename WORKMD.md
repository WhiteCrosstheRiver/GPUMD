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
| B4 | 无 virial/stress 训练项 | `lambda_v` 已解析但未用 | 待处理 |
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

### P2 (推理加速) — 待做

- P2-1: 分离 3B 邻居表 (用 rc_3b 构建)
- P2-2: 重写 3B kernel (shared memory tensor + float 累加 + 更好并行粒度)
- P2-3: ncu profiling 验证

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
| `src/force/uf3.cu` | P0修复 | 3B virial, r_min 排斥保护, 2B virial 0.5 修正, hoisted 3B kernel |
| `src/force/uf3.cuh` | 恢复 | sym_3b 成员 + 声明 (从 stash@{0} 恢复) |
| `src/main_uf3/uf3.cu` | P0+P1修复 | canonical pair map, 非 canonical slot frozen, 3B 对称化 |
| `src/main_uf3/uf3.cuh` | P0+P1修复 | type_map_host_ 成员, sym_3b 公开, symmetrize_3b_gradient |
| `src/main_uf3/lstsq.cu` | P1修复 | double 能量行, cusolver, lambda_1/lambda_2 接入, tmap 传导 |
| `src/main_uf3/main.cu` | P0修复 | write_uf3_file 从 canonical slot 镜像 |
| `src/main_uf3/fitness.cu` | P1修复 | 梯度 3B 对称化 (从 stash@{0} 恢复) |
| `src/main_uf3/parameters.cuh` | P0修复 | trim_3b 默认值 0→3 |
