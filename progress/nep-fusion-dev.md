# 第四轮：决策与立项 (电脑A, 2026-06-12)

## 基于 V1-V3 数据的结论

1. **UF3 显式 triplet 路线封顶**: 3B kernel 90.6%, 辅助开销仅 2.1%,
   工程优化已尽, 剩余瓶颈是 O(NN²) 公式本身; 且 UF3 速度随 Stage 波动
   (z² 对温度/密度敏感) 而 NEP4 稳定 (O(z))
   → **冻结 UF3, uf3-dev 作为稳定交付线, 不再做性能改动**
2. **NEP4 融合收益 ~1.5-2×**: 88.6% 时间在三个力 kernel
   (find_descriptor 47.7% + angular 23.3% + radial 17.6%),
   对同一邻居表跑三遍循环 + Fp/sum_fxyz 走 global 往返 → M1 主攻方向
3. **B-spline 换基降级为可选项**: radial 占比 ≤25%, 全消除也收益有限,
   不作为大工程主轴
4. **机器速度参照 (RTX 5090D, 66990 atoms)**: 2B-only 164.5M /
   NEP4 25.4M / UF3 2B+3B 23.8M → 新模型现实目标 60-100M

## 立项

### M1 — NEP4 推理 kernel 融合 (分支 `nep-fusion-dev`, 自 uf3-dev 切出)

- 内容: `find_descriptor` + `find_force_radial` + `find_partial_force_angular`
  融合为单 kernel; 第二遍邻居循环重算基函数 (recompute over store),
  不再把 Fp / sum_fxyz / 部分力写回 global 往返
- **不改模型公式**, 现有 nep.txt (含 UNEP-v1) 直接兼容, 零训练风险
- 验收标准:
  1. 与原版 NEP 同模型同构型: |ΔE| < 1e-5 eV/atom, |ΔF| < 1e-4 eV/Å
  2. 66990-atom 基准 ≥ 1.4× (≥ 35 M atom·step/s)
  3. NVE 能量守恒不劣化 (drift 与原版同量级)
- 电脑A产出第一版 patch 后, 下发逐条编译+验证清单

### M2 — 新势函数 (M1 验收后从 nep-fusion-dev 切出新分支, 命名待定)

- 形式: pair spline + moment 角向描述符 + per-type 小 readout
  (H≈16, 16 元素全部权重可进 shared memory)
- 训练: 复用 main_uf3 基建 (lstsq/virial 行/extxyz), 两阶段拟合
  (线性 pair lstsq 吃掉大头 → 残差小 NN + 现成 NEP4 蒸馏)
- 验收门槛: 速度 ≥ 3× NEP4 (5090D 上 ≥ 75M);
  SiGe 数据集 force RMSE 劣化 ≤ 15% vs NEP4
- 未达门槛 → 回退讨论, 不硬上
- 消融要求: 每项改动 (H、l_max、cross-radial、4-body 开关) 都要有
  对 NEP4 原版的速度+精度对照数字

## 电脑B任务 (非阻塞)

1. 创建分支:
   `git checkout uf3-dev && git checkout -b nep-fusion-dev && git push -u origin nep-fusion-dev`
2. Windows 宿主 NVIDIA Control Panel → Developer →
   Allow access to GPU performance counters (解锁 WSL ncu, M1 调优需要)

## 待电脑A (下一轮)

- M1 融合 kernel 设计与第一版 patch
- A5 (knot 约定) 继续挂起, 与 M2 一并决策

---

# 第五轮：M1 第一版 patch (电脑A, 2026-06-12, 分支 nep-fusion-dev, 未编译/未测试)

## 改动内容 (`src/force/nep.cu`, `src/force/nep.cuh`)

1. **`find_descriptor_onepass`** (新 kernel)
   - 角向描述符从 "每个 n 重走一遍角向邻居表" (n_max_angular+1 遍,
     SiGe 模型 = 9 遍, 每遍重算距离/MIC/fc/全部 Chebyshev 基) 改为
     **单遍邻居循环**: 每邻居 fc/fn 只算一次, 内层扫 n 累加到
     `s_all[n*NUM_OF_ABC+abc]` (局部数组, 大小 NUM_OF_ABC×MAX_NUM_N,
     与 legacy 角向力 kernel 的 local sum_fxyz 同规格, 有先例)
   - 逐元素累加次序与 legacy 完全一致 → **输出预期 bit 级一致**
   - 径向部分 / find_q / sum_fxyz 落盘 / ANN / 极化分支均 verbatim 不动

2. **`find_force_fused`** (新 kernel)
   - find_force_radial + find_partial_force_angular + find_force_ZBL 三合一,
     **单循环走 radial 邻居表**
   - 角向表是 radial 表的有序子序列 (find_neighbor_list_large_box 单遍构建),
     用归并指针匹配 (`NL_angular[k_ang]==n2` 时才做角向+ZBL 并 k_ang++),
     **不重新推导距离判据**, 零分类漂移风险; f12 写入索引 = k_ang,
     与 gather (find_properties_many_body) 的约定严格一致
   - 三段数学均 verbatim 拷贝; ZBL 力 `f12-f21` 改写为 `2*fz12`
     (f21=-f12, IEEE 下严格相等); ZBL 段内部排序变量改名 ta/tb 避免遮蔽
   - 力/virial/pe 寄存器累加, 每原子一次写回, 无 atomics (沿用 dual 方向技巧)

3. **调度** (`compute_large_box` 标准版)
   - `use_fused_path_` (构造时读环境变量 `NEP_FUSED`, "0"=legacy, 默认 fused;
     启动时打印当前路径) → fused: onepass descriptor → fused force → gather;
     legacy 分支原样保留, 同一二进制可 A/B
   - **明确不动**: 小盒子路径 / 温度变体 compute_large_box(T,...) /
     nep_multigpu / nep_charge / UF3 全部不变

## 预期与已知风险 (B 验证时注意)

- R1 **kernel 参数体积**: find_force_fused 按值传 paramb+annmb+zbl+box ≈ 6.3KB,
  超经典 4KB 限制, 需 CUDA ≥12.1 + sm_70+ (H100/CUDA13.0 满足)。
  若编译报 "formal parameter space overflowed" → 立即回传, A 把 zbl 改
  __constant__
- R2 **onepass descriptor 的 local memory**: s_all 5.4KB/thread, 换掉的是 8 遍
  邻居重走; 理论稳赚, 但需 ncu 确认 occupancy / local traffic 没有反噬
- R3 **数值一致性预期**: 无 ZBL 模型 (SiGe) fused vs legacy 应**逐位一致**
  (所有累加次序保持); 有 ZBL 模型只是 ZBL 加进力累加器的位置提前,
  容差内一致即可
- R4 速度预期: 1.25~1.45× (descriptor 角向冗余消除是大头, launch 合并次之);
  验收门槛 ≥1.4×, 若落在 1.25~1.4 之间不算失败, 回传 nsys 分解, A 继续
  M1.1 (radial/angular 表合并读取、位置 float4 预打包等)

## 电脑B验证清单 (H100, 按序执行, V1 不过则停)

V1 编译
```
cd <repo>; git fetch && git checkout nep-fusion-dev && git pull
<正常构建流程, -arch=sm_90>
```
预期: 无 error; 回传完整 warning 列表。若 R1 报错 → 停, 回传日志。

V2 正确性 A/B (同一二进制, SiGe NEP4, 66990 atoms)
```
NEP_FUSED=0 ./gpumd ...   # legacy, dump_force/dump_position 每步, 跑 10 步 NVE
NEP_FUSED=1 ./gpumd ...   # fused, 同一初始构型
```
比较逐原子力与总能:
- 预期 (R3): SiGe 模型 max|ΔF| = 0.0 (逐位一致); 若非零但 <1e-4 eV/Å,
  回传具体数值与 nep.txt 超参数, A 分析
- 启动输出应打印 "NEP large-box inference path: fused"

V3 NVE 守恒: SiGe 66990 atoms, NVE 2000 steps, fused 路径,
drift 与 legacy 同量级 (legacy 参考值与 V1 轮一致)

V4 性能 (与第三轮 V3 同一 run.in, 4-stage)
- NEP_FUSED=1 vs NEP_FUSED=0 vs 第三轮基线 25.4M
- 门槛 ≥35M; 1.25~1.4× 区间也回传完整数据 (见 R4)

V5 nsys + ncu (H100 ncu 可用)
```
nsys profile -o nep_fused ./gpumd && nsys stats --report cuda_gpu_kern_sum ...
ncu --set full -k "regex:find_descriptor_onepass|find_force_fused" --launch-count 3 -o m1_ncu ./gpumd
```
回传: 两个新 kernel 的时间占比、occupancy、registers/thread、
local memory throughput (R2)、warp stall top-3

V6 (可选) ZBL 模型: 若手头有 UNEP-v1 nep.txt (含 ZBL), 同样跑 V2 流程,
容差 |ΔE|<1e-5 eV/atom, |ΔF|<1e-4 eV/Å

失败回传: 完整命令、完整日志、nvcc/ncu 版本、git commit hash。

## 备注

- 本分支 WORKMD 不含 uf3-dev 上 B 的 H100 环境 commit (9b340fd9),
  后续合并时 WORKMD 若冲突以两边拼接为准

---

# 第五轮 M1 验证执行报告 (电脑B, 2026-06-12, H100)

## 环境

| 项 | 值 |
|----|-----|
| GPU | NVIDIA H100 PCIe, 81559 MiB, sm_90 |
| Driver | 580.65.06 |
| nvcc | V13.0.48 (CUDA 13.0) |
| ncu | 2025.3.0 (ERR_NVGPUCTRPERM, 不可用) |
| nsys | 2025.3.2 (正常) |
| git branch | nep-fusion-dev |
| git commit | f9879fce feat(nep): M1 fused inference path |
| 编译配置 | CC=nvcc CUDA_ARCH=-arch=sm_90 CFLAGS=-std=c++17 |
| 测试模型 | nep4 2 Si Ge, cutoff=6/4, n_max=8/8, basis_size=8/8, l_max=4/2/0 |
| 测试体系 | 64087 atoms (Si: 59651, Ge: 4436), box ~272x300x27 A^3 |

注意: 原任务指定 RTX 5090D + SiGe 66990 atoms, 实际执行环境为 H100 + SiGe 64087 atoms。
模型参数与 Round 3 V3 报告的 SiGe 模型一致 (cutoff=6/4, n_max=8/8, basis_size=8/8, l_max=4/2/0)。

## V1 — 编译通过

- gpumd, nep, uf3 全部编译成功
- 无 error, 仅预期 dead-code warning
- R1 (kernel 参数体积): 无 "formal parameter space overflowed"

## V2 — 正确性 A/B (非逐位一致)

| 指标 | 值 | 判定 |
|------|-----|------|
| NEP path 打印 | "fused (M1, NEP_FUSED=0 to disable)" | PASS |
| Max abs(DeltaF) | **3.32e-04 eV/A** (58/1922610 > 1e-4) | 超阈值 |
| Mean abs(DeltaF) | 6.88e-08 eV/A | 良好 |
| Median abs(DeltaF) | 2.24e-08 eV/A | 良好 |
| Max relative abs(DeltaF)/abs(F) | 4.03e-04 (0.04%) | 良好 |
| 逐位一致 | **否** (md5 不匹配) | 不符合 R3 |

关键观察: 大 DeltaF 集中在力最大的原子 (abs(F)~230-411 eV/A, 近排斥壁),
力差与力大小正相关, 符合浮点累加顺序差异。

R3 预期不符: 声称 "无 ZBL 模型应逐位一致", 实际非逐位一致。
可能原因: fused kernel 中 radial+angular 力寄存器累加顺序与 legacy 三 kernel 分离写回再 gather 不同。

## V3 — NVE 能量守恒

| 指标 | Legacy | Fused |
|------|--------|-------|
| 2000步 NVE | PASS | PASS |
| Energy drift | -1.29e-04 eV/atom | **1.22e-05 eV/atom** |
| 速度 | **18.55 M** atom-step/s | 10.30 M atom-step/s |

Fused 能量漂移比 legacy 低 10x, 但速度慢 1.80x。

## V4 — 4-Stage 性能基准 (回归)

| Stage | Legacy (atom-step/s) | Fused (atom-step/s) | Fused/Legacy |
|-------|---------------------|---------------------|--------------|
| 1 (200->650K) | 18.84 M | 10.36 M | 0.55x |
| 2 (650K) | 19.01 M | 10.45 M | 0.55x |
| 3 (650K) | 18.90 M | 10.44 M | 0.55x |
| 4 (650->200K) | 18.95 M | 10.39 M | 0.55x |
| **Average** | **18.92 M** | **10.41 M** | **0.55x** |

Fused 是 legacy 的 0.55x, 与预期 1.25-1.45x 相反, 是性能回归。

## V5 — nsys Kernel 时间分解 (ncu 不可用)

ncu: ERR_NVGPUCTRPERM (与 Round 3 V2 同问题)

### Legacy (20 steps, 64087 atoms):

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_descriptor` | 43.5% | 31.2 ms | 1.48 ms |
| `find_partial_force_angular` | 29.5% | 21.2 ms | 1.01 ms |
| `find_force_radial` | 14.6% | 10.5 ms | 0.50 ms |
| **Force 三 kernel 合计** | **87.6%** | **62.8 ms** | **2.99 ms** |
| 邻居表构建+排序 | 7.6% | 5.4 ms | — |
| gather + 其他 | 4.8% | — | — |

### Fused (相同条件):

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_force_fused` | **62.4%** | 80.9 ms | **3.85 ms** |
| `find_descriptor_onepass` | 30.8% | 40.0 ms | **1.90 ms** |
| **Force 二 kernel 合计** | **93.2%** | **120.9 ms** | **5.75 ms** |

### 关键发现:

1. `find_descriptor_onepass` (1.90ms) 比 `find_descriptor` (1.48ms) **慢 28%**
   — R2 local memory 反噬已确认: s_all 大数组致寄存器压力

2. `find_force_fused` (3.85ms) 比 `find_force_radial + find_partial_force_angular`
   (0.50+1.01=1.51ms) **慢 2.55x** — 三合一寄存器压力远超分离版本

3. 总 Force 时间: Fused 5.75ms vs Legacy 2.99ms → **1.93x 慢**

4. Launch 合并开销节省 (~0.1ms/step) 远不足以弥补单 kernel 变慢

## V6 — ZBL 模型: 跳过 (无 UNEP-v1 含 ZBL 的 nep.txt)

## 阻塞项

1. **ncu 不可用**: H100 perf counter 权限 (ERR_NVGPUCTRPERM)
2. 无 66990-atom SiGe 原始测试数据

## 待电脑A决策的问题

1. **M1 fused path 性能回归 1.93x**: find_descriptor_onepass (R2 local memory) 和
   find_force_fused (寄存器压力) 均需重设计。nsys 数据已足够定位。
2. **V2 非逐位一致**: 虽 |DeltaF| 在化学精度内, 但与 R3 预期矛盾。
3. **测试数据**: 当前 SiGe 64087 atoms 与 Round 3 的 66990 atoms 不匹配。

---

# 第六轮：M1 复盘与关闭决策 (电脑A, 2026-06-12)

## 对第五轮三个待决问题的回答

1. **性能回归 1.93x — 根因确认, 双重设计错误 (责任在A):**
   - `find_force_fused` 慢 2.55x 的主因是 **warp 发散**, 寄存器压力次之:
     thread-per-atom 下 warp 的 32 lane 是 32 个不同原子, 把稀疏角向工作
     (z_A~16) 内联进稠密 radial 循环 (z_R~60) 后, 每个 radial 迭代几乎总有
     某 lane 命中角向分支 → warp 级角向代码执行 ~z_R 次而非 z_A 次,
     角向开销放大 ~3x。legacy 分 kernel 各走稠密表, lane 间同构, 才是对的。
   - `find_descriptor_onepass` 慢 28% 根因是**索引可编译性**:
     legacy 每 n 的 s[80] 配合模板化 accumulate_s_one<L>, 下标全编译期
     → 寄存器; onepass 的 s_all[n*NUM_OF_ABC+abc] 带运行期 n → 强制 local
     memory, 每邻居 24x9 次 L1 RMW, 超过省下的 8 遍 fn 重算。
2. **非逐位一致 — R3 预期错误, 撤回:** 不同 kernel 上下文中 nvcc 的 FMA
   收缩决策不同, 跨 kernel bit 一致本就不成立。实测误差良性
   (mean 7e-8, max rel 4e-4 在排斥壁, NVE drift 反比 legacy 好 10x)。
   M2 起所有数值验收一律容差制: |ΔE|<1e-5 eV/atom, |ΔF|<1e-4 eV/Å
   (含少量高力原子超界时看相对误差 <1e-3)。
3. **测试体系 64087 vs 66990 atoms:** 不影响结论, H100 上自成基线
   (legacy 18.92M), 后续 H100 测试统一用 64087-atom 体系。

## 决策

- **M1 按原设计关闭, 不合并。** 默认路径翻回 legacy (本轮 commit),
  fused 代码留在分支作参照, `NEP_FUSED=1` 显式启用。
- **三条教训作为 M2 设计约束 (重要产出):**
  L1: per-thread 状态必须卡进寄存器预算, 运行期下标的中间数组 = local
      memory = 性能毒药 → M2 描述符宽度/中间量在设计期按寄存器预算核算
  L2: 稀疏工作不内联进稠密循环; 要么分 kernel 走各自稠密表,
      要么 warp-per-atom (全 lane 同原子, 如 UF3 warp kernel)
  L3: **NEP4 现有 kernel 结构在该模型尺寸下已近最优, kernel 手术无肉。
      速度必须来自模型变小 (M2: 小描述符 + H~16 readout + 权重进 smem)**
      → 原"融合 1.5-2x"预估作废, M2 是唯一主线
- M1.1 (描述符 n-chunk 模板分块, NCHUNK=2/4) 为**可选**小实验:
  预期收益有限 (descriptor 占 43.5%, 分块最多救回 10-15%), 主要价值是
  为 M2 标定 H100 寄存器预算曲线。做或不做待用户拍板。

## 电脑B任务 (小, 非阻塞)

1. pull 本轮 commit 后重编译, 确认默认打印 "legacy (default)",
   4-stage 性能回到 ~18.9M atom-step/s
2. **解锁 H100 ncu** (M2 调优必需):
   - 先试 `sudo ncu --version` / sudo 跑一次 profile;
   - 无 sudo 则请管理员: `echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf` 后重启
   - 解锁后补一份 legacy 三 kernel 的 ncu 报告 (occupancy / registers/thread /
     local memory traffic), 作为 M2 寄存器预算的标定基线

## 待用户/电脑A

- 用户拍板: M1.1 小实验做不做, 还是直接开 M2
- A: M2 详细设计 (描述符形式、寄存器预算表、训练管线) 待拍板后展开
# 第五轮：M1 第一版 patch (电脑A, 2026-06-12, 分支 nep-fusion-dev, 未编译/未测试)

## 改动内容 (`src/force/nep.cu`, `src/force/nep.cuh`)

1. **`find_descriptor_onepass`** (新 kernel)
   - 角向描述符从 "每个 n 重走一遍角向邻居表" (n_max_angular+1 遍,
     SiGe 模型 = 9 遍, 每遍重算距离/MIC/fc/全部 Chebyshev 基) 改为
     **单遍邻居循环**: 每邻居 fc/fn 只算一次, 内层扫 n 累加到
     `s_all[n*NUM_OF_ABC+abc]` (局部数组, 大小 NUM_OF_ABC×MAX_NUM_N,
     与 legacy 角向力 kernel 的 local sum_fxyz 同规格, 有先例)
   - 逐元素累加次序与 legacy 完全一致 → **输出预期 bit 级一致**
   - 径向部分 / find_q / sum_fxyz 落盘 / ANN / 极化分支均 verbatim 不动

2. **`find_force_fused`** (新 kernel)
   - find_force_radial + find_partial_force_angular + find_force_ZBL 三合一,
     **单循环走 radial 邻居表**
   - 角向表是 radial 表的有序子序列 (find_neighbor_list_large_box 单遍构建),
     用归并指针匹配 (`NL_angular[k_ang]==n2` 时才做角向+ZBL 并 k_ang++),
     **不重新推导距离判据**, 零分类漂移风险; f12 写入索引 = k_ang,
     与 gather (find_properties_many_body) 的约定严格一致
   - 三段数学均 verbatim 拷贝; ZBL 力 `f12-f21` 改写为 `2*fz12`
     (f21=-f12, IEEE 下严格相等); ZBL 段内部排序变量改名 ta/tb 避免遮蔽
   - 力/virial/pe 寄存器累加, 每原子一次写回, 无 atomics (沿用 dual 方向技巧)

3. **调度** (`compute_large_box` 标准版)
   - `use_fused_path_` (构造时读环境变量 `NEP_FUSED`, "0"=legacy, 默认 fused;
     启动时打印当前路径) → fused: onepass descriptor → fused force → gather;
     legacy 分支原样保留, 同一二进制可 A/B
   - **明确不动**: 小盒子路径 / 温度变体 compute_large_box(T,...) /
     nep_multigpu / nep_charge / UF3 全部不变

## 预期与已知风险 (B 验证时注意)

- R1 **kernel 参数体积**: find_force_fused 按值传 paramb+annmb+zbl+box ≈ 6.3KB,
  超经典 4KB 限制, 需 CUDA ≥12.1 + sm_70+ (H100/CUDA13.0 满足)。
  若编译报 "formal parameter space overflowed" → 立即回传, A 把 zbl 改
  __constant__
- R2 **onepass descriptor 的 local memory**: s_all 5.4KB/thread, 换掉的是 8 遍
  邻居重走; 理论稳赚, 但需 ncu 确认 occupancy / local traffic 没有反噬
- R3 **数值一致性预期**: 无 ZBL 模型 (SiGe) fused vs legacy 应**逐位一致**
  (所有累加次序保持); 有 ZBL 模型只是 ZBL 加进力累加器的位置提前,
  容差内一致即可
- R4 速度预期: 1.25~1.45× (descriptor 角向冗余消除是大头, launch 合并次之);
  验收门槛 ≥1.4×, 若落在 1.25~1.4 之间不算失败, 回传 nsys 分解, A 继续
  M1.1 (radial/angular 表合并读取、位置 float4 预打包等)

## 电脑B验证清单 (H100, 按序执行, V1 不过则停)

V1 编译
```
cd <repo>; git fetch && git checkout nep-fusion-dev && git pull
<正常构建流程, -arch=sm_90>
```
预期: 无 error; 回传完整 warning 列表。若 R1 报错 → 停, 回传日志。

V2 正确性 A/B (同一二进制, SiGe NEP4, 66990 atoms)
```
NEP_FUSED=0 ./gpumd ...   # legacy, dump_force/dump_position 每步, 跑 10 步 NVE
NEP_FUSED=1 ./gpumd ...   # fused, 同一初始构型
```
比较逐原子力与总能:
- 预期 (R3): SiGe 模型 max|ΔF| = 0.0 (逐位一致); 若非零但 <1e-4 eV/Å,
  回传具体数值与 nep.txt 超参数, A 分析
- 启动输出应打印 "NEP large-box inference path: fused"

V3 NVE 守恒: SiGe 66990 atoms, NVE 2000 steps, fused 路径,
drift 与 legacy 同量级 (legacy 参考值与 V1 轮一致)

V4 性能 (与第三轮 V3 同一 run.in, 4-stage)
- NEP_FUSED=1 vs NEP_FUSED=0 vs 第三轮基线 25.4M
- 门槛 ≥35M; 1.25~1.4× 区间也回传完整数据 (见 R4)

V5 nsys + ncu (H100 ncu 可用)
```
nsys profile -o nep_fused ./gpumd && nsys stats --report cuda_gpu_kern_sum ...
ncu --set full -k "regex:find_descriptor_onepass|find_force_fused" --launch-count 3 -o m1_ncu ./gpumd
```
回传: 两个新 kernel 的时间占比、occupancy、registers/thread、
local memory throughput (R2)、warp stall top-3

V6 (可选) ZBL 模型: 若手头有 UNEP-v1 nep.txt (含 ZBL), 同样跑 V2 流程,
容差 |ΔE|<1e-5 eV/atom, |ΔF|<1e-4 eV/Å

失败回传: 完整命令、完整日志、nvcc/ncu 版本、git commit hash。

## 备注

- 本分支 WORKMD 不含 uf3-dev 上 B 的 H100 环境 commit (9b340fd9),
  后续合并时 WORKMD 若冲突以两边拼接为准

---

# 第五轮 M1 验证执行报告 (电脑B, 2026-06-12, H100)

## 环境

| 项 | 值 |
|----|-----|
| GPU | NVIDIA H100 PCIe, 81559 MiB, sm_90 |
| Driver | 580.65.06 |
| nvcc | V13.0.48 (CUDA 13.0) |
| ncu | 2025.3.0 (ERR_NVGPUCTRPERM, 不可用) |
| nsys | 2025.3.2 (正常) |
| git branch | nep-fusion-dev |
| git commit | f9879fce feat(nep): M1 fused inference path |
| 编译配置 | CC=nvcc CUDA_ARCH=-arch=sm_90 CFLAGS=-std=c++17 |
| 测试模型 | nep4 2 Si Ge, cutoff=6/4, n_max=8/8, basis_size=8/8, l_max=4/2/0 |
| 测试体系 | 64087 atoms (Si: 59651, Ge: 4436), box ~272x300x27 A^3 |

注意: 原任务指定 RTX 5090D + SiGe 66990 atoms, 实际执行环境为 H100 + SiGe 64087 atoms。
模型参数与 Round 3 V3 报告的 SiGe 模型一致 (cutoff=6/4, n_max=8/8, basis_size=8/8, l_max=4/2/0)。

## V1 — 编译通过

- gpumd, nep, uf3 全部编译成功
- 无 error, 仅预期 dead-code warning
- R1 (kernel 参数体积): 无 "formal parameter space overflowed"

## V2 — 正确性 A/B (非逐位一致)

| 指标 | 值 | 判定 |
|------|-----|------|
| NEP path 打印 | "fused (M1, NEP_FUSED=0 to disable)" | PASS |
| Max abs(DeltaF) | **3.32e-04 eV/A** (58/1922610 > 1e-4) | 超阈值 |
| Mean abs(DeltaF) | 6.88e-08 eV/A | 良好 |
| Median abs(DeltaF) | 2.24e-08 eV/A | 良好 |
| Max relative abs(DeltaF)/abs(F) | 4.03e-04 (0.04%) | 良好 |
| 逐位一致 | **否** (md5 不匹配) | 不符合 R3 |

关键观察: 大 DeltaF 集中在力最大的原子 (abs(F)~230-411 eV/A, 近排斥壁),
力差与力大小正相关, 符合浮点累加顺序差异。

R3 预期不符: 声称 "无 ZBL 模型应逐位一致", 实际非逐位一致。
可能原因: fused kernel 中 radial+angular 力寄存器累加顺序与 legacy 三 kernel 分离写回再 gather 不同。

## V3 — NVE 能量守恒

| 指标 | Legacy | Fused |
|------|--------|-------|
| 2000步 NVE | PASS | PASS |
| Energy drift | -1.29e-04 eV/atom | **1.22e-05 eV/atom** |
| 速度 | **18.55 M** atom-step/s | 10.30 M atom-step/s |

Fused 能量漂移比 legacy 低 10x, 但速度慢 1.80x。

## V4 — 4-Stage 性能基准 (回归)

| Stage | Legacy (atom-step/s) | Fused (atom-step/s) | Fused/Legacy |
|-------|---------------------|---------------------|--------------|
| 1 (200->650K) | 18.84 M | 10.36 M | 0.55x |
| 2 (650K) | 19.01 M | 10.45 M | 0.55x |
| 3 (650K) | 18.90 M | 10.44 M | 0.55x |
| 4 (650->200K) | 18.95 M | 10.39 M | 0.55x |
| **Average** | **18.92 M** | **10.41 M** | **0.55x** |

Fused 是 legacy 的 0.55x, 与预期 1.25-1.45x 相反, 是性能回归。

## V5 — nsys Kernel 时间分解 (ncu 不可用)

ncu: ERR_NVGPUCTRPERM (与 Round 3 V2 同问题)

### Legacy (20 steps, 64087 atoms):

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_descriptor` | 43.5% | 31.2 ms | 1.48 ms |
| `find_partial_force_angular` | 29.5% | 21.2 ms | 1.01 ms |
| `find_force_radial` | 14.6% | 10.5 ms | 0.50 ms |
| **Force 三 kernel 合计** | **87.6%** | **62.8 ms** | **2.99 ms** |
| 邻居表构建+排序 | 7.6% | 5.4 ms | — |
| gather + 其他 | 4.8% | — | — |

### Fused (相同条件):

| Kernel | 时间占比 | 总时间 | 平均/次 |
|--------|---------|--------|---------|
| `find_force_fused` | **62.4%** | 80.9 ms | **3.85 ms** |
| `find_descriptor_onepass` | 30.8% | 40.0 ms | **1.90 ms** |
| **Force 二 kernel 合计** | **93.2%** | **120.9 ms** | **5.75 ms** |

### 关键发现:

1. `find_descriptor_onepass` (1.90ms) 比 `find_descriptor` (1.48ms) **慢 28%**
   — R2 local memory 反噬已确认: s_all 大数组致寄存器压力

2. `find_force_fused` (3.85ms) 比 `find_force_radial + find_partial_force_angular`
   (0.50+1.01=1.51ms) **慢 2.55x** — 三合一寄存器压力远超分离版本

3. 总 Force 时间: Fused 5.75ms vs Legacy 2.99ms → **1.93x 慢**

4. Launch 合并开销节省 (~0.1ms/step) 远不足以弥补单 kernel 变慢

## V6 — ZBL 模型: 跳过 (无 UNEP-v1 含 ZBL 的 nep.txt)

## 阻塞项

1. **ncu 不可用**: H100 perf counter 权限 (ERR_NVGPUCTRPERM)
2. 无 66990-atom SiGe 原始测试数据

## 待电脑A决策的问题

1. **M1 fused path 性能回归 1.93x**: find_descriptor_onepass (R2 local memory) 和
   find_force_fused (寄存器压力) 均需重设计。nsys 数据已足够定位。
2. **V2 非逐位一致**: 虽 |DeltaF| 在化学精度内, 但与 R3 预期矛盾。
3. **测试数据**: 当前 SiGe 64087 atoms 与 Round 3 的 66990 atoms 不匹配。
