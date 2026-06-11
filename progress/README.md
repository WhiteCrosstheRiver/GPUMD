# 工作进度总览

> 协作约定: 电脑A (Mac, 无 nvcc) 负责设计/写码/分析; GPU 机器负责编译/测试/剖析。
> 本文件夹是唯一协作记录, 取代旧 WORKMD.md (2026-06-12 重构)。
> 结构: 本文件 = 总进度 + 跨分支决策; 各分支进度在同名 .md 文件。
> 规则: 验证报告追加到对应分支文件; 影响路线的决策同步到本文件。

## 机器环境

| 机器 | GPU | 平台 | 工具链 | 备注 |
|------|-----|------|--------|------|
| 电脑A | 无 | macOS | 无 nvcc | 设计/写码/审查 |
| 电脑B-1 | RTX 5090 D v2 (sm_120) | WSL, CUDA 13.1 | ncu 不可用 (NVGPUCTRPERM) | 第1-3轮测试机 |
| 电脑B-2 | H100 PCIe (sm_90) | Linux el8, CUDA 13.0 | ncu 待解锁 (sudo/modprobe) | 第5轮起主力, **后续基线以此为准** |

H100 基线 (SiGe 64087 atoms, nep4 cutoff=6/4 n_max=8/8): NEP4 legacy **18.92M** atom·step/s。
5090D 基线 (SiGe 66990 atoms): UF3 2B+3B 23.8M / 2B-only 164.5M / NEP4 25.4M。
**两机数字不可互比。**

## 分支地图

| 分支 | 状态 | 内容 |
|------|------|------|
| `uf3-dev` | **活跃 (主线)** | UF3 修复+训练质量+推理优化 (P0/P1/P2 完成, **P3 进行中**) |
| `nep-fusion-dev` | 挂起 (与 uf3-dev 平级) | M1 NEP 融合实验, 已关闭 (慢 1.93×), 留作参照与教训 |
| (未建) M2 新势函数分支 | 待立项 | pair spline + moment 角向 + 小 readout, 设计约束见决策日志 R6 |

## 决策日志

- **R1-2 (06-10/11)**: UF3 P0 正确性 4 项 + P1 训练质量 4 项修复, 全部验证通过
- **R2 (06-11)**: P2 推理加速 (warp kernel + 3B 紧凑表 + smem), 5090D 实测 3.27×
- **R3 (06-11)**: 全量回归 + nsys 剖析: UF3 3B kernel 占 90.6%; NEP4 三力 kernel 占 88.6%
- **R4 (06-12)**: 立项 M1 (NEP 融合) + M2 (新势函数); 当时决定冻结 UF3 性能改动
- **R5 (06-12)**: M1 验证失败 — fused 比 legacy 慢 1.93× (H100)
- **R6 (06-12)**: **M1 关闭**, 默认翻回 legacy。三条教训成为 M2 设计约束:
  L1 per-thread 状态必须卡进寄存器 (运行期下标中间数组 = local memory = 毒药);
  L2 稀疏工作不内联进稠密循环 (分 kernel 或 warp-per-atom);
  L3 NEP4 kernel 结构已近最优, 速度必须来自模型变小, kernel 手术无肉
- **R7 (06-12, 本轮)**: 文档重构为 progress/ 文件夹; **解除 UF3 冻结, 立项 P3**
  (UF3 推理加速二期, 数据驱动: 先 ncu 定位 3B warp kernel 瓶颈再动刀);
  nep-fusion-dev 保持平级挂起; M2 立项推迟到 P3 告一段落

## 当前焦点

P3 (uf3-dev): 详见 `progress/uf3-dev.md` P3 章节。第一步是 B 解锁 H100 ncu
并提供 3B warp kernel 的微架构剖析 + UF3 在 H100 的基线。
