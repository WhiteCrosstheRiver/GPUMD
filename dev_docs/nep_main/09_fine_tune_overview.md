# Fine_tune功能概述

本文档详细说明fine_tune功能的整体设计、执行流程和使用场景。

## 功能简介

fine_tune（微调）功能允许从预训练的基础模型（foundation model）开始训练，而不是从随机初始化开始。这对于：

1. **快速收敛**: 利用基础模型的知识，减少训练时间
2. **小数据集训练**: 在有限数据上快速获得好的结果
3. **领域适应**: 将通用模型适应到特定体系

## 用户手册参考

根据[GPUMD官方文档](https://gpumd.org/nep/input_parameters/fine_tune.html)，fine_tune的语法为：

```
fine_tune <nep_model_file> <nep_restart_file>
```

其中：
- `<nep_model_file>`: 基础模型的势函数文件（如`nep89_20250409.txt`）
- `<nep_restart_file>`: 基础模型的重启文件（如`nep89_20250409.restart`）

## 基础模型要求

当前GPUMD提供的基础模型位于`GPUMD/potentials/nep/nep89_20250409`，包含89种元素（H到Pu）。

### 必须匹配的参数

以下参数必须与基础模型完全一致，不能修改：

- `version`: 必须为4
- `zbl`: 必须为2（启用ZBL）
- `cutoff`: 必须为6 5（径向6Å，角向5Å）
- `n_max`: 必须为4 4
- `basis_size`: 必须为8 8
- `l_max`: 必须为4 2 1（3体、4体、5体）
- `neuron`: 必须为80

### 可以修改的参数

以下参数可以根据需要调整：

- `lambda_1`, `lambda_2`: 正则化权重
- `lambda_e`, `lambda_f`, `lambda_v`: 损失权重
- `batch`: 批次大小
- `population`: 种群大小
- `generation`: 最大代数
- `save_potential`: 保存设置

## 完整执行流程

### 阶段1: 参数解析和验证

```
nep.in中设置: fine_tune nep89_20250409.txt nep89_20250409.restart
    │
    ├─ Parameters::parse_fine_tune()
    │   ├─ 输入: "fine_tune", "nep89_20250409.txt", "nep89_20250409.restart"
    │   ├─ 输出: fine_tune=1, fine_tune_nep_txt, fine_tune_nep_restart
    │   └─ Why: 解析fine_tune关键字，保存文件路径
    │
    └─ Parameters::report_inputs()
        └─ Parameters::check_foundation_model()
            ├─ 输入: fine_tune_nep_txt文件
            ├─ 输出: 验证结果（通过或报错）
            └─ Why: 确保当前配置与基础模型兼容
```

**验证内容**:
1. ZBL截断半径匹配
2. NEP截断半径匹配
3. n_max参数匹配
4. basis_size参数匹配
5. l_max参数匹配
6. 神经元数量匹配

### 阶段2: q_scaler加载

**位置**: `src/main_nep/parameters.cu` 第258-277行

```
Parameters::calculate_parameters()
    │
    └─ if (fine_tune)
        ├─ 打开fine_tune_nep_txt文件
        ├─ 跳过前7行（模型类型、ZBL、cutoff等）
        ├─ 跳过所有参数（num_tot行）
        ├─ 读取q_scaler值
        │   ├─ 输入: fine_tune_nep_txt文件
        │   ├─ 输出: q_scaler_cpu数组
        │   └─ Why: q_scaler是描述符缩放因子，从基础模型继承确保数值稳定性
        │
        └─ 复制到GPU: q_scaler_gpu[device_id]
```

**q_scaler的作用**:
- 描述符缩放因子，用于数值稳定性
- 在训练过程中自动优化
- 从基础模型继承可以保持数值范围的一致性

### 阶段3: 参数分布初始化

**位置**: `src/main_nep/snes.cu` 第80-84行和第144-238行

```
SNES::SNES()
    │
    └─ if (para.fine_tune)
        └─ SNES::initialize_mu_and_sigma_fine_tune()
            ├─ 输入: fine_tune_nep_restart文件
            ├─ 输出: mu和sigma数组（参数分布）
            └─ Why: 从基础模型的参数分布开始，而不是随机初始化
```

**初始化流程**:

1. **读取基础模型restart文件**
   ```
   文件格式: 每行两个浮点数（mu sigma）
   总行数: num_tot = num_ann + num_cnk_radial + num_cnk_angular
   ```

2. **元素映射**
   ```
   基础模型: 89种元素（H到Pu，但缺少5种：Po, At, Rn, Fr, Ra）
   当前模型: 用户指定的元素子集
   映射: 通过element_map数组将原子序数映射到基础模型的索引
   ```

3. **提取神经网络参数**
   ```
   对每个用户指定的元素类型:
      从基础模型中提取该元素的所有ANN参数
      包括: 权重、偏置等
   ```

4. **提取描述符参数**
   ```
   径向描述符: 对所有元素对提取c_n^k系数
   角向描述符: 对所有元素对提取c_n^k系数
   
   可选: 如果定义了FINE_TUNE_DESCRIPTOR宏，描述符参数可以继续优化
        否则，sigma设为0，描述符参数被冻结
   ```

### 阶段4: 训练过程

fine_tune模式下的训练与普通训练基本相同，但有以下区别：

1. **初始参数分布**: 从基础模型加载，而不是随机初始化
2. **描述符参数**: 默认冻结（除非定义了`FINE_TUNE_DESCRIPTOR`）
3. **q_scaler**: 从基础模型继承

## 关键代码位置

### 参数解析
- **文件**: `src/main_nep/parameters.cu`
- **函数**: `Parameters::parse_fine_tune()` (第1321-1331行)

### 基础模型验证
- **文件**: `src/main_nep/parameters.cu`
- **函数**: `Parameters::check_foundation_model()` (第288-375行)

### q_scaler加载
- **文件**: `src/main_nep/parameters.cu`
- **函数**: `Parameters::calculate_parameters()` (第258-277行)

### 参数分布初始化
- **文件**: `src/main_nep/snes.cu`
- **函数**: `SNES::initialize_mu_and_sigma_fine_tune()` (第144-238行)

### 训练中的特殊处理
- **文件**: `src/main_nep/fitness.cu`
- **位置**: 第165行
- **代码**: `(para.fine_tune ? false : true)`
- **说明**: fine_tune模式下，第0代不计算q_scaler（因为已从基础模型加载）

## 使用示例

### nep.in配置示例

```bash
# Fine-tuning from foundation model
fine_tune nep89_20250409.txt nep89_20250409.restart

# 必须匹配的参数
version    4
zbl        2
cutoff     6 5
n_max      4 4
basis_size 8 8
l_max      4 2 1
neuron     80

# 用户指定的元素（可以是基础模型的子集）
type 3 Si C O

# 可以调整的参数
lambda_1   0      # 可以设为0，因为基础模型已经正则化
lambda_e   1
lambda_f   1
lambda_v   1
batch      5000
population 50
generation 5000
save_potential 1000 0
```

## 优势与限制

### 优势

1. **快速收敛**: 从好的初始点开始，通常只需几千代就能收敛
2. **小数据集友好**: 适合数据有限的情况
3. **知识迁移**: 利用基础模型在大数据集上学到的知识
4. **稳定性**: q_scaler从基础模型继承，数值更稳定

### 限制

1. **参数约束**: 必须与基础模型的某些参数完全匹配
2. **元素限制**: 只能使用基础模型支持的89种元素
3. **描述符冻结**: 默认情况下描述符参数被冻结（可通过宏启用优化）

## 相关文档

- [参数映射机制](10_fine_tune_parameter_mapping.md) - 详细的参数提取和映射逻辑
- [元素映射机制](11_fine_tune_element_mapping.md) - 元素索引映射详解
- [参数初始化策略](12_fine_tune_initialization.md) - mu和sigma的初始化细节

