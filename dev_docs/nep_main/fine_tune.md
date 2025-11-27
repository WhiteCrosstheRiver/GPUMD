# Fine_tune功能完整详解

本文档详细说明NEP的fine_tune（微调）功能，包括功能概述、参数映射机制、元素映射、参数初始化策略，以及如何基于基础模型进行微调训练。

## 目录

1. [功能概述](#功能概述)
2. [基础模型文件结构](#基础模型文件结构)
3. [参数验证与兼容性检查](#参数验证与兼容性检查)
4. [元素映射机制](#元素映射机制)
5. [参数映射与提取](#参数映射与提取)
6. [参数初始化策略](#参数初始化策略)
7. [参数冻结机制](#参数冻结机制)
8. [完整执行流程](#完整执行流程)
9. [数据维度变化详解](#数据维度变化详解)
10. [实际应用示例](#实际应用示例)
11. [优化策略建议](#优化策略建议)

---

## 功能概述

### 什么是Fine_tune？

Fine_tune（微调）功能允许从预训练的基础模型（foundation model）开始训练，而不是从随机初始化开始。

**核心优势**：
1. **快速收敛**: 利用基础模型的知识，减少训练时间（通常只需几千代）
2. **小数据集友好**: 在有限数据上快速获得好的结果
3. **知识迁移**: 将通用模型适应到特定体系
4. **数值稳定性**: q_scaler从基础模型继承，数值更稳定

### 用户接口

**语法**：
```bash
fine_tune <nep_model_file> <nep_restart_file>
```

**参数说明**：
- `<nep_model_file>`: 基础模型的势函数文件（如`nep89_20250409.txt`）
- `<nep_restart_file>`: 基础模型的重启文件（如`nep89_20250409.restart`）

**代码位置**：`src/main_nep/parameters.cu` - `Parameters::parse_fine_tune()` (第1321-1331行)

### 基础模型信息

当前GPUMD提供的基础模型位于`GPUMD/potentials/nep/nep89_20250409`：
- **元素数量**: 89种元素（H到Pu）
- **缺失元素**: Po, At, Rn, Fr, Ra（5种元素）
- **实际支持**: 84种元素

---

## 基础模型文件结构

### nep.txt文件结构

**位置**：基础模型文件，用于读取模型配置和参数值

**文件格式**（按顺序）：

| 行号 | 内容 | 示例 | 说明 |
|------|------|------|------|
| 1 | 模型类型和元素列表 | `nep4_zbl 89 H He Li ... Pu` | 模型类型、元素数量、元素符号 |
| 2 | ZBL设置 | `zbl 1.0 2.0` | ZBL内截断、外截断（Å） |
| 3 | 截断半径 | `cutoff 6.0 5.0 200 150` | 径向截断、角向截断、最大邻居数 |
| 4 | n_max参数 | `n_max 4 4` | 径向n_max、角向n_max |
| 5 | basis_size参数 | `basis_size 8 8` | 径向basis_size、角向basis_size |
| 6 | l_max参数 | `l_max 4 2 1` | 3-body、4-body、5-body的L_max |
| 7 | ANN结构 | `ANN 80 0` | 隐藏层神经元数、其他信息 |
| 8+ | 参数值 | 每行一个浮点数 | 神经网络参数、描述符参数 |
| ... | q_scaler | 每行一个浮点数 | 描述符缩放因子（dim个） |

**参数部分结构**：
```
参数总数 = num_ann + num_cnk_radial + num_cnk_angular + dim

[0 ... num_ann-1]                    : 神经网络参数
  [0 ... 89×ann_1-1]                 : 89种元素的ANN参数
  [89×ann_1]                         : 全局偏置
[num_ann ... num_ann+num_cnk_radial-1] : 径向描述符参数
[num_ann+num_cnk_radial ... num_ann+num_cnk_radial+num_cnk_angular-1] : 角向描述符参数
[num_ann+num_cnk_radial+num_cnk_angular ... num_tot-1] : q_scaler
```

### nep.restart文件结构

**位置**：基础模型重启文件，用于读取参数分布（μ和σ）

**文件格式**：
```
每行: mu_value sigma_value
总行数: num_tot = num_ann + num_cnk_radial + num_cnk_angular
```

**数据结构**：
```cpp
// 文件内容示例（前几行）
-0.123456  0.045678  // 第1个参数的mu和sigma
0.234567   0.056789  // 第2个参数的mu和sigma
...
```

**参数索引结构**：
```
restart_mu/sigma数组索引:
[0 ... num_ann-1]                    : 神经网络参数分布
  [0 ... 89×ann_1-1]                 : 89种元素的ANN参数分布
  [89×ann_1]                         : 全局偏置分布
[num_ann ... num_ann+num_cnk_radial-1] : 径向描述符参数分布
[num_ann+num_cnk_radial ... num_tot-1] : 角向描述符参数分布
```

---

## 参数验证与兼容性检查

### 必须匹配的参数

在fine_tune模式下，以下参数必须与基础模型**完全一致**，不能修改：

| 参数 | 基础模型值 | 说明 | 代码位置 |
|------|-----------|------|---------|
| `version` | 4 | NEP版本 | `src/main_nep/parameters.cu` 第288行 |
| `zbl` | 2 | ZBL内截断1.0Å，外截断2.0Å | 第298-310行 |
| `cutoff` | 6 5 | 径向6Å，角向5Å | 第312-324行 |
| `n_max` | 4 4 | 径向和角向的最大阶数 | 第326-336行 |
| `basis_size` | 8 8 | 径向和角向的基函数数量 | 第338-348行 |
| `l_max` | 4 2 1 | 3-body、4-body、5-body的L_max | 第350-363行 |
| `neuron` | 80 | 隐藏层神经元数量 | 第365-372行 |

### 验证函数

**位置**：`src/main_nep/parameters.cu` - `Parameters::check_foundation_model()` (第288-375行)

**验证流程**：

```cpp
void Parameters::check_foundation_model()
{
  // 1. 打开基础模型文件
  std::ifstream input(fine_tune_nep_txt);
  
  // 2. 跳过第1行（模型类型）
  tokens = get_tokens(input);
  
  // 3. 验证ZBL参数
  tokens = get_tokens(input);
  if (zbl_rc_inner != 1.0 || zbl_rc_outer != 2.0) {
    PRINT_INPUT_ERROR("ZBL cutoff mismatches");
  }
  
  // 4. 验证cutoff
  tokens = get_tokens(input);
  if (rc_radial != 6.0 || rc_angular != 5.0) {
    PRINT_INPUT_ERROR("NEP cutoff mismatches");
  }
  
  // 5. 验证n_max
  tokens = get_tokens(input);
  if (n_max_radial != 4 || n_max_angular != 4) {
    PRINT_INPUT_ERROR("n_max mismatches");
  }
  
  // 6. 验证basis_size
  tokens = get_tokens(input);
  if (basis_size_radial != 8 || basis_size_angular != 8) {
    PRINT_INPUT_ERROR("basis_size mismatches");
  }
  
  // 7. 验证l_max
  tokens = get_tokens(input);
  if (L_max != 4 || L_max_4body != 2 || L_max_5body != 1) {
    PRINT_INPUT_ERROR("l_max mismatches");
  }
  
  // 8. 验证neuron
  tokens = get_tokens(input);
  if (num_neurons1 != 80) {
    PRINT_INPUT_ERROR("neuron mismatches");
  }
}
```

**验证时机**：在`Parameters::report_inputs()`中调用（第383行）

### 可以修改的参数

以下参数可以根据需要调整：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `lambda_1`, `lambda_2` | 正则化权重 | 0（基础模型已正则化）或0.001 |
| `lambda_e`, `lambda_f`, `lambda_v` | 损失权重 | 1（默认） |
| `batch` | 批次大小 | 5000（可根据数据量调整） |
| `population` | 种群大小 | 50（默认） |
| `generation` | 最大代数 | 5000（fine_tune通常收敛快） |
| `save_potential` | 保存设置 | 1000 0（每1000代保存） |
| `sigma0` | 初始标准差 | 0.01（比默认0.1小，fine_tune需要更小的学习率） |

---

## 元素映射机制

### 映射的必要性

基础模型支持89种元素，但：
1. **索引不连续**: 缺少5种元素（Po, At, Rn, Fr, Ra）
2. **需要映射**: 将原子序数（1-94）映射到基础模型的元素索引（0-88）
3. **用户子集**: 用户可能只需要基础模型的元素子集

### 元素映射数组

**位置**：`src/main_nep/snes.cu` 第147-153行

```cpp
const int element_map[94] = {
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
  20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
  40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
  60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
  80,81,82,0,0,0,0,0,83,84,85,86,87,88
};
```

**映射规则**：
```cpp
element_index = element_map[atomic_number - 1]
```

其中：
- `atomic_number`: 元素的原子序数（1-94）
- `element_index`: 在基础模型中的索引（0-88，或0表示不存在）

### 完整元素映射表

| 原子序数 | 元素符号 | 基础模型索引 | 映射值 | 说明 |
|---------|---------|------------|--------|------|
| 1 | H | 0 | 0 | ✓ 存在 |
| 2 | He | 1 | 1 | ✓ 存在 |
| ... | ... | ... | ... | ... |
| 82 | Pb | 81 | 81 | ✓ 存在 |
| 83 | Po | - | 0 | ✗ 不存在 |
| 84 | At | - | 0 | ✗ 不存在 |
| 85 | Rn | - | 0 | ✗ 不存在 |
| 86 | Fr | - | 0 | ✗ 不存在 |
| 87 | Ra | - | 0 | ✗ 不存在 |
| 88 | Ac | 82 | 82 | ✓ 存在 |
| 89 | Th | 83 | 83 | ✓ 存在 |
| ... | ... | ... | ... | ... |
| 94 | Pu | 87 | 87 | ✓ 存在 |

**缺失元素**：Po (83), At (84), Rn (85), Fr (86), Ra (87)

### 映射使用示例

#### 示例1: 提取Si的参数

```cpp
// 用户指定: type 1 Si
// Si的原子序数 = 14
int atomic_number = 14;
int element_index = element_map[14 - 1];  // element_map[13] = 13

// 从基础模型提取Si的ANN参数
for (int j = 0; j < number_of_variables_ann_1; ++j) {
  mu[count] = restart_mu[13 * number_of_variables_ann_1 + j];
  sigma[count] = restart_sigma[13 * number_of_variables_ann_1 + j];
  ++count;
}
```

#### 示例2: 提取Si-C对的描述符参数

```cpp
// Si: 原子序数14 → element_index_1 = 13
// C: 原子序数6 → element_index_2 = 5
int element_index_1 = element_map[14 - 1];  // 13
int element_index_2 = element_map[6 - 1];   // 5

// 计算元素对索引（基础模型中）
int t12 = element_index_1 * 89 + element_index_2;  // 13*89 + 5 = 1162

// 提取径向描述符参数（对于n=0, k=0）
int nk = 0 * (basis_size_radial + 1) + 0;  // 0
int index = nk * 89 * 89 + t12 + num_ann;  // 0*7921 + 1162 + num_ann
mu[count] = restart_mu[index];
```

### 边界情况处理

**H元素（原子序数1）**：
- H的`element_index = 0`，这与"不存在"的标记相同
- 需要特殊判断：如果`atomic_number == 1`，则`element_index = 0`是有效的
- 其他元素的`element_index = 0`表示不存在

**缺失元素的检测**：
- 使用缺失元素会导致从错误位置提取参数
- 建议：在使用fine_tune前，确保所有元素都在基础模型的89种元素中

---

## 参数映射与提取

### 基础模型参数数量计算

**位置**：`src/main_nep/snes.cu` 第155-159行

```cpp
const int NUM89 = 89;  // 基础模型支持89种元素
const int num_ann = NUM89 * para.number_of_variables_ann_1 + (para.charge_mode ? 2 : 1);
const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
```

**参数数量详解**：

#### 1. 神经网络参数 (num_ann)

```
num_ann = 89 × number_of_variables_ann_1 + (charge_mode ? 2 : 1)
```

其中：
- `89`: 基础模型支持的元素数量
- `number_of_variables_ann_1`: 每种元素的ANN参数数量
  - `= (dim + 2) × num_neurons1`
  - `= (30 + 2) × 80 = 2560`（假设dim=30, num_neurons1=80）
- `+1`: 全局偏置（charge_mode时+2）

**示例计算**：
```
num_ann = 89 × 2560 + 1 = 227,841
```

#### 2. 径向描述符参数 (num_cnk_radial)

```
num_cnk_radial = 89 × 89 × (n_max_radial + 1) × (basis_size_radial + 1)
                = 89² × 5 × 9
                = 7,921 × 45
                = 356,445
```

其中：
- `89² = 7,921`: 元素对总数
- `n_max_radial + 1 = 5`: 径向阶数（n = 0, 1, 2, 3, 4）
- `basis_size_radial + 1 = 9`: 基函数数量（k = 0, 1, ..., 8）

#### 3. 角向描述符参数 (num_cnk_angular)

```
num_cnk_angular = 89 × 89 × (n_max_angular + 1) × (basis_size_angular + 1)
                 = 89² × 5 × 9
                 = 7,921 × 45
                 = 356,445
```

**总参数数量**：
```
num_tot = num_ann + num_cnk_radial + num_cnk_angular
        = 227,841 + 356,445 + 356,445
        = 940,731
```

### 参数提取流程

**位置**：`src/main_nep/snes.cu` - `SNES::initialize_mu_and_sigma_fine_tune()` (第144-238行)

#### 步骤1: 读取restart文件

```cpp
std::ifstream input(para.fine_tune_nep_restart);
std::vector<float> restart_mu(num_tot);
std::vector<float> restart_sigma(num_tot);

for (int n = 0; n < num_tot; ++n) {
  tokens = get_tokens(input);
  restart_mu[n] = get_double_from_token(tokens[0], ...);
  restart_sigma[n] = get_double_from_token(tokens[1], ...);
}
```

**维度信息**：

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `restart_mu` | `[num_tot]` = `[940,731]` | `float` | CPU | 基础模型所有参数的均值 |
| `restart_sigma` | `[num_tot]` = `[940,731]` | `float` | CPU | 基础模型所有参数的标准差 |

#### 步骤2: 提取神经网络参数

**位置**：`src/main_nep/snes.cu` 第181-190行

```cpp
int count = 0;
for (int i = 0; i < para.num_types; ++i) {
  int element_index = element_map[para.atomic_numbers[i] - 1];
  for (int j = 0; j < para.number_of_variables_ann_1; ++j) {
    mu[count] = restart_mu[element_index * para.number_of_variables_ann_1 + j];
    sigma[count] = restart_sigma[element_index * para.number_of_variables_ann_1 + j];
    ++count;
  }
}
++count; // 全局偏置
```

**映射逻辑**：

1. **元素索引映射**：
   ```
   element_index = element_map[atomic_number - 1]
   ```

2. **参数提取**：
   ```
   mu[count] = restart_mu[element_index × number_of_variables_ann_1 + j]
   ```
   - 从基础模型中提取该元素的所有ANN参数
   - 包括：输入层权重、隐藏层偏置、输出层权重等

3. **全局偏置**：
   ```
   mu[count] = restart_mu[89 × number_of_variables_ann_1]  // 全局偏置位置
   ```

**维度变化**：

| 阶段 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 基础模型 | `restart_mu[0...num_ann-1]` | `[227,841]` | 89种元素的ANN参数 |
| 提取后 | `mu[0...num_types×ann_1]` | `[num_types × 2560 + 1]` | 仅用户指定元素的ANN参数 |

**示例**（用户指定Si, C, O三种元素）：
- 提取前：227,841个参数（89种元素）
- 提取后：3 × 2560 + 1 = 7,681个参数（3种元素）

#### 步骤3: 提取径向描述符参数

**位置**：`src/main_nep/snes.cu` 第192-211行

```cpp
for (int n = 0; n <= para.n_max_radial; ++n) {
  for (int k = 0; k <= para.basis_size_radial; ++k) {
    int nk = n * (para.basis_size_radial + 1) + k;
    for (int t1 = 0; t1 < para.num_types; ++t1) {
      for (int t2 = 0; t2 < para.num_types; ++t2) {
        int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
        int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
        int t12 = element_index_1 * NUM89 + element_index_2;
        
        mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann];
        
        #ifdef FINE_TUNE_DESCRIPTOR
          sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
        #else
          sigma[count] = 0.0f;  // 冻结描述符参数
        #endif
        
        ++count;
      }
    }
  }
}
```

**映射逻辑**：

1. **元素对索引计算**：
   ```
   t12 = element_index_1 × 89 + element_index_2
   ```

2. **参数位置计算**：
   ```
   index_in_restart = nk × 89² + t12 + num_ann
   ```
   - `nk`: n和k的组合索引 = `n × (basis_size_radial + 1) + k`
   - `89²`: 基础模型的元素对总数
   - `num_ann`: 跳过神经网络参数部分

**维度变化**：

| 阶段 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 基础模型 | `restart_mu[num_ann...num_ann+num_cnk_radial-1]` | `[356,445]` | 89×89种元素对的径向参数 |
| 提取后 | `mu[...]` | `[num_types² × 5 × 9]` | 仅用户指定元素对的径向参数 |

**示例**（用户指定Si, C, O三种元素）：
- 提取前：356,445个参数（7,921种元素对）
- 提取后：3² × 5 × 9 = 405个参数（9种元素对：Si-Si, Si-C, Si-O, C-Si, C-C, C-O, O-Si, O-C, O-O）

#### 步骤4: 提取角向描述符参数

**位置**：`src/main_nep/snes.cu` 第213-232行

```cpp
for (int n = 0; n <= para.n_max_angular; ++n) {
  for (int k = 0; k <= para.basis_size_angular; ++k) {
    int nk = n * (para.basis_size_angular + 1) + k;
    for (int t1 = 0; t1 < para.num_types; ++t1) {
      for (int t2 = 0; t2 < para.num_types; ++t2) {
        int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
        int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
        int t12 = element_index_1 * NUM89 + element_index_2;
        mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
        
        #ifdef FINE_TUNE_DESCRIPTOR
          sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
        #else
          sigma[count] = 0.0f;
        #endif
        
        ++count;
      }
    }
  }
}
```

**与径向描述符的区别**：
- 参数位置需要额外加上`num_cnk_radial`，跳过径向描述符部分

**维度变化**：

| 阶段 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 基础模型 | `restart_mu[num_ann+num_cnk_radial...num_tot-1]` | `[356,445]` | 89×89种元素对的角向参数 |
| 提取后 | `mu[...]` | `[num_types² × 5 × 9]` | 仅用户指定元素对的角向参数 |

**示例**（用户指定Si, C, O三种元素）：
- 提取前：356,445个参数（7,921种元素对）
- 提取后：3² × 5 × 9 = 405个参数（9种元素对）

### q_scaler加载

**位置**：`src/main_nep/parameters.cu` 第258-277行

```cpp
if (fine_tune) {
  std::ifstream input(fine_tune_nep_txt);
  // 跳过前7行（模型信息）
  for (int n = 0; n < 7; ++n) {
    tokens = get_tokens(input);
  }
  // 跳过所有参数
  const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
  for (int n = 0; n < num_tot; ++n) {
    tokens = get_tokens(input);
  }
  // 读取q_scaler
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    tokens = get_tokens(input);
    q_scaler_cpu[n] = get_double_from_token(tokens[0], ...);
  }
  // 复制到GPU
  for (int device_id = 0; device_id < deviceCount; device_id++) {
    gpuSetDevice(device_id);
    q_scaler_gpu[device_id].copy_from_host(q_scaler_cpu.data());
  }
}
```

**维度信息**：

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `q_scaler_cpu` | `[dim]` = `[30]` | `float` | CPU | 描述符缩放因子 |
| `q_scaler_gpu` | `[dim]` = `[30]` | `float` | GPU | 描述符缩放因子（每个设备） |

**说明**：
- q_scaler在nep.txt文件的最后部分
- 直接读取，不需要映射（因为描述符维度相同）
- 从基础模型继承，确保数值稳定性

---

## 参数初始化策略

### 初始化概述

fine_tune模式下的参数初始化与普通训练模式的主要区别：

| 特性 | 普通训练 | Fine_tune模式 |
|------|---------|--------------|
| mu初始化 | 随机（-1到1） | 从基础模型加载 |
| sigma初始化 | 固定值（sigma0=0.1） | 从基础模型加载 |
| q_scaler | 大值（1e10） | 从基础模型加载 |
| 描述符参数 | 可优化 | 默认冻结（可选优化） |

### 初始化函数调用

**位置**：`src/main_nep/snes.cu` 第80-84行

```cpp
if (para.fine_tune) {
  initialize_mu_and_sigma_fine_tune(para);
} else {
  initialize_mu_and_sigma(para);
}
```

### mu和sigma的初始化

#### 神经网络参数

**位置**：`src/main_nep/snes.cu` 第181-190行

```cpp
int count = 0;
for (int i = 0; i < para.num_types; ++i) {
  int element_index = element_map[para.atomic_numbers[i] - 1];
  for (int j = 0; j < para.number_of_variables_ann_1; ++j) {
    mu[count] = restart_mu[element_index * para.number_of_variables_ann_1 + j];
    sigma[count] = restart_sigma[element_index * para.number_of_variables_ann_1 + j];
    ++count;
  }
}
++count; // 全局偏置
```

**说明**：
- **mu**: 从基础模型继承，作为优化的起点
- **sigma**: 从基础模型继承，保持原有的不确定性估计

#### 描述符参数

**位置**：`src/main_nep/snes.cu` 第192-211行（径向）和第213-232行（角向）

```cpp
// 径向描述符
mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann];

#ifdef FINE_TUNE_DESCRIPTOR
  sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
#else
  sigma[count] = 0.0f;  // 冻结描述符参数
#endif
```

**关键点**：
- **mu**: 总是从基础模型继承
- **sigma**: 
  - 如果定义了`FINE_TUNE_DESCRIPTOR`宏：从基础模型继承，允许继续优化
  - 否则：设为0，冻结参数（不会更新）

---

## 参数冻结机制

### 为什么冻结描述符参数？

1. **描述符是通用的**: 基础模型在大数据集上训练，描述符已经很好地编码了原子环境
2. **减少过拟合风险**: 冻结描述符参数，只优化神经网络参数，减少可优化变量
3. **快速适应**: 通过调整神经网络权重，快速适应新体系

### sigma=0的含义

#### 在SNES更新中的行为

**位置**：`src/main_nep/snes.cu` 第610-637行（update_mu_and_sigma）

**更新公式**：
```
sigma_new = min(sigma0, sigma_old × exp(eta_sigma × gradient_sigma))
```

**当sigma=0时**：
- `sigma_new = min(sigma0, 0 × exp(...)) = min(sigma0, 0) = 0`
- 无论gradient如何，sigma始终保持为0
- 这意味着该参数被冻结，不会在训练中更新

#### 在种群生成中的行为

**位置**：`src/main_nep/snes.cu` 第413-431行（create_population）

**生成公式**：
```
population[i] = sigma[i] × s + mu[i]
```

**当sigma=0时**：
- `population[i] = 0 × s + mu[i] = mu[i]`
- 所有个体的该参数都等于mu，不会变化
- 参数被完全冻结

### 如何启用描述符优化？

如果需要优化描述符参数，需要在编译时定义宏：

```cpp
#define FINE_TUNE_DESCRIPTOR
```

然后重新编译程序。这样描述符参数的sigma会从基础模型继承，允许继续优化。

**编译选项**：
```bash
# 在Makefile或CMakeLists.txt中添加
-D FINE_TUNE_DESCRIPTOR
```

---

## 完整执行流程

### 流程图

```
nep.in中设置: fine_tune nep89_20250409.txt nep89_20250409.restart
    │
    ├─ 阶段1: 参数解析和验证
    │   ├─ Parameters::parse_fine_tune()
    │   │   ├─ 输入: "fine_tune", "nep89_20250409.txt", "nep89_20250409.restart"
    │   │   ├─ 输出: fine_tune=1, fine_tune_nep_txt, fine_tune_nep_restart
    │   │   └─ 代码位置: src/main_nep/parameters.cu 第1321-1331行
    │   │
    │   └─ Parameters::check_foundation_model()
    │       ├─ 输入: fine_tune_nep_txt文件
    │       ├─ 输出: 验证结果（通过或报错）
    │       ├─ 验证: ZBL, cutoff, n_max, basis_size, l_max, neuron
    │       └─ 代码位置: src/main_nep/parameters.cu 第288-375行
    │
    ├─ 阶段2: q_scaler加载
    │   └─ Parameters::calculate_parameters()
    │       ├─ 打开fine_tune_nep_txt文件
    │       ├─ 跳过前7行（模型类型、ZBL、cutoff等）
    │       ├─ 跳过所有参数（num_tot行）
    │       ├─ 读取q_scaler值（dim个）
    │       ├─ 输出: q_scaler_cpu数组 [dim]
    │       └─ 代码位置: src/main_nep/parameters.cu 第258-277行
    │
    ├─ 阶段3: 参数分布初始化
    │   └─ SNES::initialize_mu_and_sigma_fine_tune()
    │       ├─ 输入: fine_tune_nep_restart文件
    │       ├─ 读取: restart_mu [num_tot], restart_sigma [num_tot]
    │       ├─ 元素映射: element_map[atomic_number - 1]
    │       ├─ 提取神经网络参数: mu [num_types×ann_1+1], sigma [num_types×ann_1+1]
    │       ├─ 提取描述符参数: mu [num_types²×45×2], sigma [num_types²×45×2] (或0)
    │       ├─ 输出: mu和sigma数组（当前模型的参数分布）
    │       └─ 代码位置: src/main_nep/snes.cu 第144-238行
    │
    └─ 阶段4: 训练过程
        └─ SNES::compute()
            ├─ 与普通训练相同，但：
            │   ├─ 初始mu和sigma来自基础模型
            │   ├─ 描述符参数可能被冻结（sigma=0）
            │   └─ q_scaler保持不变
            └─ 代码位置: src/main_nep/snes.cu 第299-411行
```

### 关键代码位置总结

| 功能 | 文件位置 | 函数/位置 | 行号 |
|------|---------|----------|------|
| 参数解析 | `src/main_nep/parameters.cu` | `Parameters::parse_fine_tune()` | 1321-1331 |
| 基础模型验证 | `src/main_nep/parameters.cu` | `Parameters::check_foundation_model()` | 288-375 |
| q_scaler加载 | `src/main_nep/parameters.cu` | `Parameters::calculate_parameters()` | 258-277 |
| 参数分布初始化 | `src/main_nep/snes.cu` | `SNES::initialize_mu_and_sigma_fine_tune()` | 144-238 |
| 元素映射数组 | `src/main_nep/snes.cu` | `element_map[94]` | 147-153 |
| 训练特殊处理 | `src/main_nep/fitness.cu` | `Fitness::compute()` | 165 |

---

## 数据维度变化详解

### 参数数量变化

#### 基础模型参数数量

| 参数类型 | 数量 | 计算公式 | 示例值 |
|---------|------|---------|--------|
| 神经网络参数 | `num_ann` | `89 × 2560 + 1` | 227,841 |
| 径向描述符参数 | `num_cnk_radial` | `89² × 5 × 9` | 356,445 |
| 角向描述符参数 | `num_cnk_angular` | `89² × 5 × 9` | 356,445 |
| **总计** | `num_tot` | - | **940,731** |

#### 用户模型参数数量（示例：Si, C, O）

| 参数类型 | 数量 | 计算公式 | 示例值 |
|---------|------|---------|--------|
| 神经网络参数 | `num_ann_user` | `3 × 2560 + 1` | 7,681 |
| 径向描述符参数 | `num_cnk_radial_user` | `3² × 5 × 9` | 405 |
| 角向描述符参数 | `num_cnk_angular_user` | `3² × 5 × 9` | 405 |
| **总计** | `num_tot_user` | - | **8,491** |

**参数减少比例**：8,491 / 940,731 ≈ 0.9%（仅提取需要的参数）

### 内存使用

#### 基础模型文件大小

| 文件 | 内容 | 大小估算 |
|------|------|---------|
| `nep.txt` | 模型配置 + 参数值 + q_scaler | ~3.8 MB (940,731 × 4 bytes) |
| `nep.restart` | mu和sigma（每行2个浮点数） | ~7.5 MB (940,731 × 2 × 4 bytes) |

#### 用户模型内存使用

| 数据结构 | 维度 | 内存大小 |
|---------|------|---------|
| `mu` | `[8,491]` | 8,491 × 4 = 34 KB |
| `sigma` | `[8,491]` | 8,491 × 4 = 34 KB |
| `q_scaler` | `[30]` | 30 × 4 = 120 B |
| **总计** | - | **~68 KB** |

### 参数索引映射表

#### 基础模型索引结构

```
restart_mu/sigma数组索引:
[0 ... num_ann-1]                    : 神经网络参数
  [0 ... 89×ann_1-1]                 : 89种元素的ANN参数
  [89×ann_1]                         : 全局偏置
[num_ann ... num_ann+num_cnk_radial-1] : 径向描述符参数
  [num_ann + nk×89² + t12]           : 元素对(t1,t2)的径向参数
[num_ann+num_cnk_radial ... num_tot-1] : 角向描述符参数
  [num_ann+num_cnk_radial + nk×89² + t12] : 元素对(t1,t2)的角向参数
```

#### 当前模型索引结构

```
mu/sigma数组索引:
[0 ... num_types×ann_1]              : 神经网络参数（仅用户指定的元素）
[num_types×ann_1+1 ... num_types×ann_1+1+num_types²×45] : 径向描述符参数
[num_types×ann_1+1+num_types²×45 ... num_tot_user-1] : 角向描述符参数
```

---

## 实际应用示例

### 示例1: 从基础模型微调Si-C-O体系

#### nep.in配置

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

# 用户指定的元素（基础模型的子集）
type 3 Si C O

# 可以调整的参数
lambda_1   0      # 可以设为0，因为基础模型已经正则化
lambda_2   0
lambda_e   1
lambda_f   1
lambda_v   1
batch      5000
population 50
generation 5000
save_potential 1000 0
```

#### 参数继承过程

1. **验证配置**：检查所有必须匹配的参数是否一致
2. **加载q_scaler**：从`nep89_20250409.txt`读取30个q_scaler值
3. **提取神经网络参数**：
   - Si（原子序数14）：从基础模型索引13提取2,560个参数
   - C（原子序数6）：从基础模型索引5提取2,560个参数
   - O（原子序数8）：从基础模型索引7提取2,560个参数
   - 全局偏置：1个参数
   - 总计：7,681个ANN参数
4. **提取描述符参数**：
   - 9个元素对（Si-Si, Si-C, Si-O, C-Si, C-C, C-O, O-Si, O-C, O-O）
   - 每个元素对45个径向系数 + 45个角向系数 = 90个系数
   - 总计：810个描述符参数
   - **默认行为**：所有描述符参数的sigma设为0（冻结）
5. **开始训练**：只优化7,681个ANN参数，描述符参数保持不变

#### 参数数量对比

| 模式 | ANN参数 | 描述符参数 | 总参数 | 可优化参数 |
|------|---------|-----------|--------|-----------|
| 普通训练 | 7,681 | 810 | 8,491 | 8,491 |
| Fine_tune（默认） | 7,681 | 810 | 8,491 | 7,681（描述符冻结） |
| Fine_tune（启用描述符优化） | 7,681 | 810 | 8,491 | 8,491 |

### 示例2: 启用描述符优化

如果需要同时优化描述符参数：

1. **编译时定义宏**：
   ```bash
   # 在Makefile或CMakeLists.txt中添加
   -D FINE_TUNE_DESCRIPTOR
   ```

2. **调整正则化参数**：
   ```bash
   lambda_1   0.001  # 需要一些正则化防止过拟合
   lambda_2   0.001
   ```

3. **可能需要更多训练代数**：
   ```bash
   generation 10000  # 因为要优化的参数更多
   ```

---

## 优化策略建议

### 策略1: 默认策略（描述符冻结）

**适用场景**：
- 小数据集（< 10,000个结构）
- 快速适应新体系
- 减少过拟合风险

**参数设置**：
```bash
lambda_1   0      # 可以设为0，因为基础模型已正则化
lambda_2   0
lambda_e   1
lambda_f   1
lambda_v   1
sigma0     0.01   # 比默认值0.1小，fine_tune需要更小的学习率
generation 5000   # 通常几千代就足够
```

**优势**：
- 快速收敛（通常1,000-3,000代）
- 稳定
- 不易过拟合

### 策略2: 描述符优化（需要编译时定义宏）

**适用场景**：
- 数据集较大（> 50,000个结构）
- 需要精细调整描述符
- 体系与基础模型差异较大

**编译选项**：
```bash
# 在编译时添加
-D FINE_TUNE_DESCRIPTOR
```

**参数设置**：
```bash
lambda_1   0.001  # 需要一些正则化
lambda_2   0.001
lambda_e   1
lambda_f   1
lambda_v   1
sigma0     0.01
generation 10000  # 可能需要更多代数
```

**优势**：
- 更灵活
- 可以适应更大差异
- 可能获得更好的结果

### 策略3: 混合策略

可以手动修改代码，选择性地冻结某些描述符参数：

```cpp
// 只冻结径向描述符，优化角向描述符
if (is_radial_descriptor) {
  sigma[count] = 0.0f;
} else {
  sigma[count] = restart_sigma[...];
}
```

### 实际应用建议

#### 1. 选择合适的lambda值

由于基础模型已经在大数据集上训练，通常：
- `lambda_1 = 0`: 不需要L1正则化
- `lambda_2 = 0`: 不需要L2正则化
- 或者使用很小的值（如0.001）

#### 2. 调整学习率

fine_tune通常需要较小的学习率（通过sigma0控制）：
```bash
sigma0 0.01  # 比默认值0.1小
```

#### 3. 监控训练

观察`loss.out`文件，确保：
- 损失快速下降
- 没有过拟合迹象
- 测试集误差也在下降

#### 4. 检查点恢复

fine_tune模式下，如果训练中断，可以从检查点恢复：
```bash
# 训练中断后，使用nep.restart恢复
# nep.restart包含更新后的mu和sigma
```

**注意**: 恢复的mu和sigma是fine_tune后的值，不是基础模型的值。

---

## 总结

### Fine_tune核心机制

1. **参数验证**：确保配置与基础模型兼容
2. **参数继承**：
   - q_scaler：从基础模型的nep.txt文件读取
   - mu和sigma：从基础模型的nep.restart文件读取
3. **元素映射**：通过element_map将用户指定的元素映射到基础模型的索引
4. **参数提取**：提取对应元素的ANN参数和描述符参数
5. **参数冻结**：默认情况下，描述符参数被冻结（sigma=0），只优化ANN参数

### 优势与限制

**优势**：
1. **快速收敛**：从好的初始点开始，通常只需几千代就能收敛
2. **小数据集友好**：适合数据有限的情况
3. **知识迁移**：利用基础模型在大数据集上学到的知识
4. **数值稳定性**：q_scaler从基础模型继承，数值更稳定

**限制**：
1. **参数约束**：必须与基础模型的某些参数完全匹配
2. **元素限制**：只能使用基础模型支持的89种元素
3. **描述符冻结**：默认情况下描述符参数被冻结（可通过宏启用优化）

### 相关文档

- [NEP训练流程详解](training.md) - 完整的训练流程和数据维度变化
- [SNES算法](04_snes_algorithm.md) - 优化算法详解
- [Parameters类详解](02_parameters_class.md) - 参数计算逻辑

