# Fine_tune元素映射机制

本文档详细说明fine_tune功能中元素索引的映射机制，这是参数提取的关键步骤。

## 元素映射的必要性

### 问题背景

基础模型（foundation model）支持89种元素（H到Pu），但：
1. **不连续**: 缺少5种元素（Po, At, Rn, Fr, Ra），实际只有84种
2. **索引映射**: 需要将原子序数（1-94）映射到基础模型的元素索引（0-88）

### 映射数组定义

**位置**: `src/main_nep/snes.cu` 第147-153行

```cpp
const int element_map[94] = {
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
  20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
  40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
  60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
  80,81,82,0,0,0,0,0,83,84,85,86,87,88
};
```

## 映射规则详解

### 数组索引

- **数组索引**: 0-93（对应原子序数1-94）
- **数组值**: 基础模型的元素索引（0-88，或0表示不存在）

### 映射逻辑

```cpp
element_index = element_map[atomic_number - 1]
```

其中：
- `atomic_number`: 元素的原子序数（1-94）
- `element_index`: 在基础模型中的索引（0-88）

### 缺失元素处理

映射数组中值为0的位置表示该元素在基础模型中不存在：

```
element_map[82] = 0  // Po (原子序数83)
element_map[83] = 0  // At (原子序数84)
element_map[84] = 0  // Rn (原子序数85)
element_map[85] = 0  // Fr (原子序数86)
element_map[86] = 0  // Ra (原子序数87)
```

**注意**: 如果用户尝试使用这些元素，程序会报错（因为无法从基础模型提取参数）。

## 完整元素映射表

| 原子序数 | 元素符号 | 基础模型索引 | 映射值 |
|---------|---------|------------|--------|
| 1 | H | 0 | 0 |
| 2 | He | 1 | 1 |
| 3 | Li | 2 | 2 |
| ... | ... | ... | ... |
| 82 | Pb | 81 | 81 |
| 83 | Po | - | 0 (不存在) |
| 84 | At | - | 0 (不存在) |
| 85 | Rn | - | 0 (不存在) |
| 86 | Fr | - | 0 (不存在) |
| 87 | Ra | - | 0 (不存在) |
| 88 | Ac | 82 | 82 |
| 89 | Th | 83 | 83 |
| 90 | Pa | 84 | 84 |
| 91 | U | 85 | 85 |
| 92 | Np | 86 | 86 |
| 93 | Pu | 87 | 87 |
| 94 | (Am) | 88 | 88 |

## 映射使用示例

### 示例1: 提取Si的参数

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

### 示例2: 提取Si-C对的描述符参数

```cpp
// Si: 原子序数14 → element_index_1 = 13
// C: 原子序数6 → element_index_2 = 5
int element_index_1 = element_map[14 - 1];  // 13
int element_index_2 = element_map[6 - 1];   // 5

// 计算元素对索引
int t12 = element_index_1 * 89 + element_index_2;  // 13*89 + 5 = 1162

// 提取径向描述符参数（对于n=0, k=0）
int nk = 0 * (basis_size_radial + 1) + 0;  // 0
int index = nk * 89 * 89 + t12 + num_ann;  // 0*7921 + 1162 + num_ann
mu[count] = restart_mu[index];
```

### 示例3: 不支持的元素

```cpp
// 用户尝试使用Po (原子序数84)
int atomic_number = 84;
int element_index = element_map[84 - 1];  // element_map[83] = 0

// element_index = 0 表示该元素不存在
// 如果继续使用，会导致从错误位置提取参数
```

## 参数提取中的映射

### 神经网络参数映射

**位置**: `src/main_nep/snes.cu` 第182-189行

```cpp
for (int i = 0; i < para.num_types; ++i) {
  int element_index = element_map[para.atomic_numbers[i] - 1];
  
  // 检查元素是否存在
  if (element_index == 0 && para.atomic_numbers[i] != 1) {
    // 错误：元素不在基础模型中（H的索引也是0，需要特殊处理）
  }
  
  // 提取该元素的所有ANN参数
  for (int j = 0; j < para.number_of_variables_ann_1; ++j) {
    mu[count] = restart_mu[element_index * para.number_of_variables_ann_1 + j];
    sigma[count] = restart_sigma[element_index * para.number_of_variables_ann_1 + j];
    ++count;
  }
}
```

### 描述符参数映射

**位置**: `src/main_nep/snes.cu` 第196-210行（径向）和第217-231行（角向）

```cpp
for (int t1 = 0; t1 < para.num_types; ++t1) {
  for (int t2 = 0; t2 < para.num_types; ++t2) {
    int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
    int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
    
    // 计算基础模型中的元素对索引
    int t12 = element_index_1 * NUM89 + element_index_2;
    
    // 提取参数
    mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + offset];
    sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + offset];
    ++count;
  }
}
```

## 边界情况处理

### H元素（原子序数1）

H的`element_index = 0`，这与"不存在"的标记相同。但H确实存在于基础模型中，所以：
- 需要特殊判断：如果`atomic_number == 1`，则`element_index = 0`是有效的
- 其他元素的`element_index = 0`表示不存在

### 缺失元素的检测

虽然代码中没有显式检查，但使用缺失元素会导致：
1. 从错误位置提取参数（索引0）
2. 可能提取到其他元素的参数
3. 训练结果不正确

**建议**: 在使用fine_tune前，确保所有元素都在基础模型的89种元素中。

## 元素对索引计算

### 基础模型中的元素对索引

对于基础模型的89种元素，元素对(t1, t2)的索引为：

```
t12 = t1 × 89 + t2
```

其中：
- `t1, t2 ∈ [0, 88]`（基础模型的元素索引）
- `t12 ∈ [0, 7920]`（89×89 = 7921种元素对）

### 当前模型中的元素对索引

对于当前模型的`num_types`种元素，元素对(t1, t2)的索引为：

```
t12_local = t1 × num_types + t2
```

但在提取参数时，需要映射回基础模型的索引：

```
t12_foundation = element_map[atomic_numbers[t1] - 1] × 89 + 
                 element_map[atomic_numbers[t2] - 1]
```

## 实际应用场景

### 场景1: 单元素体系

```bash
# 用户只想微调Si的模型
type 1 Si

# 映射:
# Si (Z=14) → element_index = 13
# 提取: 基础模型索引13的所有参数
```

### 场景2: 多元素体系

```bash
# 用户要微调Si-C-O体系
type 3 Si C O

# 映射:
# Si (Z=14) → 13
# C (Z=6) → 5
# O (Z=8) → 7

# 提取:
# - ANN参数: 索引13, 5, 7
# - 描述符参数: 所有元素对 (13,13), (13,5), (13,7), (5,5), (5,7), (7,7)
```

### 场景3: 包含缺失元素（错误）

```bash
# 用户尝试使用Po（不存在于基础模型）
type 1 Po

# 映射:
# Po (Z=84) → element_index = 0 (不存在标记)
# 结果: 会错误地从索引0（H）提取参数
```

## 相关文档

- [Fine_tune功能概述](09_fine_tune_overview.md) - 整体功能说明
- [参数映射机制](10_fine_tune_parameter_mapping.md) - 参数提取逻辑

