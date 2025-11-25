# Fine_tune参数映射机制

本文档详细说明fine_tune功能中如何从基础模型提取和映射参数到当前模型。

## 基础模型文件格式

### nep.txt文件结构

基础模型的`nep.txt`文件包含以下内容（按顺序）：

1. **第1行**: 模型类型和元素数量
   ```
   nep4_zbl 89 H He Li ... Pu
   ```

2. **第2行**: ZBL设置
   ```
   zbl 1.0 2.0
   ```

3. **第3行**: 截断半径和邻居列表大小
   ```
   cutoff 6.0 5.0 max_NN_radial max_NN_angular
   ```

4. **第4行**: n_max参数
   ```
   n_max 4 4
   ```

5. **第5行**: basis_size参数
   ```
   basis_size 8 8
   ```

6. **第6行**: l_max参数
   ```
   l_max 4 2 1
   ```

7. **第7行**: ANN结构
   ```
   ANN 80 0
   ```

8. **参数部分**: 所有可优化参数（每行一个浮点数）
   - 神经网络参数（num_ann个）
   - 描述符参数（num_cnk_radial + num_cnk_angular个）
   - q_scaler（dim个）

### nep.restart文件结构

基础模型的`nep.restart`文件格式：

```
每行: mu_value sigma_value
总行数: num_tot = num_ann + num_cnk_radial + num_cnk_angular
```

## 参数数量计算

### 基础模型参数数量

**位置**: `src/main_nep/snes.cu` 第155-159行

```cpp
const int NUM89 = 89;  // 基础模型支持89种元素
const int num_ann = NUM89 * para.number_of_variables_ann_1 + (para.charge_mode ? 2 : 1);
const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
```

**计算说明**:

1. **神经网络参数 (num_ann)**:
   ```
   num_ann = 89 × number_of_variables_ann_1 + (charge_mode ? 2 : 1)
   ```
   - 89种元素，每种有`number_of_variables_ann_1`个参数
   - 加上全局偏置（charge_mode时加2）

2. **径向描述符参数 (num_cnk_radial)**:
   ```
   num_cnk_radial = 89 × 89 × (n_max_radial + 1) × (basis_size_radial + 1)
   ```
   - 89×89种元素对
   - 每种对：`(n_max_radial + 1)`个n值
   - 每个n：`(basis_size_radial + 1)`个k值

3. **角向描述符参数 (num_cnk_angular)**:
   ```
   num_cnk_angular = 89 × 89 × (n_max_angular + 1) × (basis_size_angular + 1)
   ```
   - 类似径向，但用于角向描述符

## 参数提取流程

### 步骤1: 读取restart文件

**位置**: `src/main_nep/snes.cu` 第163-178行

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

**功能**: 将整个基础模型的参数分布读入内存

### 步骤2: 提取神经网络参数

**位置**: `src/main_nep/snes.cu` 第181-190行

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

**映射逻辑**:

1. **元素索引映射**:
   ```
   element_index = element_map[atomic_number - 1]
   ```
   - 将当前元素的原子序数映射到基础模型的元素索引（0-88）

2. **参数提取**:
   ```
   mu[count] = restart_mu[element_index × number_of_variables_ann_1 + j]
   ```
   - 从基础模型中提取该元素的所有ANN参数
   - 包括：输入层权重、隐藏层偏置、输出层权重等

3. **全局偏置**:
   ```
   mu[count] = restart_mu[89 × number_of_variables_ann_1]  // 全局偏置位置
   ```

### 步骤3: 提取径向描述符参数

**位置**: `src/main_nep/snes.cu` 第192-211行

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

**映射逻辑**:

1. **元素对索引计算**:
   ```
   t12 = element_index_1 × 89 + element_index_2
   ```
   - 将当前模型的元素对(t1, t2)映射到基础模型的元素对索引

2. **参数位置计算**:
   ```
   index_in_restart = nk × 89² + t12 + num_ann
   ```
   - `nk`: n和k的组合索引
   - `89²`: 基础模型的元素对总数
   - `num_ann`: 跳过神经网络参数部分

3. **描述符参数冻结**:
   - 默认情况下（未定义`FINE_TUNE_DESCRIPTOR`），`sigma = 0`
   - 这意味着描述符参数被冻结，不会在训练中更新
   - 如果定义了宏，描述符参数可以继续优化

### 步骤4: 提取角向描述符参数

**位置**: `src/main_nep/snes.cu` 第213-232行

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

**与径向描述符的区别**:
- 参数位置需要额外加上`num_cnk_radial`，跳过径向描述符部分

## 参数索引映射表

### 基础模型索引结构

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

### 当前模型索引结构

```
mu/sigma数组索引:
[0 ... num_types×ann_1]              : 神经网络参数（仅用户指定的元素）
[num_types×ann_1+1 ...]              : 描述符参数（仅用户指定的元素对）
```

## q_scaler加载

**位置**: `src/main_nep/parameters.cu` 第258-277行

```cpp
if (fine_tune) {
  std::ifstream input(fine_tune_nep_txt);
  // 跳过前7行（模型信息）
  for (int n = 0; n < 7; ++n) {
    tokens = get_tokens(input);
  }
  // 跳过所有参数
  for (int n = 0; n < num_tot; ++n) {
    tokens = get_tokens(input);
  }
  // 读取q_scaler
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    tokens = get_tokens(input);
    q_scaler_cpu[n] = get_double_from_token(tokens[0], ...);
  }
}
```

**说明**:
- q_scaler在nep.txt文件的最后部分
- 直接读取，不需要映射（因为描述符维度相同）

## 实际示例

假设用户要微调一个包含Si、C、O三种元素的模型：

### 基础模型
- 89种元素（索引0-88）
- Si: 原子序数14 → 基础模型索引13
- C: 原子序数6 → 基础模型索引5
- O: 原子序数8 → 基础模型索引7

### 参数提取

1. **神经网络参数**:
   - 从基础模型的索引13、5、7位置提取Si、C、O的ANN参数
   - 提取全局偏置（索引89×ann_1）

2. **描述符参数**:
   - Si-Si对: 从基础模型索引`13×89+13`提取
   - Si-C对: 从基础模型索引`13×89+5`提取
   - Si-O对: 从基础模型索引`13×89+7`提取
   - C-C对: 从基础模型索引`5×89+5`提取
   - C-O对: 从基础模型索引`5×89+7`提取
   - O-O对: 从基础模型索引`7×89+7`提取

## 相关文档

- [Fine_tune功能概述](09_fine_tune_overview.md) - 整体功能说明
- [元素映射机制](11_fine_tune_element_mapping.md) - 元素索引映射详解

