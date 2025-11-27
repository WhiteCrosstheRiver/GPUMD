# NEP训练参数详解与Fine_tune机制

本文档详细说明NEP训练时具体优化的参数，以及fine_tune功能如何基于NEP训练结果进行微调。

## 目录

1. [NEP训练参数概述](#nep训练参数概述)
2. [神经网络参数（ANN参数）](#神经网络参数ann参数)
3. [描述符参数](#描述符参数)
4. [描述符缩放因子（q_scaler）](#描述符缩放因子q_scaler)
5. [参数数量计算](#参数数量计算)
6. [Fine_tune机制详解](#fine_tune机制详解)
7. [参数继承与映射](#参数继承与映射)
8. [实际应用示例](#实际应用示例)

---

## NEP训练参数概述

NEP训练过程中，通过SNES（Separable Natural Evolution Strategy）算法优化的参数主要包括三大类：

1. **神经网络参数（ANN参数）**：用于从描述符预测原子能量的神经网络权重和偏置
2. **描述符参数**：用于构建原子环境描述符的系数
3. **描述符缩放因子（q_scaler）**：用于数值稳定性的缩放因子

总参数数量由以下公式决定：

```
number_of_variables = number_of_variables_ann + number_of_variables_descriptor
```

其中：
- `number_of_variables_ann`：神经网络参数数量
- `number_of_variables_descriptor`：描述符参数数量

---

## 神经网络参数（ANN参数）

### 参数结构

NEP使用一个单隐藏层的前馈神经网络，结构为：`输入层 → 隐藏层 → 输出层`

#### NEP3版本

所有元素类型共享同一个神经网络：

```
number_of_variables_ann_1 = (dim + 2) × num_neurons1
number_of_variables_ann = (dim + 2) × num_neurons1 + 1
```

其中：
- `dim`：描述符向量维度
- `num_neurons1`：隐藏层神经元数量
- `+2`：包含激活函数的参数（每个神经元2个参数）
- `+1`：全局偏置（b1）

**参数组成**：
- `w0[dim × num_neurons1]`：输入层到隐藏层的权重矩阵
- `b0[num_neurons1]`：隐藏层偏置向量
- `w1[num_neurons1]`：隐藏层到输出层的权重向量
- `b1[1]`：输出层偏置（全局偏置）

#### NEP4版本

每种元素类型有独立的神经网络参数：

```
number_of_variables_ann_1 = (dim + 2) × num_neurons1
number_of_variables_ann = (dim + 2) × num_neurons1 × num_types + 1
```

**参数组成**（对每种元素类型）：
- `w0[type][dim × num_neurons1]`：该类型的输入层到隐藏层权重
- `b0[type][num_neurons1]`：该类型的隐藏层偏置
- `w1[type][num_neurons1]`：该类型的隐藏层到输出层权重
- `b1[1]`：全局偏置（所有类型共享）

**代码位置**：`src/main_nep/parameters.cu` 第216-230行

### 参数初始化

#### 普通训练模式

**位置**：`src/main_nep/snes.cu` 第100-142行

```cpp
// 从nep.restart文件读取（如果存在）
if (fid_restart != NULL) {
  for (int n = 0; n < number_of_variables; ++n) {
    fscanf(fid_restart, "%f%f", &mu[n], &sigma[n]);
  }
}
// 否则随机初始化
else {
  for (int n = 0; n < number_of_variables; ++n) {
    mu[n] = (r1(rng) - 0.5f) * 2.0f;  // [-1, 1]
    sigma[n] = para.sigma0;            // 默认0.1
  }
}
```

#### Fine_tune模式

从基础模型继承参数分布（详见[Fine_tune机制详解](#fine_tune机制详解)部分）。

---

## 描述符参数

描述符参数用于构建原子环境的数学表示，包括径向描述符和角向描述符的系数。

### 径向描述符参数

径向描述符（2-body）的系数为 `c_n^k`，其中：
- `n = 0, 1, 2, ..., n_max_radial`：径向多项式的阶数
- `k = 0, 1, 2, ..., basis_size_radial`：基函数的索引

**参数数量**：
```
num_cnk_radial = num_types × num_types × (n_max_radial + 1) × (basis_size_radial + 1)
```

**物理意义**：
- 描述原子对之间的径向相互作用
- 每个元素对（type1, type2）都有独立的系数集合

### 角向描述符参数

角向描述符（3-body及以上）的系数也为 `c_n^k`，其中：
- `n = 0, 1, 2, ..., n_max_angular`：角向多项式的阶数
- `k = 0, 1, 2, ..., basis_size_angular`：基函数的索引

**参数数量**：
```
num_cnk_angular = num_types × num_types × (n_max_angular + 1) × (basis_size_angular + 1)
```

**物理意义**：
- 描述多体（3-body, 4-body, 5-body）相互作用
- 每个元素对都有独立的系数集合

### 总描述符参数数量

**代码位置**：`src/main_nep/parameters.cu` 第232-234行

```cpp
number_of_variables_descriptor = 
  num_types * num_types *
  (dim_radial * (basis_size_radial + 1) + 
   (n_max_angular + 1) * (basis_size_angular + 1));
```

其中：
- `dim_radial = n_max_radial + 1`：径向描述符维度
- 第一项：径向描述符参数
- 第二项：角向描述符参数

---

## 描述符缩放因子（q_scaler）

### 作用

`q_scaler`是描述符的缩放因子，用于：
1. **数值稳定性**：防止描述符值过大或过小导致数值溢出
2. **优化效率**：将描述符值缩放到合适的范围，提高优化收敛速度

### 初始化

#### 普通训练模式

**位置**：`src/main_nep/parameters.cu` 第257行

```cpp
q_scaler_cpu.resize(dim, 1.0e10f);  // 初始化为大值
```

在训练过程中，`q_scaler`会被自动优化（如果`calculate_q_scaler = true`）。

#### Fine_tune模式

**位置**：`src/main_nep/parameters.cu` 第258-277行

```cpp
if (fine_tune) {
  // 从基础模型的nep.txt文件读取q_scaler
  std::ifstream input(fine_tune_nep_txt);
  // 跳过前7行（模型类型、ZBL、cutoff等）
  // 跳过所有参数（num_tot行）
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    tokens = get_tokens(input);
    q_scaler_cpu[n] = get_double_from_token(tokens[0], ...);
  }
}
```

**训练中的处理**：

**位置**：`src/main_nep/fitness.cu` 第165行

```cpp
potential->find_force(
  para,
  dummy_solution.data(),
  train_set[n],
  (para.fine_tune ? false : true),  // fine_tune时不重新计算q_scaler
  true,
  deviceCount);
```

当`calculate_q_scaler = false`时，直接使用从基础模型加载的值，不进行优化。

---

## 参数数量计算

### 完整计算公式

**代码位置**：`src/main_nep/parameters.cu` 第200-239行

#### 1. 计算描述符维度

```cpp
dim_radial = n_max_radial + 1;
dim_angular = (n_max_angular + 1) * L_max;
if (L_max_4body == 2) {
  dim_angular += n_max_angular + 1;  // 4-body描述符
}
if (L_max_5body == 1) {
  dim_angular += n_max_angular + 1;  // 5-body描述符
}
dim = dim_radial + dim_angular;
if (train_mode == 3) {
  dim += 1;  // 温度依赖自由能模式，添加温度维度
}
```

#### 2. 计算神经网络参数数量

**NEP3**：
```cpp
number_of_variables_ann_1 = (dim + 2) * num_neurons1;
number_of_variables_ann = (dim + 2) * num_neurons1 + 1;
```

**NEP4**：
```cpp
number_of_variables_ann_1 = (dim + 2) * num_neurons1;
number_of_variables_ann = (dim + 2) * num_neurons1 * num_types + 1;

// 如果启用电荷模式
if (charge_mode) {
  number_of_variables_ann_1 += num_neurons1;
  number_of_variables_ann += num_neurons1 * num_types + 1;
  if (charge_mode >= 4) {
    number_of_variables_ann_1 += num_neurons1;
    number_of_variables_ann += num_neurons1 * num_types;
  }
}
```

#### 3. 计算描述符参数数量

```cpp
number_of_variables_descriptor = 
  num_types * num_types *
  (dim_radial * (basis_size_radial + 1) + 
   (n_max_angular + 1) * (basis_size_angular + 1));
```

#### 4. 计算总参数数量

```cpp
number_of_variables = number_of_variables_ann + number_of_variables_descriptor;

// 如果是极化率模式（train_mode == 2），需要两套ANN参数
if (train_mode == 2) {
  number_of_variables += number_of_variables_ann;
}
```

### 典型示例

假设配置为：
- `num_types = 3`（Si, C, O）
- `n_max_radial = 4`, `n_max_angular = 4`
- `basis_size_radial = 8`, `basis_size_angular = 8`
- `L_max = 4`, `L_max_4body = 2`, `L_max_5body = 0`
- `num_neurons1 = 30`
- `version = 4`（NEP4）

**计算过程**：

1. **描述符维度**：
   ```
   dim_radial = 4 + 1 = 5
   dim_angular = (4 + 1) × 4 + (4 + 1) = 25
   dim = 5 + 25 = 30
   ```

2. **神经网络参数（NEP4）**：
   ```
   number_of_variables_ann_1 = (30 + 2) × 30 = 960
   number_of_variables_ann = 960 × 3 + 1 = 2881
   ```

3. **描述符参数**：
   ```
   number_of_variables_descriptor = 3 × 3 × (5 × 9 + 5 × 9) = 3 × 3 × 90 = 810
   ```

4. **总参数数量**：
   ```
   number_of_variables = 2881 + 810 = 3691
   ```

**代码验证位置**：`src/main_nep/parameters.cu` 第610-615行会打印这些信息。

---

## Fine_tune机制详解

Fine_tune功能允许从预训练的基础模型（foundation model）开始训练，而不是从随机初始化开始。

### 基础模型文件

基础模型包含两个文件：
1. **`nep.txt`**：包含训练好的参数值和模型配置
2. **`nep.restart`**：包含SNES算法的参数分布（mu和sigma）

### Fine_tune执行流程

#### 阶段1：参数验证

**位置**：`src/main_nep/parameters.cu` 第288-375行

在开始fine_tune之前，系统会验证当前配置与基础模型的兼容性：

```cpp
void Parameters::check_foundation_model()
{
  // 检查ZBL截断半径
  // 检查NEP截断半径（rc_radial, rc_angular）
  // 检查n_max参数
  // 检查basis_size参数
  // 检查l_max参数
  // 检查神经元数量
}
```

**必须匹配的参数**：
- `version = 4`
- `zbl = 2`
- `cutoff = 6 5`
- `n_max = 4 4`
- `basis_size = 8 8`
- `l_max = 4 2 1`
- `neuron = 80`

#### 阶段2：加载q_scaler

**位置**：`src/main_nep/parameters.cu` 第258-277行

从基础模型的`nep.txt`文件读取`q_scaler`值：

```cpp
if (fine_tune) {
  std::ifstream input(fine_tune_nep_txt);
  // 跳过前7行（模型类型、ZBL、cutoff等）
  // 跳过所有参数（num_tot行）
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    tokens = get_tokens(input);
    q_scaler_cpu[n] = get_double_from_token(tokens[0], ...);
  }
}
```

#### 阶段3：初始化参数分布（mu和sigma）

**位置**：`src/main_nep/snes.cu` 第144-238行

从基础模型的`nep.restart`文件读取参数分布：

```cpp
void SNES::initialize_mu_and_sigma_fine_tune(Parameters& para)
{
  // 1. 读取基础模型的restart文件
  const int NUM89 = 89;  // 基础模型包含89种元素
  const int num_ann = NUM89 * para.number_of_variables_ann_1 + (para.charge_mode ? 2 : 1);
  const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
  const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
  
  std::vector<float> restart_mu(num_tot);
  std::vector<float> restart_sigma(num_tot);
  
  // 读取所有参数
  for (int n = 0; n < num_tot; ++n) {
    tokens = get_tokens(input);
    restart_mu[n] = get_double_from_token(tokens[0], ...);
    restart_sigma[n] = get_double_from_token(tokens[1], ...);
  }
  
  // 2. 提取神经网络参数
  int count = 0;
  for (int i = 0; i < para.num_types; ++i) {
    int element_index = element_map[para.atomic_numbers[i] - 1];
    for (int j = 0; j < para.number_of_variables_ann_1; ++j) {
      mu[count] = restart_mu[element_index * para.number_of_variables_ann_1 + j];
      sigma[count] = restart_sigma[element_index * para.number_of_variables_ann_1 + j];
      ++count;
    }
  }
  ++count;  // 全局偏置
  
  // 3. 提取描述符参数
  // 径向描述符
  for (int n = 0; n <= para.n_max_radial; ++n) {
    for (int k = 0; k <= para.basis_size_radial; ++k) {
      for (int t1 = 0; t1 < para.num_types; ++t1) {
        for (int t2 = 0; t2 < para.num_types; ++t2) {
          int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
          int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
          mu[count] = restart_mu[...];
          #ifdef FINE_TUNE_DESCRIPTOR
            sigma[count] = restart_sigma[...];  // 允许优化
          #else
            sigma[count] = 0.0f;  // 冻结参数
          #endif
          ++count;
        }
      }
    }
  }
  
  // 角向描述符（类似处理）
}
```

---

## 参数继承与映射

### 元素映射

基础模型包含89种元素（H到Pu，但缺少5种：Po, At, Rn, Fr, Ra），而用户可能只需要其中的一个子集。

**元素映射表**：

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

**映射逻辑**：
```cpp
int element_index = element_map[para.atomic_numbers[i] - 1];
```

其中`atomic_numbers[i]`是原子序数（1=H, 2=He, ..., 94=Pu）。

### 参数提取策略

#### 1. 神经网络参数

对用户指定的每种元素类型：
- 从基础模型中提取该元素的所有ANN参数
- 包括：`w0`, `b0`, `w1`（每种元素独立）
- 全局偏置`b1`从基础模型继承

#### 2. 描述符参数

对用户指定的所有元素对（type1, type2）：
- 从基础模型中提取对应的描述符系数
- 包括：径向描述符系数和角向描述符系数

**关键点**：
- `mu`（均值）：总是从基础模型继承，作为优化的起点
- `sigma`（标准差）：
  - 如果定义了`FINE_TUNE_DESCRIPTOR`宏：从基础模型继承，允许继续优化
  - 否则：设为0，冻结参数（不会在训练中更新）

### 参数冻结机制

当`sigma = 0`时，参数被冻结：

1. **种群生成时**：
   ```cpp
   population[i] = sigma[i] × s + mu[i]
   // 当sigma[i] = 0时，population[i] = mu[i]（固定不变）
   ```

2. **分布更新时**：
   ```cpp
   sigma_new = min(sigma0, sigma_old × exp(eta_sigma × gradient_sigma))
   // 当sigma_old = 0时，sigma_new = 0（始终保持为0）
   ```

**默认行为**：描述符参数被冻结，只优化神经网络参数。

**启用描述符优化**：编译时定义`FINE_TUNE_DESCRIPTOR`宏。

---

## 实际应用示例

### 示例1：从基础模型微调Si-C-O体系

**nep.in配置**：

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

**参数继承过程**：

1. **验证配置**：检查所有必须匹配的参数是否一致
2. **加载q_scaler**：从`nep89_20250409.txt`读取30个q_scaler值
3. **提取神经网络参数**：
   - Si（原子序数14）：从基础模型索引13提取960个参数
   - C（原子序数6）：从基础模型索引5提取960个参数
   - O（原子序数8）：从基础模型索引7提取960个参数
   - 全局偏置：1个参数
   - 总计：2881个ANN参数
4. **提取描述符参数**：
   - 9个元素对（Si-Si, Si-C, Si-O, C-Si, C-C, C-O, O-Si, O-C, O-O）
   - 每个元素对90个系数（径向45 + 角向45）
   - 总计：810个描述符参数
   - **默认行为**：所有描述符参数的sigma设为0（冻结）
5. **开始训练**：只优化2881个ANN参数，描述符参数保持不变

### 示例2：启用描述符优化

如果需要同时优化描述符参数，需要：

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

### 参数数量对比

| 模式 | ANN参数 | 描述符参数 | 总参数 | 可优化参数 |
|------|---------|-----------|--------|-----------|
| 普通训练 | 2881 | 810 | 3691 | 3691 |
| Fine_tune（默认） | 2881 | 810 | 3691 | 2881（描述符冻结） |
| Fine_tune（启用描述符优化） | 2881 | 810 | 3691 | 3691 |

---

## 总结

### NEP训练参数

1. **神经网络参数**：
   - NEP3：所有元素共享，参数数量 = `(dim + 2) × num_neurons1 + 1`
   - NEP4：每种元素独立，参数数量 = `(dim + 2) × num_neurons1 × num_types + 1`
   - 包括：权重矩阵（w0, w1）和偏置向量（b0, b1）

2. **描述符参数**：
   - 径向描述符系数：`num_types² × (n_max_radial + 1) × (basis_size_radial + 1)`
   - 角向描述符系数：`num_types² × (n_max_angular + 1) × (basis_size_angular + 1)`

3. **q_scaler**：
   - 描述符缩放因子，用于数值稳定性
   - 在普通训练中自动优化
   - 在fine_tune中从基础模型继承

### Fine_tune机制

1. **参数验证**：确保配置与基础模型兼容
2. **参数继承**：
   - q_scaler：从基础模型的nep.txt文件读取
   - mu和sigma：从基础模型的nep.restart文件读取
3. **元素映射**：通过element_map将用户指定的元素映射到基础模型的索引
4. **参数提取**：提取对应元素的ANN参数和描述符参数
5. **参数冻结**：默认情况下，描述符参数被冻结（sigma=0），只优化ANN参数

### 优势

1. **快速收敛**：从好的初始点开始，通常只需几千代就能收敛
2. **小数据集友好**：适合数据有限的情况
3. **知识迁移**：利用基础模型在大数据集上学到的知识
4. **数值稳定性**：q_scaler从基础模型继承，数值更稳定

### 相关文档

- [Fine_tune功能概述](09_fine_tune_overview.md) - 整体功能说明
- [参数初始化策略](12_fine_tune_initialization.md) - mu和sigma的初始化细节
- [SNES算法](04_snes_algorithm.md) - 优化算法详解
- [Parameters类详解](02_parameters_class.md) - 参数计算逻辑

