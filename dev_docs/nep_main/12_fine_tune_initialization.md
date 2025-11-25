# Fine_tune参数初始化和优化策略

本文档详细说明fine_tune模式下参数分布（mu和sigma）的初始化策略和优化行为。

## 初始化概述

fine_tune模式下的参数初始化与普通训练模式的主要区别：

| 特性 | 普通训练 | Fine_tune模式 |
|------|---------|--------------|
| mu初始化 | 随机（-1到1） | 从基础模型加载 |
| sigma初始化 | 固定值（sigma0） | 从基础模型加载 |
| q_scaler | 大值（1e10） | 从基础模型加载 |
| 描述符参数 | 可优化 | 默认冻结（可选优化） |

## mu和sigma的初始化

### 初始化函数

**位置**: `src/main_nep/snes.cu` 第80-84行

```cpp
if (para.fine_tune) {
  initialize_mu_and_sigma_fine_tune(para);
} else {
  initialize_mu_and_sigma(para);
}
```

### 详细初始化流程

**位置**: `src/main_nep/snes.cu` 第144-238行

#### 步骤1: 读取基础模型restart文件

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

**文件格式**: 每行两个浮点数，分别表示mu和sigma

**总行数**: `num_tot = num_ann + num_cnk_radial + num_cnk_angular`

#### 步骤2: 提取神经网络参数的mu和sigma

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

**说明**:
- mu: 从基础模型继承，作为优化的起点
- sigma: 从基础模型继承，保持原有的不确定性估计

#### 步骤3: 提取描述符参数的mu和sigma

**径向描述符**:

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
          sigma[count] = 0.0f * restart_sigma[...];  // 设为0，冻结参数
        #endif
        
        ++count;
      }
    }
  }
}
```

**关键点**:
- mu: 总是从基础模型继承
- sigma: 
  - 如果定义了`FINE_TUNE_DESCRIPTOR`宏：从基础模型继承，允许继续优化
  - 否则：设为0，冻结参数（不会更新）

## 参数冻结机制

### 为什么冻结描述符参数？

1. **描述符是通用的**: 基础模型在大数据集上训练，描述符已经很好地编码了原子环境
2. **减少过拟合风险**: 冻结描述符参数，只优化神经网络参数，减少可优化变量
3. **快速适应**: 通过调整神经网络权重，快速适应新体系

### 如何启用描述符优化？

如果需要优化描述符参数，需要在编译时定义宏：

```cpp
#define FINE_TUNE_DESCRIPTOR
```

然后重新编译程序。这样描述符参数的sigma会从基础模型继承，允许继续优化。

## sigma=0的含义

### 在SNES更新中的行为

**位置**: `src/main_nep/snes.cu` 第610-637行（update_mu_and_sigma）

```cpp
gpu_update_mu_and_sigma<<<...>>>(
  population_size,
  number_of_variables,
  eta_sigma,
  para.sigma0,
  gpu_type_of_variable.data(),
  gpu_index.data(),
  gpu_utility.data(),
  gpu_s.data(),
  gpu_mu.data(),
  gpu_sigma.data());
```

**更新公式**:
```
sigma_new = min(sigma0, sigma_old × exp(eta_sigma × gradient_sigma))
```

**当sigma=0时**:
- `sigma_new = min(sigma0, 0 × exp(...)) = min(sigma0, 0) = 0`
- 无论gradient如何，sigma始终保持为0
- 这意味着该参数被冻结，不会在训练中更新

### 在种群生成中的行为

**位置**: `src/main_nep/snes.cu` 第413-431行（create_population）

```cpp
gpu_create_population<<<...>>>(
  N,
  number_of_variables,
  gpu_mu.data(),
  gpu_sigma.data(),
  curand_states.data(),
  gpu_s.data(),
  gpu_population.data());
```

**生成公式**:
```
population[i] = sigma[i] × s + mu[i]
```

**当sigma=0时**:
- `population[i] = 0 × s + mu[i] = mu[i]`
- 所有个体的该参数都等于mu，不会变化
- 参数被完全冻结

## 优化策略建议

### 策略1: 默认策略（描述符冻结）

**适用场景**:
- 小数据集
- 快速适应新体系
- 减少过拟合风险

**参数设置**:
```bash
lambda_1   0      # 可以设为0，因为基础模型已正则化
lambda_2   0
lambda_e   1
lambda_f   1
lambda_v   1
generation 5000   # 通常几千代就足够
```

**优势**:
- 快速收敛
- 稳定
- 不易过拟合

### 策略2: 描述符优化（需要编译时定义宏）

**适用场景**:
- 数据集较大
- 需要精细调整描述符
- 体系与基础模型差异较大

**编译选项**:
```bash
# 在编译时添加
-D FINE_TUNE_DESCRIPTOR
```

**参数设置**:
```bash
lambda_1   0.001  # 需要一些正则化
lambda_2   0.001
lambda_e   1
lambda_f   1
lambda_v   1
generation 10000  # 可能需要更多代数
```

**优势**:
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

## q_scaler的特殊处理

### 加载q_scaler

**位置**: `src/main_nep/parameters.cu` 第258-277行

```cpp
if (fine_tune) {
  // 从基础模型加载q_scaler
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    q_scaler_cpu[n] = get_double_from_token(tokens[0], ...);
  }
}
```

### 训练中的处理

**位置**: `src/main_nep/fitness.cu` 第165行

```cpp
potential->find_force(
  para,
  dummy_solution.data(),
  train_set[n],
  (para.fine_tune ? false : true),  // fine_tune时不计算q_scaler
  true,
  deviceCount);
```

**说明**:
- `calculate_q_scaler = false`: 不重新计算q_scaler
- 直接使用从基础模型加载的值
- 这样可以保持数值稳定性

## 初始化后的优化行为

### 第0代的行为

**位置**: `src/main_nep/fitness.cu` 第158-168行

```cpp
if (generation == 0) {
  std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
  for (int n = 0; n < num_batches; ++n) {
    potential->find_force(
      para,
      dummy_solution.data(),
      train_set[n],
      (para.fine_tune ? false : true),  // 不计算q_scaler
      true,  // 计算邻居列表
      deviceCount);
  }
}
```

**目的**: 初始化GPU内存，构建邻居列表

### 后续代的优化

与普通训练相同，但：
1. 初始mu和sigma来自基础模型
2. 描述符参数可能被冻结（sigma=0）
3. q_scaler保持不变

## 检查点恢复

fine_tune模式下，如果训练中断，可以从检查点恢复：

```bash
# 训练中断后，使用nep.restart恢复
# nep.restart包含更新后的mu和sigma
```

**注意**: 恢复的mu和sigma是fine_tune后的值，不是基础模型的值。

## 实际应用建议

### 1. 选择合适的lambda值

由于基础模型已经在大数据集上训练，通常：
- `lambda_1 = 0`: 不需要L1正则化
- `lambda_2 = 0`: 不需要L2正则化
- 或者使用很小的值（如0.001）

### 2. 调整学习率

fine_tune通常需要较小的学习率（通过sigma0控制）：
```bash
sigma0 0.01  # 比默认值0.1小
```

### 3. 监控训练

观察loss.out文件，确保：
- 损失快速下降
- 没有过拟合迹象
- 测试集误差也在下降

## 相关文档

- [Fine_tune功能概述](09_fine_tune_overview.md) - 整体功能说明
- [参数映射机制](10_fine_tune_parameter_mapping.md) - 参数提取逻辑
- [元素映射机制](11_fine_tune_element_mapping.md) - 元素索引映射

