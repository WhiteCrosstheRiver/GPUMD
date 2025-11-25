# SNES算法详解

本文档详细记录SNES (Separable Natural Evolution Strategy) 优化算法的原理和实现。

## 文件位置

- **头文件**: `src/main_nep/snes.cuh` (第29-74行)
- **实现文件**: `src/main_nep/snes.cu` (第44-670行)

## 算法原理

### 参考文献

T. Schaul, T. Glasmachers, and J. Schmidhuber,
"High Dimensions and Heavy Tails for Natural Evolution Strategies",
https://doi.org/10.1145/2001576.2001692

### 核心思想

SNES是一种基于自然进化策略的优化算法，用于优化高维参数空间。它使用高斯分布来采样候选解，并根据适应度排序更新分布参数。

### 算法流程

1. **初始化**: 设置参数分布 μ (均值) 和 σ (标准差)
2. **生成种群**: 从分布中采样生成候选解
3. **评估适应度**: 计算每个候选解的适应度
4. **排序**: 按适应度对种群排序
5. **更新分布**: 根据排序结果更新 μ 和 σ
6. **重复**: 返回步骤2，直到达到最大代数

## 类结构

### 成员变量

#### 优化参数
- `maximum_generation`: 最大代数
- `number_of_variables`: 变量数量
- `population_size`: 种群大小
- `eta_sigma`: 学习率参数

#### 分布参数
- `mu`: `std::vector<float>` - 均值向量（CPU）
- `sigma`: `std::vector<float>` - 标准差向量（CPU）
- `gpu_mu`: `GPU_Vector<float>` - 均值向量（GPU）
- `gpu_sigma`: `GPU_Vector<float>` - 标准差向量（GPU）

#### 种群数据
- `population`: `std::vector<float>` - 当前种群（CPU）
- `gpu_population`: `GPU_Vector<float>` - 当前种群（GPU）
- `gpu_s`: `GPU_Vector<float>` - 标准正态分布采样值

#### 适应度和排序
- `fitness`: `std::vector<float>` - 适应度值
- `index`: `std::vector<int>` - 排序索引
- `utility`: `std::vector<float>` - utility函数值

#### 正则化
- `cost_L1reg`: `std::vector<float>` - L1正则化成本
- `cost_L2reg`: `std::vector<float>` - L2正则化成本

#### 变量类型
- `type_of_variable`: `std::vector<int>` - 每个变量的类型（用于NEP4）

#### 随机数生成
- `rng`: `std::mt19937` - CPU随机数生成器
- `curand_states`: `GPU_Vector<gpurandState>` - GPU随机数状态

## 构造函数

### SNES::SNES(Parameters& para, Fitness* fitness_function)

**位置**: `src/main_nep/snes.cu` 第44-89行

**执行流程**:

#### 1. 初始化参数

```cpp
maximum_generation = para.maximum_generation;
number_of_variables = para.number_of_variables;
population_size = para.population_size;
```

#### 2. 计算学习率

```cpp
int num = number_of_variables;
if (para.version != 3) {
  num /= para.num_types;  // NEP4按类型归一化
}
eta_sigma = (3.0f + std::log(num * 1.0f)) / (5.0f * sqrt(num * 1.0f)) / 2.0f;
```

**公式**:
```
η_σ = (3 + ln(n)) / (5 * √n) / 2
```

其中 n 是归一化后的变量数量。

#### 3. 分配内存

```cpp
fitness.resize(population_size * 7 * (para.num_types + 1));
index.resize(population_size * (para.num_types + 1));
population.resize(population_size * number_of_variables);
mu.resize(number_of_variables);
sigma.resize(number_of_variables);
// ... GPU内存分配
```

#### 4. 初始化随机数生成器

```cpp
initialize_rng();  // CPU随机数生成器
// GPU随机数状态初始化
initialize_curand_states<<<...>>>(curand_states.data(), N, 1234567);
```

#### 5. 初始化分布参数

```cpp
if (para.fine_tune) {
  initialize_mu_and_sigma_fine_tune(para);
} else {
  initialize_mu_and_sigma(para);
}
```

#### 6. 计算utility函数

```cpp
calculate_utility();
```

#### 7. 确定变量类型

```cpp
find_type_of_variable(para);
```

#### 8. 开始优化

```cpp
compute(para, fitness_function);
```

## 核心方法

### initialize_mu_and_sigma()

**位置**: `src/main_nep/snes.cu` 第100-142行

**功能**: 初始化参数分布

**流程**:

1. **从重启文件读取**（如果存在`nep.restart`）:
```cpp
FILE* fid_restart = fopen("nep.restart", "r");
if (fid_restart != NULL) {
  for (int n = 0; n < number_of_variables; ++n) {
    fscanf(fid_restart, "%f%f", &mu[n], &sigma[n]);
  }
  // 处理电荷翻转（如果启用）
  fclose(fid_restart);
}
```

2. **随机初始化**（如果不存在重启文件）:
```cpp
std::uniform_real_distribution<float> r1(0, 1);
for (int n = 0; n < number_of_variables; ++n) {
  mu[n] = (r1(rng) - 0.5f) * 2.0f;  // [-1, 1]
  sigma[n] = para.sigma0;
}
```

3. **电荷模式特殊处理**:
```cpp
if (para.charge_mode) {
  // 确保初始电荷为0
  // 确保初始sqrt(epsilon_inf) > 0
}
```

### initialize_mu_and_sigma_fine_tune()

**位置**: `src/main_nep/snes.cu` 第144-238行

**功能**: 从基础模型加载参数进行微调

**流程**:
1. 读取基础模型的`nep.restart`文件
2. 根据元素映射提取对应类型的参数
3. 对于描述符参数，可选择是否冻结（通过`FINE_TUNE_DESCRIPTOR`宏）

### calculate_utility()

**位置**: `src/main_nep/snes.cu` 第240-250行

**功能**: 计算utility函数值

**公式**:
```cpp
utility[n] = max(0, ln(population_size * 0.5 + 1) - ln(n + 1))
utility_sum = Σ utility[n]
utility[n] = utility[n] / utility_sum - 1 / population_size
```

**物理意义**: 
- 排名越靠前（适应度越好），utility值越大
- utility值归一化后均值为0

### find_type_of_variable()

**位置**: `src/main_nep/snes.cu` 第252-297行

**功能**: 确定每个变量属于哪个原子类型（用于NEP4的类型特定正则化）

**分类**:
1. **神经网络参数**: 按元素类型分组
2. **描述符参数**: 按第一个元素类型分组

### create_population()

**位置**: `src/main_nep/snes.cu` 第433-447行

**功能**: 从当前分布生成新种群

**GPU Kernel**: `gpu_create_population`

**公式**:
```cpp
s ~ N(0, 1)  // 标准正态分布采样
population[i] = sigma[i] * s + mu[i]  // 变换到目标分布
```

### regularize() / regularize_NEP4()

**位置**: `src/main_nep/snes.cu` 第488-579行

**功能**: 计算L1和L2正则化损失

#### NEP3版本 (regularize)

```cpp
cost_L1 = lambda_1 * Σ|θ_i| / number_of_variables
cost_L2 = lambda_2 * sqrt(Σ(θ_i^2) / number_of_variables)
```

#### NEP4版本 (regularize_NEP4)

按类型分别计算正则化：
```cpp
for (int t = 0; t <= num_types; ++t) {
  // 计算属于类型t的变量的L1和L2损失
  cost_L1 = lambda_1 * Σ|θ_i| / num_variables_type_t
  cost_L2 = lambda_2 * sqrt(Σ(θ_i^2) / num_variables_type_t)
}
```

**GPU Kernel**: `gpu_find_L1_L2` 或 `gpu_find_L1_L2_NEP4`

### sort_population()

**位置**: `src/main_nep/snes.cu` 第596-608行

**功能**: 对种群按适应度排序

**实现**: 使用插入排序（`insertion_sort`）

**排序方式**: 对每个类型分别排序，得到多个排序索引

### update_mu_and_sigma()

**位置**: `src/main_nep/snes.cu` 第639-657行

**功能**: 根据适应度排序更新分布参数

**GPU Kernel**: `gpu_update_mu_and_sigma`

**更新公式**:

对于每个变量 i:

```cpp
gradient_mu[i] = Σ(s[i][p] * utility[p])
gradient_sigma[i] = Σ((s[i][p]^2 - 1) * utility[p])

mu[i] += sigma[i] * gradient_mu[i]
sigma[i] = min(sigma0, sigma[i] * exp(eta_sigma * gradient_sigma[i]))
```

其中:
- `s[i][p]` 是第p个个体中变量i的标准正态采样值
- `utility[p]` 是第p个个体的utility值（按适应度排序后）

**物理意义**:
- μ更新: 向适应度好的方向移动
- σ更新: 根据适应度梯度调整探索范围
- σ有上限: 防止过度探索

### compute()

**位置**: `src/main_nep/snes.cu` 第299-411行

**功能**: 主优化循环

**训练模式流程**:
```cpp
for (int n = 0; n < maximum_generation; ++n) {
  create_population(para);                    // 生成种群
  fitness_function->compute(...);             // 计算适应度
  regularize_NEP4(para) or regularize(para);  // 计算正则化
  sort_population(para);                       // 排序
  fitness_function->report_error(...);        // 报告误差
  update_mu_and_sigma(para);                  // 更新分布
  // 每100代保存nep.restart
  if (0 == (n + 1) % 100) {
    output_mu_and_sigma(para, "nep.restart");
  }
}
```

**预测模式流程**:
```cpp
// 从nep.txt读取参数
// 调用fitness_function->predict()进行预测
```

## 数学公式总结

### Utility函数

```
u_i = max(0, ln(λ/2 + 1) - ln(i + 1))
u_i = u_i / Σu_j - 1/λ
```

其中 λ 是种群大小，i 是排名（0为最好）。

### 参数更新

```
μ ← μ + σ · Σ(s_i · u_i)
σ ← σ · exp(η_σ · Σ((s_i² - 1) · u_i))
σ ← min(σ_max, σ)
```

其中:
- s_i ~ N(0,1) 是标准正态采样
- u_i 是utility值
- η_σ 是学习率

### 正则化损失

**L1正则化**:
```
L1 = λ_1 · (1/n) · Σ|θ_i|
```

**L2正则化**:
```
L2 = λ_2 · sqrt((1/n) · Σ(θ_i²))
```

### 总适应度

```
fitness_total = L1 + L2 + λ_e·RMSE_energy + λ_f·RMSE_force + λ_v·RMSE_virial + λ_q·RMSE_charge
```

## GPU并行化

### 种群生成

- 每个线程处理一个变量
- 并行生成所有个体的所有变量

### 正则化计算

- 每个block处理一个个体
- 使用shared memory进行归约

### 分布更新

- 每个线程处理一个变量
- 并行计算所有变量的梯度

## 相关文档

- [Fitness类详解](03_fitness_class.md) - 适应度计算
- [数学公式集合](07_mathematical_formulas.md) - 完整数学公式

