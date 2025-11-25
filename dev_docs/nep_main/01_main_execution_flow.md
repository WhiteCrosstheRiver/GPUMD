# 主程序执行流程

本文档详细分析`main.cu`中`main()`函数的执行流程。

## 文件位置

- **源文件**: `src/main_nep/main.cu`
- **函数**: `int main(int argc, char* argv[])` (第30-68行)

## 完整执行流程

### 1. 程序入口和初始化信息输出

```cpp
int main(int argc, char* argv[])
{
  print_welcome_information();      // 第32行
  print_gpu_information();          // 第33行
```

**功能**:
- `print_welcome_information()`: 打印欢迎信息，显示GPUMD版本和NEP可执行文件标识
- `print_gpu_information()`: 检测并打印所有可用GPU的信息

**输出示例**:
```
***************************************************************
*                 Welcome to use GPUMD                        *
*    (Graphics Processing Units Molecular Dynamics)           *
*                     version 4.5                             *
*              This is the nep executable                     *
***************************************************************
```

### 2. 阶段1: 初始化阶段

```cpp
  const auto time_begin1 = std::chrono::high_resolution_clock::now();  // 第39行
  Parameters para;                                                     // 第40行
  Fitness fitness(para);                                               // 第41行
  const auto time_finish1 = std::chrono::high_resolution_clock::now(); // 第42行
```

#### 2.1 Parameters对象构造

**位置**: `src/main_nep/parameters.cu` 第33-50行

**执行步骤**:
1. 调用`set_default_parameters()` - 设置所有参数的默认值
2. 调用`read_nep_in()` - 从`nep.in`文件读取用户配置
3. 如果启用ZBL，调用`read_zbl_in()` - 读取`zbl.in`文件
4. 调用`calculate_parameters()` - 计算派生参数（描述符维度、变量数量等）
5. 调用`report_inputs()` - 报告所有输入参数

**关键操作**:
- 解析`nep.in`中的关键字（如`type`, `cutoff`, `n_max`, `lambda_e`等）
- 验证参数合法性
- 计算描述符维度：`dim = dim_radial + dim_angular`
- 计算需要优化的变量总数：`number_of_variables`

#### 2.2 Fitness对象构造

**位置**: `src/main_nep/fitness.cu` 第38-142行

**执行步骤**:
1. 检测GPU设备数量
2. 读取训练数据：`read_structures(true, para, structures_train)`
   - 从`train.xyz`读取结构数据
   - 解析原子坐标、力、能量、维里等信息
3. 计算批次数：`num_batches = (structures_train.size() - 1) / para.batch_size + 1`
4. 为每个批次和每个GPU设备构建Dataset对象
5. 可选读取测试数据：`read_structures(false, para, structures_test)`
6. 根据训练模式选择Potential模型：
   - `train_mode == 1 or 2`: 创建`TNEP`对象
   - `charge_mode != 0`: 创建`NEP_Charge`对象
   - 否则: 创建`NEP`对象
7. 如果不是预测模式，打开`loss.out`文件用于记录训练误差

**关键数据结构**:
- `train_set`: `std::vector<std::vector<Dataset>>` - 每个批次每个GPU一个Dataset
- `test_set`: `std::vector<Dataset>` - 每个GPU一个Dataset
- `potential`: `std::unique_ptr<Potential>` - 势函数模型指针

#### 2.3 初始化时间统计

```cpp
  const std::chrono::duration<double> time_used1 = time_finish1 - time_begin1;  // 第44行
  print_line_1();                                                               // 第45行
  printf("Time used for initialization = %f s.\n", time_used1.count());        // 第46行
  print_line_2();                                                               // 第47行
```

**功能**: 计算并输出初始化阶段耗时

### 3. 阶段2: 训练/预测阶段

```cpp
  const auto time_begin2 = std::chrono::high_resolution_clock::now();  // 第49行
  SNES snes(para, &fitness);                                            // 第50行
  const auto time_finish2 = std::chrono::high_resolution_clock::now(); // 第51行
```

#### 3.1 SNES对象构造

**位置**: `src/main_nep/snes.cu` 第44-89行

**执行步骤**:
1. 从`Parameters`读取优化参数：
   - `maximum_generation`: 最大代数
   - `number_of_variables`: 变量数量
   - `population_size`: 种群大小
2. 计算学习率参数：`eta_sigma = (3.0 + log(num)) / (5.0 * sqrt(num)) / 2.0`
3. 初始化GPU随机数生成器状态
4. 初始化参数分布：
   - 如果存在`nep.restart`文件，从中读取mu和sigma
   - 如果启用`fine_tune`，从基础模型加载参数
   - 否则随机初始化
5. 计算utility函数值
6. 确定每个变量的类型（用于NEP4的类型特定正则化）
7. **调用`compute()`函数开始训练或预测**

#### 3.2 SNES::compute() - 核心训练循环

**位置**: `src/main_nep/snes.cu` 第299-411行

**训练模式流程** (prediction == 0):
```cpp
for (int n = 0; n < maximum_generation; ++n) {
  create_population(para);                    // 生成种群
  fitness_function->compute(...);             // 计算适应度
  regularize_NEP4(para) or regularize(para); // 计算正则化损失
  sort_population(para);                      // 排序种群
  fitness_function->report_error(...);       // 报告误差
  update_mu_and_sigma(para);                  // 更新参数分布
  // 每100代保存nep.restart
}
```

**预测模式流程** (prediction == 1):
```cpp
// 从nep.txt读取参数
// 调用fitness_function->predict()进行预测
```

#### 3.3 训练时间统计

```cpp
  const std::chrono::duration<double> time_used2 = time_finish2 - time_begin2;  // 第53行
  print_line_1();                                                                 // 第54行
  if (para.prediction == 0) {                                                     // 第55行
    printf("Time used for training = %f s.\n", time_used2.count());              // 第56行
  } else {
    printf("Time used for predicting = %f s.\n", time_used2.count());             // 第58行
  }
  print_line_2();                                                                 // 第61行
```

### 4. 程序结束

```cpp
  print_line_1();                    // 第63行
  printf("Finished running nep.\n"); // 第64行
  print_line_2();                     // 第65行

  return EXIT_SUCCESS;                // 第67行
}
```

## 函数调用链

### 初始化阶段调用链

```
main()
  ├─ print_welcome_information()
  │   └─ printf() - 打印欢迎信息
  ├─ print_gpu_information()
  │   ├─ gpuGetDeviceCount()
  │   ├─ gpuGetDeviceProperties()
  │   └─ gpuDeviceCanAccessPeer()
  ├─ Parameters::Parameters()
  │   ├─ set_default_parameters()
  │   ├─ read_nep_in()
  │   │   └─ parse_one_keyword() - 解析每个关键字
  │   ├─ read_zbl_in() [可选]
  │   ├─ calculate_parameters()
  │   └─ report_inputs()
  └─ Fitness::Fitness()
      ├─ read_structures() - 读取train.xyz
      ├─ Dataset::construct() - 构建训练数据集
      ├─ read_structures() - 读取test.xyz [可选]
      └─ new NEP/TNEP/NEP_Charge() - 创建势函数模型
```

### 训练阶段调用链

```
SNES::SNES()
  └─ SNES::compute()
      └─ [训练循环]
          ├─ SNES::create_population()
          │   └─ gpu_create_population() [GPU kernel]
          ├─ Fitness::compute()
          │   └─ Potential::find_force()
          │       ├─ 计算描述符
          │       ├─ 前向传播神经网络
          │       └─ 计算能量、力、维里
          ├─ SNES::regularize_NEP4() or regularize()
          │   └─ gpu_find_L1_L2_NEP4() [GPU kernel]
          ├─ SNES::sort_population()
          ├─ Fitness::report_error()
          │   ├─ Dataset::get_rmse_energy()
          │   ├─ Dataset::get_rmse_force()
          │   ├─ Dataset::get_rmse_virial()
          │   └─ Fitness::write_nep_txt()
          └─ SNES::update_mu_and_sigma()
              └─ gpu_update_mu_and_sigma() [GPU kernel]
```

## 关键时间点

1. **初始化开始**: `time_begin1` (第39行)
2. **初始化结束**: `time_finish1` (第42行)
3. **训练开始**: `time_begin2` (第49行)
4. **训练结束**: `time_finish2` (第51行)

## 条件分支

### prediction模式判断

- **prediction == 0**: 训练模式
  - SNES执行优化循环
  - 输出训练误差
  - 保存`nep.txt`和`nep.restart`

- **prediction == 1**: 预测模式
  - 从`nep.txt`加载参数
  - 调用`Fitness::predict()`进行预测
  - 输出预测结果到`*_train.out`文件

## 相关文档

- [Parameters类详解](02_parameters_class.md) - 参数初始化细节
- [Fitness类详解](03_fitness_class.md) - 数据加载和适应度计算
- [SNES算法](04_snes_algorithm.md) - 优化算法实现

