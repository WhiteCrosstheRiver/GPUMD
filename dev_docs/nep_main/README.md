# NEP主程序开发者文档

本文档详细记录了NEP (Neuroevolution Potential) 主程序的执行逻辑、依赖关系和实现细节。

## 文档导航

### 核心文档

1. **[主程序执行流程](01_main_execution_flow.md)** - `main.cu`函数的完整执行流程分析
2. **[Parameters类详解](02_parameters_class.md)** - 参数读取、验证和计算
3. **[Fitness类详解](03_fitness_class.md)** - 适应度计算、误差报告和预测
4. **[SNES算法](04_snes_algorithm.md)** - 可分离自然进化策略优化算法

### 数据结构与模型

5. **[Dataset数据结构](05_dataset_structure.md)** - 训练数据组织和管理
6. **[Potential模型](06_potential_models.md)** - NEP/TNEP/NEP_Charge模型实现

### 参考文档

7. **[数学公式集合](07_mathematical_formulas.md)** - 所有涉及的数学公式
8. **[工具函数说明](08_utilities_functions.md)** - 辅助函数和宏定义
14. **[NEP训练流程详解](training.md)** - 从数据到神经网络的完整流程和SNES训练方法

### Fine_tune功能文档

9. **[Fine_tune功能完整详解](fine_tune.md)** - 微调功能的完整文档（整合版，包含功能概述、参数映射、元素映射、初始化策略等）

## NEP执行总体逻辑图

### 完整执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤1: 程序启动和信息输出                                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ print_welcome_information()
    │   ├─ 输入: 无
    │   ├─ 输出: 欢迎信息（版本、程序标识）
    │   └─ Why: 标识程序身份，提供用户友好的启动信息
    │
    ├─ print_gpu_information()
    │   ├─ 输入: 无
    │   ├─ 输出: GPU设备信息（数量、名称、计算能力、内存、SM数量）
    │   └─ Why: 检测可用GPU资源，为多GPU并行做准备，验证硬件环境
    │
    └─ print_line_1/2()
        ├─ 输入: 无
        ├─ 输出: 分隔线
        └─ Why: 格式化输出，提高可读性

┌─────────────────────────────────────────────────────────────────┐
│ 步骤2: 初始化阶段 - Parameters对象构造                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Parameters::Parameters()
    │   ├─ 输入: 无（从nep.in文件读取）
    │   ├─ 输出: 完整的Parameters对象（所有配置参数）
    │   └─ Why: 统一管理所有配置参数，提供参数验证和默认值
    │
    │   ├─ set_default_parameters()
    │   │   ├─ 输入: 无
    │   │   ├─ 输出: 所有参数设置为默认值
    │   │   └─ Why: 确保所有参数都有初始值，即使未在nep.in中指定
    │   │
    │   ├─ read_nep_in()
    │   │   ├─ 输入: nep.in文件
    │   │   ├─ 输出: 解析后的参数值
    │   │   └─ Why: 读取用户配置，覆盖默认值
    │   │
    │   │   └─ parse_one_keyword()
    │   │       ├─ 输入: 关键字和参数tokens
    │   │       ├─ 输出: 解析并设置对应参数
    │   │       └─ Why: 统一解析接口，支持30+种关键字
    │   │
    │   │       └─ parse_*() 系列函数
    │   │           ├─ 输入: 参数tokens
    │   │           ├─ 输出: 验证并设置参数值
    │   │           └─ Why: 类型检查、范围验证、逻辑检查
    │   │
    │   ├─ read_zbl_in() [可选]
    │   │   ├─ 输入: zbl.in文件（如果启用ZBL）
    │   │   ├─ 输出: ZBL参数数组
    │   │   └─ Why: 支持可调ZBL势，允许优化ZBL参数
    │   │
    │   ├─ calculate_parameters()
    │   │   ├─ 输入: 基础参数（cutoff, n_max, L_max等）
    │   │   ├─ 输出: 派生参数（dim, number_of_variables等）
    │   │   └─ Why: 计算描述符维度、神经网络参数数量等，为后续内存分配提供依据
    │   │
    │   │   ├─ 计算dim_radial, dim_angular, dim
    │   │   │   └─ Why: 确定描述符向量维度，影响神经网络输入层大小
    │   │   │
    │   │   ├─ 计算number_of_variables_ann
    │   │   │   └─ Why: 确定神经网络参数数量，影响优化变量总数
    │   │   │
    │   │   ├─ 计算number_of_variables_descriptor
    │   │   │   └─ Why: 确定描述符系数数量，影响优化变量总数
    │   │   │
    │   │   ├─ 自动计算lambda_1, lambda_2 [如果未设置]
    │   │   │   └─ Why: 根据变量数量自动调整正则化强度，避免过拟合
    │   │   │
    │   │   └─ 初始化q_scaler
    │   │       └─ Why: 描述符缩放因子，用于数值稳定性
    │   │
    │   └─ report_inputs()
    │       ├─ 输入: 所有参数值
    │       ├─ 输出: 打印所有参数（区分输入和默认值）
    │       └─ Why: 让用户确认配置，便于调试和复现

┌─────────────────────────────────────────────────────────────────┐
│ 步骤3: 初始化阶段 - Fitness对象构造                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Fitness::Fitness(Parameters& para)
    │   ├─ 输入: Parameters对象
    │   ├─ 输出: Fitness对象（包含训练/测试数据集和势函数模型）
    │   └─ Why: 管理训练数据、计算适应度、报告误差
    │
    │   ├─ gpuGetDeviceCount()
    │   │   ├─ 输入: 无
    │   │   ├─ 输出: GPU设备数量
    │   │   └─ Why: 确定可用GPU数量，用于多GPU并行
    │   │
    │   ├─ read_structures(true, para, structures_train)
    │   │   ├─ 输入: train.xyz文件, Parameters
    │   │   ├─ 输出: structures_train向量（所有训练结构）
    │   │   └─ Why: 读取训练数据，包括原子坐标、力、能量、维里等
    │   │
    │   │   └─ read_one_structure()
    │   │       ├─ 输入: 文件流, Parameters
    │   │       ├─ 输出: Structure对象
    │   │       └─ Why: 解析单个结构，处理周期性边界条件
    │   │
    │   │       └─ change_box()
    │   │           ├─ 输入: 原始盒子
    │   │           ├─ 输出: 扩展盒子（用于周期性边界）
    │   │           └─ Why: 扩展盒子确保截断半径内的所有镜像原子都被考虑
    │   │
    │   ├─ 计算num_batches
    │   │   ├─ 输入: 结构数量, batch_size
    │   │   ├─ 输出: 批次数
    │   │   └─ Why: 将大数据集分批处理，减少内存占用，支持增量训练
    │   │
    │   ├─ Dataset::construct() [对每个批次和GPU]
    │   │   ├─ 输入: Parameters, structures向量, 批次范围, device_id
    │   │   ├─ 输出: Dataset对象（GPU数据已初始化）
    │   │   └─ Why: 将CPU结构数据转换为GPU格式，构建邻居列表
    │   │
    │   │   ├─ copy_structures()
    │   │   │   ├─ 输入: structures向量, 范围
    │   │   │   ├─ 输出: Dataset.structures
    │   │   │   └─ Why: 复制结构数据到Dataset
    │   │   │
    │   │   ├─ find_Na()
    │   │   │   ├─ 输入: structures
    │   │   │   ├─ 输出: Na, Na_sum（原子数和前缀和）
    │   │   │   └─ Why: 计算每个配置的原子数，用于索引计算
    │   │   │
    │   │   ├─ initialize_gpu_data()
    │   │   │   ├─ 输入: structures
    │   │   │   ├─ 输出: GPU向量（type, r, box, energy_ref等）
    │   │   │   └─ Why: 将数据复制到GPU，为GPU计算做准备
    │   │   │
    │   │   └─ find_neighbor()
    │   │       ├─ 输入: GPU数据（r, box, type等）
    │   │       ├─ 输出: NN_radial, NL_radial, NN_angular, NL_angular
    │   │       └─ Why: 构建邻居列表，避免每次计算都搜索邻居
    │   │
    │   ├─ read_structures(false, para, structures_test) [可选]
    │   │   ├─ 输入: test.xyz文件, Parameters
    │   │   ├─ 输出: structures_test向量
    │   │   └─ Why: 读取测试数据，用于评估模型泛化能力
    │   │
    │   └─ 创建Potential对象
    │       ├─ 根据train_mode和charge_mode选择:
    │       │   ├─ train_mode == 1 or 2: new TNEP()
    │       │   ├─ charge_mode != 0: new NEP_Charge()
    │       │   └─ 否则: new NEP()
    │       │
    │       ├─ 输入: Parameters, N, N_times_max_NN_radial等
    │       ├─ 输出: Potential对象（势函数模型）
    │       └─ Why: 根据训练模式选择对应的势函数实现

┌─────────────────────────────────────────────────────────────────┐
│ 步骤4: 训练/预测阶段 - SNES对象构造和优化循环                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ SNES::SNES(Parameters& para, Fitness* fitness_function)
    │   ├─ 输入: Parameters对象, Fitness指针
    │   ├─ 输出: SNES对象（优化完成或预测完成）
    │   └─ Why: 执行参数优化（训练模式）或预测（预测模式）
    │
    │   ├─ 初始化SNES参数
    │   │   ├─ 从Parameters读取: maximum_generation, number_of_variables, population_size
    │   │   ├─ 计算: eta_sigma = (3 + ln(n)) / (5√n) / 2
    │   │   └─ Why: 设置优化算法参数，eta_sigma控制学习率
    │   │
    │   ├─ 分配内存
    │   │   ├─ fitness, population, mu, sigma等向量
    │   │   └─ Why: 为种群、适应度、分布参数分配内存
    │   │
    │   ├─ initialize_rng()
    │   │   ├─ 输入: 无（或DEBUG模式固定种子）
    │   │   ├─ 输出: 初始化的随机数生成器
    │   │   └─ Why: 用于生成随机种群，可重现性（DEBUG模式）
    │   │
    │   ├─ initialize_curand_states() [GPU kernel]
    │   │   ├─ 输入: 种子
    │   │   ├─ 输出: GPU随机数状态数组
    │   │   └─ Why: 初始化GPU随机数生成器，用于并行生成种群
    │   │
    │   ├─ initialize_mu_and_sigma()
    │   │   ├─ 输入: nep.restart文件（如果存在）或随机初始化
    │   │   ├─ 输出: mu（均值）和sigma（标准差）向量
    │   │   └─ Why: 初始化参数分布，支持从检查点恢复训练
    │   │
    │   ├─ calculate_utility()
    │   │   ├─ 输入: population_size
    │   │   ├─ 输出: utility数组
    │   │   └─ Why: 计算utility函数，用于加权更新分布参数
    │   │
    │   ├─ find_type_of_variable()
    │   │   ├─ 输入: Parameters
    │   │   ├─ 输出: type_of_variable数组
    │   │   └─ Why: 确定每个变量属于哪个原子类型（NEP4类型特定正则化）
    │   │
    │   └─ SNES::compute() - 主优化循环
    │       ├─ 输入: Parameters, Fitness指针
    │       ├─ 输出: 优化后的参数（保存在nep.txt）
    │       └─ Why: 执行SNES优化算法，迭代优化参数
    │
    │       ┌─ 训练模式 (prediction == 0) ───────────────────┐
    │       │                                                 │
    │       │   for (generation = 0; generation < max_gen; ++generation) {
    │       │                                                   │
    │       │       ├─ create_population(para)                  │
    │       │       │   ├─ 输入: mu, sigma (GPU)                │
    │       │       │   ├─ 输出: population (GPU)               │
    │       │       │   └─ Why: 从当前分布采样生成候选解        │
    │       │       │                                             │
    │       │       │   └─ gpu_create_population() [GPU kernel] │
    │       │       │       ├─ 输入: gpu_mu, gpu_sigma, curand_states │
    │       │       │       ├─ 输出: gpu_population, gpu_s     │
    │       │       │       └─ Why: 并行生成所有个体的所有变量  │
    │       │       │                                             │
    │       │       ├─ Fitness::compute(generation, para, population, fitness) │
    │       │       │   ├─ 输入: generation, Parameters, population数组 │
    │       │       │   ├─ 输出: fitness数组（每个个体的适应度） │
    │       │       │   └─ Why: 评估每个候选解的适应度          │
    │       │       │                                             │
    │       │       │   ├─ 选择批次: batch_id = generation % num_batches │
    │       │       │   │   └─ Why: 循环使用不同批次，减少计算量 │
    │       │       │   │                                             │
    │       │       │   ├─ Potential::find_force() [对每个个体] │
    │       │       │   │   ├─ 输入: parameters, Dataset        │
    │       │       │   │   ├─ 输出: energy, force, virial (GPU) │
    │       │       │   │   └─ Why: 计算能量、力、维里           │
    │       │       │   │                                             │
    │       │       │   │   ├─ 计算描述符                        │
    │       │       │   │   │   ├─ 输入: 原子坐标, 邻居列表     │
    │       │       │   │   │   ├─ 输出: descriptors (GPU)       │
    │       │       │   │   │   └─ Why: 将原子环境编码为向量     │
    │       │       │   │   │                                             │
    │       │       │   │   ├─ 神经网络前向传播                  │
    │       │       │   │   │   ├─ 输入: descriptors, ANN权重   │
    │       │       │   │   │   ├─ 输出: 原子能量               │
    │       │       │   │   │   └─ Why: 从描述符预测能量        │
    │       │       │   │   │                                             │
    │       │       │   │   └─ 计算力和维里（自动微分）          │
    │       │       │   │       ├─ 输入: 能量, 描述符梯度        │
    │       │       │   │       ├─ 输出: force, virial          │
    │       │       │   │       └─ Why: 通过链式法则计算力        │
    │       │       │   │                                             │
    │       │       │   ├─ Dataset::get_rmse_energy()            │
    │       │       │   │   ├─ 输入: energy_pred, energy_ref     │
    │       │       │   │   ├─ 输出: RMSE_energy数组（按类型）   │
    │       │       │   │   └─ Why: 计算能量误差                 │
    │       │       │   │                                             │
    │       │       │   ├─ Dataset::get_rmse_force()             │
    │       │       │   │   ├─ 输入: force_pred, force_ref       │
    │       │       │   │   ├─ 输出: RMSE_force数组              │
    │       │       │   │   └─ Why: 计算力误差                   │
    │       │       │   │                                             │
    │       │       │   └─ Dataset::get_rmse_virial()            │
    │       │       │       ├─ 输入: virial_pred, virial_ref     │
    │       │       │       ├─ 输出: RMSE_virial数组             │
    │       │       │       └─ Why: 计算维里误差                  │
    │       │       │                                             │
    │       │       ├─ regularize_NEP4() or regularize()         │
    │       │       │   ├─ 输入: population (GPU)                │
    │       │       │   ├─ 输出: cost_L1reg, cost_L2reg         │
    │       │       │   └─ Why: 计算L1和L2正则化损失，防止过拟合 │
    │       │       │                                             │
    │       │       │   └─ gpu_find_L1_L2_NEP4() [GPU kernel]   │
    │       │       │       ├─ 输入: gpu_population, type_of_variable │
    │       │       │       ├─ 输出: gpu_cost_L1reg, gpu_cost_L2reg │
    │       │       │       └─ Why: 并行计算所有个体的正则化损失 │
    │       │       │                                             │
    │       │       ├─ sort_population(para)                      │
    │       │       │   ├─ 输入: fitness数组                     │
    │       │       │   ├─ 输出: index数组（排序索引）            │
    │       │       │   └─ Why: 按适应度排序，用于更新分布       │
    │       │       │                                             │
    │       │       ├─ Fitness::report_error() [每100代]         │
    │       │       │   ├─ 输入: generation, elite参数           │
    │       │       │   ├─ 输出: 误差报告, nep.txt文件           │
    │       │       │   └─ Why: 监控训练进度，保存检查点          │
    │       │       │                                             │
    │       │       │   ├─ 计算训练/测试集RMSE                   │
    │       │       │   ├─ 校正能量偏移                          │
    │       │       │   ├─ Fitness::write_nep_txt()              │
    │       │       │   │   ├─ 输入: Parameters, elite参数       │
    │       │       │   │   ├─ 输出: nep.txt文件                 │
    │       │       │   │   └─ Why: 保存训练好的势函数参数       │
    │       │       │   │                                             │
    │       │       │   └─ 输出测试集预测结果 [如果有]            │
    │       │       │       └─ Why: 评估模型在测试集上的表现      │
    │       │       │                                             │
    │       │       └─ update_mu_and_sigma(para)                  │
    │       │           ├─ 输入: fitness, index, utility, gpu_s  │
    │       │           ├─ 输出: 更新的mu和sigma (GPU)            │
    │       │           └─ Why: 根据适应度更新参数分布            │
    │       │                                                   │
    │       │           └─ gpu_update_mu_and_sigma() [GPU kernel] │
    │       │               ├─ 输入: gpu_index, gpu_utility, gpu_s │
    │       │               ├─ 输出: gpu_mu, gpu_sigma           │
    │       │               └─ Why: 并行更新所有变量的分布参数    │
    │       │                                                   │
    │       │       └─ output_mu_and_sigma() [每100代]           │
    │       │           ├─ 输入: mu, sigma                       │
    │       │           ├─ 输出: nep.restart文件                 │
    │       │           └─ Why: 保存分布参数，支持训练恢复        │
    │       │                                                   │
    │       └─ } // 结束训练循环                                  │
    │                                                             │
    │       ┌─ 预测模式 (prediction == 1) ───────────────────┐
    │       │                                                 │
    │       │   ├─ 从nep.txt读取参数                          │
    │       │   │   ├─ 输入: nep.txt文件                      │
    │       │   │   ├─ 输出: population数组（参数值）          │
    │       │   │   └─ Why: 加载训练好的模型参数               │
    │       │   │                                             │
    │       │   └─ Fitness::predict(para, population)          │
    │       │       ├─ 输入: Parameters, 参数数组              │
    │       │       ├─ 输出: *_train.out文件                  │
    │       │       └─ Why: 对训练集进行预测并输出结果        │
    │       │                                             │
    │       └─────────────────────────────────────────────┘
    │
    └─ 程序结束
        ├─ 输出: "Finished running nep."
        └─ 返回: EXIT_SUCCESS
```

### 关键数据流

#### 输入文件
- `nep.in`: 配置文件（参数设置）
- `train.xyz`: 训练数据（原子结构、力、能量、维里）
- `test.xyz`: 测试数据（可选）
- `zbl.in`: ZBL参数（可选，如果启用flexible ZBL）
- `nep.restart`: 重启文件（可选，恢复训练）

#### 输出文件
- `nep.txt`: 最终训练好的势函数参数
- `nep.restart`: 参数分布（mu, sigma），用于恢复训练
- `loss.out`: 训练误差记录
- `*_test.out`: 测试集预测结果（energy, force, virial, stress等）
- `*_train.out`: 训练集预测结果（每1000代输出一次）
- `nep_gen*.txt`: 检查点文件（按save_potential设置）

### 为什么使用这些函数？

1. **模块化设计**: 每个类负责特定功能，便于维护和扩展
2. **GPU加速**: 关键计算在GPU上并行执行，大幅提升速度
3. **批处理**: 将大数据集分批，减少内存占用，支持增量训练
4. **多GPU支持**: 自动检测并使用所有可用GPU，提升并行效率
5. **检查点机制**: 支持训练恢复，避免长时间训练中断
6. **误差监控**: 定期报告训练/测试误差，便于调整超参数
7. **类型特定正则化**: NEP4为每种元素类型单独正则化，提高泛化能力

## 程序整体架构

### 执行流程概览

```
main() 
  ├─ 打印欢迎信息和GPU信息
  ├─ 阶段1: 初始化
  │   ├─ Parameters::Parameters() - 读取nep.in配置
  │   └─ Fitness::Fitness() - 读取训练/测试数据
  └─ 阶段2: 训练/预测
      └─ SNES::SNES() - 执行优化或预测
```

### 核心类依赖关系

```
main.cu
  ├─ Parameters (parameters.cuh/cu)
  │   └─ 读取nep.in, 计算派生参数
  ├─ Fitness (fitness.cuh/cu)
  │   ├─ Dataset (dataset.cuh/cu)
  │   │   └─ Structure (structure.cuh/cu)
  │   └─ Potential (potential.cuh)
  │       ├─ NEP (nep.cuh/cu)
  │       ├─ TNEP (tnep.cuh/cu)
  │       └─ NEP_Charge (nep_charge.cuh/cu)
  └─ SNES (snes.cuh/cu)
      └─ 使用Fitness计算适应度
```

## 主要文件位置

### 主程序
- **入口点**: `src/main_nep/main.cu` (第30-68行)

### 核心类
- **Parameters**: `src/main_nep/parameters.cuh` / `parameters.cu`
- **Fitness**: `src/main_nep/fitness.cuh` / `fitness.cu`
- **SNES**: `src/main_nep/snes.cuh` / `snes.cu`

### 数据与模型
- **Dataset**: `src/main_nep/dataset.cuh` / `dataset.cu`
- **Structure**: `src/main_nep/structure.cuh` / `structure.cu`
- **NEP**: `src/main_nep/nep.cuh` / `nep.cu`
- **TNEP**: `src/main_nep/tnep.cuh` / `tnep.cu`
- **NEP_Charge**: `src/main_nep/nep_charge.cuh` / `nep_charge.cu`

### 工具函数
- **GPU信息**: `src/utilities/main_common.cuh` / `main_common.cu`
- **错误处理**: `src/utilities/error.cuh` / `error.cu`

## 程序执行的两个主要模式

### 1. 训练模式 (prediction = 0)

1. **初始化阶段**
   - 读取`nep.in`配置文件
   - 读取`train.xyz`训练数据
   - 可选读取`test.xyz`测试数据
   - 初始化GPU数据结构

2. **训练阶段**
   - SNES算法迭代优化参数
   - 每代生成种群
   - 计算每个个体的适应度
   - 更新参数分布(mu, sigma)
   - 定期输出检查点和误差报告

3. **输出文件**
   - `nep.txt` - 最终训练好的势函数
   - `nep.restart` - 重启文件(mu和sigma)
   - `loss.out` - 训练误差记录
   - `*_test.out` - 测试集预测结果

### 2. 预测模式 (prediction = 1)

1. **加载模型**
   - 从`nep.txt`读取训练好的参数

2. **预测计算**
   - 对训练集进行预测
   - 输出预测结果到`*_train.out`文件

## 关键概念

### NEP版本
- **NEP3**: 早期版本，所有元素共享同一神经网络
- **NEP4**: 改进版本，每种元素有独立的神经网络参数

### 训练模式
- **mode 0**: 势函数训练（能量、力、维里）
- **mode 1**: 偶极矩训练
- **mode 2**: 极化率训练
- **mode 3**: 温度依赖自由能训练

### 优化算法
- **SNES**: Separable Natural Evolution Strategy
- 使用高斯分布采样生成候选解
- 基于适应度排序更新分布参数

## 数学公式快速索引

详见 [数学公式集合](07_mathematical_formulas.md)，包括：
- SNES更新公式
- NEP描述符计算
- 损失函数定义
- RMSE计算公式

## 使用建议

1. **理解整体流程**: 先阅读 [主程序执行流程](01_main_execution_flow.md)
2. **配置参数**: 参考 [Parameters类详解](02_parameters_class.md)
3. **理解优化**: 阅读 [SNES算法](04_snes_algorithm.md)
4. **深入模型**: 查看 [Potential模型](06_potential_models.md)
5. **数学细节**: 查阅 [数学公式集合](07_mathematical_formulas.md)

## 版本信息

- **GPUMD版本**: 4.5
- **文档创建日期**: 2024
- **适用NEP版本**: NEP3, NEP4

