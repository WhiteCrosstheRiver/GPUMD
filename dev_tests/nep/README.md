# NEP测试套件

本目录包含NEP主程序（`src/main_nep`）的完整测试文件，用于验证各个模块的功能正确性。

## 目录结构

```
dev_tests/nep/
├── README.md                    # 本文件
├── test_parameters.cpp         # Parameters类测试
├── test_structure.cpp          # Structure读取测试
├── test_dataset.cpp            # Dataset类测试
├── test_fitness.cpp            # Fitness类测试
├── test_snes.cpp               # SNES优化算法测试
└── test_nep.cpp                # NEP模型测试
```

## 测试文件说明

### 1. test_parameters.cpp

**测试对象**：`src/main_nep/parameters.cu`

**测试内容**：
- Parameters构造函数和默认参数
- 所有parse_*函数（参数解析）
- calculate_parameters函数（派生参数计算）
- check_foundation_model函数（基础模型验证）
- parse_one_keyword函数（关键字路由）

**主要函数测试**：
- `test_parameters_constructor()` - 构造函数测试
- `test_parse_version()` - 版本号解析
- `test_parse_type()` - 元素类型解析
- `test_parse_cutoff()` - 截断半径解析
- `test_parse_n_max()` - n_max参数解析
- `test_parse_basis_size()` - basis_size参数解析
- `test_parse_l_max()` - l_max参数解析
- `test_parse_neuron()` - 神经元数量解析
- `test_parse_lambda()` - lambda系列参数解析
- `test_parse_batch()` - 批次大小解析
- `test_parse_population()` - 种群大小解析
- `test_parse_generation()` - 最大代数解析
- `test_parse_fine_tune()` - fine_tune参数解析
- `test_calculate_parameters()` - 参数计算
- `test_parse_one_keyword()` - 关键字路由

**输入输出维度**：
- 所有parse函数：输入`const char** param, int num_param`，无返回值
- calculate_parameters：无输入，计算派生参数（dim, number_of_variables等）

---

### 2. test_structure.cpp

**测试对象**：`src/main_nep/structure.cu`

**测试内容**：
- Structure数据结构初始化
- read_structures函数（从xyz文件读取）
- change_box函数（盒子扩展）
- 多结构读取
- 带维里张量的结构读取
- 错误处理

**主要函数测试**：
- `test_structure_initialization()` - 结构初始化
- `test_read_structures_basic()` - 基本读取
- `test_read_structures_multiple()` - 多结构读取
- `test_read_structures_with_virial()` - 带维里读取
- `test_change_box()` - 盒子扩展
- `test_read_structures_file_not_found()` - 文件不存在处理
- `test_read_structures_type_validation()` - 类型验证

**输入输出维度**：
- read_structures输入：
  - `is_train: bool` - 是否为训练集
  - `para: Parameters&` - 参数对象
  - `structures: std::vector<Structure>&` - 输出结构数组
- 输出：`bool` - 是否成功
- Structure维度：
  - `type: std::vector<int> [num_atom]`
  - `x, y, z: std::vector<float> [num_atom]`
  - `fx, fy, fz: std::vector<float> [num_atom]`
  - `energy: float`
  - `virial: float[6]`
  - `box_original: float[9]`
  - `box: float[18]`

---

### 3. test_dataset.cpp

**测试对象**：`src/main_nep/dataset.cu`

**测试内容**：
- Dataset::construct完整流程
- Dataset::copy_structures（结构复制）
- Dataset::find_Na（原子数计算）
- Dataset::initialize_gpu_data（GPU数据传输）
- Dataset::find_neighbor（邻居列表构建）
- Dataset::get_rmse_*系列函数（RMSE计算）

**主要函数测试**：
- `test_dataset_copy_structures()` - 结构复制
- `test_dataset_find_Na()` - 原子数计算
- `test_dataset_initialize_gpu_data()` - GPU数据初始化
- `test_dataset_find_neighbor()` - 邻居列表
- `test_dataset_get_rmse_force()` - 力RMSE计算
- `test_dataset_get_rmse_energy()` - 能量RMSE计算
- `test_dataset_get_rmse_virial()` - 维里RMSE计算
- `test_dataset_construct()` - 完整构造流程

**输入输出维度**：
- construct输入：
  - `para: Parameters&`
  - `structures_input: std::vector<Structure>&`
  - `n1, n2: int` - 范围[n1, n2)
  - `device_id: int`
- Dataset成员维度：
  - `Nc: int` - 配置数量
  - `N: int` - 总原子数
  - `Na: GPU_Vector<int> [Nc]` - 每个配置的原子数
  - `Na_sum: GPU_Vector<int> [Nc]` - 原子数前缀和
  - `type: GPU_Vector<int> [N]` - 原子类型
  - `r: GPU_Vector<float> [N * 3]` - 原子坐标
  - `force_ref: GPU_Vector<float> [N * 3]` - 参考力
  - `energy_ref: GPU_Vector<float> [Nc]` - 参考能量
  - `virial_ref: GPU_Vector<float> [Nc * 6]` - 参考维里

---

### 4. test_fitness.cpp

**测试对象**：`src/main_nep/fitness.cu`

**测试内容**：
- Fitness构造函数（数据集加载）
- Fitness::compute（适应度计算）
- Fitness::report_error（误差报告）
- Fitness::predict（预测功能）
- Fitness::write_nep_txt（文件写入）
- 适应度数组维度验证

**主要函数测试**：
- `test_fitness_constructor()` - 构造函数
- `test_fitness_compute()` - 适应度计算
- `test_fitness_report_error()` - 误差报告
- `test_fitness_predict()` - 预测功能
- `test_fitness_write_nep_txt()` - 文件写入
- `test_fitness_dimensions()` - 维度验证

**输入输出维度**：
- compute输入：
  - `generation: int`
  - `para: Parameters&`
  - `population: const float* [population_size * number_of_variables]`
  - `fitness: float* [population_size * 7 * (num_types + 1)]`
- fitness数组组织：
  - `fitness[p + (metric * (num_types + 1) + type) * population_size]`
  - metric: 0=total, 1=L1, 2=L2, 3=energy, 4=force, 5=virial, 6=charge

---

### 5. test_snes.cpp

**测试对象**：`src/main_nep/snes.cu`

**测试内容**：
- SNES构造函数
- SNES::create_population（种群生成）
- SNES::regularize_NEP4（正则化计算）
- SNES::sort_population（种群排序）
- SNES::update_mu_and_sigma（分布更新）
- SNES::initialize_mu_and_sigma_fine_tune（Fine_tune初始化）
- SNES::calculate_utility（Utility计算）
- SNES::find_type_of_variable（变量分类）
- 数组维度验证

**主要函数测试**：
- `test_snes_constructor()` - 构造函数
- `test_snes_create_population()` - 种群生成
- `test_snes_regularize_NEP4()` - 正则化计算
- `test_snes_sort_population()` - 种群排序
- `test_snes_update_mu_and_sigma()` - 分布更新
- `test_snes_initialize_fine_tune()` - Fine_tune初始化
- `test_snes_calculate_utility()` - Utility计算
- `test_snes_find_type_of_variable()` - 变量分类
- `test_snes_dimensions()` - 维度验证

**输入输出维度**：
- create_population输出：
  - `population: std::vector<float> [population_size * number_of_variables]`
- update_mu_and_sigma输出：
  - `mu: std::vector<float> [number_of_variables]`
  - `sigma: std::vector<float> [number_of_variables]`
- utility: `std::vector<float> [population_size]`
- index: `std::vector<int> [population_size * (num_types + 1)]`

---

### 6. test_nep.cpp

**测试对象**：`src/main_nep/nep.cu`

**测试内容**：
- NEP构造函数
- NEP::update_potential（参数更新）
- 径向描述符计算
- 角向描述符计算
- 神经网络前向传播
- NEP::find_force（力和能量计算）
- q_scaler应用
- 力和维里维度验证

**主要函数测试**：
- `test_nep_constructor()` - 构造函数
- `test_nep_update_potential()` - 参数更新
- `test_nep_radial_descriptors()` - 径向描述符
- `test_nep_angular_descriptors()` - 角向描述符
- `test_nep_neural_network()` - 神经网络
- `test_nep_find_force()` - 力和能量计算
- `test_nep_q_scaler()` - q_scaler应用
- `test_nep_force_virial_dimensions()` - 维度验证

**输入输出维度**：
- find_force输出：
  - `energy: GPU_Vector<float> [N]` - 原子能量
  - `force: GPU_Vector<float> [N * 3]` - 原子力
  - `virial: GPU_Vector<float> [N * 6]` - 原子维里
- descriptors: `GPU_Vector<float> [dim * N]`
  - `dim = dim_radial + dim_angular`
  - `dim_radial = n_max_radial + 1`
  - `dim_angular = (n_max_angular + 1) * L_max + ...`

---

## 编译和运行

### 编译单个测试

```bash
# 编译test_parameters
nvcc -o test_parameters test_parameters.cpp \
  ../src/main_nep/parameters.cu \
  -I../src -I../src/utilities \
  -std=c++11

# 编译test_structure
nvcc -o test_structure test_structure.cpp \
  ../src/main_nep/structure.cu \
  ../src/main_nep/parameters.cu \
  -I../src -I../src/utilities \
  -std=c++11

# 类似地编译其他测试文件
```

### 运行测试

```bash
# 运行单个测试
./test_parameters
./test_structure
./test_dataset
./test_fitness
./test_snes
./test_nep

# 运行所有测试
for test in test_*; do
  echo "Running $test..."
  ./$test
done
```

---

## 测试覆盖范围

### 已覆盖的功能

| 模块 | 测试文件 | 覆盖率 |
|------|---------|--------|
| Parameters类 | test_parameters.cpp | 所有parse函数、参数计算 |
| Structure读取 | test_structure.cpp | 基本读取、多结构、维里 |
| Dataset类 | test_dataset.cpp | 构造、GPU传输、RMSE计算 |
| Fitness类 | test_fitness.cpp | 适应度计算、误差报告 |
| SNES算法 | test_snes.cpp | 种群生成、排序、更新 |
| NEP模型 | test_nep.cpp | 描述符、神经网络、力计算 |

### 测试类型

1. **单元测试**：测试单个函数的功能
2. **集成测试**：测试多个函数的协作
3. **维度验证**：验证数组维度和索引计算
4. **边界测试**：测试边界条件和错误处理
5. **数值验证**：验证计算结果正确性

---

## 注意事项

### 1. GPU环境要求

大部分测试需要GPU环境，特别是：
- Dataset类的GPU数据传输
- NEP模型的GPU计算
- SNES算法的GPU并行

### 2. 文件依赖

某些测试需要创建临时文件：
- `test_structure.cpp` 需要创建测试xyz文件
- `test_fitness.cpp` 需要nep.in和train.xyz文件

### 3. 私有函数测试

某些函数是私有的，需要通过公共接口间接测试：
- `Dataset::copy_structures` - 通过`construct`测试
- `Dataset::find_Na` - 通过`construct`测试
- `Fitness::write_nep_txt` - 通过`report_error`测试

### 4. 完整环境测试

某些测试需要完整的运行环境：
- Fitness构造函数需要完整的nep.in配置
- SNES构造函数需要完整的训练数据集

---

## 扩展测试

### 添加新测试

1. 在相应测试文件中添加新的测试函数
2. 遵循命名规范：`test_<模块>_<功能>()`
3. 在主函数中调用新测试
4. 添加详细的注释说明输入输出维度

### 测试最佳实践

1. **独立性**：每个测试应该独立，不依赖其他测试
2. **可重复性**：测试结果应该可重复
3. **清晰性**：测试代码应该清晰易懂
4. **完整性**：测试应该覆盖正常情况和边界情况
5. **文档化**：每个测试都应该有详细的注释

---

## 相关文档

- [NEP训练流程详解](../../dev_docs/nep_main/training.md) - 完整的训练流程
- [Fine_tune功能详解](../../dev_docs/nep_main/fine_tune.md) - Fine_tune机制
- [SNES算法详解](../../dev_docs/nep_main/04_snes_algorithm.md) - SNES算法原理

---

## 版本信息

- **创建日期**：2024
- **适用GPUMD版本**：4.5+
- **测试框架**：自定义测试框架（基于assert）

