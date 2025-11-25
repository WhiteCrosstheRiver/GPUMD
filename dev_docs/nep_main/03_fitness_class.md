# Fitness类详解

本文档详细记录`Fitness`类的结构、数据加载、适应度计算和误差报告功能。

## 文件位置

- **头文件**: `src/main_nep/fitness.cuh` (第26-71行)
- **实现文件**: `src/main_nep/fitness.cu` (第38-689行)

## 类结构概览

`Fitness`类负责：
1. 读取和构建训练/测试数据集
2. 计算每个个体的适应度（fitness）
3. 报告训练误差
4. 执行预测并输出结果

## 成员变量

### 核心数据
- `train_set`: `std::vector<std::vector<Dataset>>` - 训练数据集，按批次和GPU组织
- `test_set`: `std::vector<Dataset>` - 测试数据集，按GPU组织
- `potential`: `std::unique_ptr<Potential>` - 势函数模型指针

### 统计信息
- `num_batches`: 批次数
- `max_NN_radial`: 最大径向邻居数
- `max_NN_angular`: 最大角向邻居数
- `has_test_set`: 是否有测试集

### 输出文件
- `fid_loss_out`: `FILE*` - 损失输出文件指针

## 构造函数

### Fitness::Fitness(Parameters& para)

**位置**: `src/main_nep/fitness.cu` 第38-142行

**执行流程**:

#### 1. 检测GPU设备

```cpp
int deviceCount;
CHECK(gpuGetDeviceCount(&deviceCount));
```

#### 2. 读取训练数据

```cpp
std::vector<Structure> structures_train;
read_structures(true, para, structures_train);
```

**功能**: 从`train.xyz`文件读取所有训练结构

**读取内容**:
- 原子类型、坐标
- 参考力（fx, fy, fz）
- 参考能量
- 参考维里（virial）
- 可选：原子维里、BEC、电荷、温度

#### 3. 计算批次大小

```cpp
num_batches = (structures_train.size() - 1) / para.batch_size + 1;
```

**调整逻辑**: 如果批次大小不能整除，会自动调整：
```cpp
para.batch_size = (structures_train.size() - 1) / num_batches + 1;
```

#### 4. 构建训练数据集

```cpp
train_set.resize(num_batches);
for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
  train_set[batch_id].resize(deviceCount);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    train_set[batch_id][device_id].construct(
      para, structures_train, count - batch_size, count, device_id);
  }
}
```

**组织方式**:
- 第一维：批次ID
- 第二维：GPU设备ID
- 每个Dataset包含该批次在该GPU上的数据

#### 5. 读取测试数据（可选）

```cpp
std::vector<Structure> structures_test;
has_test_set = read_structures(false, para, structures_test);
if (has_test_set) {
  test_set.resize(deviceCount);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
  }
}
```

#### 6. 确定最大尺寸

```cpp
int N = -1;  // 最大原子数
int Nc = -1;  // 最大配置数（用于charge_mode）
int N_times_max_NN_radial = -1;
int N_times_max_NN_angular = -1;
max_NN_radial = -1;
max_NN_angular = -1;
```

遍历所有批次，找到最大值，用于分配GPU内存。

#### 7. 创建势函数模型

```cpp
if (para.train_mode == 1 || para.train_mode == 2) {
  potential.reset(new TNEP(...));  // 偶极/极化率
} else {
  if (para.charge_mode) {
    potential.reset(new NEP_Charge(...));  // 电荷模式
  } else {
    potential.reset(new NEP(...));  // 标准NEP
  }
}
```

#### 8. 打开损失输出文件

```cpp
if (para.prediction == 0) {
  fid_loss_out = my_fopen("loss.out", "a");
}
```

## 核心方法

### Fitness::compute()

**位置**: `src/main_nep/fitness.cu` 第151-252行

**函数签名**:
```cpp
void compute(const int generation, Parameters& para, const float* population, float* fitness);
```

**功能**: 计算种群中每个个体的适应度

**执行流程**:

#### 1. 第0代初始化

```cpp
if (generation == 0) {
  std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
  for (int n = 0; n < num_batches; ++n) {
    potential->find_force(para, dummy_solution.data(), train_set[n], 
                         (para.fine_tune ? false : true), true, deviceCount);
  }
}
```

**目的**: 初始化GPU内存，计算邻居列表

#### 2. 选择批次

```cpp
int batch_id = generation % num_batches;
bool calculate_neighbor = (num_batches > 1) || (generation % 100 == 0);
```

**批次选择**: 循环使用不同批次
**邻居计算**: 多批次或每100代重新计算邻居列表

#### 3. 计算每个个体的适应度

```cpp
for (int n = 0; n < population_iter; ++n) {
  const float* individual = population + deviceCount * n * para.number_of_variables;
  potential->find_force(para, individual, train_set[batch_id], 
                        false, calculate_neighbor, deviceCount);
  
  for (int m = 0; m < deviceCount; ++m) {
    // 计算RMSE
    auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(...);
    auto rmse_force_array = train_set[batch_id][m].get_rmse_force(...);
    auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(...);
    auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(...);
    
    // 存储到fitness数组
    for (int t = 0; t <= para.num_types; ++t) {
      fitness[...] = para.lambda_e * rmse_energy_array[t];
      fitness[...] = para.lambda_f * rmse_force_array[t];
      fitness[...] = para.lambda_v * rmse_virial_array[t];
      if (para.charge_mode) {
        fitness[...] = para.lambda_q * rmse_charge_array[t];
      }
    }
  }
}
```

**fitness数组布局**:
- 每个个体有 `7 * (num_types + 1)` 个fitness值
- 对于每个类型t (0到num_types):
  - `fitness[7*t + 0]`: 总损失（稍后计算）
  - `fitness[7*t + 1]`: L1正则化损失
  - `fitness[7*t + 2]`: L2正则化损失
  - `fitness[7*t + 3]`: 能量损失
  - `fitness[7*t + 4]`: 力损失
  - `fitness[7*t + 5]`: 维里损失
  - `fitness[7*t + 6]`: 电荷损失（如果启用）

#### 4. 有效全批次模式

```cpp
if (para.use_full_batch) {
  // 计算其他批次的RMSE
  // 使用RMS方式合并多批次结果
  new_value = sqrt((old_value^2 * count_batch + new_value^2) / (count_batch + 1));
}
```

**目的**: 即使使用批次训练，也能获得接近全批次的梯度估计

### Fitness::report_error()

**位置**: `src/main_nep/fitness.cu` 第431-570行

**函数签名**:
```cpp
void report_error(Parameters& para, const int generation, 
                  const float loss_total, const float loss_L1, const float loss_L2, 
                  float* elite);
```

**功能**: 每100代报告训练误差并保存模型

**执行流程**:

#### 1. 计算训练集误差

```cpp
if (0 == (generation + 1) % 100) {
  int batch_id = generation % num_batches;
  potential->find_force(para, elite, train_set[batch_id], false, true, 1);
  
  float energy_shift_per_structure;
  auto rmse_energy_train_array = train_set[batch_id][0].get_rmse_energy(...);
  auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(...);
  auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(...);
  
  float rmse_energy_train = rmse_energy_train_array.back();  // 全局RMSE
  float rmse_force_train = rmse_force_train_array.back();
  float rmse_virial_train = rmse_virial_train_array.back();
}
```

#### 2. 能量偏移校正

```cpp
if (para.train_mode == 0 || para.train_mode == 3) {
  elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
}
```

**目的**: 校正神经网络偏置，使预测能量与参考能量对齐

#### 3. 计算测试集误差（如果有）

```cpp
if (has_test_set) {
  potential->find_force(para, elite, test_set, false, true, 1);
  // 计算测试集RMSE
}
```

#### 4. 保存模型文件

```cpp
FILE* fid_nep = my_fopen("nep.txt", "w");
write_nep_txt(fid_nep, para, elite);
fclose(fid_nep);
```

#### 5. 定期保存检查点

```cpp
if (0 == (generation + 1) % para.save_potential) {
  std::string filename;
  get_save_potential_label(para, generation, filename);
  filename += ".txt";
  FILE* fid_nep = my_fopen(filename.c_str(), "w");
  write_nep_txt(fid_nep, para, elite);
  fclose(fid_nep);
}
```

#### 6. 输出误差到控制台和文件

```cpp
printf("%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
       generation + 1, loss_total, loss_L1, loss_L2,
       rmse_energy_train, rmse_force_train, rmse_virial_train,
       rmse_energy_test, rmse_force_test, rmse_virial_test);
fprintf(fid_loss_out, ...);
```

#### 7. 输出测试集预测结果

```cpp
if (has_test_set) {
  if (para.train_mode == 0 || para.train_mode == 3) {
    FILE* fid_force = my_fopen("force_test.out", "w");
    FILE* fid_energy = my_fopen("energy_test.out", "w");
    FILE* fid_virial = my_fopen("virial_test.out", "w");
    FILE* fid_stress = my_fopen("stress_test.out", "w");
    update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
    // ...
  }
}
```

#### 8. 每1000代输出训练集预测

```cpp
if (0 == (generation + 1) % 1000) {
  predict(para, elite);
}
```

### Fitness::predict()

**位置**: `src/main_nep/fitness.cu` 第637-688行

**功能**: 对训练集进行预测并输出结果

**输出文件**:
- `energy_train.out`: 能量预测
- `force_train.out`: 力预测
- `virial_train.out`: 维里预测
- `stress_train.out`: 应力预测
- `charge_train.out`: 电荷预测（如果启用）
- `bec_train.out`: BEC预测（如果启用）

### Fitness::write_nep_txt()

**位置**: `src/main_nep/fitness.cu` 第315-416行

**功能**: 将训练好的参数写入`nep.txt`文件

**文件格式**:
1. 第一行: 模型类型和元素数量
2. 第二行: ZBL设置（如果启用）
3. 第三行: 截断半径和邻居列表大小
4. 第四行: n_max参数
5. 第五行: basis_size参数
6. 第六行: l_max参数
7. 第七行: ANN结构
8. 后续行: 所有参数值（每行一个）
9. q_scaler值
10. ZBL参数（如果flexible_zbl）

## RMSE计算

### get_rmse_energy()

**功能**: 计算能量RMSE

**计算方式**:
1. 对每个配置计算能量误差
2. 考虑能量权重和配置权重
3. 按原子类型分组计算
4. 返回每个类型的RMSE和全局RMSE

**公式**:
```
RMSE_energy = sqrt(Σ((E_pred - E_ref)^2 * weight) / Σ(weight))
```

### get_rmse_force()

**功能**: 计算力RMSE

**计算方式**:
1. 对每个原子计算力误差
2. 考虑类型权重
3. 按原子类型分组

**公式**:
```
RMSE_force = sqrt(Σ(|F_pred - F_ref|^2 * type_weight) / Σ(type_weight))
```

### get_rmse_virial()

**功能**: 计算维里RMSE

**计算方式**:
1. 对每个配置计算维里误差
2. 考虑剪切维里权重
3. 按类型分组

**公式**:
```
RMSE_virial = sqrt(Σ((V_pred - V_ref)^2 * weight) / Σ(weight))
```

## 输出文件格式

### loss.out

每行格式（训练模式）:
```
generation  total_loss  L1_loss  L2_loss  RMSE_E_train  RMSE_F_train  RMSE_V_train  RMSE_E_test  RMSE_F_test  RMSE_V_test
```

### force_test.out / force_train.out

每行格式:
```
fx_pred  fy_pred  fz_pred  fx_ref  fy_ref  fz_ref
```

### energy_test.out / energy_train.out

每行格式:
```
E_pred_per_atom  E_ref_per_atom
```

### virial_test.out / virial_train.out

每行格式:
```
Vxx_pred  Vyy_pred  Vzz_pred  Vxy_pred  Vyz_pred  Vzx_pred  Vxx_ref  Vyy_ref  Vzz_ref  Vxy_ref  Vyz_ref  Vzx_ref
```

## 相关文档

- [Dataset数据结构](05_dataset_structure.md) - Dataset类的详细说明
- [Potential模型](06_potential_models.md) - 势函数模型实现
- [SNES算法](04_snes_algorithm.md) - 优化算法如何调用Fitness

