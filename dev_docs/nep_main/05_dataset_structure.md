# Dataset数据结构

本文档详细记录`Dataset`类的数据结构、数据组织方式和关键方法。

## 文件位置

- **头文件**: `src/main_nep/dataset.cuh` (第22-71行)
- **实现文件**: `src/main_nep/dataset.cu` (第27-1025行)

## 类结构概览

`Dataset`类负责：
1. 存储多个原子配置的结构数据
2. 管理GPU内存中的数据
3. 计算RMSE误差
4. 构建邻居列表

## 核心成员变量

### 配置信息
- `Nc`: 配置数量
- `N`: 总原子数（所有配置的原子数之和）
- `max_Na`: 单个配置中的最大原子数
- `max_NN_radial`: 最大径向邻居数
- `max_NN_angular`: 最大角向邻居数

### 原子数据（GPU）
- `Na`: `GPU_Vector<int>` - 每个配置的原子数
- `Na_sum`: `GPU_Vector<int>` - 原子数的前缀和
- `type`: `GPU_Vector<int>` - 原子类型
- `r`: `GPU_Vector<float>` - 原子坐标 (N×3)

### 盒子数据（GPU）
- `box`: `GPU_Vector<float>` - 扩展盒子矩阵 (Nc×18)
  - 前9个: 扩展后的盒子向量
  - 后9个: 逆盒子矩阵
- `box_original`: `GPU_Vector<float>` - 原始盒子 (Nc×9)
- `num_cell`: `GPU_Vector<int>` - 扩展盒子的单元数 (Nc×3)

### 计算结果（GPU）
- `energy`: `GPU_Vector<float>` - 计算的能量 (N)
- `force`: `GPU_Vector<float>` - 计算的力 (N×3)
- `virial`: `GPU_Vector<float>` - 计算的维里 (N×6)
- `charge`: `GPU_Vector<float>` - 计算的电荷 (N，如果启用)
- `bec`: `GPU_Vector<float>` - 计算的BEC (N×9，如果启用)

### 参考数据（GPU）
- `energy_ref_gpu`: `GPU_Vector<float>` - 参考能量 (Nc)
- `force_ref_gpu`: `GPU_Vector<float>` - 参考力 (N×3)
- `virial_ref_gpu`: `GPU_Vector<float>` - 参考维里 (Nc×6)
- `charge_ref_gpu`: `GPU_Vector<float>` - 参考电荷 (Nc，如果启用)
- `bec_ref_gpu`: `GPU_Vector<float>` - 参考BEC (N×9，如果启用)

### CPU副本
- `*_cpu`: 对应的CPU数据，用于数据传输和误差计算

### 权重
- `weight_cpu`: `std::vector<float>` - 配置权重
- `energy_weight_cpu`: `std::vector<float>` - 能量权重
- `type_weight_gpu`: `GPU_Vector<float>` - 类型权重

### 结构数据
- `structures`: `std::vector<Structure>` - 原始结构数据

## 关键方法

### Dataset::construct()

**位置**: `src/main_nep/dataset.cu` (主要逻辑)

**功能**: 构建Dataset对象

**执行流程**:
1. `copy_structures()` - 复制结构数据
2. `find_has_type()` - 确定每个配置包含哪些原子类型
3. `find_Na()` - 计算原子数和前缀和
4. `initialize_gpu_data()` - 初始化GPU数据
5. `find_neighbor()` - 构建邻居列表

### find_Na()

**位置**: `src/main_nep/dataset.cu` 第121-156行

**功能**: 计算每个配置的原子数和前缀和

```cpp
for (int nc = 0; nc < Nc; ++nc) {
  Na_cpu[nc] = structures[nc].num_atom;
  N += structures[nc].num_atom;
  if (structures[nc].num_atom > max_Na) {
    max_Na = structures[nc].num_atom;
  }
}

// 计算前缀和
for (int nc = 1; nc < Nc; ++nc) {
  Na_sum_cpu[nc] = Na_sum_cpu[nc - 1] + Na_cpu[nc - 1];
}
```

### initialize_gpu_data()

**位置**: `src/main_nep/dataset.cu` 第158-274行

**功能**: 将CPU数据复制到GPU并分配GPU内存

**数据组织**:
- 所有配置的原子数据连续存储
- 使用`Na_sum`索引访问特定配置的原子

### find_neighbor()

**位置**: `src/main_nep/dataset.cu` 第347-1025行

**功能**: 构建径向和角向邻居列表

**GPU Kernel**: `gpu_find_neighbor_number` 和 `gpu_find_neighbor_list`

**邻居列表结构**:
- `NN_radial[i]`: 原子i的径向邻居数
- `NL_radial[NN_sum[i] ... NN_sum[i+1]-1]`: 原子i的径向邻居索引
- 类似地处理角向邻居

**周期性边界条件**: 使用扩展盒子处理

### get_rmse_energy()

**功能**: 计算能量RMSE

**计算方式**:
1. 对每个配置计算能量误差
2. 考虑能量权重和配置权重
3. 按原子类型分组

**公式**:
```
RMSE_energy = sqrt(Σ((E_pred - E_ref)^2 * weight_energy * weight_config) / Σ(weight_energy * weight_config))
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
RMSE_virial = sqrt(Σ((V_pred - V_ref)^2 * weight * shear_weight) / Σ(weight * shear_weight))
```

## 数据组织方式

### 内存布局

所有配置的原子数据连续存储：

```
原子索引: [0 ... Na[0]-1] [Na[0] ... Na[0]+Na[1]-1] ... [N-Na[Nc-1] ... N-1]
配置索引:    0                 1                        ...      Nc-1
```

### 访问模式

访问配置nc的原子na:
```cpp
int global_index = Na_sum_cpu[nc] + na;
float x = r_cpu[global_index];
float y = r_cpu[global_index + N];
float z = r_cpu[global_index + N * 2];
```

### 邻居列表组织

```
NN_radial[i]: 原子i的径向邻居数
NL_radial[NN_sum[i] + j]: 原子i的第j个径向邻居的全局索引
```

## GPU内存管理

### 数据传输

- **CPU → GPU**: 使用`copy_from_host()`
- **GPU → CPU**: 使用`copy_to_host()`

### 内存分配

所有GPU向量在`initialize_gpu_data()`中分配，大小根据实际数据确定。

## 相关文档

- [Fitness类详解](03_fitness_class.md) - Dataset的使用
- [Structure数据结构](structure.cuh) - 单个结构的数据格式

