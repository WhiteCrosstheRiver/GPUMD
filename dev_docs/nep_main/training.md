# NEP训练流程详解：从数据到神经网络

本文档详细说明NEP训练过程中，数据集（包含原子坐标、力、能量信息）如何一步步处理形成输入到神经网络的变量，以及SNES训练算法如何训练权重。

## 目录

1. [数据预处理流程](#数据预处理流程)
2. [数据维度变化与信息详解](#数据维度变化与信息详解)
3. [描述符计算（输入层预处理）](#描述符计算输入层预处理)
4. [神经网络结构](#神经网络结构)
5. [前向传播：从描述符到能量](#前向传播从描述符到能量)
6. [反向传播：力和维里的计算](#反向传播力和维里的计算)
7. [SNES训练算法](#snes训练算法)
8. [完整训练流程示意](#完整训练流程示意)

---

## 数据预处理流程

### 1. 原始数据读取

**输入文件**：`train.xyz`（训练集）和`test.xyz`（测试集，可选）

**数据格式**：每个结构包含
- 原子数量
- 盒子信息（周期性边界条件）
- 每个原子的：类型、坐标(x,y,z)、力(fx,fy,fz)
- 结构级：总能量、维里张量（可选）

**代码位置**：`src/main_nep/structure.cu` - `read_one_structure()`

### 2. 数据结构组织

**位置**：`src/main_nep/dataset.cu` - `copy_structures()`

数据被组织为：
```
Dataset
├── Nc: 配置数量
├── N: 总原子数
├── structures[Nc]: 结构数组
│   ├── num_atom: 原子数
│   ├── type[Na]: 原子类型
│   ├── x[Na], y[Na], z[Na]: 原子坐标
│   ├── fx[Na], fy[Na], fz[Na]: 参考力
│   ├── energy: 参考总能量
│   ├── virial[6]: 参考维里张量
│   └── box[18]: 盒子信息（扩展盒子用于周期性边界）
```

### 3. GPU数据传输

**位置**：`src/main_nep/dataset.cu` - `initialize_gpu_data()`

将CPU数据复制到GPU：
- `g_type[N]`: 原子类型
- `g_x[N]`, `g_y[N]`, `g_z[N]`: 原子坐标
- `g_fx_ref[N]`, `g_fy_ref[N]`, `g_fz_ref[N]`: 参考力
- `g_energy_ref[Nc]`: 参考能量
- `g_virial_ref[Nc*6]`: 参考维里
- `g_box[Nc*18]`: 盒子信息

### 4. 邻居列表构建

**位置**：`src/main_nep/nep.cu` - `gpu_find_neighbor_list()`

对每个原子，找到截断半径内的邻居：

```
对每个原子 n1:
  对每个可能的邻居 n2（包括周期性镜像）:
    计算距离 d12
    if (d12 < rc_radial):
      添加到径向邻居列表 NL_radial
    if (d12 < rc_angular):
      添加到角向邻居列表 NL_angular
```

**输出**：
- `NN_radial[N]`: 每个原子的径向邻居数量
- `NL_radial[max_NN_radial * N]`: 径向邻居列表
- `NN_angular[N]`: 每个原子的角向邻居数量
- `NL_angular[max_NN_angular * N]`: 角向邻居列表
- `x12_radial`, `y12_radial`, `z12_radial`: 相对位置向量
- `x12_angular`, `y12_angular`, `z12_angular`: 相对位置向量

---

## 数据维度变化与信息详解

本节详细说明数据在每一步处理后的维度变化、存储方式和包含的信息。这是理解NEP训练流程的关键。

### 假设配置

为了具体说明，我们假设以下配置：
- **Nc = 100**: 训练配置数量
- **N = 10000**: 总原子数（所有配置的原子总和）
- **max_Na = 200**: 单个配置的最大原子数
- **num_types = 3**: 元素类型数（例如：Si, C, O）
- **n_max_radial = 4**: 径向描述符最大阶数
- **n_max_angular = 4**: 角向描述符最大阶数
- **basis_size_radial = 8**: 径向基函数数量
- **basis_size_angular = 8**: 角向基函数数量
- **L_max = 4**: 球谐函数最大阶数
- **L_max_4body = 2**: 4-body描述符阶数
- **L_max_5body = 0**: 5-body描述符（未启用）
- **num_neurons1 = 30**: 隐藏层神经元数量
- **rc_radial = 6.0 Å**: 径向截断半径
- **rc_angular = 5.0 Å**: 角向截断半径
- **max_NN_radial = 200**: 最大径向邻居数
- **max_NN_angular = 150**: 最大角向邻居数

**计算得到的维度**：
- **dim_radial = n_max_radial + 1 = 5**: 径向描述符维度
- **dim_angular = (n_max_angular + 1) × L_max + (n_max_angular + 1) = 5 × 4 + 5 = 25**: 角向描述符维度
- **dim = dim_radial + dim_angular = 5 + 25 = 30**: 总描述符维度

---

### 步骤1：原始数据读取（train.xyz）

**位置**：`src/main_nep/structure.cu` - `read_one_structure()`

#### 数据结构

每个结构在CPU内存中存储为：

```cpp
Structure {
  int num_atom;                    // 该结构的原子数（例如：150）
  int type[Na];                    // 原子类型数组 [Na]
  float x[Na], y[Na], z[Na];       // 原子坐标 [Na] × 3
  float fx[Na], fy[Na], fz[Na];   // 参考力 [Na] × 3
  float energy;                    // 参考总能量（标量）
  float virial[6];                // 参考维里张量 [6]
  float box[18];                  // 扩展盒子信息 [18]
  float box_original[9];          // 原始盒子信息 [9]
  int num_cell[3];                // 扩展盒子单元数 [3]
}
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `type` | `[Na]` | `int` | CPU | 每个原子的类型索引（0, 1, 2, ...） |
| `x, y, z` | `[Na]` × 3 | `float` | CPU | 每个原子的笛卡尔坐标（单位：Å） |
| `fx, fy, fz` | `[Na]` × 3 | `float` | CPU | 每个原子的参考力（单位：eV/Å） |
| `energy` | `1` | `float` | CPU | 结构的总能量（单位：eV） |
| `virial` | `[6]` | `float` | CPU | 维里张量的6个独立分量（xx, yy, zz, xy, xz, yz） |
| `box` | `[18]` | `float` | CPU | 扩展盒子的9个基向量 + 9个逆矩阵元素 |
| `box_original` | `[9]` | `float` | CPU | 原始盒子的3×3基向量矩阵 |
| `num_cell` | `[3]` | `int` | CPU | 扩展盒子在x, y, z方向的单元数 |

**总内存（单个结构，假设Na=150）**：
- 类型：150 × 4 bytes = 600 bytes
- 坐标：150 × 3 × 4 bytes = 1800 bytes
- 力：150 × 3 × 4 bytes = 1800 bytes
- 其他：约200 bytes
- **总计**：约4.4 KB per structure

---

### 步骤2：数据集组织（Dataset类）

**位置**：`src/main_nep/dataset.cu` - `copy_structures()`

#### 数据结构

多个结构被组织到Dataset类中：

```cpp
Dataset {
  int Nc;                          // 配置数量（例如：100）
  int N;                           // 总原子数（例如：10000）
  int max_Na;                      // 最大配置原子数（例如：200）
  std::vector<Structure> structures; // 结构数组 [Nc]
  GPU_Vector<int> Na;              // 每个配置的原子数 [Nc]
  GPU_Vector<int> Na_sum;          // 原子数前缀和 [Nc]
}
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `structures` | `[Nc]` | `Structure` | CPU | 所有结构的数组 |
| `Na` | `[Nc]` | `int` | GPU | 每个配置的原子数，例如：[150, 180, 120, ...] |
| `Na_sum` | `[Nc]` | `int` | GPU | 前缀和，用于索引计算，例如：[0, 150, 330, 450, ...] |

**索引关系**：
- 配置`c`的原子索引范围：`[Na_sum[c], Na_sum[c] + Na[c])`
- 例如：配置0的原子索引为0-149，配置1的原子索引为150-329

---

### 步骤3：GPU数据传输

**位置**：`src/main_nep/dataset.cu` - `initialize_gpu_data()`

#### 数据结构

数据被展平并复制到GPU：

```cpp
Dataset {
  // 原子级数据（展平到一维数组）
  GPU_Vector<int> type;            // [N]
  GPU_Vector<float> r;             // [N * 3] (x, y, z交错存储)
  
  // 配置级数据
  GPU_Vector<float> box;           // [Nc * 18]
  GPU_Vector<float> box_original;   // [Nc * 9]
  GPU_Vector<int> num_cell;        // [Nc * 3]
  
  // 参考数据
  GPU_Vector<float> energy_ref;    // [Nc]
  GPU_Vector<float> force_ref;     // [N * 3]
  GPU_Vector<float> virial_ref;    // [Nc * 6]
}
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `type` | `[N]` | `int` | GPU | 所有原子的类型索引 | 10000 × 4 = 40 KB |
| `r` | `[N * 3]` | `float` | GPU | 所有原子的坐标（x,y,z交错） | 10000 × 3 × 4 = 120 KB |
| `box` | `[Nc * 18]` | `float` | GPU | 所有配置的扩展盒子信息 | 100 × 18 × 4 = 7.2 KB |
| `box_original` | `[Nc * 9]` | `float` | GPU | 所有配置的原始盒子信息 | 100 × 9 × 4 = 3.6 KB |
| `num_cell` | `[Nc * 3]` | `int` | GPU | 所有配置的扩展单元数 | 100 × 3 × 4 = 1.2 KB |
| `energy_ref` | `[Nc]` | `float` | GPU | 所有配置的参考能量 | 100 × 4 = 400 B |
| `force_ref` | `[N * 3]` | `float` | GPU | 所有原子的参考力 | 10000 × 3 × 4 = 120 KB |
| `virial_ref` | `[Nc * 6]` | `float` | GPU | 所有配置的参考维里 | 100 × 6 × 4 = 2.4 KB |

**存储方式说明**：
- `r`数组：`[x₀, y₀, z₀, x₁, y₁, z₁, ..., x_{N-1}, y_{N-1}, z_{N-1}]`
- `force_ref`数组：`[fx₀, fy₀, fz₀, fx₁, fy₁, fz₁, ..., fx_{N-1}, fy_{N-1}, fz_{N-1}]`
- 原子`i`的坐标：`r[i*3]`, `r[i*3+1]`, `r[i*3+2]`
- 原子`i`的力：`force_ref[i*3]`, `force_ref[i*3+1]`, `force_ref[i*3+2]`

**总GPU内存（示例）**：约295 KB

---

### 步骤4：邻居列表构建

**位置**：`src/main_nep/nep.cu` - `gpu_find_neighbor_list()`

#### 数据结构

```cpp
NEP_Data {
  GPU_Vector<int> NN_radial;       // [N]
  GPU_Vector<int> NL_radial;       // [max_NN_radial * N]
  GPU_Vector<int> NN_angular;      // [N]
  GPU_Vector<int> NL_angular;      // [max_NN_angular * N]
  GPU_Vector<float> x12_radial;   // [max_NN_radial * N]
  GPU_Vector<float> y12_radial;   // [max_NN_radial * N]
  GPU_Vector<float> z12_radial;   // [max_NN_radial * N]
  GPU_Vector<float> x12_angular;  // [max_NN_angular * N]
  GPU_Vector<float> y12_angular;  // [max_NN_angular * N]
  GPU_Vector<float> z12_angular;  // [max_NN_angular * N]
}
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `NN_radial` | `[N]` | `int` | GPU | 每个原子的径向邻居数量 | 10000 × 4 = 40 KB |
| `NL_radial` | `[max_NN_radial * N]` | `int` | GPU | 径向邻居列表 | 200 × 10000 × 4 = 8 MB |
| `NN_angular` | `[N]` | `int` | GPU | 每个原子的角向邻居数量 | 10000 × 4 = 40 KB |
| `NL_angular` | `[max_NN_angular * N]` | `int` | GPU | 角向邻居列表 | 150 × 10000 × 4 = 6 MB |
| `x12_radial` | `[max_NN_radial * N]` | `float` | GPU | 径向相对位置x分量 | 200 × 10000 × 4 = 8 MB |
| `y12_radial` | `[max_NN_radial * N]` | `float` | GPU | 径向相对位置y分量 | 200 × 10000 × 4 = 8 MB |
| `z12_radial` | `[max_NN_radial * N]` | `float` | GPU | 径向相对位置z分量 | 200 × 10000 × 4 = 8 MB |
| `x12_angular` | `[max_NN_angular * N]` | `float` | GPU | 角向相对位置x分量 | 150 × 10000 × 4 = 6 MB |
| `y12_angular` | `[max_NN_angular * N]` | `float` | GPU | 角向相对位置y分量 | 150 × 10000 × 4 = 6 MB |
| `z12_angular` | `[max_NN_angular * N]` | `float` | GPU | 角向相对位置z分量 | 150 × 10000 × 4 = 6 MB |

**存储方式说明**：

邻居列表使用**列优先**存储（column-major）：
- `NL_radial[i * N + n1]`：原子`n1`的第`i`个径向邻居的索引
- `x12_radial[i * N + n1]`：原子`n1`到其第`i`个径向邻居的x方向相对位置

**访问模式**：
```cpp
// 对原子n1，访问其第i个径向邻居
int neighbor_index = NL_radial[i * N + n1];
float x_rel = x12_radial[i * N + n1];
float y_rel = y12_radial[i * N + n1];
float z_rel = z12_radial[i * N + n1];
float distance = sqrt(x_rel*x_rel + y_rel*y_rel + z_rel*z_rel);
```

**总GPU内存（示例）**：约52 MB

---

### 步骤5：径向描述符计算

**位置**：`src/main_nep/nep.cu` - `find_descriptors_radial()`

#### 计算过程

对每个原子`n1`，计算径向描述符：

```cpp
// 临时变量（每个原子）
float q_radial[n_max_radial + 1] = {0.0f};  // [5] 局部变量

// 对每个径向邻居
for (i = 0; i < NN_radial[n1]; ++i):
  int n2 = NL_radial[i * N + n1];
  float x12 = x12_radial[i * N + n1];
  float y12 = y12_radial[i * N + n1];
  float z12 = z12_radial[i * N + n1];
  float d12 = sqrt(x12² + y12² + z12²);
  
  // 计算基函数和描述符系数
  for (n = 0; n <= n_max_radial; ++n):
    float gn12 = ...;  // 使用参数c_n^k计算
    q_radial[n] += gn12;

// 存储到全局数组
for (n = 0; n <= n_max_radial; ++n):
  descriptors[n * N + n1] = q_radial[n];
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `q_radial`（局部） | `[n_max_radial + 1]` = `[5]` | `float` | 寄存器/共享内存 | 单个原子的径向描述符（临时） | 5 × 4 = 20 B |
| `descriptors`（径向部分） | `[dim_radial * N]` = `[5 * N]` | `float` | GPU | 所有原子的径向描述符 | 5 × 10000 × 4 = 200 KB |

**存储方式**：
- `descriptors[n * N + i]`：原子`i`的第`n`个径向描述符分量（`n = 0, 1, ..., n_max_radial`）
- 存储顺序：`[q₀⁰, q₀¹, ..., q₀^{N-1}, q₁⁰, q₁¹, ..., q₁^{N-1}, ..., q₄⁰, q₄¹, ..., q₄^{N-1}]`

**信息内容**：
- `q_n^i`：原子`i`的第`n`阶径向描述符，编码了该原子周围径向环境的特征
- 每个分量是标量，累加了所有径向邻居的贡献

---

### 步骤6：角向描述符计算

**位置**：`src/main_nep/nep.cu` - `find_descriptors_angular()`

#### 计算过程

对每个原子`n1`，计算角向描述符：

```cpp
// 临时变量（每个原子）
float q_angular[dim_angular] = {0.0f};  // [25] 局部变量
float s[NUM_OF_ABC] = {0.0f};           // [80] 球谐函数系数（临时）

// 对每个n（角向阶数）
for (n = 0; n <= n_max_angular; ++n):
  // 重置球谐函数系数
  s[...] = {0.0f};
  
  // 对每个角向邻居
  for (i = 0; i < NN_angular[n1]; ++i):
    int n2 = NL_angular[i * N + n1];
    float x12 = x12_angular[i * N + n1];
    float y12 = y12_angular[i * N + n1];
    float z12 = z12_angular[i * N + n1];
    float d12 = sqrt(x12² + y12² + z12²);
    
    // 计算gn12并累加到球谐函数系数
    float gn12 = ...;  // 使用参数c_n^k计算
    accumulate_s(L_max, d12, x12, y12, z12, gn12, s);
  
  // 从球谐函数系数计算描述符
  find_q(L_max, num_L, n_max_angular + 1, n, s, q_angular);

// 存储到全局数组
for (ln = 0; ln < dim_angular; ++ln):
  descriptors[(dim_radial + ln) * N + n1] = q_angular[ln];
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `q_angular`（局部） | `[dim_angular]` = `[25]` | `float` | 寄存器/共享内存 | 单个原子的角向描述符（临时） | 25 × 4 = 100 B |
| `s`（局部） | `[NUM_OF_ABC]` = `[80]` | `float` | 寄存器/共享内存 | 球谐函数系数（临时） | 80 × 4 = 320 B |
| `sum_fxyz` | `[N * (n_max_angular + 1) * NUM_OF_ABC]` = `[N * 5 * 80]` | `float` | GPU | 用于力计算的中间变量 | 10000 × 5 × 80 × 4 = 16 MB |
| `descriptors`（角向部分） | `[dim_angular * N]` = `[25 * N]` | `float` | GPU | 所有原子的角向描述符 | 25 × 10000 × 4 = 1 MB |

**存储方式**：
- `descriptors[(dim_radial + ln) * N + i]`：原子`i`的第`ln`个角向描述符分量
- `ln`的索引：`ln = l * (n_max_angular + 1) + n`，其中`l = 0, 1, ..., L_max-1`，`n = 0, 1, ..., n_max_angular`
- 对于4-body描述符，有额外的`n`值

**信息内容**：
- `q_{nl}^i`：原子`i`的第`(n,l)`阶角向描述符，编码了多体（3-body, 4-body, 5-body）相互作用
- 每个分量是标量，通过球谐函数展开编码角度信息

---

### 步骤7：描述符归一化

**位置**：`src/force/nep.cu` - `find_descriptor()` 第711-713行

#### 计算过程

```cpp
// 对每个原子n1
for (d = 0; d < dim; ++d):
  descriptors[d * N + n1] *= q_scaler[d];
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `descriptors`（完整） | `[dim * N]` = `[30 * N]` | `float` | GPU | 所有原子的完整描述符向量 | 30 × 10000 × 4 = 1.2 MB |
| `q_scaler` | `[dim]` = `[30]` | `float` | GPU常量内存 | 描述符缩放因子（可训练） | 30 × 4 = 120 B |

**归一化后的描述符结构**：
```
对原子i，描述符向量为：
q^i = [q_scaler[0] * q_0^i, 
       q_scaler[1] * q_1^i,
       ...
       q_scaler[4] * q_4^i,                    // 径向描述符 (5个)
       q_scaler[5] * q_{0,0}^i,
       q_scaler[6] * q_{0,1}^i,
       ...
       q_scaler[29] * q_{4,4}^i]               // 角向描述符 (25个)
```

**信息内容**：
- 归一化后的描述符值通常在合理范围内（例如：-10到10）
- `q_scaler[d]`在训练过程中自动优化，确保数值稳定性

---

### 步骤8：神经网络输入

**位置**：`src/utilities/nep_utilities.cuh` - `apply_ann_one_layer()`

#### 数据结构

```cpp
// 对每个原子n1，提取描述符向量
float q[dim] = {0.0f};  // [30] 局部变量

// 从全局数组复制
for (d = 0; d < dim; ++d):
  q[d] = descriptors[d * N + n1];
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `q`（局部） | `[dim]` = `[30]` | `float` | 寄存器 | 单个原子的描述符向量（神经网络输入） |

**输入向量结构**：
```
q = [q_0, q_1, q_2, q_3, q_4,                    // 径向描述符 (5维)
     q_{0,0}, q_{0,1}, ..., q_{0,4},            // l=0角向描述符 (5维)
     q_{1,0}, q_{1,1}, ..., q_{1,4},            // l=1角向描述符 (5维)
     q_{2,0}, q_{2,1}, ..., q_{2,4},            // l=2角向描述符 (5维)
     q_{3,0}, q_{3,1}, ..., q_{3,4},            // l=3角向描述符 (5维)
     q_{4,0}, q_{4,1}, ..., q_{4,4}]             // l=4角向描述符 (5维，如果L_max=4)
```

**信息内容**：
- 30维向量完整编码了原子的局部环境
- 径向部分（5维）：编码2-body相互作用
- 角向部分（25维）：编码3-body及以上相互作用

---

### 步骤9：神经网络前向传播

**位置**：`src/utilities/nep_utilities.cuh` - `apply_ann_one_layer()`

#### 计算过程

```cpp
// 输入
float q[dim] = {0.0f};              // [30] 描述符向量

// 隐藏层
float h[num_neurons1] = {0.0f};     // [30] 隐藏层输出（临时）
float energy_derivative[dim] = {0.0f}; // [30] 能量对描述符的导数（临时）

// 输出
float energy = 0.0f;                // 原子能量（标量）
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `q` | `[dim]` = `[30]` | `float` | 寄存器 | 输入：描述符向量 |
| `h`（局部） | `[num_neurons1]` = `[30]` | `float` | 寄存器 | 隐藏层输出：激活后的特征 |
| `energy_derivative`（局部） | `[dim]` = `[30]` | `float` | 寄存器 | 能量对描述符的导数（用于计算力） |
| `energy` | `1` | `float` | 寄存器 | 输出：原子能量（单位：eV） |

**计算流程**：
1. **隐藏层输入**：`h_input[n] = Σ_d (w0[n][d] * q[d]) - b0[n]`（30维 → 30维）
2. **隐藏层输出**：`h[n] = tanh(h_input[n])`（30维）
3. **输出层**：`energy = Σ_n (w1[n] * h[n]) - b1`（30维 → 1维）

**参数维度**（NEP4，每种元素独立）：
- `w0[t]`: `[num_neurons1 * dim]` = `[30 * 30]` = `[900]` per type
- `b0[t]`: `[num_neurons1]` = `[30]` per type
- `w1[t]`: `[num_neurons1]` = `[30]` per type
- `b1`: `[1]`（全局，所有类型共享）

---

### 步骤10：力和维里计算

**位置**：`src/force/nep.cu` - `find_force_radial()` 和 `find_force_angular()`

#### 数据结构

```cpp
// 对每个原子n1
float fx = 0.0f, fy = 0.0f, fz = 0.0f;  // 力（3个标量）
float virial[6] = {0.0f};                // 维里张量（6个独立分量）
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 | 内存大小（示例） |
|------|------|---------|---------|---------|----------------|
| `force` | `[N * 3]` | `float` | GPU | 所有原子的力 | 10000 × 3 × 4 = 120 KB |
| `virial`（原子级） | `[N * 6]` | `float` | GPU | 所有原子的维里贡献 | 10000 × 6 × 4 = 240 KB |
| `energy`（原子级） | `[N]` | `float` | GPU | 所有原子的能量 | 10000 × 4 = 40 KB |

**存储方式**：
- `force[i*3]`, `force[i*3+1]`, `force[i*3+2]`：原子`i`的力（fx, fy, fz）
- `virial[i*6 + 0]`到`virial[i*6 + 5]`：原子`i`的维里张量（xx, yy, zz, xy, xz, yz）

**信息内容**：
- **力**：每个原子受到的力（单位：eV/Å），通过链式法则从能量导数计算
- **维里**：每个原子对维里张量的贡献（单位：eV），用于计算应力
- **能量**：每个原子的能量贡献（单位：eV）

---

### 步骤11：结构级量聚合

**位置**：`src/main_nep/dataset.cu` - `get_rmse_energy()` 等

#### 计算过程

```cpp
// 对每个配置c
float E_total = 0.0f;
for (i = Na_sum[c]; i < Na_sum[c] + Na[c]; ++i):
  E_total += energy[i];

// 计算误差
float error = E_total - energy_ref[c];
```

#### 维度信息

| 变量 | 维度 | 数据类型 | 存储位置 | 信息内容 |
|------|------|---------|---------|---------|
| `energy`（结构级） | `[Nc]` | `float` | CPU | 每个配置的总能量（从原子能量求和） |
| `virial`（结构级） | `[Nc * 6]` | `float` | CPU | 每个配置的维里张量（从原子维里求和） |

**聚合关系**：
- 结构`c`的总能量：`E_c = Σ_{i ∈ config_c} E_i`
- 结构`c`的维里：`W_c = Σ_{i ∈ config_c} W_i`

---

### 维度变化总结表

| 步骤 | 主要变量 | 输入维度 | 输出维度 | 维度变化 | 关键信息 |
|------|---------|---------|---------|---------|---------|
| 1. 原始数据 | `type`, `x,y,z`, `fx,fy,fz` | `[Na]`, `[Na]×3` | - | - | 原子坐标、类型、参考力 |
| 2. 数据集组织 | `structures` | `[Nc]` | - | - | 多个结构组织 |
| 3. GPU传输 | `type`, `r`, `force_ref` | `[N]`, `[N×3]` | - | - | 展平到一维数组 |
| 4. 邻居列表 | `NN_radial`, `NL_radial` | - | `[N]`, `[max_NN×N]` | 新增 | 邻居索引和相对位置 |
| 5. 径向描述符 | `q_radial` | - | `[5×N]` | 5维/原子 | 2-body环境编码 |
| 6. 角向描述符 | `q_angular` | - | `[25×N]` | 25维/原子 | 多体环境编码 |
| 7. 描述符归一化 | `descriptors` | `[30×N]` | `[30×N]` | 不变 | 缩放后的描述符 |
| 8. 神经网络输入 | `q` | `[30×N]` | `[30]` per atom | 提取 | 单个原子的描述符向量 |
| 9. 神经网络输出 | `energy` | `[30]` | `1` | 30→1 | 原子能量 |
| 10. 力和维里 | `force`, `virial` | - | `[N×3]`, `[N×6]` | 新增 | 力和维里张量 |
| 11. 结构级聚合 | `E_total`, `W_total` | `[N]` | `[Nc]` | 聚合 | 配置级能量和维里 |

---

### 内存使用总结（示例配置）

| 数据类型 | 内存大小 | 说明 |
|---------|---------|------|
| 原始数据（CPU） | ~440 KB | 100个结构，平均150原子/结构 |
| GPU基础数据 | ~295 KB | 类型、坐标、参考值 |
| 邻居列表 | ~52 MB | 邻居索引和相对位置 |
| 描述符 | ~1.2 MB | 所有原子的描述符向量 |
| 中间变量 | ~16 MB | sum_fxyz等 |
| 力和能量 | ~400 KB | 计算的力和能量 |
| **总计** | **~70 MB** | 单个GPU设备的内存使用 |

**注意**：实际内存使用取决于配置大小、邻居数量等，上述为示例估算。

---

## 描述符计算（输入层预处理）

描述符将原子的局部环境编码为固定维度的向量，作为神经网络的输入。

### 1. 径向描述符（2-body）

**位置**：`src/main_nep/nep.cu` - `find_descriptors_radial()`

对每个原子`n1`，计算径向描述符：

```cpp
float q[n_max_radial + 1] = {0.0f};

for (每个邻居 n2 in NL_radial[n1]):
  float d12 = sqrt(x12² + y12² + z12²);  // 距离
  int t1 = type[n1], t2 = type[n2];      // 原子类型
  
  // 1. 计算截断函数
  float fc12 = (1 - d12/rc)² * (1 + 2*d12/rc);  // 平滑截断
  
  // 2. 计算基函数 fn12[k]
  float fn12[basis_size_radial + 1];
  for (k = 0; k <= basis_size_radial; ++k):
    fn12[k] = Chebyshev多项式(d12/rc) * fc12
  
  // 3. 计算 gn12[n] = Σ_k c_n^k * fn12[k]
  for (n = 0; n <= n_max_radial; ++n):
    float gn12 = 0.0f;
    for (k = 0; k <= basis_size_radial; ++k):
      int c_index = (n * (basis_size_radial + 1) + k) * num_types²
                    + t1 * num_types + t2;
      gn12 += c[c_index] * fn12[k];
    q[n] += gn12;  // 累加所有邻居的贡献
```

**数学公式**：

径向描述符的第`n`个分量为：
```
q_n^i = Σ_{j≠i} g_n^{ij}(r_{ij})
```

其中：
```
g_n^{ij}(r) = Σ_{k=0}^{K} c_n^{ij,k} · f_k(r/rc) · fc(r)
```

- `f_k(x)`: Chebyshev多项式基函数
- `fc(r)`: 截断函数，确保在`rc`处平滑截断
- `c_n^{ij,k}`: **可训练的描述符参数**（元素对`(i,j)`，阶数`n`，基函数`k`）

**输出**：`g_descriptors[n1 + n * N] = q[n]`，`n = 0, 1, ..., n_max_radial`

### 2. 角向描述符（3-body及以上）

**位置**：`src/main_nep/nep.cu` - `find_descriptors_angular()`

角向描述符考虑多体相互作用（3-body, 4-body, 5-body）：

```cpp
float q[dim_angular] = {0.0f};

for (n = 0; n <= n_max_angular; ++n):
  float s[NUM_OF_ABC] = {0.0f};  // 球谐函数系数
  
  for (每个邻居 n2 in NL_angular[n1]):
    float d12 = sqrt(x12² + y12² + z12²);
    int t2 = type[n2];
    
    // 1. 计算截断函数和基函数（类似径向）
    float fc12 = ...;
    float fn12[basis_size_angular + 1] = ...;
    
    // 2. 计算 gn12 = Σ_k c_n^k * fn12[k]
    float gn12 = 0.0f;
    for (k = 0; k <= basis_size_angular; ++k):
      int c_index = (n * (basis_size_angular + 1) + k) * num_types²
                    + t1 * num_types + t2 + num_c_radial;
      gn12 += c[c_index] * fn12[k];
    
    // 3. 累加球谐函数项
    accumulate_s(L_max, d12, x12, y12, z12, gn12, s);
  
  // 4. 从球谐函数系数计算描述符
  find_q(L_max, num_L, n_max_angular + 1, n, s, q);
```

**数学公式**：

角向描述符使用球谐函数展开：
```
q_{nl}^i = Σ_{j≠i} Σ_{k≠i,j} g_n^{ij}(r_{ij}) · g_n^{ik}(r_{ik}) · Y_l^m(θ_{jik})
```

其中`Y_l^m`是球谐函数，`θ_{jik}`是角度。

**输出**：`g_descriptors[n1 + (dim_radial + ln) * N] = q[ln]`

### 3. 描述符归一化（q_scaler）

**位置**：`src/force/nep.cu` - `find_descriptor()` 第711-713行

```cpp
// 归一化描述符
for (int d = 0; d < annmb.dim; ++d) {
  q[d] = q[d] * paramb.q_scaler[d];
}
```

**作用**：
- 将描述符值缩放到合适的范围，提高数值稳定性
- `q_scaler[d]`是**可训练的参数**，在训练过程中自动优化

**完整描述符向量**：
```
q = [q_0, q_1, ..., q_{n_max_radial},     // 径向描述符 (dim_radial个)
     q_{0,0}, q_{0,1}, ..., q_{L_max,n_max_angular}]  // 角向描述符 (dim_angular个)
```

总维度：`dim = dim_radial + dim_angular`

---

## 神经网络结构

### 网络架构

NEP使用**单隐藏层前馈神经网络**：

```
输入层 (dim维) → 隐藏层 (num_neurons1个神经元) → 输出层 (1维，原子能量)
```

**代码位置**：`src/utilities/nep_utilities.cuh` - `apply_ann_one_layer()`

### 参数结构

#### NEP3版本（所有元素共享）

```
参数数量 = (dim + 2) × num_neurons1 + 1
```

参数组成：
- `w0[dim × num_neurons1]`: 输入层到隐藏层的权重矩阵
- `b0[num_neurons1]`: 隐藏层偏置向量
- `w1[num_neurons1]`: 隐藏层到输出层的权重向量
- `b1[1]`: 输出层偏置（全局偏置）

#### NEP4版本（每种元素独立）

```
参数数量 = (dim + 2) × num_neurons1 × num_types + 1
```

对每种元素类型`t`：
- `w0[t][dim × num_neurons1]`: 该类型的输入层到隐藏层权重
- `b0[t][num_neurons1]`: 该类型的隐藏层偏置
- `w1[t][num_neurons1]`: 该类型的隐藏层到输出层权重
- `b1[1]`: 全局偏置（所有类型共享）

---

## 前向传播：从描述符到能量

### 计算流程

**位置**：`src/utilities/nep_utilities.cuh` - `apply_ann_one_layer()`

对每个原子`i`，给定描述符`q`，计算原子能量`E_i`：

```cpp
float energy = 0.0f;
float energy_derivative[dim] = {0.0f};  // 用于后续计算力

for (int n = 0; n < num_neurons1; ++n) {
  // 1. 隐藏层输入：线性组合
  float w0_times_q = 0.0f;
  for (int d = 0; d < dim; ++d) {
    w0_times_q += w0[n * dim + d] * q[d];
  }
  
  // 2. 隐藏层输出：激活函数（tanh）
  float x1 = tanh(w0_times_q - b0[n]);
  float tanh_der = 1.0f - x1 * x1;  // tanh的导数，用于反向传播
  
  // 3. 输出层：线性组合
  energy += w1[n] * x1;
  
  // 4. 计算能量对描述符的导数（用于计算力）
  for (int d = 0; d < dim; ++d) {
    energy_derivative[d] += w1[n] * tanh_der * w0[n * dim + d];
  }
}

// 5. 减去全局偏置
energy -= b1[0];
```

### 数学公式

**隐藏层**：
```
h_n = tanh(Σ_d w0_{n,d} · q_d - b0_n)
```

**输出层**：
```
E_i = Σ_n w1_n · h_n - b1
```

**能量对描述符的导数**：
```
∂E_i/∂q_d = Σ_n w1_n · (1 - h_n²) · w0_{n,d}
```

---

## 反向传播：力和维里的计算

### 力的计算

**位置**：`src/force/nep.cu` - `find_force_radial()` 和 `find_force_angular()`

力通过链式法则计算：

```cpp
// 对每个原子 n1
float fx = 0.0f, fy = 0.0f, fz = 0.0f;

// 径向描述符对力的贡献
for (每个邻居 n2 in NL_radial[n1]):
  float d12 = sqrt(x12² + y12² + z12²);
  float x12_norm = x12 / d12, y12_norm = y12 / d12, z12_norm = z12 / d12;
  
  for (n = 0; n <= n_max_radial; ++n):
    // 计算 ∂q_n/∂r_{12}
    float dq_dr = ...;  // 描述符对距离的导数
    
    // 力 = -∂E/∂r = -Σ_n (∂E/∂q_n) · (∂q_n/∂r_{12}) · (r_{12}/|r_{12}|)
    fx -= energy_derivative[n] * dq_dr * x12_norm;
    fy -= energy_derivative[n] * dq_dr * y12_norm;
    fz -= energy_derivative[n] * dq_dr * z12_norm;

// 角向描述符对力的贡献（类似处理）
...
```

**数学公式**：

对原子`i`的力：
```
F_i = -∇_i E = -Σ_j≠i (∂E/∂r_{ij}) · (r_{ij}/|r_{ij}|)
```

其中：
```
∂E/∂r_{ij} = Σ_d (∂E/∂q_d) · (∂q_d/∂r_{ij})
```

### 维里的计算

**位置**：`src/force/nep.cu` - `find_force_radial()` 和 `find_force_angular()`

维里张量（用于计算应力）：

```cpp
// 对每个原子 n1
float virial[6] = {0.0f};  // xx, yy, zz, xy, xz, yz

for (每个邻居 n2):
  // 维里 = -0.5 * r_{ij} ⊗ F_{ij}
  virial[0] -= 0.5 * x12 * f12x;  // xx
  virial[1] -= 0.5 * y12 * f12y;  // yy
  virial[2] -= 0.5 * z12 * f12z;  // zz
  virial[3] -= 0.5 * x12 * f12y;  // xy
  virial[4] -= 0.5 * x12 * f12z;  // xz
  virial[5] -= 0.5 * y12 * f12z;  // yz
```

**数学公式**：

维里张量：
```
W = -0.5 Σ_{i,j} r_{ij} ⊗ F_{ij}
```

---

## SNES训练算法

### 算法概述

SNES（Separable Natural Evolution Strategy）是一种基于自然进化策略的优化算法，用于优化高维参数空间。

**核心思想**：
- 使用高斯分布`N(μ, σ²)`来采样候选解
- 根据适应度排序更新分布参数`μ`和`σ`
- 不需要计算梯度，适合不可微分的优化问题

### 训练的参数

SNES训练以下参数：

1. **神经网络参数**：
   - `w0`: 输入层到隐藏层的权重
   - `b0`: 隐藏层偏置
   - `w1`: 隐藏层到输出层的权重
   - `b1`: 输出层偏置

2. **描述符参数**：
   - `c_n^k`: 径向描述符系数
   - `c_n^k`: 角向描述符系数

3. **q_scaler**：
   - 描述符缩放因子（在训练过程中自动优化）

### SNES训练流程

**位置**：`src/main_nep/snes.cu` - `compute()`

#### 步骤1：初始化参数分布

```cpp
// 初始化 μ (均值) 和 σ (标准差)
if (fine_tune) {
  initialize_mu_and_sigma_fine_tune(para);  // 从基础模型加载
} else {
  initialize_mu_and_sigma(para);  // 随机初始化
}
```

**初始化方式**：
- `μ[n] = random(-1, 1)` 或从`nep.restart`读取
- `σ[n] = sigma0`（默认0.1）或从`nep.restart`读取

##### Fine_tune模式下的参数复制机制

在fine_tune模式下，所有参数（包括径向/角向描述符参数和NN权重）都是从原始NEP89基础模型**直接复制（copy）**过来的。**即使fine_tune之后元素数量减少了（例如从89种减少到3种），复制的参数仍然基于原始NEP89模型的对应元素或元素对，参数数值直接继承，不做重新初始化。**

**代码位置**：`src/main_nep/snes.cu` - `SNES::initialize_mu_and_sigma_fine_tune()` (第144-238行)

###### 1. 参数数组的存储结构

在SNES算法中，所有可优化参数存储在一个线性数组中：

```cpp
// SNES类中的成员变量
std::vector<float> mu;      // [number_of_variables] 参数分布的均值
std::vector<float> sigma;   // [number_of_variables] 参数分布的标准差
```

**参数数组结构**（按顺序排列）：
```
parameters[0 ... number_of_variables_ann - 1]           : 神经网络参数
  [0 ... num_types × number_of_variables_ann_1 - 1]    : 每种元素的ANN参数
  [num_types × number_of_variables_ann_1]              : 全局偏置
parameters[number_of_variables_ann ... number_of_variables_ann + num_cnk_radial - 1]  : 径向描述符参数
parameters[number_of_variables_ann + num_cnk_radial ... number_of_variables - 1]      : 角向描述符参数
```

其中：
- `number_of_variables_ann_1 = (dim + 2) × num_neurons1`：每种元素的ANN参数数量
- `number_of_variables_ann = num_types × number_of_variables_ann_1 + 1`：所有ANN参数数量
- `num_cnk_radial = num_types² × (n_max_radial + 1) × (basis_size_radial + 1)`：径向描述符参数数量
- `num_cnk_angular = num_types² × (n_max_angular + 1) × (basis_size_angular + 1)`：角向描述符参数数量

###### 2. 元素映射机制

基础模型（NEP89）支持89种元素，但缺少5种元素（Po, At, Rn, Fr, Ra）。需要通过元素映射将原子序数映射到基础模型的元素索引。

**元素映射数组**（代码位置：`src/main_nep/snes.cu` 第147-153行）：
```cpp
const int element_map[94] = {
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
  20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
  40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
  60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
  80,81,82,0,0,0,0,0,83,84,85,86,87,88  // 缺失元素映射为0
};
```

**映射规则**：
```cpp
element_index = element_map[atomic_number - 1]
```

其中：
- `atomic_number`：元素的原子序数（1-94）
- `element_index`：在基础模型中的索引（0-88，或0表示不存在）

**示例**：
- Si（原子序数14）→ `element_index = element_map[13] = 13`
- C（原子序数6）→ `element_index = element_map[5] = 5`
- O（原子序数8）→ `element_index = element_map[7] = 7`

###### 3. 神经网络参数的复制

**位置**：`src/main_nep/snes.cu` 第181-190行

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

**复制逻辑**：
1. **元素索引映射**：通过`element_map`将用户指定的元素原子序数映射到基础模型的元素索引
2. **参数提取**：从基础模型中提取该元素的所有ANN参数（`w0`, `b0`, `w1`）
3. **直接复制**：`mu`和`sigma`直接从基础模型的`restart_mu`和`restart_sigma`中复制对应位置的值

**关键点**：
- **数值完全继承**：`mu[count]`和`sigma[count]`的值**直接等于**基础模型中对应元素的参数值
- **参数不变性**：即使fine_tune后元素数量从89种减少到3种，复制的参数仍然是**原始NEP89模型中该元素的训练结果**
- **全局偏置共享**：全局偏置`b1`从基础模型的全局偏置位置直接复制

**维度变化示例**（用户指定Si, C, O三种元素）：
- 基础模型：89种元素 × 2560参数/元素 + 1（全局偏置）= 227,841个ANN参数
- 用户模型：提取Si（索引13）、C（索引5）、O（索引7）的参数
  - Si的参数：`restart_mu[13 × 2560 ... 14 × 2560 - 1]` → `mu[0 ... 2559]`
  - C的参数：`restart_mu[5 × 2560 ... 6 × 2560 - 1]` → `mu[2560 ... 5119]`
  - O的参数：`restart_mu[7 × 2560 ... 8 × 2560 - 1]` → `mu[5120 ... 7679]`
  - 全局偏置：`restart_mu[89 × 2560]` → `mu[7680]`
  - 总计：3 × 2560 + 1 = 7,681个ANN参数

###### 4. 径向描述符参数的复制

**位置**：`src/main_nep/snes.cu` 第192-211行

```cpp
for (int n = 0; n <= para.n_max_radial; ++n) {
  for (int k = 0; k <= para.basis_size_radial; ++k) {
    int nk = n * (para.basis_size_radial + 1) + k;
    for (int t1 = 0; t1 < para.num_types; ++t1) {
      for (int t2 = 0; t2 < para.num_types; ++t2) {
        int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
        int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
        int t12 = element_index_1 * NUM89 + element_index_2;  // 基础模型中的元素对索引
        mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann];
        #ifdef FINE_TUNE_DESCRIPTOR
          sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
        #else
          sigma[count] = 0.0f;  // 默认冻结描述符参数
        #endif
        ++count;
      }
    }
  }
}
```

**复制逻辑**：
1. **元素对索引计算**：
   - 用户模型中的元素对：`(t1, t2)`，例如`(Si, C)`
   - 映射到基础模型：`element_index_1 = 13`（Si），`element_index_2 = 5`（C）
   - 基础模型中的元素对索引：`t12 = 13 × 89 + 5 = 1,162`
2. **参数位置计算**：
   ```
   index_in_restart = nk × 89² + t12 + num_ann
   ```
   - `nk = n × (basis_size_radial + 1) + k`：n和k的组合索引
   - `89²`：基础模型的元素对总数（89 × 89 = 7,921）
   - `num_ann`：跳过神经网络参数部分
3. **直接复制**：`mu[count] = restart_mu[index_in_restart]`

**关键点**：
- **元素对映射**：用户模型中的元素对`(Si, C)`直接映射到基础模型中的元素对`(13, 5)`
- **参数完全继承**：描述符参数`c[n][k][t1][t2]`的值**直接等于**基础模型中对应元素对的参数值
- **基于NEP89模型**：即使fine_tune后只有3种元素，所有描述符参数仍然是**原始NEP89模型在89×89=7,921种元素对上训练的结果**

**维度变化示例**（用户指定Si, C, O三种元素）：
- 基础模型：89²种元素对 × 5阶(n) × 9基函数(k) = 7,921 × 45 = 356,445个径向描述符参数
- 用户模型：提取9种元素对（Si-Si, Si-C, Si-O, C-Si, C-C, C-O, O-Si, O-C, O-O）的参数
  - 每个元素对的参数：`restart_mu[nk × 7,921 + t12 + num_ann]`
  - 例如Si-C对：`restart_mu[nk × 7,921 + (13×89+5) + num_ann]` → `mu[...]`
  - 总计：3² × 5 × 9 = 405个径向描述符参数

###### 5. 角向描述符参数的复制

**位置**：`src/main_nep/snes.cu` 第213-232行

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

**与径向描述符的区别**：
- 参数位置需要额外加上`num_cnk_radial`，跳过径向描述符部分
- 其他逻辑完全相同：通过元素对索引从基础模型提取对应参数

###### 6. 参数使用流程

复制后的参数在训练过程中的使用：

```cpp
// 1. SNES生成种群（每次迭代）
for (int p = 0; p < population_size; ++p) {
  for (int v = 0; v < number_of_variables; ++v) {
    float s = gpurand_normal(&state);  // s ~ N(0, 1)
    population[p][v] = sigma[v] * s + mu[v];  // ~ N(mu[v], sigma[v]²)
  }
}

// 2. 使用参数计算描述符和能量
potential->find_force(para, population[p], train_set, ...);

// 3. update_potential解析参数数组
void update_potential(float* parameters, ...) {
  // 提取ANN参数
  for (int t = 0; t < num_types; ++t) {
    w0[t] = parameters[offset...];
    b0[t] = parameters[offset...];
    w1[t] = parameters[offset...];
  }
  b1 = parameters[offset...];
  
  // 提取描述符参数
  c_radial[n][k][t1][t2] = parameters[offset...];
  c_angular[n][k][t1][t2] = parameters[offset...];
}
```

**关键点**：
- **参数值的连续性**：虽然参数数组的维度减少了（从940,731减少到8,491），但提取的参数值**完全继承**自基础模型
- **基于NEP89的知识**：所有参数都携带了原始NEP89模型在大数据集上训练得到的知识
- **微调而非重训练**：fine_tune是在基础模型参数的基础上进行小幅调整（通过更新`mu`和`sigma`），而不是从零开始训练

###### 7. 参数复制的核心结论

1. **直接数值复制**：所有参数（ANN和描述符）通过**直接内存复制**从基础模型继承，不是重新初始化或重新计算
2. **基于NEP89模型**：即使fine_tune后元素数量减少了，所有参数仍然是**原始NEP89模型在该元素或元素对上训练得到的参数值**
3. **元素映射保证一致性**：通过`element_map`确保用户指定的元素正确映射到基础模型的对应元素索引
4. **知识迁移**：fine_tune利用了基础模型在大数据集上学到的通用知识，只在小数据集上进行微调适应

**数学表达**：
```
用户模型参数[用户元素i] = 基础模型参数[element_map[用户元素i]]
```

例如：
```
mu_user[Si的所有ANN参数] = mu_NEP89[element_map[14-1] = 13的所有ANN参数]
                          = mu_NEP89[基础模型中Si元素的ANN参数]
```

#### 步骤2：生成种群

**位置**：`src/main_nep/snes.cu` - `create_population()`

```cpp
for (generation = 0; generation < max_generation; ++generation) {
  // 从当前分布采样生成种群
  for (每个个体 p = 0; p < population_size; ++p):
    for (每个参数 v = 0; v < number_of_variables; ++v):
      s[p][v] ~ N(0, 1)  // 标准正态分布采样
      population[p][v] = σ[v] * s[p][v] + μ[v]  // 变换到目标分布
}
```

**数学公式**：
```
θ_p^v ~ N(μ_v, σ_v²)
```

其中`θ_p^v`是第`p`个个体中第`v`个参数的值。

#### 步骤3：评估适应度

**位置**：`src/main_nep/fitness.cu` - `compute()`

对每个个体（参数集合），计算适应度：

```cpp
for (每个个体 p in population):
  // 1. 使用当前参数计算能量、力、维里
  potential->find_force(para, population[p], train_set, ...);
  
  // 2. 计算RMSE
  rmse_energy = sqrt(Σ_i (E_pred[i] - E_ref[i])² / N)
  rmse_force = sqrt(Σ_i |F_pred[i] - F_ref[i]|² / (3N))
  rmse_virial = sqrt(Σ_i (W_pred[i] - W_ref[i])² / 6)
  
  // 3. 计算正则化损失
  L1_loss = λ₁ · (1/n) · Σ_v |θ_v|
  L2_loss = λ₂ · sqrt((1/n) · Σ_v θ_v²)
  
  // 4. 总适应度（越小越好）
  fitness[p] = L1_loss + L2_loss 
               + λ_e · rmse_energy 
               + λ_f · rmse_force 
               + λ_v · rmse_virial
```

**数学公式**：

总损失函数：
```
L = λ₁·L₁ + λ₂·L₂ + λ_e·RMSE_E + λ_f·RMSE_F + λ_v·RMSE_V
```

其中：
- `L₁ = (1/n) Σ_v |θ_v|`：L1正则化
- `L₂ = sqrt((1/n) Σ_v θ_v²)`：L2正则化
- `RMSE_E`, `RMSE_F`, `RMSE_V`：能量、力、维里的均方根误差

#### 步骤4：排序和计算Utility

**位置**：`src/main_nep/snes.cu` - `sort_population()`

```cpp
// 按适应度排序（从小到大，适应度越小越好）
sort_population(para);

// 计算utility函数
for (int n = 0; n < population_size; ++n) {
  utility[n] = max(0, ln(population_size * 0.5 + 1) - ln(n + 1));
}
// 归一化
utility_sum = Σ utility[n];
for (int n = 0; n < population_size; ++n) {
  utility[n] = utility[n] / utility_sum - 1 / population_size;
}
```

**数学公式**：

Utility函数：
```
u_i = max(0, ln(λ/2 + 1) - ln(i + 1))
u_i = u_i / Σ_j u_j - 1/λ
```

其中`i`是排名（0为最好），`λ`是种群大小。

#### 步骤5：更新分布参数

**位置**：`src/main_nep/snes.cu` - `update_mu_and_sigma()`

```cpp
for (每个参数 v = 0; v < number_of_variables; ++v):
  // 计算梯度
  gradient_mu[v] = Σ_p (s[p][v] · utility[p])
  gradient_sigma[v] = Σ_p ((s[p][v]² - 1) · utility[p])
  
  // 更新均值
  μ[v] += σ[v] · gradient_mu[v]
  
  // 更新标准差
  σ[v] = min(σ_max, σ[v] · exp(η_σ · gradient_sigma[v]))
```

**数学公式**：

参数更新：
```
μ_v ← μ_v + σ_v · Σ_p (s_p^v · u_p)
σ_v ← min(σ_max, σ_v · exp(η_σ · Σ_p ((s_p^v)² - 1) · u_p))
```

其中：
- `s_p^v ~ N(0,1)`是标准正态采样值
- `u_p`是utility值
- `η_σ = (3 + ln(n)) / (5√n) / 2`是学习率

### 训练循环

**完整训练流程**：

```
初始化: μ, σ
for generation = 0 to max_generation:
  1. 生成种群: population ~ N(μ, σ²)
  2. 评估适应度: fitness = loss(energy, force, virial) + regularization
  3. 排序种群: 按适应度从小到大排序
  4. 计算utility: u_i = f(rank_i)
  5. 更新分布: μ, σ ← update(μ, σ, utility, population)
  6. 报告误差: 每100代输出RMSE
  7. 保存检查点: 每100代保存nep.restart
```

---

## 完整训练流程示意

### 数据流图

```
原始数据 (train.xyz)
    │
    ├─ 读取结构
    │   ├─ 原子坐标 (x, y, z)
    │   ├─ 原子类型 (type)
    │   ├─ 参考力 (fx, fy, fz)
    │   ├─ 参考能量 (energy)
    │   └─ 参考维里 (virial)
    │
    ├─ GPU数据传输
    │   └─ 复制到GPU内存
    │
    ├─ 构建邻居列表
    │   ├─ 径向邻居 (rc_radial内)
    │   └─ 角向邻居 (rc_angular内)
    │
    ├─ 计算描述符
    │   ├─ 径向描述符 q_n (2-body)
    │   │   └─ 使用参数 c_n^k
    │   └─ 角向描述符 q_{nl} (3-body+)
    │       └─ 使用参数 c_n^k
    │
    ├─ 描述符归一化
    │   └─ q[d] *= q_scaler[d]
    │
    └─ 神经网络前向传播
        ├─ 输入: q (dim维)
        ├─ 隐藏层: h = tanh(W0·q - b0)
        ├─ 输出: E = W1·h - b1
        └─ 使用参数: w0, b0, w1, b1
```

### 训练流程图

```
SNES训练循环
    │
    ├─ 初始化: μ, σ (参数分布)
    │
    └─ for generation = 0 to max_generation:
        │
        ├─ 1. 生成种群
        │   └─ population[p][v] ~ N(μ[v], σ[v]²)
        │
        ├─ 2. 评估适应度 (对每个个体)
        │   ├─ 使用population[p]作为参数
        │   ├─ 计算描述符 (使用c参数)
        │   ├─ 神经网络前向传播 (使用w0,b0,w1,b1)
        │   ├─ 计算能量、力、维里
        │   ├─ 计算RMSE (与参考值比较)
        │   ├─ 计算正则化损失
        │   └─ fitness[p] = total_loss
        │
        ├─ 3. 排序种群
        │   └─ 按fitness从小到大排序
        │
        ├─ 4. 计算utility
        │   └─ u[i] = f(rank[i])
        │
        ├─ 5. 更新参数分布
        │   ├─ μ[v] += σ[v] · Σ_p (s[p][v] · u[p])
        │   └─ σ[v] = min(σ_max, σ[v] · exp(η_σ · gradient))
        │
        ├─ 6. 报告误差
        │   └─ 每100代输出RMSE到loss.out
        │
        └─ 7. 保存检查点
            └─ 每100代保存nep.restart (μ, σ)
```

### 参数更新示意

```
训练前:
  μ = [μ₁, μ₂, ..., μₙ]  (参数均值)
  σ = [σ₁, σ₂, ..., σₙ]  (参数标准差)

第g代:
  生成种群: θ_p ~ N(μ, σ²)
  评估适应度: fitness[p]
  排序: fitness[0] < fitness[1] < ... < fitness[λ-1]
  计算utility: u[p] (排名越靠前，u越大)
  
  更新:
    μ ← μ + σ · Σ_p (s_p · u_p)
    σ ← σ · exp(η_σ · Σ_p ((s_p² - 1) · u_p))

第g+1代:
  使用更新后的μ, σ生成新种群
  ...
```

### 关键代码位置总结

| 功能 | 文件位置 | 关键函数 |
|------|---------|---------|
| 数据读取 | `src/main_nep/structure.cu` | `read_one_structure()` |
| 邻居列表 | `src/main_nep/nep.cu` | `gpu_find_neighbor_list()` |
| 径向描述符 | `src/main_nep/nep.cu` | `find_descriptors_radial()` |
| 角向描述符 | `src/main_nep/nep.cu` | `find_descriptors_angular()` |
| 神经网络 | `src/utilities/nep_utilities.cuh` | `apply_ann_one_layer()` |
| 力计算 | `src/force/nep.cu` | `find_force_radial()`, `find_force_angular()` |
| 适应度计算 | `src/main_nep/fitness.cu` | `compute()` |
| SNES优化 | `src/main_nep/snes.cu` | `compute()`, `update_mu_and_sigma()` |

---

## 总结

### 数据到神经网络的完整流程

1. **原始数据** → 读取原子坐标、类型、力、能量
2. **邻居列表** → 找到截断半径内的邻居
3. **描述符计算** → 将局部环境编码为固定维度向量
   - 使用可训练参数`c_n^k`
4. **描述符归一化** → 使用`q_scaler`缩放
5. **神经网络输入** → 归一化后的描述符向量
6. **神经网络前向传播** → 计算原子能量
   - 使用可训练参数`w0`, `b0`, `w1`, `b1`
7. **力和维里** → 通过链式法则计算

### SNES训练方法

1. **参数表示**：使用高斯分布`N(μ, σ²)`表示参数的不确定性
2. **种群采样**：从分布中采样生成候选解
3. **适应度评估**：计算每个个体的损失（RMSE + 正则化）
4. **排序选择**：按适应度排序，好的个体获得更高的utility
5. **分布更新**：根据utility更新`μ`和`σ`，使分布向好的方向移动
6. **迭代优化**：重复上述过程，直到收敛

### 训练的参数

- **神经网络参数**：`w0`, `b0`, `w1`, `b1`（通过SNES优化）
- **描述符参数**：`c_n^k`（径向和角向，通过SNES优化）
- **q_scaler**：描述符缩放因子（在训练过程中自动优化）

所有参数都通过SNES算法同时优化，目标是最小化总损失函数（能量、力、维里的RMSE + 正则化项）。

