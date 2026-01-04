# Run类初始化详解

本文档详细记录`Run`类构造函数的执行流程，包括位置初始化、内存分配和速度初始化。

## 文件位置

- **头文件**: `src/main_gpumd/run.cuh` (第38-77行)
- **实现文件**: `src/main_gpumd/run.cu` (第145-177行)

## 构造函数执行流程

### Run::Run()

**位置**: `src/main_gpumd/run.cu` 第145-177行

```cpp
Run::Run()
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  initialize_position(has_velocity_in_xyz, number_of_types, box, group, atom);

  allocate_memory_gpu(group, atom, thermo);

  velocity.initialize(
    has_velocity_in_xyz,
    300,
    atom.cpu_mass,
    atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom,
    atom.velocity_per_atom,
    false,
    123);
  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in();
}
```

## 步骤1: 初始化位置和相关参数

### initialize_position()

**位置**: `src/model/read_xyz.cu` 第482-530行

**功能**: 从`model.xyz`文件读取原子结构信息

**执行步骤**:

#### 1.1 打开文件并获取原子类型列表

```cpp
std::string filename("model.xyz");
std::ifstream input(filename);
if (!input.is_open()) {
  PRINT_INPUT_ERROR("Failed to open model.xyz.");
}

std::vector<std::string> atom_symbols;
auto filename_potential = get_filename_potential();
atom_symbols = get_atom_symbols(filename_potential);
```

**功能**:
- 打开`model.xyz`文件
- 从势函数文件（如`model.txt`）读取允许的原子类型列表
- 确定`number_of_types`

**Why**: 需要知道系统中有哪些原子类型，以便验证`model.xyz`中的原子是否合法

#### 1.2 读取第一行：原子数量

```cpp
read_xyz_line_1(input, atom.number_of_atoms);
```

**功能**: 读取第一行，获取原子数量N

**格式**: `N` (整数)

**Why**: 确定系统大小，用于后续内存分配

#### 1.3 读取第二行：盒子信息和属性定义

```cpp
int property_offset[6] = {0, 0, 0, 0, 0, 0};
int num_columns = 0;
bool has_mass = true;
bool has_charge = true;
read_xyz_line_2(
  input, box, has_velocity_in_xyz, has_mass, has_charge, 
  num_columns, property_offset, group);
```

**功能**:
- 解析属性定义字符串（如`property:species:S:pos:R:mass:R:velocity:R:group:I`）
- 读取盒子矩阵h（3x3矩阵，定义周期性边界条件）
- 确定各属性的列偏移量

**属性定义格式**:
- `species:S` - 原子符号（字符串）
- `pos:R` - 位置（3个实数：x, y, z）
- `mass:R` - 质量（实数，可选）
- `velocity:R` - 速度（3个实数：vx, vy, vz，可选）
- `group:I` - 分组标签（整数，可选，可以有多个）

**盒子矩阵h**:
- 3x3矩阵，定义周期性边界条件的基向量
- 格式：`h11 h12 h13 h21 h22 h23 h31 h32 h33`

**Why**: 
- 属性定义告诉程序如何解析后续的原子数据行
- 盒子矩阵用于周期性边界条件计算

#### 1.4 读取原子数据（N行）

```cpp
read_xyz_in_line_3(
  input,
  atom.number_of_atoms,
  has_velocity_in_xyz,
  has_mass,
  has_charge,
  num_columns,
  property_offset,
  number_of_types,
  atom_symbols,
  atom.cpu_atom_symbol,
  atom.cpu_type,
  atom.cpu_mass,
  atom.cpu_charge,
  atom.cpu_position_per_atom,
  atom.cpu_velocity_per_atom,
  group);
```

**功能**: 读取每个原子的数据

**对每个原子（n = 0 到 N-1）**:

1. **读取原子符号并转换为类型索引**
   ```cpp
   cpu_atom_symbol[n] = tokens[property_offset[0]];
   for (int t = 0; t < number_of_types; ++t) {
     if (cpu_atom_symbol[n] == atom_symbols[t]) {
       cpu_type[n] = t;
       is_allowed_element = true;
     }
   }
   ```
   - 将原子符号（如"Si", "O"）转换为类型索引（0, 1, ...）
   - 验证原子类型是否在势函数允许的列表中

2. **读取位置坐标**
   ```cpp
   for (int d = 0; d < 3; ++d) {
     cpu_position_per_atom[n + N * d] = 
       get_double_from_token(tokens[property_offset[1] + d], ...);
   }
   ```
   - 存储格式：`[x0, x1, ..., xN-1, y0, y1, ..., yN-1, z0, z1, ..., zN-1]`

3. **读取或计算质量**
   ```cpp
   if (has_mass) {
     cpu_mass[n] = get_double_from_token(tokens[property_offset[2]], ...);
   } else {
     cpu_mass[n] = MASS_TABLE.at(cpu_atom_symbol[n]);
   }
   ```
   - 如果文件中提供质量，直接使用
   - 否则从质量表（MASS_TABLE）中查找

4. **读取电荷（如果提供）**
   ```cpp
   if (has_charge) {
     cpu_charge[n] = get_double_from_token(tokens[property_offset[3]], ...);
   }
   ```

5. **读取速度（如果提供）**
   ```cpp
   if (has_velocity_in_xyz) {
     const double A_per_fs_to_natural = TIME_UNIT_CONVERSION;
     for (int d = 0; d < 3; ++d) {
       cpu_velocity_per_atom[n + N * d] = 
         get_double_from_token(tokens[property_offset[4] + d], ...) *
         A_per_fs_to_natural;
     }
   }
   ```
   - 单位转换：从 Å/fs 转换为自然单位

6. **读取分组标签**
   ```cpp
   for (int m = 0; m < group.size(); ++m) {
     group[m].cpu_label[n] = 
       get_int_from_token(tokens[property_offset[5] + m], ...);
     if ((group[m].cpu_label[n] + 1) > group[m].number) {
       group[m].number = group[m].cpu_label[n] + 1;
     }
   }
   ```
   - 每个分组方法可以有多个标签
   - 确定每个分组的最大标签值（用于确定分组数量）

**Why**: 建立完整的原子数据结构，包括位置、类型、质量、速度、分组等信息

#### 1.5 处理分组信息

```cpp
for (int m = 0; m < group.size(); ++m) {
  group[m].find_size(atom.number_of_atoms, m);
  group[m].find_contents(atom.number_of_atoms);
}
```

**功能**:
- `find_size()`: 统计每个分组标签对应的原子数量
- `find_contents()`: 建立每个分组包含的原子索引列表

**Why**: 分组信息用于后续的选择性操作（如对特定组施加力、设置温度等）

#### 1.6 统计类型数量

```cpp
find_type_size(atom.number_of_atoms, number_of_types, 
               atom.cpu_type, atom.cpu_type_size);
```

**功能**: 统计每种原子类型的数量

**输出**: `atom.cpu_type_size[t]` = 类型t的原子数量

**Why**: 用于后续的类型相关操作和统计

## 步骤2: 分配GPU内存

### allocate_memory_gpu()

**位置**: `src/model/read_xyz.cu` 第532-559行

**功能**: 为GPU计算分配内存并复制CPU数据

**执行步骤**:

#### 2.1 分配原子类型数组

```cpp
atom.type.resize(N);
atom.type.copy_from_host(atom.cpu_type.data());
```

**功能**: 将原子类型索引复制到GPU

**Why**: 力计算需要知道每个原子的类型

#### 2.2 分配分组数据

```cpp
for (int m = 0; m < group.size(); ++m) {
  group[m].label.resize(N);
  group[m].size.resize(group[m].number);
  group[m].size_sum.resize(group[m].number);
  group[m].contents.resize(N);
  group[m].label.copy_from_host(group[m].cpu_label.data());
  group[m].size.copy_from_host(group[m].cpu_size.data());
  group[m].size_sum.copy_from_host(group[m].cpu_size_sum.data());
  group[m].contents.copy_from_host(group[m].cpu_contents.data());
}
```

**功能**: 将分组信息复制到GPU

**数据结构**:
- `label[N]`: 每个原子的分组标签
- `size[G]`: 每个分组的原子数量
- `size_sum[G]`: 分组大小的前缀和（用于索引）
- `contents[N]`: 按分组组织的原子索引列表

**Why**: 分组操作（如对特定组施加力）需要在GPU上执行

#### 2.3 分配原子基本属性

```cpp
atom.mass.resize(N);
atom.mass.copy_from_host(atom.cpu_mass.data());
atom.charge.resize(N);
atom.charge.copy_from_host(atom.cpu_charge.data());
atom.position_per_atom.resize(N * 3);
atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
atom.velocity_per_atom.resize(N * 3);
atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
```

**功能**: 将质量、电荷、位置、速度复制到GPU

**Why**: 这些是力计算和积分所需的基本数据

#### 2.4 分配计算数据

```cpp
atom.force_per_atom.resize(N * 3, 0);
atom.virial_per_atom.resize(N * 9);
atom.potential_per_atom.resize(N);
thermo.resize(12);
```

**功能**: 为力、维里、势能、热力学量分配GPU内存

**数据结构**:
- `force_per_atom[N*3]`: 每个原子的力（fx, fy, fz）
- `virial_per_atom[N*9]`: 每个原子的维里张量（3x3矩阵，按行存储）
- `potential_per_atom[N]`: 每个原子的势能
- `thermo[12]`: 热力学量（温度、压力等）

**Why**: 这些是MD模拟中需要计算和更新的量

## 步骤3: 初始化速度

### velocity.initialize()

**位置**: `src/main_gpumd/velocity.cu` 第317-354行

**功能**: 初始化原子速度

**执行流程**:

#### 3.1 检查是否已有速度

```cpp
if (!has_velocity_in_xyz) {
  // 需要生成速度
} else {
  // 直接使用xyz文件中的速度
}
```

#### 3.2 生成随机速度（如果需要）

```cpp
if (use_seed) {
  get_random_velocities_by_seed(N, vx, vy, vz, seed);
} else {
  get_random_velocities(N, vx, vy, vz);
}
```

**功能**: 生成随机速度

**方法**:
- 每个方向的速度在[-1, 1]范围内随机
- 如果指定种子，使用固定种子生成（可重现）

**Why**: 为后续的修正和缩放提供初始速度

#### 3.3 修正动量和角动量

```cpp
correct_velocity(cpu_mass, cpu_position_per_atom, cpu_velocity_per_atom);
```

**功能**: 确保总线性动量和总角动量为零

**步骤**:
1. **修正线性动量**
   ```cpp
   // 计算总动量
   double momentum[3] = {0, 0, 0};
   for (int n = 0; n < N; ++n) {
     for (int d = 0; d < 3; ++d) {
       momentum[d] += mass[n] * velocity[n + d*N];
     }
   }
   // 从每个原子减去平均动量
   for (int n = 0; n < N; ++n) {
     for (int d = 0; d < 3; ++d) {
       velocity[n + d*N] -= momentum[d] / (N * mass[n]);
     }
   }
   ```

2. **修正角动量**
   ```cpp
   // 计算总角动量
   double angular_momentum[3] = {0, 0, 0};
   for (int n = 0; n < N; ++n) {
     // L = r × p
     // 计算并累加
   }
   // 通过调整速度使角动量为零
   ```

**Why**: 
- 零线性动量：系统不会整体移动
- 零角动量：系统不会整体旋转

#### 3.4 缩放到目标温度

```cpp
scale(initial_temperature, cpu_mass, vx, vy, vz);
```

**功能**: 将速度缩放到目标温度

**步骤**:
1. **计算当前温度**
   ```cpp
   double temperature = 0.0;
   for (int n = 0; n < N; ++n) {
     double v2 = vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n];
     temperature += mass[n] * v2;
   }
   temperature /= 3.0 * K_B * N;
   ```
   根据动能定理：`T = (2/3) * <E_kin> / (N * k_B)`

2. **计算缩放因子并应用**
   ```cpp
   double factor = sqrt(initial_temperature / temperature);
   for (int n = 0; n < N; ++n) {
     vx[n] *= factor;
     vy[n] *= factor;
     vz[n] *= factor;
   }
   ```

**Why**: 使系统的初始温度达到用户指定的值（默认300K）

#### 3.5 复制到GPU

```cpp
velocity_per_atom.copy_from_host(cpu_velocity_per_atom.data());
```

**功能**: 将初始化后的速度复制到GPU

**Why**: 后续的积分和力计算在GPU上进行

## 步骤4: 执行run.in文件

### execute_run_in()

**位置**: `src/main_gpumd/run.cu` 第179-213行

**功能**: 解析并执行`run.in`文件中的所有命令

**详细流程**: 参见 [run.in文件解析](03_run_in_parsing.md)

## 关键数据结构

### Atom类（CPU数据）

```cpp
class Atom {
  int number_of_atoms;
  std::vector<std::string> cpu_atom_symbol;      // 原子符号
  std::vector<int> cpu_type;                      // 类型索引
  std::vector<double> cpu_mass;                  // 质量
  std::vector<double> cpu_charge;                 // 电荷
  std::vector<double> cpu_position_per_atom;     // 位置 [N*3]
  std::vector<double> cpu_velocity_per_atom;     // 速度 [N*3]
  std::vector<int> cpu_type_size;                // 每种类型的数量
};
```

### Atom类（GPU数据）

```cpp
class Atom {
  GPU_Vector<int> type;                          // 类型索引
  GPU_Vector<double> mass;                       // 质量
  GPU_Vector<double> charge;                     // 电荷
  GPU_Vector<double> position_per_atom;          // 位置 [N*3]
  GPU_Vector<double> velocity_per_atom;          // 速度 [N*3]
  GPU_Vector<double> force_per_atom;             // 力 [N*3]
  GPU_Vector<double> virial_per_atom;            // 维里 [N*9]
  GPU_Vector<double> potential_per_atom;         // 势能 [N]
};
```

### Box类

```cpp
class Box {
  double cpu_h[9];      // 盒子矩阵（3x3，按行存储）
  GPU_Vector<double> h;  // GPU上的盒子矩阵
  GPU_Vector<double> h_inv; // 逆矩阵（用于周期性边界条件）
};
```

### Group类

```cpp
class Group {
  int number;                                    // 分组数量
  std::vector<int> cpu_label;                   // 每个原子的标签
  std::vector<int> cpu_size;                    // 每个分组的原子数
  std::vector<int> cpu_size_sum;                // 前缀和
  std::vector<int> cpu_contents;                // 分组内容（原子索引）
  
  GPU_Vector<int> label;                        // GPU上的标签
  GPU_Vector<int> size;                          // GPU上的大小
  GPU_Vector<int> size_sum;                      // GPU上的前缀和
  GPU_Vector<int> contents;                      // GPU上的内容
};
```

## 相关文档

- [主程序执行流程](01_main_execution_flow.md) - 整体流程
- [run.in文件解析](03_run_in_parsing.md) - 命令执行
- [Atom数据结构](08_atom_structure.md) - 原子数据组织
- [Box数据结构](09_box_structure.md) - 周期性边界条件
- [Group数据结构](10_group_structure.md) - 分组机制

