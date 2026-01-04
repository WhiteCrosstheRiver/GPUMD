# 主程序执行流程

本文档详细分析`main.cu`中`main()`函数的执行流程。

## 文件位置

- **源文件**: `src/main_gpumd/main.cu`
- **函数**: `int main(int argc, char* argv[])` (第29-57行)

## 完整执行流程

### 1. 程序入口和初始化信息输出

```cpp
int main(int argc, char* argv[])
{
  print_welcome_information();      // 第31行
  print_compile_information();      // 第32行
  print_gpu_information();          // 第33行
```

**功能**:
- `print_welcome_information()`: 打印欢迎信息，显示GPUMD版本和可执行文件标识
- `print_compile_information()`: 打印编译信息，包括CUDA版本、编译选项等
- `print_gpu_information()`: 检测并打印所有可用GPU的信息

**输出示例**:
```
***************************************************************
*                 Welcome to use GPUMD                        *
*     (Graphics Processing Units Molecular Dynamics)          *
*                     version 4.5                             *
*              This is the gpumd executable                   *
***************************************************************
```

### 2. 开始运行提示

```cpp
  print_line_1();
  printf("Started running GPUMD.\n");
  print_line_2();
```

**功能**: 输出开始运行的提示信息

### 3. 同步GPU并开始计时

```cpp
  CHECK(gpuDeviceSynchronize());
  const auto time_begin = std::chrono::high_resolution_clock::now();
```

**功能**:
- `gpuDeviceSynchronize()`: 确保GPU完成所有之前的操作
- `time_begin`: 记录开始时间，用于计算总耗时

### 4. 核心执行：Run对象构造

```cpp
  Run run;
```

**位置**: `src/main_gpumd/run.cu` 第145-177行

**执行步骤**:
1. **初始化位置和相关参数** (`Run::Run()` 构造函数)
   - 调用`initialize_position()` - 从`model.xyz`读取原子结构
   - 调用`allocate_memory_gpu()` - 分配GPU内存并复制数据
   - 调用`velocity.initialize()` - 初始化速度（默认300K或从xyz读取）

2. **执行run.in文件** (`execute_run_in()`)
   - 打开并解析`run.in`文件
   - 按顺序执行所有命令（potential, ensemble, run等）

**关键操作**:
- 读取`model.xyz`文件获取初始结构
- 从势函数文件确定原子类型
- 分配GPU内存用于力计算
- 根据温度初始化Maxwell-Boltzmann速度分布
- 解析并执行`run.in`中的所有命令

### 5. 同步GPU并结束计时

```cpp
  CHECK(gpuDeviceSynchronize());
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
```

**功能**:
- 确保GPU完成所有计算
- 计算总耗时

### 6. 输出耗时信息

```cpp
  print_line_1();
  printf("Time used = %f s.\n", time_used.count());
  print_line_2();
```

**功能**: 输出程序总运行时间

### 7. 程序结束

```cpp
  print_line_1();
  printf("Finished running GPUMD.\n");
  print_line_2();

  return EXIT_SUCCESS;
}
```

**功能**: 输出结束信息并返回成功状态

## 函数调用链

### 主程序调用链

```
main()
  ├─ print_welcome_information()
  │   └─ printf() - 打印欢迎信息
  ├─ print_compile_information()
  │   └─ printf() - 打印编译信息
  ├─ print_gpu_information()
  │   ├─ gpuGetDeviceCount()
  │   ├─ gpuGetDeviceProperties()
  │   └─ gpuDeviceCanAccessPeer()
  └─ Run::Run()
      ├─ initialize_position()
      │   ├─ read_xyz_line_1() - 读取原子数量
      │   ├─ read_xyz_line_2() - 读取盒子信息
      │   ├─ read_xyz_in_line_3() - 读取原子数据
      │   └─ find_type_size() - 统计类型数量
      ├─ allocate_memory_gpu()
      │   └─ GPU内存分配和数据复制
      ├─ velocity.initialize()
      │   └─ 根据温度生成速度
      └─ execute_run_in()
          └─ parse_one_keyword() - 解析每个命令
              ├─ "potential" → force.parse_potential()
              ├─ "ensemble" → integrate.parse_ensemble()
              ├─ "run" → perform_a_run()
              └─ 其他命令...
```

### Run对象构造详细流程

```
Run::Run()
  ├─ 打印初始化开始信息
  ├─ initialize_position()
  │   ├─ 打开model.xyz文件
  │   ├─ 从势函数文件获取原子类型列表
  │   ├─ 读取第一行：原子数量N
  │   ├─ 读取第二行：盒子矩阵h和属性信息
  │   ├─ 读取N行原子数据：
  │   │   ├─ 原子符号 → 转换为类型索引
  │   │   ├─ 位置坐标 (x, y, z)
  │   │   ├─ 质量 (如果提供)
  │   │   ├─ 速度 (如果提供)
  │   │   └─ 分组标签
  │   ├─ 统计每种类型的原子数量
  │   └─ 确定分组大小和内容
  │
  ├─ allocate_memory_gpu()
  │   ├─ 分配原子类型数组 (type)
  │   ├─ 分配位置数组 (position_per_atom)
  │   ├─ 分配速度数组 (velocity_per_atom)
  │   ├─ 分配质量数组 (mass)
  │   ├─ 分配电荷数组 (charge)
  │   ├─ 分配力数组 (force_per_atom)
  │   ├─ 分配势能数组 (potential_per_atom)
  │   ├─ 分配维里数组 (virial_per_atom)
  │   ├─ 分配分组数据 (label, size, contents)
  │   └─ 分配热力学量数组 (thermo)
  │
  ├─ velocity.initialize()
  │   ├─ 如果xyz中有速度，直接使用
  │   └─ 否则根据温度生成Maxwell-Boltzmann分布
  │
  └─ execute_run_in()
      ├─ 打开run.in文件
      ├─ 循环读取每一行
      ├─ 解析tokens（忽略注释）
      └─ 调用parse_one_keyword()执行命令
```

## 关键时间点

1. **程序开始**: `time_begin` (第40行)
2. **Run对象构造完成**: Run构造函数返回
3. **程序结束**: `time_finish` (第45行)
4. **总耗时**: `time_used` (第46行)

## 关键数据结构

### Run类成员变量

```cpp
class Run {
  int number_of_types;              // 原子类型数量
  int has_velocity_in_xyz = 0;      // xyz文件中是否包含速度
  int number_of_steps;              // 当前run的步数
  double global_time = 0.0;        // 全局时间（fs）
  double initial_temperature;      // 初始温度
  double time_step;                // 时间步长
  double max_distance_per_step;    // 每步最大距离限制
  
  Atom atom;                       // 原子数据
  GPU_Vector<double> thermo;      // 热力学量
  Velocity velocity;                // 速度管理
  Box box;                         // 盒子（周期性边界）
  std::vector<Group> group;        // 原子分组
  
  Force force;                     // 力计算
  Integrate integrate;             // 积分器
  MC mc;                           // Monte Carlo
  Measure measure;                 // 测量和输出
  
  Electron_Stop electron_stop;      // 电子阻止
  Add_Force add_force;             // 添加力
  Add_Random_Force add_random_force; // 随机力
  Add_Efield add_efield;            // 电场
};
```

## 输入文件

### model.xyz格式

```
N
property:species:S:pos:R:mass:R:velocity:R:group:I:...
x1 y1 z1 mass1 vx1 vy1 vz1 group1_1 group1_2 ...
x2 y2 z2 mass2 vx2 vy2 vz2 group2_1 group2_2 ...
...
```

- **第一行**: 原子数量N
- **第二行**: 属性定义和盒子信息
- **后续N行**: 每个原子的数据

### run.in格式

```
potential 1 model.txt
velocity 300
ensemble nvt 300 300 100
time_step 1.0
run 10000
```

## 相关文档

- [Run类初始化详解](02_run_initialization.md) - 初始化过程细节
- [run.in文件解析](03_run_in_parsing.md) - 命令解析机制
- [MD模拟循环详解](04_md_simulation_loop.md) - MD循环实现

