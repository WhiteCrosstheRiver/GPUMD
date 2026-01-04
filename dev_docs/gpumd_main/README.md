# GPUMD主程序开发者文档

本文档详细记录了GPUMD主程序的执行逻辑、依赖关系和实现细节，重点关注分子动力学模拟的完整工作流程。

## 文档导航

### 核心文档

1. **[主程序执行流程](01_main_execution_flow.md)** - `main.cu`函数的完整执行流程分析
2. **[Run类初始化详解](02_run_initialization.md)** - 位置初始化、内存分配和速度初始化
3. **[run.in文件解析](03_run_in_parsing.md)** - 命令解析和执行机制
4. **[MD模拟循环详解](04_md_simulation_loop.md)** - 完整的MD步进循环流程

### 核心组件

5. **[力计算详解](05_force_computation.md)** - Force类的实现和势函数调用
6. **[积分器详解](06_integration.md)** - 各种系综的积分算法实现
7. **[测量和输出](07_measurement.md)** - 各种物理量的计算和输出

### 数据结构

8. **[Atom数据结构](08_atom_structure.md)** - 原子数据在CPU和GPU上的组织
9. **[Box数据结构](09_box_structure.md)** - 周期性边界条件的实现
10. **[Group数据结构](10_group_structure.md)** - 原子分组机制

## GPUMD执行总体逻辑图

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
    ├─ print_compile_information()
    │   ├─ 输入: 无
    │   ├─ 输出: 编译信息（CUDA版本、编译选项等）
    │   └─ Why: 帮助诊断编译相关问题
    │
    ├─ print_gpu_information()
    │   ├─ 输入: 无
    │   ├─ 输出: GPU设备信息（数量、名称、计算能力、内存、SM数量）
    │   └─ Why: 检测可用GPU资源，验证硬件环境
    │
    └─ print_line_1/2()
        ├─ 输入: 无
        ├─ 输出: 分隔线
        └─ Why: 格式化输出，提高可读性

┌─────────────────────────────────────────────────────────────────┐
│ 步骤2: Run对象构造和初始化                                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Run::Run()
    │   ├─ 输入: 无（从model.xyz和势函数文件读取）
    │   ├─ 输出: 初始化的Run对象（包含所有模拟数据）
    │   └─ Why: 统一管理整个MD模拟的所有组件和数据
    │
    │   ├─ initialize_position()
    │   │   ├─ 输入: model.xyz文件, 势函数文件
    │   │   ├─ 输出: 原子位置、类型、质量、速度、分组信息
    │   │   └─ Why: 读取初始结构，建立原子数据
    │   │
    │   │   ├─ read_xyz_line_1()
    │   │   │   ├─ 输入: model.xyz第一行
    │   │   │   ├─ 输出: 原子数量N
    │   │   │   └─ Why: 确定系统大小
    │   │   │
    │   │   ├─ read_xyz_line_2()
    │   │   │   ├─ 输入: model.xyz第二行（盒子信息）
    │   │   │   ├─ 输出: 盒子矩阵h, 列数, 属性偏移
    │   │   │   └─ Why: 设置周期性边界条件
    │   │   │
    │   │   ├─ read_xyz_in_line_3()
    │   │   │   ├─ 输入: model.xyz后续N行
    │   │   │   ├─ 输出: 每个原子的符号、位置、速度、分组标签
    │   │   │   └─ Why: 读取每个原子的详细信息
    │   │   │
    │   │   └─ find_type_size()
    │   │       ├─ 输入: 原子类型数组
    │   │       ├─ 输出: 每种类型的原子数量
    │   │       └─ Why: 统计各类型原子数量，用于后续处理
    │   │
    │   ├─ allocate_memory_gpu()
    │   │   ├─ 输入: CPU数据（位置、类型、质量等）
    │   │   ├─ 输出: GPU内存分配和数据复制
    │   │   └─ Why: 为GPU计算准备数据，包括力、势能、维里等
    │   │
    │   │   ├─ 分配原子数据: type, mass, charge, position, velocity
    │   │   │   └─ Why: 原子基本属性需要在GPU上
    │   │   │
    │   │   ├─ 分配计算数据: force, virial, potential
    │   │   │   └─ Why: 力计算的结果存储在GPU上
    │   │   │
    │   │   ├─ 分配分组数据: group label, size, contents
    │   │   │   └─ Why: 分组信息用于选择性操作
    │   │   │
    │   │   └─ 分配热力学量: thermo (12个元素)
    │   │       └─ Why: 存储温度、压力等热力学量
    │   │
    │   └─ velocity.initialize()
    │       ├─ 输入: 温度、质量、位置（可选）
    │       ├─ 输出: 初始速度（Maxwell-Boltzmann分布）
    │       └─ Why: 根据温度初始化速度，启动MD模拟
    │
    └─ execute_run_in()
        ├─ 输入: run.in文件
        ├─ 输出: 解析并执行所有命令
        └─ Why: 执行用户定义的模拟流程

┌─────────────────────────────────────────────────────────────────┐
│ 步骤3: 解析和执行run.in文件                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Run::execute_run_in()
    │   ├─ 输入: run.in文件
    │   ├─ 输出: 所有命令已解析和执行
    │   └─ Why: 按顺序执行用户配置的模拟步骤
    │
    │   ├─ 读取run.in文件
    │   │   └─ Why: 获取用户定义的模拟流程
    │   │
    │   └─ 循环解析每个关键字
    │       ├─ get_tokens() - 读取一行并分词
    │       │   └─ Why: 解析命令和参数
    │       │
    │       └─ parse_one_keyword() - 解析单个命令
    │           ├─ 输入: tokens数组
    │           ├─ 输出: 执行对应操作
    │           └─ Why: 根据命令类型执行相应功能
    │
    │           ├─ "potential" - 加载势函数
    │           │   └─ Why: 指定用于计算的势函数文件
    │           │
    │           ├─ "replicate" - 复制系统
    │           │   └─ Why: 扩大模拟系统尺寸
    │           │
    │           ├─ "deposit" - 沉积原子
    │           │   └─ Why: 动态添加原子到系统
    │           │
    │           ├─ "delete" - 删除原子
    │           │   └─ Why: 动态移除原子
    │           │
    │           ├─ "minimize" - 能量最小化
    │           │   └─ Why: 优化初始结构
    │           │
    │           ├─ "velocity" - 设置速度
    │           │   └─ Why: 指定初始温度或速度
    │           │
    │           ├─ "ensemble" - 设置系综
    │           │   └─ Why: 指定NVE/NVT/NPT等系综
    │           │
    │           ├─ "time_step" - 设置时间步长
    │           │   └─ Why: 控制积分精度和稳定性
    │           │
    │           ├─ "run" - 执行MD模拟
    │           │   └─ Why: 启动MD循环
    │           │
    │           └─ "dump_*" / "compute_*" - 测量和输出
    │               └─ Why: 计算和保存物理量

┌─────────────────────────────────────────────────────────────────┐
│ 步骤4: MD模拟循环 (当执行"run"命令时)                           │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Run::perform_a_run()
    │   ├─ 输入: number_of_steps, time_step, ensemble等
    │   ├─ 输出: 模拟轨迹和测量结果
    │   └─ Why: 执行核心的MD模拟循环
    │
    │   ├─ 初始化阶段
    │   │   ├─ integrate.initialize()
    │   │   │   ├─ 输入: time_step, atom, box, group, thermo
    │   │   │   ├─ 输出: 积分器准备就绪
    │   │   │   └─ Why: 初始化积分算法（Verlet, Nose-Hoover等）
    │   │   │
    │   │   ├─ mc.initialize()
    │   │   │   ├─ 输入: 无
    │   │   │   ├─ 输出: Monte Carlo模块准备就绪
    │   │   │   └─ Why: 初始化MC操作（如果需要）
    │   │   │
    │   │   ├─ measure.initialize()
    │   │   │   ├─ 输入: number_of_steps, time_step, integrate等
    │   │   │   ├─ 输出: 测量模块准备就绪
    │   │   │   └─ Why: 初始化所有测量和输出功能
    │   │   │
    │   │   └─ force.compute() [首次计算]
    │   │       ├─ 输入: position, type, box, group
    │   │       ├─ 输出: force, potential, virial
    │   │       └─ Why: 计算初始力，为积分做准备
    │   │
    │   └─ MD循环 (for step = 0; step < number_of_steps; ++step)
    │       │
    │       ├─ velocity.correct_velocity() [可选]
    │       │   ├─ 输入: step, group, mass, position, velocity
    │       │   ├─ 输出: 修正后的速度
    │       │   └─ Why: 修正线性动量和角动量（每N步）
    │       │
    │       ├─ calculate_time_step() [可选]
    │       │   ├─ 输入: max_distance_per_step, velocity
    │       │   ├─ 输出: 调整后的time_step
    │       │   └─ Why: 根据最大速度限制步长，保证稳定性
    │       │
    │       ├─ global_time += time_step
    │       │   └─ Why: 更新全局时间
    │       │
    │       ├─ integrate.compute1()
    │       │   ├─ 输入: time_step, progress, group, box, atom, thermo
    │       │   ├─ 输出: 更新的position, velocity (第一步)
    │       │   └─ Why: 积分算法的第一步（更新位置和部分速度）
    │       │
    │       ├─ force.compute()
    │       │   ├─ 输入: position, type, box, group
    │       │   ├─ 输出: force, potential, virial
    │       │   └─ Why: 根据新位置计算力
    │       │
    │       │   ├─ 构建邻居列表 (如果需要)
    │       │   │   └─ Why: 确定每个原子的邻居，用于力计算
    │       │   │
    │       │   ├─ 调用势函数计算
    │       │   │   ├─ NEP/Tersoff/SW/等
    │       │   │   └─ Why: 根据势函数类型计算能量和力
    │       │   │
    │       │   └─ 计算维里张量
    │       │       └─ Why: 用于压力计算
    │       │
    │       ├─ electron_stop.compute() [可选]
    │       │   ├─ 输入: time_step, atom
    │       │   ├─ 输出: 修正的force
    │       │   └─ Why: 模拟电子阻止效应
    │       │
    │       ├─ add_force.compute() [可选]
    │       │   ├─ 输入: step, group, atom
    │       │   ├─ 输出: 添加的force
    │       │   └─ Why: 对特定原子组添加外力
    │       │
    │       ├─ add_random_force.compute() [可选]
    │       │   ├─ 输入: step, atom
    │       │   ├─ 输出: 随机力
    │       │   └─ Why: 添加随机力（Langevin动力学）
    │       │
    │       ├─ add_efield.compute() [可选]
    │       │   ├─ 输入: step, group, atom, force
    │       │   ├─ 输出: 电场力
    │       │   └─ Why: 添加电场作用
    │       │
    │       ├─ integrate.compute2()
    │       │   ├─ 输入: time_step, progress, group, box, atom, thermo, force
    │       │   ├─ 输出: 更新的velocity, box (第二步)
    │       │   └─ Why: 积分算法的第二步（完成速度更新，处理系综）
    │       │
    │       ├─ mc.compute() [可选]
    │       │   ├─ 输入: step, number_of_steps, atom, box, group
    │       │   ├─ 输出: 可能的原子交换或删除
    │       │   └─ Why: 执行Monte Carlo操作
    │       │
    │       └─ measure.process()
    │           ├─ 输入: step, global_time, integrate, box, group, thermo, atom, force
    │           ├─ 输出: 各种测量结果和输出文件
    │           └─ Why: 计算和输出物理量
    │           │
    │           ├─ dump_thermo - 输出热力学量
    │           │   └─ Why: 记录温度、压力、能量等
    │           │
    │           ├─ dump_position/velocity/force - 输出轨迹
    │           │   └─ Why: 保存原子位置、速度、力
    │           │
    │           ├─ compute_rdf - 径向分布函数
    │           │   └─ Why: 分析结构
    │           │
    │           ├─ compute_msd - 均方位移
    │           │   └─ Why: 分析扩散
    │           │
    │           └─ 其他测量...
    │
    └─ 清理阶段
        ├─ measure.finalize()
        │   └─ Why: 完成所有测量，输出最终结果
        │
        ├─ electron_stop.finalize()
        ├─ add_force.finalize()
        ├─ add_random_force.finalize()
        ├─ add_efield.finalize()
        ├─ integrate.finalize()
        ├─ mc.finalize()
        ├─ velocity.finalize()
        └─ force.finalize()
            └─ Why: 清理所有模块的资源

┌─────────────────────────────────────────────────────────────────┐
│ 步骤5: 程序结束                                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ 输出总耗时
    ├─ 输出"Finished running GPUMD."
    └─ 返回 EXIT_SUCCESS
```

### 关键数据流

#### 输入文件
- `model.xyz`: 初始原子结构（位置、类型、速度、分组）
- `run.in`: 模拟配置文件（命令序列）
- 势函数文件: NEP/Tersoff/SW等势函数参数

#### 输出文件
- `out.xyz`: 轨迹文件（如果启用dump_xyz）
- `thermo.out`: 热力学量输出（如果启用dump_thermo）
- `rdf.out`: 径向分布函数（如果启用compute_rdf）
- 其他各种测量输出文件

### 为什么使用这些函数？

1. **模块化设计**: 每个类负责特定功能，便于维护和扩展
2. **GPU加速**: 所有计算在GPU上并行执行，大幅提升速度
3. **灵活配置**: 通过run.in文件灵活配置模拟流程
4. **多种势函数**: 支持NEP、Tersoff、SW等多种势函数
5. **多种系综**: 支持NVE、NVT、NPT等多种系综
6. **丰富的测量**: 提供多种物理量的计算和输出
7. **动态操作**: 支持沉积、删除等动态操作

## 程序整体架构

### 执行流程概览

```
main() 
  ├─ 打印欢迎信息和GPU信息
  └─ Run::Run()
      ├─ 初始化位置和内存
      ├─ 初始化速度
      └─ execute_run_in()
          └─ 解析run.in并执行命令
              └─ perform_a_run() [当遇到"run"命令]
                  └─ MD循环
                      ├─ integrate.compute1()
                      ├─ force.compute()
                      ├─ integrate.compute2()
                      └─ measure.process()
```

### 核心类依赖关系

```
main.cu
  └─ Run (run.cuh/cu)
      ├─ Atom (model/atom.cuh/cu)
      ├─ Box (model/box.cuh/cu)
      ├─ Group (model/group.cuh/cu)
      ├─ Force (force/force.cuh/cu)
      │   └─ 各种势函数 (NEP, Tersoff, SW等)
      ├─ Integrate (integrate/integrate.cuh/cu)
      │   └─ 各种系综 (NVE, NVT, NPT等)
      ├─ Measure (measure/measure.cuh/cu)
      │   └─ 各种Property (Dump_Thermo, Compute_RDF等)
      ├─ Velocity (velocity.cuh/cu)
      ├─ MC (mc/mc.cuh/cu)
      ├─ Electron_Stop (electron_stop.cuh/cu)
      ├─ Add_Force (add_force.cuh/cu)
      ├─ Add_Random_Force (add_random_force.cuh/cu)
      └─ Add_Efield (add_efield.cuh/cu)
```

## 主要文件位置

### 主程序
- **入口点**: `src/main_gpumd/main.cu` (第29-57行)

### 核心类
- **Run**: `src/main_gpumd/run.cuh` / `run.cu`
- **Force**: `src/force/force.cuh` / `force.cu`
- **Integrate**: `src/integrate/integrate.cuh` / `integrate.cu`
- **Measure**: `src/measure/measure.cuh` / `measure.cu`

### 数据模型
- **Atom**: `src/model/atom.cuh` / `atom.cu`
- **Box**: `src/model/box.cuh` / `box.cu`
- **Group**: `src/model/group.cuh` / `group.cu`

### 工具函数
- **GPU信息**: `src/utilities/main_common.cuh` / `main_common.cu`
- **错误处理**: `src/utilities/error.cuh` / `error.cu`
- **读取文件**: `src/model/read_xyz.cuh` / `read_xyz.cu`

## 关键概念

### MD模拟步骤
1. **初始化**: 读取结构、分配内存、初始化速度
2. **力计算**: 根据位置计算所有原子的力
3. **积分**: 根据力更新位置和速度
4. **测量**: 计算和输出各种物理量
5. **重复**: 重复步骤2-4直到达到指定步数

### 积分算法
- **Verlet**: 基本的NVE系综积分
- **Nose-Hoover**: NVT和NPT系综的积分
- **PIMD**: 路径积分分子动力学

### 势函数类型
- **NEP**: 神经网络势函数
- **Tersoff**: 三体势函数
- **SW**: Stillinger-Weber势函数
- **其他**: LJ、EAM等

## 使用建议

1. **理解整体流程**: 先阅读 [主程序执行流程](01_main_execution_flow.md)
2. **理解初始化**: 阅读 [Run类初始化详解](02_run_initialization.md)
3. **理解命令解析**: 阅读 [run.in文件解析](03_run_in_parsing.md)
4. **理解MD循环**: 阅读 [MD模拟循环详解](04_md_simulation_loop.md)
5. **深入组件**: 查看各个组件的详细文档

## 版本信息

- **GPUMD版本**: 4.5
- **文档创建日期**: 2024
- **适用版本**: GPUMD 4.x

