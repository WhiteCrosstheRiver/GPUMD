# MD模拟循环详解

本文档详细记录MD模拟的核心循环`perform_a_run()`的完整执行流程。

## 文件位置

- **函数**: `src/main_gpumd/run.cu` 第215-346行 (`perform_a_run`)

## 函数签名

```cpp
void Run::perform_a_run()
```

## 完整执行流程

### 阶段1: 初始化

```cpp
integrate.initialize(time_step, atom, box, group, thermo, number_of_steps);
mc.initialize();
measure.initialize(number_of_steps, time_step, integrate, group, atom, box, force);
```

#### 1.1 积分器初始化

**功能**: 初始化积分算法

**操作**:
- 根据系综类型（NVE/NVT/NPT等）设置积分参数
- 分配必要的GPU内存
- 初始化温度、压力控制器（如果适用）

**Why**: 为MD积分做准备

#### 1.2 Monte Carlo初始化

**功能**: 初始化MC模块（如果启用）

**Why**: 为混合MC/MD模拟做准备

#### 1.3 测量模块初始化

**功能**: 初始化所有测量和输出功能

**操作**:
- 初始化所有`Property`对象（dump_*, compute_*）
- 打开输出文件
- 分配测量所需的GPU内存

**Why**: 为后续的测量和输出做准备

### 阶段2: 首次力计算

```cpp
const auto time_begin = std::chrono::high_resolution_clock::now();

// compute force for the first integrate step
if (integrate.type >= 31) { // PIMD
  for (int k = 0; k < integrate.number_of_beads; ++k) {
    force.compute(
      box,
      atom.position_beads[k],
      atom.type,
      group,
      atom.potential_beads[k],
      atom.force_beads[k],
      atom.virial_beads[k],
      atom.velocity_beads[k],
      atom.mass);
  }
} else {
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom,
    atom.velocity_per_atom,
    atom.mass);
}
```

**功能**: 计算初始位置的力

**Why**: 
- 大多数积分算法需要初始力来启动
- Verlet算法在第一步需要力来更新速度

**PIMD处理**: 如果是路径积分分子动力学（PIMD），需要对每个bead计算力

### 阶段3: MD主循环

```cpp
double initial_time_step = time_step;

for (int step = 0; step < number_of_steps; ++step) {
  // 循环体
}
```

#### 3.1 速度修正（可选）

```cpp
velocity.correct_velocity(
  step,
  group,
  atom.cpu_mass,
  atom.position_per_atom,
  atom.cpu_position_per_atom,
  atom.cpu_velocity_per_atom,
  atom.velocity_per_atom);
```

**功能**: 定期修正线性动量和角动量

**执行条件**: 如果`correct_velocity`命令被设置，且`step % velocity_correction_interval == 0`

**操作**:
1. 将速度和位置复制到CPU
2. 计算总线性动量并修正
3. 计算总角动量并修正
4. 将修正后的速度复制回GPU

**Why**: 
- 防止系统整体移动（零线性动量）
- 防止系统整体旋转（零角动量）
- 保持系统的质心固定

#### 3.2 时间步长调整（可选）

```cpp
calculate_time_step(
  max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
global_time += time_step;
```

**功能**: 根据最大速度限制调整时间步长

**执行条件**: 如果`time_step`命令指定了`max_distance_per_step`

**操作**:
1. 在GPU上找到最大速度的平方
2. 计算所需的最小时间步长：`time_step_min = max_distance_per_step / v_max`
3. 如果`time_step_min < initial_time_step`，则使用`time_step_min`

**Why**: 防止原子在一步内移动过远，保证数值稳定性

#### 3.3 积分第一步：更新位置和部分速度

```cpp
integrate.current_step = step;
integrate.compute1(time_step, double(step) / number_of_steps, group, box, atom, thermo);
```

**功能**: 执行积分算法的第一步

**操作**（以Verlet算法为例）:
1. **更新位置**: `r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2`
2. **部分更新速度**: `v(t+dt/2) = v(t) + 0.5*a(t)*dt`

**系综处理**:
- **NVE**: 简单Verlet积分
- **NVT**: Nose-Hoover链，更新位置和部分速度
- **NPT**: 同时更新盒子大小

**Why**: 这是大多数积分算法的第一步，需要在新位置计算力

#### 3.4 力计算

```cpp
if (integrate.type >= 31) { // PIMD
  for (int k = 0; k < integrate.number_of_beads; ++k) {
    force.compute(
      box,
      atom.position_beads[k],
      atom.type,
      group,
      atom.potential_beads[k],
      atom.force_beads[k],
      atom.virial_beads[k],
      atom.velocity_beads[k],
      atom.mass);
  }
} else {
  force.compute(
    box,
    atom.position_per_atom,
    atom.type,
    group,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom,
    atom.velocity_per_atom,
    atom.mass);
}
```

**功能**: 根据新位置计算所有原子的力

**详细流程**: 参见 [力计算详解](05_force_computation.md)

**输出**:
- `force_per_atom[N*3]`: 每个原子的力
- `potential_per_atom[N]`: 每个原子的势能
- `virial_per_atom[N*9]`: 每个原子的维里张量

**Why**: 力是驱动MD模拟的核心，用于更新速度

#### 3.5 电子阻止（可选）

```cpp
electron_stop.compute(time_step, atom);
```

**功能**: 模拟电子阻止效应

**执行条件**: 如果`electron_stop`命令被设置

**操作**:
- 根据原子速度和类型计算电子阻止力
- 将阻止力添加到总力中

**Why**: 模拟离子注入等过程中的电子阻止效应

#### 3.6 添加外力（可选）

```cpp
add_force.compute(step, group, atom);
```

**功能**: 对特定原子组添加恒定外力

**执行条件**: 如果`add_force`命令被设置

**Why**: 模拟外力作用（如拉伸、压缩）

#### 3.7 添加随机力（可选）

```cpp
add_random_force.compute(step, atom);
```

**功能**: 添加随机力（Langevin动力学）

**执行条件**: 如果`add_random_force`命令被设置

**操作**:
- 生成随机力（高斯分布）
- 添加到总力中

**Why**: 实现Langevin动力学或温度控制

#### 3.8 添加电场（可选）

```cpp
add_efield.compute(step, group, atom, force);
```

**功能**: 对特定原子组添加电场力

**执行条件**: 如果`add_efield`命令被设置

**操作**:
- 根据电场强度和原子电荷计算电场力
- 添加到总力中

**Why**: 模拟电场作用

#### 3.9 积分第二步：完成速度更新

```cpp
integrate.compute2(time_step, double(step) / number_of_steps, group, box, atom, thermo, force);
```

**功能**: 执行积分算法的第二步

**操作**（以Verlet算法为例）:
1. **完成速度更新**: `v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt`
2. **系综控制**: 
   - NVT: 通过Nose-Hoover链控制温度
   - NPT: 控制压力和体积

**Why**: 完成一个完整的积分步骤

#### 3.10 Monte Carlo操作（可选）

```cpp
mc.compute(step, number_of_steps, atom, box, group);
```

**功能**: 执行Monte Carlo操作

**执行条件**: 如果`mc`命令被设置

**操作**:
- 可能包括原子交换、删除等操作

**Why**: 实现混合MC/MD模拟

#### 3.11 测量和输出

```cpp
measure.process(
  number_of_steps,
  step,
  integrate.fixed_group,
  integrate.move_group,
  global_time,
  integrate.temperature2,
  integrate,
  box,
  group,
  thermo,
  atom,
  force);
```

**功能**: 计算和输出各种物理量

**操作**:
- 遍历所有`Property`对象
- 根据设定的间隔执行测量
- 输出结果到文件

**详细流程**: 参见 [测量和输出](07_measurement.md)

#### 3.12 进度输出

```cpp
int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
if (0 == (step + 1) % base) {
  printf("    %d steps completed.\n", step + 1);
  fflush(stdout);
}
```

**功能**: 每完成10%的步数输出一次进度

**Why**: 让用户了解模拟进度

### 阶段4: 清理和统计

```cpp
print_line_1();
const auto time_finish = std::chrono::high_resolution_clock::now();
const std::chrono::duration<double> time_used = time_finish - time_begin;

printf("Time used for this run = %g second.\n", time_used.count());
double run_speed = atom.number_of_atoms * (number_of_steps * 1.0 / time_used.count());
printf("Speed of this run = %g atom*step/second.\n", run_speed);
print_line_2();
```

**功能**: 输出本次运行的统计信息

**输出**:
- 总耗时（秒）
- 运行速度（原子·步/秒）

**Why**: 评估模拟性能和效率

### 阶段5: 最终化

```cpp
measure.finalize(atom, box, integrate, number_of_steps, time_step, integrate.temperature2);

electron_stop.finalize();
add_force.finalize();
add_random_force.finalize();
add_efield.finalize();
integrate.finalize();
mc.finalize();
velocity.finalize();
force.finalize();
max_distance_per_step = 0.0;
```

**功能**: 清理所有模块的资源

**操作**:
- 关闭输出文件
- 释放GPU内存
- 重置状态变量

**Why**: 确保资源正确释放，为下一次`run`命令做准备

## MD循环流程图

```
开始
  │
  ├─ 初始化 (integrate, mc, measure)
  │
  ├─ 首次力计算
  │
  └─ MD循环 (for step = 0; step < number_of_steps; ++step)
      │
      ├─ 速度修正 (可选)
      │
      ├─ 时间步长调整 (可选)
      │
      ├─ integrate.compute1() ──→ 更新位置和部分速度
      │
      ├─ force.compute() ──────→ 计算力
      │
      ├─ electron_stop.compute() (可选)
      ├─ add_force.compute() (可选)
      ├─ add_random_force.compute() (可选)
      ├─ add_efield.compute() (可选)
      │
      ├─ integrate.compute2() ──→ 完成速度更新和系综控制
      │
      ├─ mc.compute() (可选)
      │
      └─ measure.process() ─────→ 测量和输出
      │
  └─ 清理和统计
  │
结束
```

## 关键时间点

1. **循环开始**: `time_begin` (第221行)
2. **每次迭代**: `step` 从0到`number_of_steps-1`
3. **循环结束**: `time_finish` (第327行)
4. **总耗时**: `time_used` (第328行)

## 积分算法示例

### Verlet算法（NVE系综）

**第一步** (`compute1`):
```cpp
r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
v(t+dt/2) = v(t) + 0.5*a(t)*dt
```

**第二步** (`compute2`):
```cpp
v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
```

### Nose-Hoover算法（NVT系综）

**扩展系统**: 引入虚拟粒子（thermostat）

**第一步**: 更新位置和部分速度，同时更新thermostat变量

**第二步**: 完成速度更新，通过thermostat控制温度

## 性能优化

### GPU并行化

- **力计算**: 所有原子的力并行计算
- **积分**: 位置和速度更新并行进行
- **测量**: 大多数测量在GPU上并行计算

### 内存管理

- 数据主要在GPU上，减少CPU-GPU传输
- 只在必要时（如速度修正）进行数据传输

### 邻居列表

- 力计算使用邻居列表，避免重复搜索
- 定期更新邻居列表（根据原子移动距离）

## 相关文档

- [力计算详解](05_force_computation.md) - 力计算的详细实现
- [积分器详解](06_integration.md) - 各种积分算法的实现
- [测量和输出](07_measurement.md) - 测量功能的详细说明

