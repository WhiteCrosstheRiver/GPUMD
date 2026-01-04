# run.in文件解析

本文档详细记录`run.in`文件的解析机制和所有支持的命令。

## 文件位置

- **解析函数**: `src/main_gpumd/run.cu` 第179-213行 (`execute_run_in`)
- **命令解析**: `src/main_gpumd/run.cu` 第348-595行 (`parse_one_keyword`)

## 解析流程

### execute_run_in()

**位置**: `src/main_gpumd/run.cu` 第179-213行

```cpp
void Run::execute_run_in()
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  std::ifstream input("run.in");
  if (!input.is_open()) {
    std::cout << "Failed to open run.in." << std::endl;
    exit(1);
  }

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (tokens_without_comments.size() > 0) {
      parse_one_keyword(tokens_without_comments);
    }
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  input.close();
}
```

**执行步骤**:

1. **打开run.in文件**
   - 如果文件不存在，程序退出

2. **循环读取每一行**
   - 使用`get_tokens()`将行分割为tokens（空格分隔）
   - 忽略以`#`开头的注释

3. **解析并执行命令**
   - 调用`parse_one_keyword()`解析每个命令
   - 根据命令类型执行相应操作

## 命令解析机制

### parse_one_keyword()

**位置**: `src/main_gpumd/run.cu` 第348-595行

**功能**: 根据命令关键字执行相应操作

**基本结构**:

```cpp
void Run::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param)
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");
  
  const char* param[max_num_param];
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }

  if (strcmp(param[0], "potential") == 0) {
    // 处理potential命令
  } else if (strcmp(param[0], "run") == 0) {
    // 处理run命令
  } else if ... {
    // 其他命令
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}
```

## 支持的命令

### 1. potential - 加载势函数

**语法**: `potential <type> <filename> [parameters...]`

**功能**: 加载势函数文件

**示例**:
```
potential 1 model.txt
potential 2 tersoff.txt
```

**处理**: `force.parse_potential(param, num_param, box, atom.type.size())`

**Why**: 指定用于力计算的势函数类型和参数文件

### 2. replicate - 复制系统

**语法**: `replicate <nx> <ny> <nz>`

**功能**: 将系统在x、y、z方向分别复制nx、ny、nz倍

**示例**:
```
replicate 2 2 2
```

**处理**: `Replicate(param, num_param, box, atom, group)`

**后续操作**: `allocate_memory_gpu()` - 重新分配内存

**Why**: 扩大模拟系统尺寸，用于研究更大系统或减少边界效应

### 3. deposit - 沉积原子

**语法**: `deposit <style> <parameters...>`

**功能**: 动态添加原子到系统

**示例**:
```
deposit element Si 100 0.0 0.0 10.0
```

**处理**: `Deposit(param, num_param, atom, group)`

**后续操作**: `allocate_memory_gpu()` - 重新分配内存

**Why**: 模拟原子沉积过程（如物理气相沉积）

### 4. delete - 删除原子

**语法**: `delete <style> <parameters...>`

**功能**: 动态删除系统中的原子

**示例**:
```
delete element Si
delete cubic 0.0 0.0 0.0 10.0 10.0 10.0
```

**处理**: `Delete(param, num_param, atom, group)`

**后续操作**: `allocate_memory_gpu()` - 重新分配内存

**Why**: 模拟原子移除过程

### 5. minimize - 能量最小化

**语法**: `minimize <method> <parameters...>`

**功能**: 优化原子位置，使系统能量最小

**示例**:
```
minimize cg 1.0e-6 1000
```

**处理**: 创建`Minimize`对象并调用`parse_minimize()`

**Why**: 优化初始结构，去除不合理的原子重叠或高能构型

### 6. compute_phonon - 计算声子

**语法**: `compute_phonon <parameters...>`

**功能**: 计算声子色散关系

**处理**: 创建`Hessian`对象并调用`compute()`

**Why**: 分析晶格动力学性质

### 7. compute_cohesive - 计算内聚能

**语法**: `compute_cohesive <parameters...>`

**功能**: 计算系统的内聚能

**处理**: 创建`Cohesive`对象并调用`compute()`

**Why**: 评估材料稳定性

### 8. compute_elastic - 计算弹性常数

**语法**: `compute_elastic <parameters...>`

**功能**: 计算弹性常数矩阵

**处理**: 创建`Cohesive`对象（mode=1）并调用`compute()`

**Why**: 分析材料的力学性质

### 9. change_box - 改变盒子

**语法**: `change_box <dx> [dy] [dz] [yz] [xz] [xy]`

**功能**: 改变周期性边界条件的盒子大小

**示例**:
```
change_box 1.0
change_box 1.0 1.0 1.0
change_box 1.0 1.0 1.0 0.0 0.0 0.0
```

**处理**: `parse_change_box(param, num_param)`

**Why**: 模拟体积变化（如等静压压缩）

### 10. velocity - 设置速度

**语法**: `velocity <temperature> [seed <seed_number>]`

**功能**: 根据温度初始化或重新初始化速度

**示例**:
```
velocity 300
velocity 500 seed 12345
```

**处理**: `parse_velocity(param, num_param)`

**Why**: 设置或改变系统的初始温度

### 11. ensemble - 设置系综

**语法**: `ensemble <type> <parameters...>`

**功能**: 设置MD模拟的系综类型

**示例**:
```
ensemble nve
ensemble nvt 300 300 100
ensemble npt 300 300 100 1.0 1.0 100
```

**处理**: `integrate.parse_ensemble(param, num_param, time_step, atom, box, group, thermo)`

**Why**: 指定模拟的统计系综（NVE/NVT/NPT等）

### 12. time_step - 设置时间步长

**语法**: `time_step <dt> [max_distance]`

**功能**: 设置MD模拟的时间步长

**示例**:
```
time_step 1.0
time_step 1.0 0.1
```

**处理**: `parse_time_step(param, num_param)`

**Why**: 控制积分精度和稳定性

### 13. correct_velocity - 速度修正

**语法**: `correct_velocity <interval> [group_method]`

**功能**: 定期修正线性动量和角动量

**示例**:
```
correct_velocity 100
correct_velocity 100 0
```

**处理**: `parse_correct_velocity(param, num_param, group)`

**Why**: 防止系统整体移动或旋转

### 14. run - 执行MD模拟

**语法**: `run <number_of_steps>`

**功能**: 执行指定步数的MD模拟

**示例**:
```
run 10000
```

**处理**: `parse_run(param, num_param)` → `perform_a_run()`

**Why**: 启动核心的MD模拟循环

**详细流程**: 参见 [MD模拟循环详解](04_md_simulation_loop.md)

### 15. dump_* - 输出命令

#### dump_thermo - 输出热力学量

**语法**: `dump_thermo <interval> <filename>`

**功能**: 定期输出温度、压力、能量等热力学量

**示例**:
```
dump_thermo 100 thermo.out
```

#### dump_position - 输出位置

**语法**: `dump_position <interval> <filename> [group_id]`

**功能**: 定期输出原子位置

#### dump_velocity - 输出速度

**语法**: `dump_velocity <interval> <filename> [group_id]`

**功能**: 定期输出原子速度

#### dump_force - 输出力

**语法**: `dump_force <interval> <filename> [group_id]`

**功能**: 定期输出原子受力

#### dump_xyz - 输出XYZ格式轨迹

**语法**: `dump_xyz <interval> <filename> [group_id]`

**功能**: 定期输出XYZ格式的轨迹文件

#### dump_restart - 输出重启文件

**语法**: `dump_restart <interval> <filename>`

**功能**: 定期输出重启文件，用于恢复模拟

#### 其他dump命令

- `dump_netcdf` - NetCDF格式输出
- `dump_exyz` - 扩展XYZ格式
- `dump_beads` - PIMD的beads输出
- `dump_observer` - 观察者输出
- `dump_dipole` - 偶极矩输出
- `dump_polarizability` - 极化率输出

**处理**: 创建对应的`Property`对象并添加到`measure.properties`

**Why**: 保存模拟轨迹和结果，用于后续分析

### 16. compute_* - 计算命令

#### compute_rdf - 径向分布函数

**语法**: `compute_rdf <parameters...>`

**功能**: 计算径向分布函数

#### compute_msd - 均方位移

**语法**: `compute_msd <parameters...>`

**功能**: 计算均方位移，用于分析扩散

#### compute_dos - 态密度

**语法**: `compute_dos <parameters...>`

**功能**: 计算态密度

#### compute_hac - 热自相关函数

**语法**: `compute_hac <parameters...>`

**功能**: 计算热自相关函数，用于热导率计算

#### compute_hnemd - HNEMD方法

**语法**: `compute_hnemd <parameters...>`

**功能**: 使用HNEMD方法计算热导率

#### compute_shc - 声子热导率

**语法**: `compute_shc <parameters...>`

**功能**: 计算声子热导率

#### 其他compute命令

- `compute_adf` - 角分布函数
- `compute_orientorder` - 取向序参数
- `compute_angular_rdf` - 角向径向分布函数
- `compute_dpdt` - 压力-温度导数
- `compute_viscosity` - 粘度
- `compute_hnemdec` - HNEMDEC方法
- `compute_gkma` - GKMA方法
- `compute_hnema` - HNEMA方法
- `compute_lsqt` - LSQT方法
- `compute_extrapolation` - 外推检测

**处理**: 创建对应的`Property`对象并添加到`measure.properties`

**Why**: 计算各种物理量，用于材料性质分析

### 17. fix - 固定原子

**语法**: `fix <group_id> <style> <parameters...>`

**功能**: 固定特定原子组的位置

**处理**: `integrate.parse_fix(param, num_param, group)`

**Why**: 模拟固定边界条件或固定某些原子

### 18. move - 移动原子

**语法**: `move <group_id> <style> <parameters...>`

**功能**: 以指定方式移动特定原子组

**处理**: `integrate.parse_move(param, num_param, group)`

**Why**: 模拟外部驱动或约束

### 19. deform - 形变

**语法**: `deform <parameters...>`

**功能**: 对系统施加形变

**处理**: `integrate.parse_deform(param, num_param)`

**Why**: 模拟拉伸、压缩等力学测试

### 20. electron_stop - 电子阻止

**语法**: `electron_stop <parameters...>`

**功能**: 启用电子阻止效应

**处理**: `electron_stop.parse(param, num_param, atom.number_of_atoms, number_of_types)`

**Why**: 模拟离子注入等过程中的电子阻止

### 21. add_random_force - 添加随机力

**语法**: `add_random_force <parameters...>`

**功能**: 添加随机力（Langevin动力学）

**处理**: `add_random_force.parse(param, num_param, atom.number_of_atoms)`

**Why**: 实现Langevin动力学或温度控制

### 22. add_force - 添加外力

**语法**: `add_force <group_id> <fx> <fy> <fz> [interval]`

**功能**: 对特定原子组添加恒定外力

**处理**: `add_force.parse(param, num_param, group)`

**Why**: 模拟外力作用

### 23. add_efield - 添加电场

**语法**: `add_efield <group_id> <Ex> <Ey> <Ez> [interval]`

**功能**: 对特定原子组添加电场

**处理**: `add_efield.parse(param, num_param, group)`

**Why**: 模拟电场作用

### 24. mc - Monte Carlo

**语法**: `mc <parameters...>`

**功能**: 启用Monte Carlo操作

**处理**: `mc.parse_mc(param, num_param, group, atom)`

**Why**: 实现混合MC/MD模拟

### 25. active - 主动学习

**语法**: `active <parameters...>`

**功能**: 启用主动学习功能

**Why**: 自动识别需要进一步训练的结构

### 26. plumed - PLUMED接口

**语法**: `plumed <parameters...>`

**功能**: 与PLUMED库集成

**Why**: 使用PLUMED进行增强采样和自由能计算

## 命令执行顺序

命令按照在`run.in`文件中出现的顺序执行：

1. **初始化命令**: `potential`, `replicate`, `deposit`, `delete`
2. **结构优化**: `minimize`
3. **计算命令**: `compute_phonon`, `compute_cohesive`, `compute_elastic`
4. **设置命令**: `velocity`, `ensemble`, `time_step`, `correct_velocity`
5. **测量设置**: `dump_*`, `compute_*`
6. **约束设置**: `fix`, `move`, `deform`
7. **外力设置**: `electron_stop`, `add_random_force`, `add_force`, `add_efield`
8. **MC设置**: `mc`
9. **执行模拟**: `run`

**注意**: 
- 一个`run.in`文件中可以有多个`run`命令
- 每个`run`命令会执行一次完整的MD循环
- 在`run`命令之间可以改变设置（如温度、系综等）

## 示例run.in文件

```
# 加载势函数
potential 1 model.txt

# 能量最小化
minimize cg 1.0e-6 1000

# 设置初始温度
velocity 300

# 设置NVT系综
ensemble nvt 300 300 100

# 设置时间步长
time_step 1.0

# 设置输出
dump_thermo 100 thermo.out
dump_xyz 1000 out.xyz

# 设置测量
compute_rdf 1 200 0.01 10.0 rdf.out

# 执行模拟
run 10000

# 改变温度
ensemble nvt 500 500 100

# 继续模拟
run 10000
```

## 相关文档

- [Run类初始化详解](02_run_initialization.md) - 初始化过程
- [MD模拟循环详解](04_md_simulation_loop.md) - run命令的执行
- [测量和输出](07_measurement.md) - dump和compute命令的详细说明

