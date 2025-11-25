# Parameters类详解

本文档详细记录`Parameters`类的结构、参数含义、读取流程和计算逻辑。

## 文件位置

- **头文件**: `src/main_nep/parameters.cuh` (第21-164行)
- **实现文件**: `src/main_nep/parameters.cu` (第33-1359行)

## 类结构概览

`Parameters`类负责：
1. 从`nep.in`文件读取所有配置参数
2. 设置默认参数值
3. 验证参数合法性
4. 计算派生参数（描述符维度、变量数量等）
5. 报告所有输入参数

## 构造函数执行流程

### Parameters::Parameters()

**位置**: `src/main_nep/parameters.cu` 第33-50行

```cpp
Parameters::Parameters()
{
  print_line_1();
  printf("Started reading nep.in.\n");
  print_line_2();

  set_default_parameters();  // 设置默认值
  read_nep_in();              // 读取nep.in
  if (is_zbl_set) {
    read_zbl_in();           // 读取zbl.in（如果启用）
  }
  calculate_parameters();     // 计算派生参数
  report_inputs();            // 报告所有参数

  print_line_1();
  printf("Finished reading nep.in.\n");
  print_line_2();
}
```

## 参数分类

### 1. 基本配置参数

#### version
- **类型**: `int`
- **默认值**: `4` (NEP4)
- **可选值**: `3` (NEP3) 或 `4` (NEP4)
- **说明**: NEP版本号，NEP4为每种元素使用独立的神经网络参数

#### train_mode
- **类型**: `int`
- **默认值**: `0` (势函数)
- **可选值**:
  - `0`: 势函数训练（能量、力、维里）
  - `1`: 偶极矩训练
  - `2`: 极化率训练
  - `3`: 温度依赖自由能训练

#### prediction
- **类型**: `int`
- **默认值**: `0` (训练模式)
- **可选值**: `0` (训练) 或 `1` (预测)

#### num_types
- **类型**: `int`
- **说明**: 原子类型数量，必须通过`type`关键字设置

### 2. 截断半径参数

#### rc_radial
- **类型**: `float`
- **默认值**: `8.0` Å
- **范围**: ≤ 10.0 Å
- **说明**: 径向描述符的截断半径

#### rc_angular
- **类型**: `float`
- **默认值**: `4.0` Å
- **范围**: 2.5 ≤ rc_angular ≤ rc_radial
- **说明**: 角向描述符的截断半径

#### use_typewise_cutoff
- **类型**: `bool`
- **默认值**: `false`
- **说明**: 是否使用类型相关的截断半径

### 3. 描述符参数

#### n_max_radial
- **类型**: `int`
- **默认值**: `4`
- **范围**: 0 ≤ n_max_radial ≤ 12
- **说明**: 径向Chebyshev多项式的最大阶数

#### n_max_angular
- **类型**: `int`
- **默认值**: `4`
- **范围**: 0 ≤ n_max_angular ≤ 8
- **说明**: 角向Chebyshev多项式的最大阶数

#### L_max
- **类型**: `int`
- **默认值**: `4`
- **范围**: 0 ≤ L_max ≤ 8
- **说明**: 3体球谐函数的最大角动量量子数

#### L_max_4body
- **类型**: `int`
- **默认值**: `2`
- **可选值**: `0` 或 `2`
- **说明**: 4体描述符的最大角动量，`2`表示包含4体项

#### L_max_5body
- **类型**: `int`
- **默认值**: `0`
- **可选值**: `0` 或 `1`
- **说明**: 5体描述符的最大角动量，`1`表示包含5体项

#### basis_size_radial
- **类型**: `int`
- **默认值**: `8` (NEP3)
- **范围**: 0 ≤ basis_size_radial ≤ 16
- **说明**: 径向基函数大小（仅NEP3）

#### basis_size_angular
- **类型**: `int`
- **默认值**: `8` (NEP3)
- **范围**: 0 ≤ basis_size_angular ≤ 12
- **说明**: 角向基函数大小（仅NEP3）

### 4. 神经网络参数

#### num_neurons1
- **类型**: `int`
- **默认值**: `30`
- **范围**: 1 ≤ num_neurons1 ≤ 120
- **说明**: 隐藏层神经元数量

### 5. 损失函数权重

#### lambda_e
- **类型**: `float`
- **默认值**: `1.0`
- **范围**: ≥ 0
- **说明**: 能量RMSE损失的权重

#### lambda_f
- **类型**: `float`
- **默认值**: `1.0`
- **范围**: ≥ 0
- **说明**: 力RMSE损失的权重

#### lambda_v
- **类型**: `float`
- **默认值**: `0.1` (势函数) 或 `1.0` (偶极/极化率)
- **范围**: ≥ 0
- **说明**: 维里RMSE损失的权重

#### lambda_shear
- **类型**: `float`
- **默认值**: `1.0`
- **范围**: ≥ 0
- **说明**: 剪切维里的额外权重

#### lambda_q
- **类型**: `float`
- **默认值**: `0.1`
- **范围**: ≥ 0
- **说明**: 电荷损失的权重（仅charge_mode）

#### lambda_1
- **类型**: `float`
- **默认值**: 自动计算（见下文）
- **范围**: ≥ 0
- **说明**: L1正则化权重

#### lambda_2
- **类型**: `float`
- **默认值**: 自动计算（见下文）
- **范围**: ≥ 0
- **说明**: L2正则化权重

#### force_delta
- **类型**: `float`
- **默认值**: `0.0`
- **说明**: 用于修改力损失的参数

### 6. SNES优化参数

#### population_size
- **类型**: `int`
- **默认值**: `50`（会自动调整为GPU数量的倍数）
- **范围**: 10 ≤ population_size ≤ 200
- **说明**: SNES算法的种群大小

#### maximum_generation
- **类型**: `int`
- **默认值**: `100000`
- **范围**: 0 ≤ maximum_generation ≤ 10000000
- **说明**: 最大训练代数

#### batch_size
- **类型**: `int`
- **默认值**: `1000`
- **范围**: ≥ 1
- **说明**: 每个批次包含的配置数量

#### use_full_batch
- **类型**: `int`
- **默认值**: `0`
- **说明**: 是否启用有效全批次模式

#### initial_para
- **类型**: `float`
- **默认值**: `1.0`
- **范围**: 0.1 ≤ initial_para ≤ 1.0
- **说明**: 初始参数范围

#### sigma0
- **类型**: `float`
- **默认值**: `0.1`
- **范围**: 0.01 ≤ sigma0 ≤ 0.1
- **说明**: 初始标准差

### 7. ZBL势参数

#### enable_zbl
- **类型**: `bool`
- **默认值**: `false`
- **说明**: 是否启用ZBL势

#### flexible_zbl
- **类型**: `bool`
- **默认值**: `false`
- **说明**: 是否使用可调ZBL势（需要zbl.in文件）

#### zbl_rc_inner
- **类型**: `float`
- **默认值**: `zbl_rc_outer * 0.5`
- **范围**: 1.0 ≤ zbl_rc_outer ≤ 2.5
- **说明**: ZBL势内截断半径

#### zbl_rc_outer
- **类型**: `float`
- **说明**: ZBL势外截断半径

### 8. 电荷模式参数

#### charge_mode
- **类型**: `int`
- **默认值**: `0` (禁用)
- **可选值**:
  - `0`: 禁用电荷
  - `1`: NEP-Charge，包含实空间和k空间
  - `2`: NEP-Charge，仅k空间
  - `3`: NEP-Charge，仅实空间
  - `4`: NEP-Charge-VdW，仅k空间
  - `5`: NEP-Charge-VdW，仅实空间

### 9. 其他参数

#### atomic_v
- **类型**: `int`
- **默认值**: `0`
- **可选值**: `0` 或 `1`
- **说明**: 是否使用原子维里（仅偶极/极化率模式）

#### output_descriptor
- **类型**: `int`
- **默认值**: `false`
- **说明**: 是否输出描述符

#### save_potential
- **类型**: `int`
- **默认值**: `100000`
- **说明**: 每隔多少代保存一次势函数文件

#### save_potential_format
- **类型**: `int`
- **默认值**: `1`
- **可选值**: `0` 或 `1`
- **说明**: 保存文件名格式（0=简单，1=包含时间戳）

#### save_potential_restart
- **类型**: `int`
- **默认值**: `0`
- **说明**: 是否同时保存restart文件

#### fine_tune
- **类型**: `int`
- **默认值**: `0`
- **说明**: 是否基于基础模型进行微调

## 参数读取流程

### set_default_parameters()

**位置**: `src/main_nep/parameters.cu` 第52-135行

**功能**: 初始化所有参数为默认值，并设置所有`is_*_set`标志为`false`

### read_nep_in()

**位置**: `src/main_nep/parameters.cu` 第137-161行

**执行流程**:
1. 打开`nep.in`文件
2. 逐行读取，忽略注释（以`#`开头）
3. 对每行调用`parse_one_keyword()`解析关键字

**关键代码**:
```cpp
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
```

### parse_one_keyword()

**位置**: `src/main_nep/parameters.cu` 第618-690行

**功能**: 根据关键字调用相应的解析函数

**支持的关键字**:
- `model_type` / `mode`: 训练模式
- `prediction`: 预测模式
- `version`: NEP版本
- `type`: 原子类型
- `cutoff`: 截断半径
- `n_max`: Chebyshev多项式阶数
- `basis_size`: 基函数大小（NEP3）
- `l_max`: 球谐函数角动量
- `neuron`: 神经元数量
- `lambda_*`: 各种损失权重
- `batch`: 批次大小
- `population`: 种群大小
- `generation`: 最大代数
- `zbl`: ZBL势
- `charge_mode`: 电荷模式
- `fine_tune`: 微调模式
- `save_potential`: 保存设置

### read_zbl_in()

**位置**: `src/main_nep/parameters.cu` 第163-176行

**功能**: 如果启用ZBL且存在`zbl.in`文件，读取可调ZBL参数

**参数数量**: `num_types * (num_types + 1) / 2 * 10`

## 派生参数计算

### calculate_parameters()

**位置**: `src/main_nep/parameters.cu` 第178-286行

**计算内容**:

#### 1. 描述符维度计算

```cpp
dim_radial = n_max_radial + 1;  // 2体描述符 q^i_n
dim_angular = (n_max_angular + 1) * L_max;  // 3体描述符 q^i_nl
if (L_max_4body == 2) {
  dim_angular += n_max_angular + 1;  // 4体描述符 q^i_n222
}
if (L_max_5body == 1) {
  dim_angular += n_max_angular + 1;  // 5体描述符 q^i_n1111
}
dim = dim_radial + dim_angular;
if (train_mode == 3) {
  dim += 1;  // 温度依赖模式，添加温度维度
}
```

#### 2. 神经网络参数数量

**NEP3**:
```cpp
number_of_variables_ann_1 = (dim + 2) * num_neurons1;
number_of_variables_ann = (dim + 2) * num_neurons1 + 1;  // +1为全局偏置
```

**NEP4**:
```cpp
number_of_variables_ann_1 = (dim + 2) * num_neurons1;
number_of_variables_ann = (dim + 2) * num_neurons1 * num_types + 1;
```

**NEP4 + Charge**:
```cpp
number_of_variables_ann_1 += num_neurons1;  // 电荷网络
number_of_variables_ann += num_neurons1 * num_types + 1;
if (charge_mode >= 4) {
  number_of_variables_ann_1 += num_neurons1;  // VdW网络
  number_of_variables_ann += num_neurons1 * num_types;
}
```

#### 3. 描述符参数数量

```cpp
number_of_variables_descriptor = 
  num_types * num_types *
  (dim_radial * (basis_size_radial + 1) + 
   (n_max_angular + 1) * (basis_size_angular + 1));
```

#### 4. 总变量数量

```cpp
number_of_variables = number_of_variables_ann + number_of_variables_descriptor;
if (train_mode == 2) {
  number_of_variables += number_of_variables_ann;  // 极化率需要额外的网络
}
```

#### 5. 自动正则化权重

**NEP4** (如果未设置):
```cpp
lambda_1 = sqrt(number_of_variables * 1.0e-6 / num_types);
lambda_2 = sqrt(number_of_variables * 1.0e-6 / num_types);
```

**NEP3** (如果未设置):
```cpp
lambda_1 = sqrt(number_of_variables * 1.0e-6);
lambda_2 = sqrt(number_of_variables * 1.0e-6);
```

#### 6. q_scaler初始化

```cpp
q_scaler_cpu.resize(dim, 1.0e10f);  // 初始化为大值
// 如果fine_tune，从基础模型读取
```

### report_inputs()

**位置**: `src/main_nep/parameters.cu` 第377-616行

**功能**: 打印所有输入参数和计算得到的参数，区分用户输入和默认值

## 参数验证

每个`parse_*`函数都包含参数验证：

1. **类型检查**: 确保参数类型正确（int/float）
2. **范围检查**: 确保参数在允许范围内
3. **逻辑检查**: 确保参数组合合理（如rc_angular ≤ rc_radial）

## 特殊处理

### GPU数量适配

**位置**: `src/main_nep/parameters.cu` 第127-134行和第1160-1174行

如果`population_size`不是GPU数量的倍数，会自动调整：
```cpp
int fully_used_device = population_size % deviceCount;
if (fully_used_device != 0) {
  population_size += deviceCount - fully_used_device;
}
```

### fine_tune模式

**位置**: `src/main_nep/parameters.cu` 第258-277行和第288-375行

1. 从基础模型文件读取`q_scaler`
2. 验证基础模型参数与当前设置匹配
3. 检查ZBL截断、描述符参数、神经网络结构等

## 相关文档

- [主程序执行流程](01_main_execution_flow.md) - Parameters的调用位置
- [数学公式集合](07_mathematical_formulas.md) - 描述符维度计算公式

