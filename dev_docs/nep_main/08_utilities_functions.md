# 工具函数说明

本文档记录NEP主程序中使用的工具函数和宏定义。

## 文件位置

### 主程序工具函数
- **源文件**: `src/main_nep/main.cu` (第70-80行)

### GPU和系统信息
- **头文件**: `src/utilities/main_common.cuh`
- **实现文件**: `src/utilities/main_common.cu`

### 错误处理
- **头文件**: `src/utilities/error.cuh`
- **实现文件**: `src/utilities/error.cu`

### 文件读取
- **头文件**: `src/utilities/read_file.cuh`
- **实现文件**: `src/utilities/read_file.cu`

## 主程序工具函数

### print_welcome_information()

**位置**: `src/main_nep/main.cu` 第70-80行

**功能**: 打印欢迎信息

**输出**:
```
***************************************************************
*                 Welcome to use GPUMD                        *
*    (Graphics Processing Units Molecular Dynamics)           *
*                     version 4.5                             *
*              This is the nep executable                     *
***************************************************************
```

## GPU信息函数

### print_gpu_information()

**位置**: `src/utilities/main_common.cu` 第38-76行

**功能**: 检测并打印所有可用GPU的信息

**输出内容**:
- GPU数量
- 每个GPU的:
  - 设备ID
  - 设备名称
  - 计算能力（major.minor）
  - 全局内存大小（GB）
  - 流多处理器（SM）数量
- GPU之间的P2P访问能力

**关键代码**:
```cpp
int num_gpus;
CHECK(gpuGetDeviceCount(&num_gpus));

for (int device_id = 0; device_id < num_gpus; ++device_id) {
  gpuDeviceProp prop;
  CHECK(gpuGetDeviceProperties(&prop, device_id));
  // 打印设备信息
}

// 检查P2P访问
for (int i = 0; i < num_gpus; i++) {
  for (int j = 0; j < num_gpus; j++) {
    if (i != j) {
      CHECK(gpuDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        CHECK(gpuDeviceEnablePeerAccess(j, 0));
      }
    }
  }
}
```

## 格式化输出函数

### print_line_1()

**位置**: `src/utilities/error.cu` 第28-32行

**功能**: 打印分隔线（开始）

**输出**:
```
-------------------------------------------------------------------------------
```

### print_line_2()

**位置**: `src/utilities/error.cu` 第34-38行

**功能**: 打印分隔线（结束）

**输出**: 同上

## 错误处理宏

### CHECK()

**功能**: 检查CUDA/HIP函数调用错误

**用法**:
```cpp
CHECK(gpuMalloc(&ptr, size));
```

如果出错，会打印错误信息并退出程序。

### PRINT_INPUT_ERROR()

**功能**: 打印输入错误信息并退出

**用法**:
```cpp
PRINT_INPUT_ERROR("Error message");
```

### PRINT_KEYWORD_ERROR()

**功能**: 打印未知关键字错误

**用法**:
```cpp
PRINT_KEYWORD_ERROR(keyword);
```

### PRINT_SCANF_ERROR()

**功能**: 检查scanf读取错误

**用法**:
```cpp
PRINT_SCANF_ERROR(count, expected, "Error message");
```

## 文件读取函数

### get_tokens()

**功能**: 从输入流读取一行并分割为tokens

**返回**: `std::vector<std::string>`

**特点**:
- 自动处理空白字符
- 支持引号内的字符串

### get_tokens_without_unwanted_spaces()

**功能**: 读取tokens，去除不需要的空格

### get_double_from_token()

**功能**: 从token字符串转换为double

**参数**:
- token字符串
- 文件名（用于错误报告）
- 行号（用于错误报告）

### get_int_from_token()

**功能**: 从token字符串转换为int

### is_valid_int()

**功能**: 验证字符串是否为有效整数

### is_valid_real()

**功能**: 验证字符串是否为有效浮点数

## 文件操作函数

### my_fopen()

**功能**: 安全打开文件

**特点**:
- 检查文件是否成功打开
- 如果失败，打印错误并退出

**用法**:
```cpp
FILE* fid = my_fopen("filename.txt", "w");
```

## GPU内存管理

### GPU_Vector类

**功能**: GPU向量容器，类似`std::vector`但存储在GPU

**主要方法**:
- `resize(size)`: 调整大小
- `copy_from_host(ptr)`: 从CPU复制到GPU
- `copy_to_host(ptr)`: 从GPU复制到CPU
- `data()`: 获取GPU指针

**用法**:
```cpp
GPU_Vector<float> vec;
vec.resize(100);
vec.copy_from_host(cpu_data);
float* gpu_ptr = vec.data();
```

## 数学工具函数

### dev_apply_mic()

**功能**: 应用最小镜像约定（Minimum Image Convention）

**用途**: 计算周期性边界条件下的最短距离向量

### COVALENT_RADIUS[]

**功能**: 共价半径数组

**用途**: 用于类型相关截断半径计算

## 常量定义

### NUM_ELEMENTS

**值**: 89

**说明**: 支持的元素数量（H到Pu）

### ELEMENTS[]

**类型**: `const std::string[89]`

**内容**: 元素符号数组

### PRESSURE_UNIT_CONVERSION

**功能**: 压力单位转换因子

**用途**: 将维里转换为应力（GPa）

## 随机数生成

### gpurandState

**类型**: CUDA/HIP随机数状态

**初始化**:
```cpp
gpurand_init(seed, n, 0, &state);
```

### gpurand_normal()

**功能**: 生成标准正态分布随机数

**用法**:
```cpp
float s = gpurand_normal(&state);
```

## 调试宏

### DEBUG

**功能**: 调试模式标志

**影响**:
- 如果定义，使用固定随机数种子
- 否则使用时间戳作为种子

### GPU_CHECK_KERNEL

**功能**: 检查GPU kernel执行错误

**用法**:
```cpp
kernel<<<...>>>();
GPU_CHECK_KERNEL;
```

## 相关文档

- [主程序执行流程](01_main_execution_flow.md) - 工具函数的使用位置
- [错误处理](error.cuh) - 完整的错误处理机制

