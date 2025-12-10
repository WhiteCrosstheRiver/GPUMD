# 如何执行 test_parameters.cpp 测试

## 当前状态

测试文件 `test_parameters.cpp` 存在以下问题：

1. **私有函数访问问题**：测试代码试图直接调用 `Parameters` 类的私有 `parse_*` 函数，这些函数无法从外部访问。

2. **CUDA 编译问题**：由于 `parameters.cuh` 依赖 CUDA 头文件，需要使用 `nvcc` 编译器，并且需要正确的 CUDA 环境。

## 解决方案

### 方案1：修改测试代码（推荐）

由于 `parse_*` 函数都是私有的，测试应该通过以下方式：

1. **测试公共接口**：直接设置公共成员变量，然后验证结果
2. **通过 `read_nep_in()` 测试**：创建测试用的 `nep.in` 文件，然后调用 `read_nep_in()` 来间接测试解析功能

### 方案2：使用友元类（需要修改源代码）

在 `Parameters` 类中添加测试友元类，但这需要修改源代码。

## 编译命令

如果解决了私有函数访问问题，可以使用以下命令编译：

```bash
cd dev_tests/nep

# 方法1：使用 nvcc 编译（需要 CUDA 环境）
nvcc -o test_parameters test_parameters.cpp \
    ../../src/main_nep/parameters.cu \
    -I../../src \
    -I../../src/utilities \
    -std=c++17 \
    -O3 \
    -arch=sm_86 \
    -lcublas -lcusolver -lcufft

# 方法2：使用提供的脚本
./build_and_run_test.sh
```

## 运行测试

```bash
./test_parameters
```

## 注意事项

1. **需要 CUDA 环境**：确保已安装 CUDA 工具包，并且 `nvcc` 在 PATH 中
2. **GPU 架构**：根据你的 GPU 修改 `-arch=sm_XX` 参数
3. **私有函数**：当前测试代码无法直接测试私有函数，需要修改测试策略

## 建议的测试方法

### 方法1：测试公共成员变量

```cpp
bool test_parameters_constructor()
{
  Parameters para;
  
  // 直接测试公共成员变量的默认值
  assert(para.version == 4);
  assert(para.rc_radial == 8.0f);
  // ...
  return true;
}
```

### 方法2：通过文件读取测试

```cpp
bool test_read_nep_in()
{
  Parameters para;
  
  // 创建测试文件
  create_test_file("nep.in", "version 4\ntype 2 Si C\n");
  
  // 调用公共函数
  para.read_nep_in();
  
  // 验证结果
  assert(para.version == 4);
  assert(para.num_types == 2);
  // ...
  
  cleanup_test_file("nep.in");
  return true;
}
```

## 当前问题修复

要修复当前测试代码，需要：

1. 移除所有对私有 `parse_*` 函数的直接调用
2. 改为测试公共接口或通过文件读取间接测试
3. 确保使用 `nvcc` 编译器并正确配置 CUDA 环境

