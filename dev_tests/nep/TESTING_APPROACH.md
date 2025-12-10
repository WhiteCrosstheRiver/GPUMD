# 测试方法说明

## 问题

`Parameters` 类的所有方法（除了构造函数）都是**私有的**，包括：
- `read_nep_in()`
- `parse_*` 系列函数
- `calculate_parameters()`
- 等等

这意味着我们**无法直接测试这些函数的逻辑**。

## 解决方案

由于无法访问私有函数，我们采用以下测试策略：

### 1. 测试公共成员变量

直接设置和验证公共成员变量，测试：
- 成员变量可以被正确设置
- 成员变量的默认值
- 成员变量之间的关系

**示例：**
```cpp
bool test_parse_version()
{
  Parameters para;
  
  // 直接设置公共成员变量
  para.version = 4;
  para.is_version_set = true;
  
  // 验证设置成功
  assert(para.version == 4);
  assert(para.is_version_set == true);
  
  return true;
}
```

### 2. 测试构造函数

测试构造函数的默认值设置：

```cpp
bool test_parameters_constructor()
{
  Parameters para;
  
  // 验证默认值
  assert(para.version == 4);
  assert(para.rc_radial == 8.0f);
  // ...
  
  return true;
}
```

### 3. 测试成员变量之间的关系

验证相关成员变量的一致性：

```cpp
bool test_parameters_consistency()
{
  Parameters para;
  
  // 设置相关参数
  para.num_types = 3;
  para.elements = {"Si", "C", "O"};
  para.atomic_numbers = {14, 6, 8};
  
  // 验证一致性
  assert(para.elements.size() == para.num_types);
  assert(para.atomic_numbers.size() == para.num_types);
  // ...
  
  return true;
}
```

## 限制

⚠️ **重要限制：**

1. **无法测试解析逻辑**：我们无法测试 `parse_*` 函数是否正确解析输入字符串
2. **无法测试文件读取**：我们无法测试 `read_nep_in()` 是否正确读取文件
3. **无法测试错误处理**：我们无法测试无效输入的处理

## 建议

如果要完整测试 `Parameters` 类，有以下选项：

### 选项1：添加友元类（需要修改源代码）

在 `parameters.cuh` 中添加：
```cpp
class Parameters {
  // ...
  friend class ParametersTest;  // 添加友元类
  // ...
};
```

### 选项2：添加公共测试接口（需要修改源代码）

在 `Parameters` 类中添加公共测试方法：
```cpp
public:
  // 仅用于测试
  #ifdef TESTING
  void test_parse_version(const char** param, int num_param) {
    parse_version(param, num_param);
  }
  #endif
```

### 选项3：使用集成测试

在实际使用场景中测试（通过 NEP 主程序），但这需要完整的运行环境。

## 当前测试覆盖范围

当前测试可以验证：
- ✅ 构造函数默认值
- ✅ 公共成员变量的设置和读取
- ✅ 成员变量之间的关系
- ❌ 解析函数的逻辑（无法测试）
- ❌ 文件读取功能（无法测试）

## 结论

当前的测试方法可以验证 `Parameters` 类的**数据结构**和**公共接口**，但无法测试**解析逻辑**。如果需要测试解析逻辑，需要修改源代码添加测试接口。

