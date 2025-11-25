# Potential模型详解

本文档详细记录NEP、TNEP和NEP_Charge三种势函数模型的实现。

## 文件位置

### 基类
- **头文件**: `src/main_nep/potential.cuh` (第22-33行)

### NEP模型
- **头文件**: `src/main_nep/nep.cuh` (第40-109行)
- **实现文件**: `src/main_nep/nep.cu`

### TNEP模型
- **头文件**: `src/main_nep/tnep.cuh` (第40-104行)
- **实现文件**: `src/main_nep/tnep.cu`

### NEP_Charge模型
- **头文件**: `src/main_nep/nep_charge.cuh` (第23-133行)
- **实现文件**: `src/main_nep/nep_charge.cu`

## Potential基类

### 接口定义

```cpp
class Potential {
public:
  virtual ~Potential() = default;
  virtual void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    bool calculate_neighbor,
    int DeviceCount) = 0;
};
```

### 核心方法

所有Potential子类必须实现`find_force()`方法，该方法：
1. 接收参数数组
2. 对数据集中的每个配置计算能量、力、维里
3. 可选计算q_scaler（描述符缩放因子）
4. 可选重新计算邻居列表

## NEP模型

### 类结构

```cpp
class NEP : public Potential {
  struct ParaMB {
    float rc_radial, rc_angular;
    int n_max_radial, n_max_angular;
    int L_max, L_max_4body, L_max_5body;
    int basis_size_radial, basis_size_angular;
    int version;  // 3 or 4
    // ...
  };
  
  struct ANN {
    int dim, num_neurons1;
    const float* w0[NUM_ELEMENTS];  // 输入层到隐藏层权重
    const float* b0[NUM_ELEMENTS];  // 隐藏层偏置
    const float* w1[NUM_ELEMENTS];  // 隐藏层到输出层权重
    const float* b1;                // 输出层偏置
    const float* c;                  // 描述符系数
  };
  
  struct ZBL {
    bool enabled, flexibled;
    float rc_inner, rc_outer;
    float para[550];
  };
  
  ParaMB paramb;
  ANN annmb[16];  // 每个GPU设备一个
  NEP_Data nep_data[16];
  ZBL zbl;
};
```

### find_force()执行流程

1. **更新参数**: 从参数数组提取神经网络权重和描述符系数
2. **计算描述符**: 对每个原子计算径向和角向描述符
3. **前向传播**: 通过神经网络计算原子能量
4. **计算力**: 通过自动微分计算力
5. **计算维里**: 从力计算维里张量
6. **ZBL势**: 如果启用，添加ZBL势的贡献

### 描述符计算

#### 径向描述符（2体）

```
q^i_n = Σ_j f_cut(r_ij) · c_n^k · T_n(s_ij)
```

其中:
- `f_cut(r_ij)`: 截断函数
- `c_n^k`: 描述符系数（从参数中学习）
- `T_n(s)`: Chebyshev多项式
- `s_ij = 2(r_ij/rc) - 1`: 归一化距离

#### 角向描述符（3体）

```
q^i_nl = Σ_j Σ_k f_cut(r_ij) · f_cut(r_ik) · c_n^k · T_n(s_ij) · T_n(s_ik) · Y_lm(θ_jik)
```

其中:
- `Y_lm`: 球谐函数
- `θ_jik`: 角度

#### 4体和5体描述符

类似地定义，包含更多原子的几何信息。

### 神经网络前向传播

```
h = activation(W0 · q + b0)  // 隐藏层
E = W1 · h + b1               // 输出层
```

其中`activation`通常是tanh或类似函数。

### 力计算

通过链式法则计算：

```
F_i = -∇_i E = -Σ_j (∂E/∂q^j) · (∂q^j/∂r_i)
```

### ZBL势

如果启用，添加ZBL (Ziegler-Biersack-Littmark) 势：

```
V_ZBL(r) = (Z1·Z2·e²/4πε₀) · (a/r) · Σ_i A_i·exp(-b_i·r/a)
```

## TNEP模型

### 用途

TNEP (Tensor NEP) 用于训练：
- **train_mode = 1**: 偶极矩
- **train_mode = 2**: 极化率

### 与NEP的区别

1. **输出**: 不是标量能量，而是张量（偶极矩或极化率）
2. **网络结构**: 可能需要额外的网络分支
3. **损失函数**: 使用维里作为偶极/极化率的存储

### 结构

```cpp
class TNEP : public Potential {
  struct ANN {
    // 标准网络（用于偶极/极化率的向量部分）
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;
    
    // 极化率标量部分（仅train_mode=2）
    const float* w0_pol[10];
    const float* b0_pol[10];
    const float* w1_pol[10];
    const float* b1_pol;
  };
};
```

## NEP_Charge模型

### 用途

NEP_Charge用于包含动态电荷的体系，支持：
- 电荷-电荷相互作用（Coulomb）
- 可选的范德华相互作用（VdW）

### 电荷模式

- **charge_mode = 1**: 实空间 + k空间
- **charge_mode = 2**: 仅k空间
- **charge_mode = 3**: 仅实空间
- **charge_mode = 4**: VdW + k空间
- **charge_mode = 5**: VdW + 实空间

### 结构

```cpp
class NEP_Charge : public Potential {
  struct Charge_Para {
    int num_kpoints_max = 50000;
    float alpha = 0.5f;  // Ewald参数
    // ...
  };
  
  struct NEP_Charge_Data {
    GPU_Vector<float> charge_derivative;  // 电荷对描述符的导数
    GPU_Vector<float> C6;                // C6系数（VdW）
    GPU_Vector<float> D_C6;              // 动态C6
    GPU_Vector<float> kx, ky, kz;       // k空间向量
    GPU_Vector<float> G;                 // 结构因子
    GPU_Vector<float> S_real, S_imag;    // 实部和虚部
    // ...
  };
};
```

### 电荷计算

电荷通过独立的神经网络分支计算：

```
q_i = NN_charge(descriptors_i)
```

### Coulomb能量

#### 实空间

```
E_Coulomb_real = (1/2) · Σ_i Σ_j (q_i · q_j / r_ij) · erfc(α·r_ij)
```

#### k空间

```
E_Coulomb_k = (1/2V) · Σ_k (|S(k)|² / k²) · exp(-k²/(4α²))
```

其中`S(k)`是结构因子。

### VdW相互作用

如果启用（charge_mode >= 4）：

```
E_VdW = -Σ_i Σ_j (C6_ij / r_ij^6) · f_damp(r_ij)
```

其中`C6_ij`可以是动态的，依赖于电荷。

### BEC计算

Born有效电荷（BEC）通过电荷对位置的导数计算：

```
BEC_αβ = ∂q_α / ∂r_β
```

## 共同特性

### 多GPU支持

所有模型支持多GPU并行：
- 每个GPU设备有独立的ANN和数据结构
- 种群中的不同个体分配到不同GPU

### 邻居列表

- **径向邻居**: 用于2体描述符
- **角向邻居**: 用于3+体描述符
- 使用扩展盒子处理周期性边界条件

### 截断函数

通常使用平滑截断函数：

```
f_cut(r) = (1/2) · (cos(π·r/rc) + 1)  if r < rc
f_cut(r) = 0                           if r >= rc
```

### 类型相关截断

如果启用`use_typewise_cutoff`，截断半径根据原子类型对调整：

```
rc_ij = min((R_i + R_j) · factor, rc_global)
```

其中`R_i`是共价半径。

## 相关文档

- [数学公式集合](07_mathematical_formulas.md) - 完整的数学公式
- [Fitness类详解](03_fitness_class.md) - 模型的使用

