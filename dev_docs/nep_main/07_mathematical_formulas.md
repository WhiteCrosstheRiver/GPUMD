# 数学公式集合

本文档收集NEP主程序中涉及的所有数学公式。

## SNES算法公式

### Utility函数

对于排名为 i 的个体（0为最好）：

```
u_i = max(0, ln(λ/2 + 1) - ln(i + 1))
```

归一化后：

```
u_i = u_i / Σ_j u_j - 1/λ
```

其中 λ 是种群大小。

### 参数分布更新

对于每个变量 i：

**均值更新**:
```
μ_i ← μ_i + σ_i · Σ_p (s_{i,p} · u_p)
```

**标准差更新**:
```
σ_i ← min(σ_max, σ_i · exp(η_σ · Σ_p ((s_{i,p}² - 1) · u_p)))
```

其中:
- `s_{i,p} ~ N(0,1)` 是第p个个体中变量i的标准正态采样
- `u_p` 是第p个个体的utility值
- `η_σ = (3 + ln(n)) / (5·√n) / 2` 是学习率

### 正则化损失

**L1正则化**:
```
L1 = λ_1 · (1/n) · Σ_i |θ_i|
```

**L2正则化**:
```
L2 = λ_2 · sqrt((1/n) · Σ_i θ_i²)
```

**NEP4类型特定正则化**:
```
L1_t = λ_1 · (1/n_t) · Σ_{i∈t} |θ_i|
L2_t = λ_2 · sqrt((1/n_t) · Σ_{i∈t} θ_i²)
```

其中 `n_t` 是属于类型 t 的变量数量。

## NEP描述符公式

### 径向描述符（2体）

对于原子 i，径向描述符为：

```
q^i_n = Σ_{j≠i} f_cut(r_{ij}) · Σ_k c^k_n · T_n(s_{ij})
```

其中:
- `r_{ij}` 是原子i和j之间的距离
- `f_cut(r)` 是截断函数
- `c^k_n` 是描述符系数（可学习参数）
- `T_n(s)` 是n阶Chebyshev多项式
- `s_{ij} = 2(r_{ij}/r_c) - 1` 是归一化距离，`r_c` 是截断半径

**Chebyshev多项式**:
```
T_0(s) = 1
T_1(s) = s
T_n(s) = 2s·T_{n-1}(s) - T_{n-2}(s)  for n ≥ 2
```

### 角向描述符（3体）

```
q^i_{nl} = Σ_{j≠i} Σ_{k≠i,j} f_cut(r_{ij}) · f_cut(r_{ik}) · 
           Σ_k c^k_n · T_n(s_{ij}) · T_n(s_{ik}) · Y^l_m(θ_{jik})
```

其中:
- `θ_{jik}` 是角度
- `Y^l_m` 是球谐函数

**球谐函数**:
```
Y^l_m(θ, φ) = P^m_l(cos θ) · e^{imφ}
```

其中 `P^m_l` 是关联Legendre多项式。

### 4体描述符

```
q^i_{n222} = Σ_{j≠i} Σ_{k≠i,j} Σ_{l≠i,j,k} f_cut(r_{ij}) · f_cut(r_{ik}) · f_cut(r_{il}) ·
             T_n(s_{ij}) · T_n(s_{ik}) · T_n(s_{il}) · Y^2_2(θ_{jik}) · Y^2_2(θ_{jil})
```

### 5体描述符

```
q^i_{n1111} = Σ_{j≠i} Σ_{k≠i,j} Σ_{l≠i,j,k} Σ_{m≠i,j,k,l} 
              f_cut(r_{ij}) · f_cut(r_{ik}) · f_cut(r_{il}) · f_cut(r_{im}) ·
              T_n(s_{ij}) · T_n(s_{ik}) · T_n(s_{il}) · T_n(s_{im}) ·
              Y^1_1(θ_{jik}) · Y^1_1(θ_{jil}) · Y^1_1(θ_{jim})
```

### 截断函数

平滑截断函数：

```
f_cut(r) = (1/2) · (cos(π·r/r_c) + 1)  if r < r_c
f_cut(r) = 0                            if r ≥ r_c
```

## 神经网络公式

### 前向传播

**隐藏层**:
```
h_j = tanh(Σ_i w_{ij}^0 · q_i + b_j^0)
```

**输出层**:
```
E = Σ_j w_j^1 · h_j + b^1
```

### 激活函数

通常使用tanh：
```
tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
```

## 力计算

通过链式法则：

```
F_i^α = -∂E/∂r_i^α = -Σ_j (∂E/∂q_j) · (∂q_j/∂r_i^α)
```

其中 `α ∈ {x, y, z}`。

## 维里计算

维里张量：

```
V_{αβ} = -Σ_i r_i^α · F_i^β
```

应力：

```
σ_{αβ} = V_{αβ} / V
```

其中 V 是体积。

## 损失函数

### 能量RMSE

```
RMSE_E = sqrt(Σ_c ((E_c^{pred} - E_c^{ref})² · w_c · w_E,c) / Σ_c (w_c · w_E,c))
```

其中:
- `E_c` 是配置c的能量
- `w_c` 是配置权重
- `w_E,c` 是能量权重

### 力RMSE

```
RMSE_F = sqrt(Σ_i (|F_i^{pred} - F_i^{ref}|² · w_{type(i)}) / Σ_i w_{type(i)})
```

其中 `w_{type(i)}` 是原子类型权重。

### 维里RMSE

```
RMSE_V = sqrt(Σ_c Σ_{αβ} ((V_{c,αβ}^{pred} - V_{c,αβ}^{ref})² · w_c · w_{shear,αβ}) / 
              Σ_c Σ_{αβ} (w_c · w_{shear,αβ}))
```

其中 `w_{shear,αβ}` 是剪切维里权重（对角元为1，非对角元为`lambda_shear`）。

### 总损失

```
L_total = L1 + L2 + λ_E·RMSE_E + λ_F·RMSE_F + λ_V·RMSE_V + λ_Q·RMSE_Q
```

## NEP_Charge公式

### 电荷计算

```
q_i = NN_charge(descriptors_i)
```

### Coulomb能量（实空间）

```
E_{Coulomb}^{real} = (1/2) · Σ_{i≠j} (q_i · q_j / r_{ij}) · erfc(α·r_{ij})
```

其中 `erfc` 是互补误差函数，`α` 是Ewald参数。

### Coulomb能量（k空间）

```
E_{Coulomb}^{k} = (1/(2V)) · Σ_{k≠0} (|S(k)|² / k²) · exp(-k²/(4α²))
```

结构因子：

```
S(k) = Σ_i q_i · exp(i·k·r_i)
```

### VdW能量

```
E_{VdW} = -Σ_{i<j} (C6_{ij} / r_{ij}^6) · f_{damp}(r_{ij})
```

阻尼函数：

```
f_{damp}(r) = 1 / (1 + (R_{ij}/r)^β)
```

### Born有效电荷

```
BEC_{i,αβ} = ∂q_i^α / ∂r_i^β
```

## ZBL势公式

### 通用ZBL势

```
V_{ZBL}(r) = (Z_1·Z_2·e²/(4πε₀)) · (a/r) · Σ_{i=1}^4 A_i · exp(-b_i·r/a)
```

其中:
- `a = 0.8854·a_0 / (Z_1^{0.23} + Z_2^{0.23})`
- `a_0` 是Bohr半径
- `A_i, b_i` 是拟合参数

### 可调ZBL势

如果启用`flexible_zbl`，参数可以从`zbl.in`文件读取，允许对每种元素对进行优化。

## 描述符维度计算

### 径向维度

```
dim_{radial} = n_{max}^{radial} + 1
```

### 角向维度

```
dim_{angular} = (n_{max}^{angular} + 1) · L_{max}
```

如果包含4体：
```
dim_{angular} += n_{max}^{angular} + 1
```

如果包含5体：
```
dim_{angular} += n_{max}^{angular} + 1
```

### 总维度

```
dim = dim_{radial} + dim_{angular}
```

如果温度依赖模式：
```
dim += 1
```

## 参数数量计算

### 神经网络参数（NEP3）

```
N_{ANN} = (dim + 2) · num_{neurons} + 1
```

### 神经网络参数（NEP4）

```
N_{ANN} = (dim + 2) · num_{neurons} · num_{types} + 1
```

### 描述符参数

```
N_{descriptor} = num_{types}² · [dim_{radial} · (basis_{radial} + 1) + 
                                  (n_{max}^{angular} + 1) · (basis_{angular} + 1)]
```

### 总参数

```
N_{total} = N_{ANN} + N_{descriptor}
```

如果极化率模式：
```
N_{total} += N_{ANN}
```

## 相关文档

- [SNES算法](04_snes_algorithm.md) - SNES公式的详细说明
- [Potential模型](06_potential_models.md) - 描述符计算的实现

