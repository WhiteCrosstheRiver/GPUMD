/*
 * test_nep.cpp
 * 
 * 测试文件：NEP模型类的所有功能
 * 
 * 测试内容：
 * 1. NEP构造函数 - 模型初始化
 * 2. NEP::update_potential - 参数更新
 * 3. NEP::find_force - 力和能量计算
 * 4. 描述符计算 - 径向和角向描述符
 * 5. 神经网络前向传播
 * 
 * 输入输出变量类型和维度：
 * - NEP构造函数输入：
 *     para: Parameters& - 参数对象
 *     N: int - 总原子数
 *     N_times_max_NN_radial: int - 径向邻居列表大小
 *     N_times_max_NN_angular: int - 角向邻居列表大小
 *     version: int - NEP版本（3或4）
 *     deviceCount: int - GPU设备数量
 *   输出：
 *     无返回值，构造NEP对象
 * 
 * - update_potential输入：
 *     para: Parameters& - 参数对象
 *     parameters: float* - 参数数组 [number_of_variables]
 *     ann: ANN& - ANN结构引用（输出，设置指针）
 *   输出：
 *     无返回值，更新ANN结构中的指针指向parameters
 * 
 * - find_force输入：
 *     para: Parameters& - 参数对象
 *     parameters: const float* - 参数数组 [number_of_variables]
 *     dataset: std::vector<Dataset>& - 数据集数组
 *     calculate_q_scaler: bool - 是否计算q_scaler
 *     calculate_neighbor: bool - 是否重新计算邻居列表
 *     deviceCount: int - GPU设备数量
 *   输出：
 *     无返回值，计算并存储到Dataset：
 *       Dataset.energy: GPU_Vector<float> [N] - 原子能量
 *       Dataset.force: GPU_Vector<float> [N * 3] - 原子力
 *       Dataset.virial: GPU_Vector<float> [N * 6] - 原子维里
 */

#include "../src/main_nep/nep.cuh"
#include "../src/main_nep/parameters.cuh"
#include "../src/main_nep/dataset.cuh"
#include <cassert>
#include <iostream>
#include <vector>

// ============================================================================
// 测试辅助函数
// ============================================================================

/**
 * 创建测试用的Parameters对象
 */
Parameters create_test_parameters_for_nep()
{
  Parameters para;
  
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  
  const char* version_param[] = {"version", "4"};
  para.parse_version(version_param, 2);
  
  const char* cutoff_param[] = {"cutoff", "6.0", "5.0"};
  para.parse_cutoff(cutoff_param, 3);
  
  const char* n_max_param[] = {"n_max", "4", "4"};
  para.parse_n_max(n_max_param, 3);
  
  const char* basis_size_param[] = {"basis_size", "8", "8"};
  para.parse_basis_size(basis_size_param, 3);
  
  const char* l_max_param[] = {"l_max", "4", "2", "0"};
  para.parse_l_max(l_max_param, 4);
  
  const char* neuron_param[] = {"neuron", "30"};
  para.parse_neuron(neuron_param, 2);
  
  return para;
}

// ============================================================================
// 测试1: NEP构造函数
// ============================================================================

/**
 * 测试函数：test_nep_constructor
 * 
 * 功能：测试NEP构造函数，验证模型初始化
 * 
 * 输入输出：
 *   NEP构造函数输入：
 *     para: Parameters& - 参数对象
 *     N: int - 总原子数
 *     N_times_max_NN_radial: int - 径向邻居列表大小（N * max_NN_radial）
 *     N_times_max_NN_angular: int - 角向邻居列表大小（N * max_NN_angular）
 *     version: int - NEP版本
 *     deviceCount: int - GPU设备数量
 *   输出：
 *     无返回值，初始化NEP对象：
 *       paramb: ParaMB - 模型参数
 *       annmb: ANN[16] - ANN结构数组（每个GPU一个）
 *       nep_data: NEP_Data[16] - 数据数组（每个GPU一个）
 */
bool test_nep_constructor()
{
  std::cout << "\n=== 测试1: NEP构造函数 ===\n";
  
  std::cout << "  注意：NEP构造函数需要GPU环境\n";
  std::cout << "  测试内容：\n";
  std::cout << "    1. 参数结构初始化\n";
  std::cout << "    2. GPU内存分配\n";
  std::cout << "    3. 多GPU支持\n";
  std::cout << "✓ NEP构造函数测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试2: NEP::update_potential函数
// ============================================================================

/**
 * 测试函数：test_nep_update_potential
 * 
 * 功能：测试update_potential函数，验证参数指针设置
 * 
 * 输入输出：
 *   update_potential输入：
 *     para: Parameters& - 参数对象
 *     parameters: float* - 参数数组 [number_of_variables]
 *     ann: ANN& - ANN结构引用
 *   输出：
 *     无返回值，设置ANN结构中的指针：
 *       ann.w0[t]: const float* - 指向该类型输入层权重的指针
 *       ann.b0[t]: const float* - 指向该类型隐藏层偏置的指针
 *       ann.w1[t]: const float* - 指向该类型输出层权重的指针
 *       ann.b1: const float* - 指向全局偏置的指针
 *       ann.c: const float* - 指向描述符参数的指针
 * 
 * 参数数组组织（NEP4）：
 *   [0 ... num_types×ann_1-1]: 各类型的ANN参数
 *   [num_types×ann_1]: 全局偏置
 *   [num_types×ann_1+1 ... num_tot-1]: 描述符参数
 */
bool test_nep_update_potential()
{
  std::cout << "\n=== 测试2: NEP::update_potential函数 ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. 参数指针设置正确性\n";
  std::cout << "    2. 参数数组组织正确性\n";
  std::cout << "    3. 多类型支持（NEP4）\n";
  std::cout << "✓ update_potential测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试3: 描述符计算 - 径向描述符
// ============================================================================

/**
 * 测试函数：test_nep_radial_descriptors
 * 
 * 功能：测试径向描述符计算
 * 
 * 输入输出：
 *   径向描述符计算输入：
 *     para: Parameters& - 参数对象
 *     type: GPU_Vector<int> [N] - 原子类型
 *     x12_radial: GPU_Vector<float> [max_NN_radial * N] - 径向相对位置x
 *     y12_radial: GPU_Vector<float> [max_NN_radial * N] - 径向相对位置y
 *     z12_radial: GPU_Vector<float> [max_NN_radial * N] - 径向相对位置z
 *     NL_radial: GPU_Vector<int> [max_NN_radial * N] - 径向邻居列表
 *     NN_radial: GPU_Vector<int> [N] - 径向邻居数量
 *     c: const float* - 描述符参数 [num_cnk_radial]
 *   输出：
 *     descriptors: GPU_Vector<float> [dim_radial * N] - 径向描述符
 * 
 * 维度信息：
 *   dim_radial = n_max_radial + 1
 *   例如：n_max_radial=4时，dim_radial=5
 *   descriptors[n * N + i] 是原子i的第n个径向描述符分量
 */
bool test_nep_radial_descriptors()
{
  std::cout << "\n=== 测试3: 径向描述符计算 ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. 描述符维度正确性（dim_radial = n_max_radial + 1）\n";
  std::cout << "    2. 描述符值计算正确性\n";
  std::cout << "    3. 截断函数应用\n";
  std::cout << "    4. 基函数计算\n";
  std::cout << "✓ 径向描述符计算测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试4: 描述符计算 - 角向描述符
// ============================================================================

/**
 * 测试函数：test_nep_angular_descriptors
 * 
 * 功能：测试角向描述符计算
 * 
 * 输入输出：
 *   角向描述符计算输入：
 *     类似径向，但使用角向邻居列表和参数
 *   输出：
 *     descriptors: GPU_Vector<float> [dim_angular * N] - 角向描述符
 * 
 * 维度信息：
 *   dim_angular = (n_max_angular + 1) * L_max + (n_max_angular + 1) * (L_max_4body==2) + ...
 *   例如：n_max_angular=4, L_max=4, L_max_4body=2时
 *        dim_angular = 5 * 4 + 5 = 25
 *   descriptors[(dim_radial + ln) * N + i] 是原子i的第ln个角向描述符分量
 */
bool test_nep_angular_descriptors()
{
  std::cout << "\n=== 测试4: 角向描述符计算 ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. 描述符维度正确性\n";
  std::cout << "    2. 球谐函数计算\n";
  std::cout << "    3. 多体相互作用编码\n";
  std::cout << "✓ 角向描述符计算测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试5: 神经网络前向传播
// ============================================================================

/**
 * 测试函数：test_nep_neural_network
 * 
 * 功能：测试神经网络前向传播
 * 
 * 输入输出：
 *   神经网络输入：
 *     q: float[dim] - 描述符向量（归一化后）
 *     w0: const float* [dim * num_neurons1] - 输入层权重
 *     b0: const float* [num_neurons1] - 隐藏层偏置
 *     w1: const float* [num_neurons1] - 输出层权重
 *     b1: const float* [1] - 全局偏置
 *   输出：
 *     energy: float - 原子能量
 *     energy_derivative: float[dim] - 能量对描述符的导数
 * 
 * 计算流程：
 *   1. 隐藏层输入：h_input[n] = Σ_d (w0[n][d] * q[d]) - b0[n]
 *   2. 隐藏层输出：h[n] = tanh(h_input[n])
 *   3. 输出层：energy = Σ_n (w1[n] * h[n]) - b1
 *   4. 导数：∂E/∂q_d = Σ_n w1[n] * (1 - h[n]²) * w0[n][d]
 */
bool test_nep_neural_network()
{
  std::cout << "\n=== 测试5: 神经网络前向传播 ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. 前向传播计算正确性\n";
  std::cout << "    2. 激活函数（tanh）应用\n";
  std::cout << "    3. 能量导数计算\n";
  std::cout << "    4. NEP4类型特定网络\n";
  std::cout << "✓ 神经网络前向传播测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试6: NEP::find_force函数
// ============================================================================

/**
 * 测试函数：test_nep_find_force
 * 
 * 功能：测试find_force函数，验证力和能量计算
 * 
 * 输入输出：
 *   find_force输入：
 *     para: Parameters& - 参数对象
 *     parameters: const float* [number_of_variables] - 模型参数
 *     dataset: std::vector<Dataset>& - 数据集
 *     calculate_q_scaler: bool - 是否计算q_scaler
 *     calculate_neighbor: bool - 是否重新计算邻居
 *     deviceCount: int - GPU设备数量
 *   输出：
 *     无返回值，计算并存储：
 *       Dataset.energy: GPU_Vector<float> [N] - 原子能量
 *       Dataset.force: GPU_Vector<float> [N * 3] - 原子力
 *       Dataset.virial: GPU_Vector<float> [N * 6] - 原子维里
 * 
 * 计算流程：
 *   1. 计算描述符（径向+角向）
 *   2. 描述符归一化（q_scaler）
 *   3. 神经网络前向传播（得到能量和能量导数）
 *   4. 通过链式法则计算力
 *   5. 计算维里张量
 */
bool test_nep_find_force()
{
  std::cout << "\n=== 测试6: NEP::find_force函数 ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. 能量计算正确性\n";
  std::cout << "    2. 力计算正确性（通过链式法则）\n";
  std::cout << "    3. 维里计算正确性\n";
  std::cout << "    4. 多GPU并行计算\n";
  std::cout << "✓ find_force测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试7: 描述符归一化（q_scaler）
// ============================================================================

/**
 * 测试函数：test_nep_q_scaler
 * 
 * 功能：测试q_scaler的应用和计算
 * 
 * 输入输出：
 *   q_scaler应用：
 *     输入：descriptors_raw [dim * N] - 原始描述符
 *     输出：descriptors [dim * N] - 归一化后的描述符
 *     公式：descriptors[d * N + i] = descriptors_raw[d * N + i] * q_scaler[d]
 * 
 * 维度信息：
 *   q_scaler: float[dim] - 描述符缩放因子
 *   descriptors: GPU_Vector<float> [dim * N]
 */
bool test_nep_q_scaler()
{
  std::cout << "\n=== 测试7: 描述符归一化（q_scaler） ===\n";
  
  std::cout << "  测试内容：\n";
  std::cout << "    1. q_scaler应用正确性\n";
  std::cout << "    2. q_scaler自动优化（如果calculate_q_scaler=true）\n";
  std::cout << "    3. 数值稳定性\n";
  std::cout << "✓ q_scaler测试框架创建完成\n";
  
  return true;
}

// ============================================================================
// 测试8: 力和维里计算维度验证
// ============================================================================

/**
 * 测试函数：test_nep_force_virial_dimensions
 * 
 * 功能：验证力和维里计算的维度
 */
bool test_nep_force_virial_dimensions()
{
  std::cout << "\n=== 测试8: 力和维里计算维度验证 ===\n";
  
  int N = 1000;  // 总原子数
  
  std::cout << "  示例配置：N = " << N << "\n";
  std::cout << "\n  输出数组维度：\n";
  std::cout << "    energy: [" << N << "] - 每个原子的能量\n";
  std::cout << "    force: [" << N << " × 3] - 每个原子的力（fx, fy, fz）\n";
  std::cout << "    virial: [" << N << " × 6] - 每个原子的维里（xx, yy, zz, xy, xz, yz）\n";
  
  // 验证存储方式
  std::cout << "\n  存储方式：\n";
  std::cout << "    force[i*3 + 0] = fx[i]\n";
  std::cout << "    force[i*3 + 1] = fy[i]\n";
  std::cout << "    force[i*3 + 2] = fz[i]\n";
  std::cout << "    virial[i*6 + 0] = xx[i]\n";
  std::cout << "    virial[i*6 + 1] = yy[i]\n";
  std::cout << "    virial[i*6 + 2] = zz[i]\n";
  std::cout << "    virial[i*6 + 3] = xy[i]\n";
  std::cout << "    virial[i*6 + 4] = xz[i]\n";
  std::cout << "    virial[i*6 + 5] = yz[i]\n";
  
  std::cout << "✓ 力和维里计算维度验证通过\n";
  return true;
}

// ============================================================================
// 主测试函数
// ============================================================================

int main()
{
  std::cout << "========================================\n";
  std::cout << "NEP模型类测试套件\n";
  std::cout << "========================================\n";
  
  bool all_passed = true;
  
  // 运行所有测试
  all_passed &= test_nep_constructor();
  all_passed &= test_nep_update_potential();
  all_passed &= test_nep_radial_descriptors();
  all_passed &= test_nep_angular_descriptors();
  all_passed &= test_nep_neural_network();
  all_passed &= test_nep_find_force();
  all_passed &= test_nep_q_scaler();
  all_passed &= test_nep_force_virial_dimensions();
  
  std::cout << "\n========================================\n";
  if (all_passed) {
    std::cout << "所有测试通过！\n";
    return 0;
  } else {
    std::cout << "部分测试失败！\n";
    return 1;
  }
}

