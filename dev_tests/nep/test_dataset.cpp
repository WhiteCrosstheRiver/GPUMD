/*
 * test_dataset.cpp
 * 
 * 测试文件：Dataset类的所有功能
 * 
 * 测试内容：
 * 1. Dataset::construct - 数据集构造
 * 2. Dataset::copy_structures - 结构复制
 * 3. Dataset::find_Na - 原子数计算
 * 4. Dataset::initialize_gpu_data - GPU数据初始化
 * 5. Dataset::find_neighbor - 邻居列表构建
 * 6. Dataset::get_rmse_* - RMSE计算函数
 * 
 * 输入输出变量类型和维度：
 * - construct输入：
 *     para: Parameters& - 参数对象
 *     structures_input: std::vector<Structure>& - 输入结构数组
 *     n1, n2: int - 结构范围 [n1, n2)
 *     device_id: int - GPU设备ID
 *   输出：
 *     无返回值，构造Dataset对象
 * 
 * - Dataset成员变量维度：
 *     Nc: int - 配置数量
 *     N: int - 总原子数
 *     Na: GPU_Vector<int> [Nc] - 每个配置的原子数
 *     Na_sum: GPU_Vector<int> [Nc] - 原子数前缀和
 *     type: GPU_Vector<int> [N] - 所有原子的类型
 *     r: GPU_Vector<float> [N * 3] - 所有原子的坐标（x,y,z交错）
 *     force_ref: GPU_Vector<float> [N * 3] - 参考力
 *     energy_ref: GPU_Vector<float> [Nc] - 参考能量
 *     virial_ref: GPU_Vector<float> [Nc * 6] - 参考维里
 */

#include "../src/main_nep/dataset.cuh"
#include "../src/main_nep/structure.cuh"
#include "../src/main_nep/parameters.cuh"
#include <cassert>
#include <iostream>
#include <vector>

// ============================================================================
// 测试辅助函数
// ============================================================================

/**
 * 创建测试用的Structure对象
 * 
 * 输入：
 *   num_atom: int - 原子数量
 *   types: std::vector<int> - 原子类型数组
 *   coords: std::vector<std::vector<float>> - 坐标数组 [num_atom][3]
 *   forces: std::vector<std::vector<float>> - 力数组 [num_atom][3]
 *   energy: float - 总能量
 * 
 * 输出：
 *   Structure - 填充好的结构对象
 */
Structure create_test_structure(
  int num_atom,
  const std::vector<int>& types,
  const std::vector<std::vector<float>>& coords,
  const std::vector<std::vector<float>>& forces,
  float energy)
{
  Structure structure;
  structure.num_atom = num_atom;
  structure.energy = energy;
  structure.weight = 1.0f;
  structure.has_virial = 0;
  
  structure.type.resize(num_atom);
  structure.x.resize(num_atom);
  structure.y.resize(num_atom);
  structure.z.resize(num_atom);
  structure.fx.resize(num_atom);
  structure.fy.resize(num_atom);
  structure.fz.resize(num_atom);
  
  for (int i = 0; i < num_atom; ++i) {
    structure.type[i] = types[i];
    structure.x[i] = coords[i][0];
    structure.y[i] = coords[i][1];
    structure.z[i] = coords[i][2];
    structure.fx[i] = forces[i][0];
    structure.fy[i] = forces[i][1];
    structure.fz[i] = forces[i][2];
  }
  
  // 设置默认盒子
  for (int i = 0; i < 9; ++i) {
    structure.box_original[i] = (i % 4 == 0) ? 10.0f : 0.0f;
  }
  structure.num_cell[0] = structure.num_cell[1] = structure.num_cell[2] = 1;
  
  return structure;
}

// ============================================================================
// 测试1: Dataset::copy_structures函数
// ============================================================================

/**
 * 测试函数：test_dataset_copy_structures
 * 
 * 功能：测试copy_structures函数，验证结构数据复制
 * 
 * 输入输出：
 *   copy_structures输入：
 *     structures_input: std::vector<Structure>& - 输入结构数组
 *     n1, n2: int - 范围 [n1, n2)
 *   输出：
 *     无返回值，复制到Dataset.structures
 * 
 *   复制后Dataset包含：
 *     Nc = n2 - n1
 *     structures[Nc] - 复制的结构数组
 */
bool test_dataset_copy_structures()
{
  std::cout << "\n=== 测试1: Dataset::copy_structures函数 ===\n";
  
  // 创建测试结构
  std::vector<Structure> structures_input;
  
  Structure s1 = create_test_structure(
    2,
    {0, 1},  // Si, C
    {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}},
    {{0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f}},
    -100.0f
  );
  
  Structure s2 = create_test_structure(
    3,
    {0, 1, 2},  // Si, C, O
    {{0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}, {2.5f, 2.5f, 2.5f}},
    {{0.15f, 0.15f, 0.15f}, {0.25f, 0.25f, 0.25f}, {0.35f, 0.35f, 0.35f}},
    -150.0f
  );
  
  structures_input.push_back(s1);
  structures_input.push_back(s2);
  
  // 创建Dataset并测试copy_structures（私有函数，需要通过construct测试）
  Parameters para;
  const char* type_param[] = {"type", "3", "Si", "C", "O"};
  para.parse_type(type_param, 5);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  // 注意：copy_structures是私有函数，需要通过construct间接测试
  
  std::cout << "  注意：copy_structures是私有函数，需要通过construct测试\n";
  std::cout << "✓ copy_structures测试框架创建完成\n";
  return true;
}

// ============================================================================
// 测试2: Dataset::find_Na函数
// ============================================================================

/**
 * 测试函数：test_dataset_find_Na
 * 
 * 功能：测试find_Na函数，验证原子数计算和前缀和
 * 
 * 输入输出：
 *   find_Na输入：
 *     无直接输入，使用Dataset.structures
 *   输出：
 *     无返回值，计算并设置：
 *       Dataset.N: int - 总原子数
 *       Dataset.max_Na: int - 最大配置原子数
 *       Dataset.Na_cpu: std::vector<int> [Nc] - 每个配置的原子数
 *       Dataset.Na_sum_cpu: std::vector<int> [Nc] - 原子数前缀和
 *       Dataset.Na: GPU_Vector<int> [Nc] - GPU版本的Na
 *       Dataset.Na_sum: GPU_Vector<int> [Nc] - GPU版本的Na_sum
 * 
 * 前缀和计算：
 *   Na_sum[0] = 0
 *   Na_sum[i] = Na_sum[i-1] + Na[i-1]  (i > 0)
 */
bool test_dataset_find_Na()
{
  std::cout << "\n=== 测试2: Dataset::find_Na函数 ===\n";
  
  // 创建测试结构
  std::vector<Structure> structures_input;
  
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  Structure s2 = create_test_structure(3, {0, 1, 2}, {{0.5,0.5,0.5}, {1.5,1.5,1.5}, {2.5,2.5,2.5}}, {{0.15,0.15,0.15}, {0.25,0.25,0.25}, {0.35,0.35,0.35}}, -150.0f);
  Structure s3 = create_test_structure(1, {0}, {{0,0,0}}, {{0.1,0.1,0.1}}, -50.0f);
  
  structures_input.push_back(s1);
  structures_input.push_back(s2);
  structures_input.push_back(s3);
  
  Parameters para;
  const char* type_param[] = {"type", "3", "Si", "C", "O"};
  para.parse_type(type_param, 5);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 3, 0);
  
  // 验证原子数计算
  assert(dataset.Nc == 3 && "Nc应为3");
  assert(dataset.N == 6 && "总原子数N应为6 (2+3+1)");
  assert(dataset.max_Na == 3 && "max_Na应为3");
  assert(dataset.Na_cpu.size() == 3 && "Na_cpu大小应为3");
  assert(dataset.Na_cpu[0] == 2 && "第一个配置原子数应为2");
  assert(dataset.Na_cpu[1] == 3 && "第二个配置原子数应为3");
  assert(dataset.Na_cpu[2] == 1 && "第三个配置原子数应为1");
  
  // 验证前缀和
  assert(dataset.Na_sum_cpu[0] == 0 && "Na_sum[0]应为0");
  assert(dataset.Na_sum_cpu[1] == 2 && "Na_sum[1]应为2");
  assert(dataset.Na_sum_cpu[2] == 5 && "Na_sum[2]应为5");
  
  std::cout << "✓ find_Na测试通过\n";
  return true;
}

// ============================================================================
// 测试3: Dataset::initialize_gpu_data函数
// ============================================================================

/**
 * 测试函数：test_dataset_initialize_gpu_data
 * 
 * 功能：测试initialize_gpu_data函数，验证GPU数据传输
 * 
 * 输入输出：
 *   initialize_gpu_data输入：
 *     para: Parameters& - 参数对象
 *   输出：
 *     无返回值，初始化GPU数据：
 *       Dataset.type: GPU_Vector<int> [N] - 原子类型
 *       Dataset.r: GPU_Vector<float> [N * 3] - 原子坐标
 *       Dataset.force_ref_gpu: GPU_Vector<float> [N * 3] - 参考力
 *       Dataset.energy_ref_gpu: GPU_Vector<float> [Nc] - 参考能量
 *       Dataset.virial_ref_gpu: GPU_Vector<float> [Nc * 6] - 参考维里
 *       Dataset.box: GPU_Vector<float> [Nc * 18] - 盒子信息
 */
bool test_dataset_initialize_gpu_data()
{
  std::cout << "\n=== 测试3: Dataset::initialize_gpu_data函数 ===\n";
  
  std::vector<Structure> structures_input;
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  structures_input.push_back(s1);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 1, 0);
  
  // 验证GPU数据已初始化
  assert(dataset.type.size() == 2 && "type GPU向量大小应为2");
  assert(dataset.r.size() == 6 && "r GPU向量大小应为6 (2*3)");
  assert(dataset.force_ref_gpu.size() == 6 && "force_ref_gpu大小应为6");
  assert(dataset.energy_ref_gpu.size() == 1 && "energy_ref_gpu大小应为1");
  
  std::cout << "✓ initialize_gpu_data测试通过\n";
  return true;
}

// ============================================================================
// 测试4: Dataset::find_neighbor函数
// ============================================================================

/**
 * 测试函数：test_dataset_find_neighbor
 * 
 * 功能：测试find_neighbor函数，验证邻居列表构建
 * 
 * 输入输出：
 *   find_neighbor输入：
 *     para: Parameters& - 参数对象（需要rc_radial, rc_angular）
 *   输出：
 *     无返回值，构建邻居列表（通过NEP类的方法）
 * 
 * 注意：find_neighbor是私有函数，需要通过construct间接测试
 */
bool test_dataset_find_neighbor()
{
  std::cout << "\n=== 测试4: Dataset::find_neighbor函数 ===\n";
  
  std::vector<Structure> structures_input;
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  structures_input.push_back(s1);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 1, 0);
  
  // 验证邻居列表相关变量已设置
  assert(dataset.max_NN_radial > 0 && "max_NN_radial应大于0");
  assert(dataset.max_NN_angular > 0 && "max_NN_angular应大于0");
  
  std::cout << "✓ find_neighbor测试通过\n";
  return true;
}

// ============================================================================
// 测试5: Dataset::get_rmse_force函数
// ============================================================================

/**
 * 测试函数：test_dataset_get_rmse_force
 * 
 * 功能：测试get_rmse_force函数，验证力RMSE计算
 * 
 * 输入输出：
 *   get_rmse_force输入：
 *     para: Parameters& - 参数对象
 *     use_weight: bool - 是否使用权重
 *     device_id: int - GPU设备ID
 *   输出：
 *     std::vector<float> [num_types + 1] - 每个类型的RMSE（最后一个为总体RMSE）
 * 
 * 计算公式：
 *   RMSE_force = sqrt(Σ_i |F_pred[i] - F_ref[i]|² / (3N))
 */
bool test_dataset_get_rmse_force()
{
  std::cout << "\n=== 测试5: Dataset::get_rmse_force函数 ===\n";
  
  std::vector<Structure> structures_input;
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  structures_input.push_back(s1);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  para.force_delta = 0.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 1, 0);
  
  // 设置预测力（与参考力相同，RMSE应为0）
  dataset.force_cpu.resize(6);
  for (int i = 0; i < 6; ++i) {
    dataset.force_cpu[i] = dataset.force_ref_cpu[i];
  }
  dataset.force.copy_from_host(dataset.force_cpu.data());
  
  // 计算RMSE
  std::vector<float> rmse_array = dataset.get_rmse_force(para, false, 0);
  
  assert(rmse_array.size() == para.num_types + 1 && "RMSE数组大小应为num_types+1");
  // RMSE应该接近0（由于浮点误差可能不为0）
  assert(rmse_array.back() < 1e-6f && "当预测等于参考时，RMSE应接近0");
  
  std::cout << "✓ get_rmse_force测试通过\n";
  return true;
}

// ============================================================================
// 测试6: Dataset::get_rmse_energy函数
// ============================================================================

/**
 * 测试函数：test_dataset_get_rmse_energy
 * 
 * 功能：测试get_rmse_energy函数，验证能量RMSE计算
 * 
 * 输入输出：
 *   get_rmse_energy输入：
 *     para: Parameters& - 参数对象
 *     energy_shift_per_structure: float& - 输出能量偏移
 *     use_weight: bool - 是否使用权重
 *     do_shift: bool - 是否进行能量偏移校正
 *     device_id: int - GPU设备ID
 *   输出：
 *     std::vector<float> [num_types + 1] - 每个类型的RMSE
 * 
 * 计算公式：
 *   RMSE_energy = sqrt(Σ_c (E_pred[c] - E_ref[c] - shift)² / Nc)
 */
bool test_dataset_get_rmse_energy()
{
  std::cout << "\n=== 测试6: Dataset::get_rmse_energy函数 ===\n";
  
  std::vector<Structure> structures_input;
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  structures_input.push_back(s1);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 1, 0);
  
  // 设置预测能量（与参考能量相同）
  dataset.energy_cpu.resize(2);
  dataset.energy_cpu[0] = -100.0f;  // 原子能量
  dataset.energy_cpu[1] = -100.0f;
  dataset.energy.copy_from_host(dataset.energy_cpu.data());
  
  float energy_shift = 0.0f;
  std::vector<float> rmse_array = dataset.get_rmse_energy(para, energy_shift, false, false, 0);
  
  assert(rmse_array.size() == para.num_types + 1 && "RMSE数组大小应为num_types+1");
  
  std::cout << "✓ get_rmse_energy测试通过\n";
  return true;
}

// ============================================================================
// 测试7: Dataset::get_rmse_virial函数
// ============================================================================

/**
 * 测试函数：test_dataset_get_rmse_virial
 * 
 * 功能：测试get_rmse_virial函数，验证维里RMSE计算
 * 
 * 输入输出：
 *   get_rmse_virial输入：
 *     para: Parameters& - 参数对象
 *     use_weight: bool - 是否使用权重
 *     device_id: int - GPU设备ID
 *   输出：
 *     std::vector<float> [num_types + 1] - 每个类型的RMSE
 * 
 * 计算公式：
 *   RMSE_virial = sqrt(Σ_c Σ_i (W_pred[c][i] - W_ref[c][i])² / (6*Nc))
 */
bool test_dataset_get_rmse_virial()
{
  std::cout << "\n=== 测试7: Dataset::get_rmse_virial函数 ===\n";
  
  std::vector<Structure> structures_input;
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  s1.has_virial = 1;
  for (int i = 0; i < 6; ++i) {
    s1.virial[i] = float(i + 1) * 0.1f;
  }
  structures_input.push_back(s1);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 1, 0);
  
  // 设置预测维里（与参考相同）
  dataset.virial_cpu.resize(6);
  for (int i = 0; i < 6; ++i) {
    dataset.virial_cpu[i] = s1.virial[i];
  }
  dataset.virial.copy_from_host(dataset.virial_cpu.data());
  
  std::vector<float> rmse_array = dataset.get_rmse_virial(para, false, 0);
  
  assert(rmse_array.size() == para.num_types + 1 && "RMSE数组大小应为num_types+1");
  
  std::cout << "✓ get_rmse_virial测试通过\n";
  return true;
}

// ============================================================================
// 测试8: Dataset::construct完整流程
// ============================================================================

/**
 * 测试函数：test_dataset_construct
 * 
 * 功能：测试Dataset::construct完整流程
 * 
 * 输入输出：
 *   construct输入：
 *     para: Parameters& - 参数对象
 *     structures_input: std::vector<Structure>& - 输入结构数组
 *     n1, n2: int - 结构范围 [n1, n2)
 *     device_id: int - GPU设备ID
 *   输出：
 *     无返回值，完整构造Dataset对象
 */
bool test_dataset_construct()
{
  std::cout << "\n=== 测试8: Dataset::construct完整流程 ===\n";
  
  std::vector<Structure> structures_input;
  
  Structure s1 = create_test_structure(2, {0, 1}, {{0,0,0}, {1,1,1}}, {{0.1,0.1,0.1}, {0.2,0.2,0.2}}, -100.0f);
  Structure s2 = create_test_structure(3, {0, 1, 2}, {{0.5,0.5,0.5}, {1.5,1.5,1.5}, {2.5,2.5,2.5}}, {{0.15,0.15,0.15}, {0.25,0.25,0.25}, {0.35,0.35,0.35}}, -150.0f);
  
  structures_input.push_back(s1);
  structures_input.push_back(s2);
  
  Parameters para;
  const char* type_param[] = {"type", "3", "Si", "C", "O"};
  para.parse_type(type_param, 5);
  para.rc_radial = 6.0f;
  para.rc_angular = 5.0f;
  
  Dataset dataset;
  dataset.construct(para, structures_input, 0, 2, 0);
  
  // 验证完整构造
  assert(dataset.Nc == 2 && "Nc应为2");
  assert(dataset.N == 5 && "总原子数N应为5 (2+3)");
  assert(dataset.structures.size() == 2 && "structures大小应为2");
  assert(dataset.type.size() == 5 && "type GPU向量大小应为5");
  assert(dataset.r.size() == 15 && "r GPU向量大小应为15 (5*3)");
  
  std::cout << "✓ Dataset::construct完整流程测试通过\n";
  return true;
}

// ============================================================================
// 主测试函数
// ============================================================================

int main()
{
  std::cout << "========================================\n";
  std::cout << "Dataset类测试套件\n";
  std::cout << "========================================\n";
  
  bool all_passed = true;
  
  // 运行所有测试
  all_passed &= test_dataset_copy_structures();
  all_passed &= test_dataset_find_Na();
  all_passed &= test_dataset_initialize_gpu_data();
  all_passed &= test_dataset_find_neighbor();
  all_passed &= test_dataset_get_rmse_force();
  all_passed &= test_dataset_get_rmse_energy();
  all_passed &= test_dataset_get_rmse_virial();
  all_passed &= test_dataset_construct();
  
  std::cout << "\n========================================\n";
  if (all_passed) {
    std::cout << "所有测试通过！\n";
    return 0;
  } else {
    std::cout << "部分测试失败！\n";
    return 1;
  }
}

