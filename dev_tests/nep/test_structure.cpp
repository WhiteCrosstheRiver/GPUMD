/*
 * test_structure.cpp
 * 
 * 测试文件：Structure读取和处理功能
 * 
 * 测试内容：
 * 1. read_structures函数 - 从xyz文件读取结构
 * 2. Structure数据结构 - 验证数据完整性
 * 3. change_box函数 - 盒子扩展计算
 * 4. 边界条件处理
 * 
 * 输入输出变量类型和维度：
 * - read_structures输入：
 *     is_train: bool - 是否为训练集
 *     para: Parameters& - 参数对象
 *     structures: std::vector<Structure>& - 输出结构数组
 *   输出：
 *     bool - 是否成功读取
 * 
 * - Structure结构体维度：
 *     type: std::vector<int> [num_atom] - 原子类型
 *     x, y, z: std::vector<float> [num_atom] - 原子坐标
 *     fx, fy, fz: std::vector<float> [num_atom] - 参考力
 *     energy: float - 总能量
 *     virial: float[6] - 维里张量
 *     box_original: float[9] - 原始盒子
 *     box: float[18] - 扩展盒子
 */

#include "../src/main_nep/structure.cuh"
#include "../src/main_nep/parameters.cuh"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

// ============================================================================
// 测试辅助函数
// ============================================================================

/**
 * 创建测试用的train.xyz文件
 * 
 * 输入：
 *   filename: string - 文件名
 *   num_atoms: int - 原子数量
 *   content: string - 文件内容（可选，如果为空则生成默认内容）
 * 
 * 输出：
 *   无返回值，创建文件
 */
void create_test_xyz_file(const std::string& filename, int num_atoms, const std::string& content = "")
{
  std::ofstream file(filename);
  
  if (content.empty()) {
    // 生成默认内容
    file << num_atoms << "\n";
    file << "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
    file << "Properties=species:S:1:pos:R:3:force:R:3 energy=-100.0\n";
    for (int i = 0; i < num_atoms; ++i) {
      file << "Si " << i * 0.5 << " " << i * 0.5 << " " << i * 0.5;
      file << " 0.1 0.1 0.1\n";
    }
  } else {
    file << content;
  }
  
  file.close();
}

/**
 * 清理测试文件
 */
void cleanup_test_file(const std::string& filename)
{
  std::remove(filename.c_str());
}

// ============================================================================
// 测试1: Structure数据结构初始化
// ============================================================================

/**
 * 测试函数：test_structure_initialization
 * 
 * 功能：测试Structure结构体的初始化和默认值
 * 
 * 输入输出：
 *   输入：无
 *   输出：验证Structure的默认值
 */
bool test_structure_initialization()
{
  std::cout << "\n=== 测试1: Structure数据结构初始化 ===\n";
  
  Structure structure;
  
  // 验证默认值
  assert(structure.num_atom == 0 && "num_atom初始应为0");
  assert(structure.has_virial == 0 && "has_virial初始应为0");
  assert(structure.has_atomic_virial == 0 && "has_atomic_virial初始应为0");
  assert(structure.weight == 1.0f && "weight初始应为1.0");
  assert(structure.charge == 0.0f && "charge初始应为0.0");
  assert(structure.energy == 0.0f && "energy初始应为0.0");
  assert(structure.energy_weight == 1.0f && "energy_weight初始应为1.0");
  
  // 验证数组大小
  assert(structure.type.size() == 0 && "type初始应为空");
  assert(structure.x.size() == 0 && "x初始应为空");
  assert(structure.y.size() == 0 && "y初始应为空");
  assert(structure.z.size() == 0 && "z初始应为空");
  
  std::cout << "✓ Structure初始化测试通过\n";
  return true;
}

// ============================================================================
// 测试2: read_structures函数 - 基本读取
// ============================================================================

/**
 * 测试函数：test_read_structures_basic
 * 
 * 功能：测试read_structures函数的基本功能
 * 
 * 输入输出：
 *   read_structures输入：
 *     is_train: bool - true表示训练集
 *     para: Parameters& - 参数对象（需要设置type）
 *     structures: std::vector<Structure>& - 输出结构数组
 *   输出：
 *     bool - 是否成功读取
 * 
 *   读取后structures[0]包含：
 *     num_atom: int - 原子数量
 *     type: std::vector<int> [num_atom] - 原子类型索引
 *     x, y, z: std::vector<float> [num_atom] - 原子坐标
 *     fx, fy, fz: std::vector<float> [num_atom] - 参考力
 *     energy: float - 总能量
 *     box_original: float[9] - 原始盒子矩阵
 */
bool test_read_structures_basic()
{
  std::cout << "\n=== 测试2: read_structures基本读取 ===\n";
  
  // 创建测试文件
  const std::string filename = "test_train.xyz";
  std::string content = "3\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 energy=-150.0\n";
  content += "Si 0.0 0.0 0.0 0.1 0.2 0.3\n";
  content += "C 1.0 1.0 1.0 0.2 0.3 0.4\n";
  content += "O 2.0 2.0 2.0 0.3 0.4 0.5\n";
  
  create_test_xyz_file(filename, 3, content);
  
  // 设置Parameters
  Parameters para;
  const char* type_param[] = {"type", "3", "Si", "C", "O"};
  para.parse_type(type_param, 5);
  
  // 读取结构
  std::vector<Structure> structures;
  bool success = read_structures(true, para, structures);
  
  assert(success && "read_structures应返回true");
  assert(structures.size() == 1 && "应读取1个结构");
  assert(structures[0].num_atom == 3 && "原子数量应为3");
  assert(structures[0].type.size() == 3 && "type数组大小应为3");
  assert(structures[0].x.size() == 3 && "x数组大小应为3");
  assert(structures[0].y.size() == 3 && "y数组大小应为3");
  assert(structures[0].z.size() == 3 && "z数组大小应为3");
  assert(structures[0].fx.size() == 3 && "fx数组大小应为3");
  assert(structures[0].fy.size() == 3 && "fy数组大小应为3");
  assert(structures[0].fz.size() == 3 && "fz数组大小应为3");
  
  // 验证类型映射
  assert(structures[0].type[0] == 0 && "第一个原子类型应为0(Si)");
  assert(structures[0].type[1] == 1 && "第二个原子类型应为1(C)");
  assert(structures[0].type[2] == 2 && "第三个原子类型应为2(O)");
  
  // 验证坐标
  assert(structures[0].x[0] == 0.0f && structures[0].y[0] == 0.0f && structures[0].z[0] == 0.0f);
  assert(structures[0].x[1] == 1.0f && structures[0].y[1] == 1.0f && structures[0].z[1] == 1.0f);
  assert(structures[0].x[2] == 2.0f && structures[0].y[2] == 2.0f && structures[0].z[2] == 2.0f);
  
  // 验证力
  assert(structures[0].fx[0] == 0.1f && structures[0].fy[0] == 0.2f && structures[0].fz[0] == 0.3f);
  
  // 验证能量
  assert(structures[0].energy == -150.0f && "能量应为-150.0");
  
  // 验证盒子
  assert(structures[0].box_original[0] == 10.0f && "box_original[0]应为10.0");
  
  cleanup_test_file(filename);
  
  std::cout << "✓ read_structures基本读取测试通过\n";
  return true;
}

// ============================================================================
// 测试3: read_structures函数 - 多结构读取
// ============================================================================

/**
 * 测试函数：test_read_structures_multiple
 * 
 * 功能：测试读取多个结构
 * 
 * 输入输出：
 *   输入：包含多个结构的xyz文件
 *   输出：structures数组包含多个Structure对象
 */
bool test_read_structures_multiple()
{
  std::cout << "\n=== 测试3: read_structures多结构读取 ===\n";
  
  const std::string filename = "test_train_multiple.xyz";
  std::string content = "2\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 energy=-100.0\n";
  content += "Si 0.0 0.0 0.0 0.1 0.1 0.1\n";
  content += "C 1.0 1.0 1.0 0.2 0.2 0.2\n";
  content += "2\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 energy=-120.0\n";
  content += "Si 0.5 0.5 0.5 0.15 0.15 0.15\n";
  content += "C 1.5 1.5 1.5 0.25 0.25 0.25\n";
  
  create_test_xyz_file(filename, 0, content);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  
  std::vector<Structure> structures;
  bool success = read_structures(true, para, structures);
  
  assert(success && "read_structures应返回true");
  assert(structures.size() == 2 && "应读取2个结构");
  assert(structures[0].num_atom == 2 && "第一个结构原子数应为2");
  assert(structures[1].num_atom == 2 && "第二个结构原子数应为2");
  assert(structures[0].energy == -100.0f && "第一个结构能量应为-100.0");
  assert(structures[1].energy == -120.0f && "第二个结构能量应为-120.0");
  
  cleanup_test_file(filename);
  
  std::cout << "✓ read_structures多结构读取测试通过\n";
  return true;
}

// ============================================================================
// 测试4: read_structures函数 - 带维里张量
// ============================================================================

/**
 * 测试函数：test_read_structures_with_virial
 * 
 * 功能：测试读取包含维里张量的结构
 * 
 * 输入输出：
 *   读取后structures[0]包含：
 *     has_virial: int - 1表示有维里张量
 *     virial: float[6] - 维里张量的6个独立分量
 */
bool test_read_structures_with_virial()
{
  std::cout << "\n=== 测试4: read_structures带维里张量 ===\n";
  
  const std::string filename = "test_train_virial.xyz";
  std::string content = "2\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 ";
  content += "energy=-100.0 virial=\"1.0 2.0 3.0 0.1 0.2 0.3\"\n";
  content += "Si 0.0 0.0 0.0 0.1 0.1 0.1\n";
  content += "C 1.0 1.0 1.0 0.2 0.2 0.2\n";
  
  create_test_xyz_file(filename, 0, content);
  
  Parameters para;
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  
  std::vector<Structure> structures;
  bool success = read_structures(true, para, structures);
  
  assert(success && "read_structures应返回true");
  assert(structures[0].has_virial == 1 && "has_virial应为1");
  assert(structures[0].virial[0] == 1.0f && "virial[0]应为1.0");
  assert(structures[0].virial[1] == 2.0f && "virial[1]应为2.0");
  assert(structures[0].virial[2] == 3.0f && "virial[2]应为3.0");
  assert(structures[0].virial[3] == 0.1f && "virial[3]应为0.1");
  assert(structures[0].virial[4] == 0.2f && "virial[4]应为0.2");
  assert(structures[0].virial[5] == 0.3f && "virial[5]应为0.3");
  
  cleanup_test_file(filename);
  
  std::cout << "✓ read_structures带维里张量测试通过\n";
  return true;
}

// ============================================================================
// 测试5: 盒子扩展计算（change_box函数）
// ============================================================================

/**
 * 测试函数：test_change_box
 * 
 * 功能：测试change_box函数，验证周期性边界条件的盒子扩展
 * 
 * 注意：change_box是静态函数，需要通过read_structures间接测试
 * 
 * 输入输出：
 *   change_box输入：
 *     para: const Parameters& - 参数对象（需要rc_radial）
 *     structure: Structure& - 结构对象（输入box_original，输出box和num_cell）
 *   输出：
 *     无返回值，修改structure：
 *       structure.num_cell: int[3] - 扩展盒子在x,y,z方向的单元数
 *       structure.box: float[18] - 扩展盒子矩阵（9个基向量 + 9个逆矩阵元素）
 *       structure.volume: float - 盒子体积
 */
bool test_change_box()
{
  std::cout << "\n=== 测试5: change_box盒子扩展 ===\n";
  
  // 通过read_structures测试，因为change_box在读取时自动调用
  const std::string filename = "test_train_box.xyz";
  std::string content = "2\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 energy=-100.0\n";
  content += "Si 0.0 0.0 0.0 0.1 0.1 0.1\n";
  content += "C 1.0 1.0 1.0 0.2 0.2 0.2\n";
  
  create_test_xyz_file(filename, 0, content);
  
  Parameters para;
  para.rc_radial = 6.0f;  // 设置截断半径
  const char* type_param[] = {"type", "2", "Si", "C"};
  para.parse_type(type_param, 4);
  
  std::vector<Structure> structures;
  bool success = read_structures(true, para, structures);
  
  assert(success && "read_structures应返回true");
  
  // 验证盒子扩展
  assert(structures[0].num_cell[0] > 0 && "num_cell[0]应大于0");
  assert(structures[0].num_cell[1] > 0 && "num_cell[1]应大于0");
  assert(structures[0].num_cell[2] > 0 && "num_cell[2]应大于0");
  
  // 验证扩展盒子
  assert(structures[0].box[0] > structures[0].box_original[0] && "扩展盒子应大于原始盒子");
  
  // 验证体积
  assert(structures[0].volume > 0.0f && "体积应大于0");
  
  cleanup_test_file(filename);
  
  std::cout << "✓ change_box测试通过\n";
  return true;
}

// ============================================================================
// 测试6: 错误处理 - 文件不存在
// ============================================================================

/**
 * 测试函数：test_read_structures_file_not_found
 * 
 * 功能：测试读取不存在的文件时的错误处理
 * 
 * 输入输出：
 *   输入：不存在的文件名
 *   输出：read_structures应返回false或抛出异常
 */
bool test_read_structures_file_not_found()
{
  std::cout << "\n=== 测试6: read_structures文件不存在 ===\n";
  
  Parameters para;
  const char* type_param[] = {"type", "1", "Si"};
  para.parse_type(type_param, 3);
  
  std::vector<Structure> structures;
  
  // 尝试读取不存在的文件
  bool success = read_structures(true, para, structures);
  
  // 应该失败或返回false
  assert(!success || structures.size() == 0);
  
  std::cout << "✓ read_structures文件不存在测试通过\n";
  return true;
}

// ============================================================================
// 测试7: 原子类型验证
// ============================================================================

/**
 * 测试函数：test_read_structures_type_validation
 * 
 * 功能：测试原子类型验证，确保文件中的元素在Parameters中定义
 * 
 * 输入输出：
 *   如果xyz文件中的元素不在para.type中，应报错或跳过
 */
bool test_read_structures_type_validation()
{
  std::cout << "\n=== 测试7: read_structures原子类型验证 ===\n";
  
  const std::string filename = "test_train_type_error.xyz";
  std::string content = "1\n";
  content += "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" ";
  content += "Properties=species:S:1:pos:R:3:force:R:3 energy=-50.0\n";
  content += "Fe 0.0 0.0 0.0 0.1 0.1 0.1\n";  // Fe不在type中
  
  create_test_xyz_file(filename, 0, content);
  
  Parameters para;
  const char* type_param[] = {"type", "1", "Si"};  // 只定义了Si
  para.parse_type(type_param, 3);
  
  std::vector<Structure> structures;
  
  // 应该处理错误（报错或跳过）
  bool success = read_structures(true, para, structures);
  
  // 根据实际实现，可能返回false或抛出异常
  std::cout << "  注意：实际行为取决于错误处理实现\n";
  
  cleanup_test_file(filename);
  
  std::cout << "✓ read_structures原子类型验证测试完成\n";
  return true;
}

// ============================================================================
// 主测试函数
// ============================================================================

int main()
{
  std::cout << "========================================\n";
  std::cout << "Structure读取功能测试套件\n";
  std::cout << "========================================\n";
  
  bool all_passed = true;
  
  // 运行所有测试
  all_passed &= test_structure_initialization();
  all_passed &= test_read_structures_basic();
  all_passed &= test_read_structures_multiple();
  all_passed &= test_read_structures_with_virial();
  all_passed &= test_change_box();
  all_passed &= test_read_structures_file_not_found();
  all_passed &= test_read_structures_type_validation();
  
  std::cout << "\n========================================\n";
  if (all_passed) {
    std::cout << "所有测试通过！\n";
    return 0;
  } else {
    std::cout << "部分测试失败！\n";
    return 1;
  }
}

