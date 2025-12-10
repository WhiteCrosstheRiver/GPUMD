#!/bin/bash
# 编译和运行 test_parameters.cpp 的脚本

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}编译 test_parameters 测试${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查 nvcc 是否可用
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}错误: 未找到 nvcc 编译器${NC}"
    echo "请确保已安装 CUDA 工具包并配置了环境变量"
    exit 1
fi

# 检查源文件是否存在
if [ ! -f "test_parameters.cu" ]; then
    echo -e "${RED}错误: 未找到 test_parameters.cu${NC}"
    exit 1
fi

if [ ! -f "../../src/main_nep/parameters.cu" ]; then
    echo -e "${RED}错误: 未找到 ../../src/main_nep/parameters.cu${NC}"
    exit 1
fi

# 编译命令 - 使用与项目相同的编译标志
# 需要链接 utilities 目录中的工具函数
echo -e "${YELLOW}正在编译...${NC}"
nvcc -o test_parameters test_parameters.cu \
    ../../src/main_nep/parameters.cu \
    ../../src/utilities/error.cu \
    ../../src/utilities/read_file.cu \
    -I../../src \
    -I../../src/utilities \
    -std=c++17 \
    -O3 \
    -arch=sm_86 \
    -lcublas -lcusolver -lcufft

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 编译成功！${NC}"
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}运行测试${NC}"
    echo -e "${YELLOW}========================================${NC}"
    ./test_parameters
    
    # 检查测试是否通过
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}所有测试通过！${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}测试失败！${NC}"
        echo -e "${RED}========================================${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ 编译失败！${NC}"
    exit 1
fi

