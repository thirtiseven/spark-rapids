#!/bin/bash
# 便捷脚本：生成 nested_proto 测试数据
#
# 使用方法：
#   ./gen_nested_proto_data.sh [utf8|gbk] [count]
#
# 示例：
#   ./gen_nested_proto_data.sh          # 默认 UTF-8，100 条
#   ./gen_nested_proto_data.sh gbk      # GBK 编码，100 条
#   ./gen_nested_proto_data.sh utf8 500 # UTF-8 编码，500 条

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/nested_proto"
OUTPUT_DIR="${SCRIPT_DIR}/nested_proto/generated"

ENCODING="${1:-utf8}"
COUNT="${2:-100}"

echo "=== Protobuf Test Data Generator ==="
echo "Proto dir: ${PROTO_DIR}"
echo "Encoding: ${ENCODING}"
echo "Count: ${COUNT}"
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# Step 1: 编译 proto 文件
DESC_FILE="${OUTPUT_DIR}/main_log.desc"
echo "Step 1: Compiling proto files..."
protoc \
    --descriptor_set_out="${DESC_FILE}" \
    --include_imports \
    -I"${PROTO_DIR}" \
    "${PROTO_DIR}/main_log.proto"
echo "  Generated: ${DESC_FILE}"

# Step 2: 列出可用消息类型
echo ""
echo "Step 2: Available message types:"
python3 "${SCRIPT_DIR}/generate_test_data.py" list --desc "${DESC_FILE}"

# Step 3: 生成测试数据
echo ""
echo "Step 3: Generating test data..."

OUTPUT_FILE="${OUTPUT_DIR}/main_log_${ENCODING}_${COUNT}.pb"
python3 "${SCRIPT_DIR}/generate_test_data.py" generate \
    --desc "${DESC_FILE}" \
    --message "MainLogRecord" \
    --count "${COUNT}" \
    --encoding "${ENCODING}" \
    --output "${OUTPUT_FILE}" \
    --seed 42

echo ""
echo "=== Done ==="
echo "Descriptor: ${DESC_FILE}"
echo "Test data:  ${OUTPUT_FILE}"