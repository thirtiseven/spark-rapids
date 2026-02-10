#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protobuf Test Data Generator

通用的 protobuf 测试数据生成工具，支持：
1. 从 .desc 文件读取消息结构
2. 为每个字段生成随机数据
3. 支持 UTF-8 / GBK 字符串编码
4. 支持嵌套消息、repeated 字段、枚举等

使用方法：
    # 1. 先编译 proto 文件生成 .desc
    protoc --descriptor_set_out=output.desc --include_imports -I. main_log.proto
    
    # 2. 生成测试数据
    python generate_test_data.py --desc output.desc --message MainLogRecord --count 100 --encoding utf8 --output test_data.pb

依赖：
    pip install protobuf
"""

import argparse
import os
import random
import string
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

# Protobuf wire types
WIRE_VARINT = 0
WIRE_64BIT = 1
WIRE_LEN_DELIM = 2
WIRE_32BIT = 5


class RandomDataGenerator:
    """随机数据生成器"""
    
    def __init__(self, encoding: str = 'utf8', seed: Optional[int] = None):
        """
        Args:
            encoding: 字符串编码方式，'utf8' 或 'gbk'
            seed: 随机种子，用于可重复测试
        """
        self.encoding = encoding
        if seed is not None:
            random.seed(seed)
        
        # GBK 中文字符范围（常用汉字）
        self._gbk_chars = []
        for code in range(0xB0A1, 0xD7FA):  # 常用汉字区
            try:
                char = bytes([code >> 8, code & 0xFF]).decode('gbk')
                self._gbk_chars.append(char)
            except:
                pass
    
    def gen_bool(self) -> bool:
        return random.choice([True, False])
    
    def gen_int32(self) -> int:
        return random.randint(-2147483648, 2147483647)
    
    def gen_int64(self) -> int:
        return random.randint(-9223372036854775808, 9223372036854775807)
    
    def gen_uint32(self) -> int:
        return random.randint(0, 4294967295)
    
    def gen_uint64(self) -> int:
        return random.randint(0, 18446744073709551615)
    
    def gen_sint32(self) -> int:
        return random.randint(-2147483648, 2147483647)
    
    def gen_sint64(self) -> int:
        return random.randint(-9223372036854775808, 9223372036854775807)
    
    def gen_fixed32(self) -> int:
        return random.randint(0, 4294967295)
    
    def gen_fixed64(self) -> int:
        return random.randint(0, 18446744073709551615)
    
    def gen_sfixed32(self) -> int:
        return random.randint(-2147483648, 2147483647)
    
    def gen_sfixed64(self) -> int:
        return random.randint(-9223372036854775808, 9223372036854775807)
    
    def gen_float(self) -> float:
        return random.uniform(-1e6, 1e6)
    
    def gen_double(self) -> float:
        return random.uniform(-1e15, 1e15)
    
    def gen_string(self, min_len: int = 0, max_len: int = 50) -> str:
        """生成随机字符串，根据编码选择字符集"""
        length = random.randint(min_len, max_len)
        if length == 0:
            return ""
        
        if self.encoding == 'gbk':
            # 混合中文和ASCII
            chars = []
            for _ in range(length):
                if random.random() < 0.7 and self._gbk_chars:
                    chars.append(random.choice(self._gbk_chars))
                else:
                    chars.append(random.choice(string.ascii_letters + string.digits))
            return ''.join(chars)
        else:
            # UTF-8: 使用ASCII + 部分Unicode字符
            chars = []
            for _ in range(length):
                if random.random() < 0.3:
                    # 添加一些Unicode字符
                    chars.append(chr(random.randint(0x4E00, 0x9FA5)))  # 中文
                else:
                    chars.append(random.choice(string.ascii_letters + string.digits + '_-'))
            return ''.join(chars)
    
    def gen_bytes(self, min_len: int = 0, max_len: int = 100) -> bytes:
        """生成随机字节序列"""
        length = random.randint(min_len, max_len)
        return bytes(random.randint(0, 255) for _ in range(length))
    
    def gen_enum(self, values: List[int]) -> int:
        """从枚举值列表中随机选择"""
        return random.choice(values) if values else 0


class ProtobufEncoder:
    """Protobuf 二进制编码器"""
    
    def __init__(self, encoding: str = 'utf8'):
        self.encoding = encoding
    
    def encode_varint(self, value: int) -> bytes:
        """编码 varint（支持负数）"""
        if value < 0:
            value = value & 0xFFFFFFFFFFFFFFFF  # 转为无符号64位
        
        result = bytearray()
        while value >= 128:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value)
        return bytes(result)
    
    def encode_zigzag32(self, value: int) -> bytes:
        """编码 sint32 (zigzag)"""
        zigzag = (value << 1) ^ (value >> 31)
        return self.encode_varint(zigzag & 0xFFFFFFFF)
    
    def encode_zigzag64(self, value: int) -> bytes:
        """编码 sint64 (zigzag)"""
        zigzag = (value << 1) ^ (value >> 63)
        return self.encode_varint(zigzag & 0xFFFFFFFFFFFFFFFF)
    
    def encode_fixed32(self, value: int) -> bytes:
        """编码 fixed32/sfixed32"""
        return struct.pack('<I', value & 0xFFFFFFFF)
    
    def encode_fixed64(self, value: int) -> bytes:
        """编码 fixed64/sfixed64"""
        return struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF)
    
    def encode_sfixed32(self, value: int) -> bytes:
        """编码 sfixed32"""
        return struct.pack('<i', value)
    
    def encode_sfixed64(self, value: int) -> bytes:
        """编码 sfixed64"""
        return struct.pack('<q', value)
    
    def encode_float(self, value: float) -> bytes:
        return struct.pack('<f', value)
    
    def encode_double(self, value: float) -> bytes:
        return struct.pack('<d', value)
    
    def encode_string(self, value: str) -> bytes:
        """编码字符串（length-delimited）"""
        encoded = value.encode(self.encoding, errors='replace')
        return self.encode_varint(len(encoded)) + encoded
    
    def encode_bytes(self, value: bytes) -> bytes:
        """编码 bytes（length-delimited）"""
        return self.encode_varint(len(value)) + value
    
    def encode_field_key(self, field_number: int, wire_type: int) -> bytes:
        """编码字段 key"""
        return self.encode_varint((field_number << 3) | wire_type)


class ProtobufDataGenerator:
    """Protobuf 测试数据生成器"""
    
    def __init__(self, encoding: str = 'utf8', seed: Optional[int] = None,
                 nullable_prob: float = 0.1, max_repeated: int = 5):
        """
        Args:
            encoding: 字符串编码，'utf8' 或 'gbk'
            seed: 随机种子
            nullable_prob: optional 字段为空的概率
            max_repeated: repeated 字段最大元素数
        """
        self.encoding = encoding
        self.rand_gen = RandomDataGenerator(encoding, seed)
        self.encoder = ProtobufEncoder(encoding)
        self.nullable_prob = nullable_prob
        self.max_repeated = max_repeated
        
        # 消息定义缓存 {full_name: message_descriptor}
        self._message_defs: Dict[str, Any] = {}
        # 枚举定义缓存 {full_name: [values]}
        self._enum_defs: Dict[str, List[int]] = {}
    
    def load_descriptor_set(self, desc_path: str):
        """加载 .desc 文件"""
        from google.protobuf import descriptor_pb2
        
        with open(desc_path, 'rb') as f:
            desc_set = descriptor_pb2.FileDescriptorSet()
            desc_set.ParseFromString(f.read())
        
        # 解析所有消息和枚举定义
        for file_desc in desc_set.file:
            package = file_desc.package
            
            # 解析顶层枚举
            for enum_desc in file_desc.enum_type:
                full_name = f"{package}.{enum_desc.name}" if package else enum_desc.name
                self._enum_defs[full_name] = [v.number for v in enum_desc.value]
            
            # 解析顶层消息
            for msg_desc in file_desc.message_type:
                self._parse_message(msg_desc, package)
    
    def _parse_message(self, msg_desc, parent_name: str):
        """递归解析消息定义"""
        full_name = f"{parent_name}.{msg_desc.name}" if parent_name else msg_desc.name
        self._message_defs[full_name] = msg_desc
        
        # 解析嵌套枚举
        for enum_desc in msg_desc.enum_type:
            enum_full_name = f"{full_name}.{enum_desc.name}"
            self._enum_defs[enum_full_name] = [v.number for v in enum_desc.value]
        
        # 解析嵌套消息
        for nested_msg in msg_desc.nested_type:
            self._parse_message(nested_msg, full_name)
    
    def generate_message(self, message_name: str) -> bytes:
        """生成指定消息类型的随机数据"""
        # 尝试查找消息定义
        msg_desc = self._message_defs.get(message_name)
        if msg_desc is None:
            # 尝试添加包名前缀
            for full_name, desc in self._message_defs.items():
                if full_name.endswith('.' + message_name) or full_name == message_name:
                    msg_desc = desc
                    break
        
        if msg_desc is None:
            raise ValueError(f"Message '{message_name}' not found. Available: {list(self._message_defs.keys())}")
        
        return self._generate_message_data(msg_desc)
    
    def _generate_message_data(self, msg_desc) -> bytes:
        """生成消息数据"""
        from google.protobuf.descriptor_pb2 import FieldDescriptorProto
        
        result = bytearray()
        
        for field in msg_desc.field:
            # optional 字段有概率跳过
            if field.label != FieldDescriptorProto.LABEL_REQUIRED:
                if random.random() < self.nullable_prob:
                    continue
            
            # repeated 字段
            if field.label == FieldDescriptorProto.LABEL_REPEATED:
                count = random.randint(0, self.max_repeated)
                for _ in range(count):
                    result.extend(self._encode_field(field))
            else:
                result.extend(self._encode_field(field))
        
        return bytes(result)
    
    def _encode_field(self, field) -> bytes:
        """编码单个字段"""
        from google.protobuf.descriptor_pb2 import FieldDescriptorProto
        
        field_num = field.number
        field_type = field.type
        
        # 根据类型生成数据并编码
        if field_type == FieldDescriptorProto.TYPE_BOOL:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_varint(1 if self.rand_gen.gen_bool() else 0)
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_INT32:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_varint(self.rand_gen.gen_int32())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_INT64:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_varint(self.rand_gen.gen_int64())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_UINT32:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_varint(self.rand_gen.gen_uint32())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_UINT64:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_varint(self.rand_gen.gen_uint64())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_SINT32:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_zigzag32(self.rand_gen.gen_sint32())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_SINT64:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            value = self.encoder.encode_zigzag64(self.rand_gen.gen_sint64())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_FIXED32:
            key = self.encoder.encode_field_key(field_num, WIRE_32BIT)
            value = self.encoder.encode_fixed32(self.rand_gen.gen_fixed32())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_FIXED64:
            key = self.encoder.encode_field_key(field_num, WIRE_64BIT)
            value = self.encoder.encode_fixed64(self.rand_gen.gen_fixed64())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_SFIXED32:
            key = self.encoder.encode_field_key(field_num, WIRE_32BIT)
            value = self.encoder.encode_sfixed32(self.rand_gen.gen_sfixed32())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_SFIXED64:
            key = self.encoder.encode_field_key(field_num, WIRE_64BIT)
            value = self.encoder.encode_sfixed64(self.rand_gen.gen_sfixed64())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_FLOAT:
            key = self.encoder.encode_field_key(field_num, WIRE_32BIT)
            value = self.encoder.encode_float(self.rand_gen.gen_float())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_DOUBLE:
            key = self.encoder.encode_field_key(field_num, WIRE_64BIT)
            value = self.encoder.encode_double(self.rand_gen.gen_double())
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_STRING:
            key = self.encoder.encode_field_key(field_num, WIRE_LEN_DELIM)
            str_value = self.rand_gen.gen_string()
            value = self.encoder.encode_string(str_value)
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_BYTES:
            key = self.encoder.encode_field_key(field_num, WIRE_LEN_DELIM)
            bytes_value = self.rand_gen.gen_bytes()
            value = self.encoder.encode_bytes(bytes_value)
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_ENUM:
            key = self.encoder.encode_field_key(field_num, WIRE_VARINT)
            # 查找枚举定义
            enum_values = self._find_enum_values(field.type_name)
            value = self.encoder.encode_varint(self.rand_gen.gen_enum(enum_values))
            return key + value
        
        elif field_type == FieldDescriptorProto.TYPE_MESSAGE:
            # 递归生成嵌套消息
            nested_msg_desc = self._find_message_desc(field.type_name)
            if nested_msg_desc is None:
                return b''  # 找不到定义，跳过
            
            nested_data = self._generate_message_data(nested_msg_desc)
            key = self.encoder.encode_field_key(field_num, WIRE_LEN_DELIM)
            length = self.encoder.encode_varint(len(nested_data))
            return key + length + nested_data
        
        else:
            # 未知类型，跳过
            return b''
    
    def _find_enum_values(self, type_name: str) -> List[int]:
        """查找枚举值列表"""
        # 移除前导点
        name = type_name.lstrip('.')
        
        if name in self._enum_defs:
            return self._enum_defs[name]
        
        # 尝试部分匹配
        for full_name, values in self._enum_defs.items():
            if full_name.endswith(name) or name.endswith(full_name.split('.')[-1]):
                return values
        
        return [0]  # 默认返回 [0]
    
    def _find_message_desc(self, type_name: str):
        """查找消息描述"""
        name = type_name.lstrip('.')
        
        if name in self._message_defs:
            return self._message_defs[name]
        
        # 尝试部分匹配
        for full_name, desc in self._message_defs.items():
            if full_name.endswith(name) or name.endswith(full_name.split('.')[-1]):
                return desc
        
        return None
    
    def list_messages(self) -> List[str]:
        """列出所有可用的消息类型"""
        return list(self._message_defs.keys())
    
    def list_enums(self) -> Dict[str, List[int]]:
        """列出所有可用的枚举类型"""
        return dict(self._enum_defs)


def compile_proto(proto_dir: str, main_proto: str, output_desc: str) -> bool:
    """
    编译 proto 文件生成 .desc 文件
    
    Args:
        proto_dir: proto 文件所在目录
        main_proto: 主 proto 文件名
        output_desc: 输出的 .desc 文件路径
    
    Returns:
        是否成功
    """
    import subprocess
    
    cmd = [
        'protoc',
        f'--descriptor_set_out={output_desc}',
        '--include_imports',
        f'-I{proto_dir}',
        os.path.join(proto_dir, main_proto)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"protoc error: {result.stderr}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("Error: protoc not found. Please install protobuf compiler.", file=sys.stderr)
        return False


def generate_test_data_file(
    desc_path: str,
    message_name: str,
    output_path: str,
    count: int = 100,
    encoding: str = 'utf8',
    seed: Optional[int] = None,
    format: str = 'length_prefixed'
):
    """
    生成测试数据文件
    
    Args:
        desc_path: .desc 文件路径
        message_name: 消息类型名称
        output_path: 输出文件路径
        count: 生成的消息数量
        encoding: 字符串编码 ('utf8' 或 'gbk')
        seed: 随机种子
        format: 输出格式
            - 'length_prefixed': 每条消息前加 4 字节长度
            - 'raw': 原始消息拼接（需要自己解析）
            - 'single': 只输出单条消息
    """
    generator = ProtobufDataGenerator(encoding=encoding, seed=seed)
    generator.load_descriptor_set(desc_path)
    
    print(f"Available messages: {generator.list_messages()}")
    print(f"Available enums: {list(generator.list_enums().keys())}")
    print(f"Generating {count} messages of type '{message_name}' with {encoding} encoding...")
    
    with open(output_path, 'wb') as f:
        for i in range(count):
            data = generator.generate_message(message_name)
            
            if format == 'length_prefixed':
                # 4 字节大端长度 + 消息
                f.write(struct.pack('>I', len(data)))
                f.write(data)
            elif format == 'single':
                f.write(data)
                break
            else:  # raw
                f.write(data)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{count} messages")
    
    print(f"Done! Output: {output_path} ({os.path.getsize(output_path)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description='Protobuf Test Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. 先编译 proto 文件
  python generate_test_data.py compile --proto-dir ./nested_proto --main-proto main_log.proto --output main_log.desc
  
  # 2. 列出可用的消息类型
  python generate_test_data.py list --desc main_log.desc
  
  # 3. 生成 UTF-8 编码的测试数据
  python generate_test_data.py generate --desc main_log.desc --message MainLogRecord --count 100 --encoding utf8 --output test_utf8.pb
  
  # 4. 生成 GBK 编码的测试数据
  python generate_test_data.py generate --desc main_log.desc --message MainLogRecord --count 100 --encoding gbk --output test_gbk.pb
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # compile 命令
    compile_parser = subparsers.add_parser('compile', help='Compile .proto files to .desc')
    compile_parser.add_argument('--proto-dir', required=True, help='Directory containing .proto files')
    compile_parser.add_argument('--main-proto', required=True, help='Main .proto file name')
    compile_parser.add_argument('--output', required=True, help='Output .desc file path')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='List available message types')
    list_parser.add_argument('--desc', required=True, help='.desc file path')
    
    # generate 命令
    gen_parser = subparsers.add_parser('generate', help='Generate test data')
    gen_parser.add_argument('--desc', required=True, help='.desc file path')
    gen_parser.add_argument('--message', required=True, help='Message type name')
    gen_parser.add_argument('--count', type=int, default=100, help='Number of messages (default: 100)')
    gen_parser.add_argument('--encoding', choices=['utf8', 'gbk'], default='utf8', help='String encoding')
    gen_parser.add_argument('--output', required=True, help='Output file path')
    gen_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    gen_parser.add_argument('--format', choices=['length_prefixed', 'raw', 'single'], 
                           default='length_prefixed', help='Output format')
    
    args = parser.parse_args()
    
    if args.command == 'compile':
        success = compile_proto(args.proto_dir, args.main_proto, args.output)
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        generator = ProtobufDataGenerator()
        generator.load_descriptor_set(args.desc)
        print("Messages:")
        for name in generator.list_messages():
            print(f"  - {name}")
        print("\nEnums:")
        for name, values in generator.list_enums().items():
            print(f"  - {name}: {values}")
    
    elif args.command == 'generate':
        generate_test_data_file(
            desc_path=args.desc,
            message_name=args.message,
            output_path=args.output,
            count=args.count,
            encoding=args.encoding,
            seed=args.seed,
            format=args.format
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()