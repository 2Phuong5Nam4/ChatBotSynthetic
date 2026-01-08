#!/usr/bin/env python3
"""
Script kiểm tra tool_calls có đúng với schema không.
Xử lý 2 format:
  1. {"name": "tra_cuu_thong_tin", "arguments": {"ma_cua_hang": "123"}}
  2. {"name": "unknown", "arguments": {"raw": "{'name': 'tra_cuu_thong_tin', ...}"}}
"""
import json
import re
import ast
from pathlib import Path

VALID_TOOL_NAMES = [
    "tra_cuu_thong_tin",
    "kiem_tra_mqh",
    "kiem_tra_don_hang",
    "tao_ticket",
    "force_sync",
    "gui_huong_dan"
]

TOOL_SCHEMAS = {
    "tra_cuu_thong_tin": {
        "optional": ["ma_cua_hang", "sdt", "ma_npp"],
        "required": [],
        "enums": {}
    },
    "kiem_tra_mqh": {
        "required": ["outlet_id"],
        "optional": ["npp_subd_id"],
        "enums": {}
    },
    "kiem_tra_don_hang": {
        "required": ["ma_don_hang", "kenh"],
        "optional": [],
        "enums": {}
    },
    "tao_ticket": {
        "required": ["team", "noi_dung", "du_lieu"],
        "optional": [],
        "enums": {}
    },
    "force_sync": {
        "required": ["outlet_id"],
        "optional": ["npp_subd_id"],
        "enums": {}
    },
    "gui_huong_dan": {
        "required": ["loai_huong_dan"],
        "optional": [],
        "enums": {}
    }
}


def parse_raw_tool_call(raw_str: str) -> tuple[str, dict]:
    """Parse raw tool call string từ format như:
       "{'name': 'tra_cuu_thong_tin', 'arguments': {'ma_cua_hang': '123'}}"
       hoặc "tra_cuu_thong_tin{\"sdt\":\"0912345678\"}"
    """
    # Format 1: Dict-like string với 'name' và 'arguments'
    if "'name':" in raw_str or '"name":' in raw_str:
        try:
            # Thử parse như Python dict (dùng ast.literal_eval)
            data = ast.literal_eval(raw_str)
            if isinstance(data, dict):
                name = data.get("name", "unknown")
                args = data.get("arguments", {})
                # Parse arguments nếu là string
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                return name, args if isinstance(args, dict) else {}
        except:
            pass

        # Fallback: extract name bằng regex
        match = re.search(r"['\"]name['\"]\s*:\s*['\"](\w+)['\"]", raw_str)
        if match:
            return match.group(1), {}

    # Format 2: "tool_name{...}" hoặc "tool_name:{...}"
    for tool_name in VALID_TOOL_NAMES:
        if raw_str.startswith(tool_name):
            remaining = raw_str[len(tool_name):]
            if remaining.startswith("{") or remaining.startswith(":{"):
                if remaining.startswith(":"):
                    remaining = remaining[1:]
                try:
                    args = json.loads(remaining)
                    return tool_name, args
                except:
                    pass
            return tool_name, {}

    return "unparseable", {"_raw": raw_str}


class ToolValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_lines": 0,
            "total_tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_tool_calls": 0,
            "unparseable_calls": 0
        }

    def validate_file(self, file_path: str):
        path = Path(file_path)
        print(f"\n📂 Validating: {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                self.stats["total_lines"] += 1
                try:
                    data = json.loads(line.strip())
                    for msg in data.get("messages", []):
                        if msg.get("role") == "assistant" and "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                self._validate_tool_call(
                                    tc, path.name, line_num)
                except json.JSONDecodeError:
                    self.errors.append(
                        (path.name, line_num, "Invalid JSON line"))

    def _validate_tool_call(self, tc: dict, file_name: str, line_num: int):
        self.stats["total_tool_calls"] += 1
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})

        # Parse arguments nếu là string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {"raw": args}

        # Xử lý trường hợp name="unknown" với raw arguments
        if name == "unknown" and isinstance(args, dict) and "raw" in args:
            name, args = parse_raw_tool_call(args["raw"])

        # Check unparseable
        if name == "unparseable":
            self.stats["unparseable_calls"] += 1
            self.stats["invalid_tool_calls"] += 1
            raw_preview = str(args.get("_raw", ""))[:50]
            self.warnings.append(
                (file_name, line_num, f"Cannot parse: {raw_preview}..."))
            return

        # Check tên tool
        if name not in VALID_TOOL_NAMES:
            self.stats["invalid_tool_calls"] += 1
            self.errors.append(
                (file_name, line_num, f"Invalid tool name: '{name}'"))
            return

        schema = TOOL_SCHEMAS[name]
        is_valid = True

        # Check required params
        for req in schema["required"]:
            if req not in args:
                self.errors.append(
                    (file_name, line_num, f"'{name}' missing: '{req}'"))
                is_valid = False

        # Check enum values
        for param, valid_values in schema.get("enums", {}).items():
            if param in args and args[param] not in valid_values:
                self.errors.append((file_name, line_num,
                                    f"'{name}'.{param}='{args[param]}' not in {valid_values}"))
                is_valid = False

        if is_valid:
            self.stats["valid_tool_calls"] += 1
        else:
            self.stats["invalid_tool_calls"] += 1

    def print_report(self):
        print("\n" + "="*65)
        print("📊 TOOL CALLS VALIDATION REPORT")
        print("="*65)

        print(f"\n📈 Statistics:")
        print(f"  • Total lines: {self.stats['total_lines']}")
        print(f"  • Total tool calls: {self.stats['total_tool_calls']}")
        print(f"  • ✅ Valid: {self.stats['valid_tool_calls']}")
        print(f"  • ❌ Invalid: {self.stats['invalid_tool_calls']}")
        print(f"  • ⚠️  Unparseable: {self.stats['unparseable_calls']}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for fp, ln, msg in self.errors[:15]:
                print(f"  [{fp}] Line {ln}: {msg}")
            if len(self.errors) > 15:
                print(f"  ... and {len(self.errors) - 15} more")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for fp, ln, msg in self.warnings[:10]:
                print(f"  [{fp}] Line {ln}: {msg}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        print("\n" + "="*65)
        if self.errors:
            print(f"❌ FAILED: {len(self.errors)} schema violations found!")
        elif self.warnings:
            print(
                f"⚠️  PASSED with warnings: {len(self.warnings)} unparseable calls")
        else:
            print("✅ PASSED: All tool calls match schema!")
        print("="*65)


def main():
    validator = ToolValidator()
    data_dir = Path(__file__).parent / "data"

    for f in ["train.jsonl", "validation.jsonl"]:
        path = data_dir / f
        if path.exists():
            validator.validate_file(str(path))

    validator.print_report()
    return 1 if validator.errors else 0


if __name__ == "__main__":
    exit(main())
