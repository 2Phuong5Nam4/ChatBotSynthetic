"""
Script to validate reasoning_content format in JSONL files.

Expected pattern:
```
Tình huống: [content]
Quy trình: [content]
Bước: [optional or "N - description" or "ngoại lệ - description"]
Thông tin có: [content]
Thông tin cần thêm: [content]
Hành động: [content]
```
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a validation error"""
    file_path: str
    conversation_idx: int
    message_idx: int
    field: str
    error_type: str
    actual_value: str
    reasoning_content: str


def validate_reasoning_content(reasoning: str) -> List[Tuple[str, str, str]]:
    """
    Validate reasoning_content against the expected pattern.
    
    Returns list of (field, error_type, actual_value) tuples for errors found.
    """
    errors = []
    
    if not reasoning or not reasoning.strip():
        return [("reasoning_content", "EMPTY", "")]
    
    # Normalize line endings
    reasoning = reasoning.replace('\r\n', '\n').strip()
    
    # Define the field order for extraction
    field_names = ["Tình huống", "Quy trình", "Bước", "Thông tin có", "Thông tin cần thêm", "Hành động"]
    
    # Build regex to find each field and extract content until next field or end
    # This handles multi-line content (e.g., bullet points on next lines)
    def get_field_content(field_name: str, text: str) -> tuple[bool, str]:
        """
        Extract content for a field, supporting multi-line content.
        Returns (found, content).
        """
        # Build pattern: field_name[optional stuff]: content until next field or end
        # For "Thông tin có", allow variants like "Thông tin có (tổng hợp):" or "Thông tin có chi tiết:"
        if field_name == "Thông tin có":
            field_pattern = r"Thông tin có[^:\n]*:"
        else:
            field_pattern = re.escape(field_name) + r":"
        
        # Build the list of next possible fields
        next_fields = []
        for f in field_names:
            if f == "Thông tin có":
                next_fields.append(r"Thông tin có[^:\n]*:")
            else:
                next_fields.append(re.escape(f) + r":")
        next_field_pattern = "|".join(next_fields)
        
        # Find the field
        field_match = re.search(field_pattern, text)
        if not field_match:
            return (False, "")
        
        # Extract content from after the field until next field or end
        start_pos = field_match.end()
        
        # Find the next field after this position
        remaining_text = text[start_pos:]
        next_match = re.search(next_field_pattern, remaining_text)
        
        if next_match:
            content = remaining_text[:next_match.start()]
        else:
            content = remaining_text
        
        return (True, content.strip())
    
    # Check each required field exists and has content
    for field in field_names:
        found, content = get_field_content(field, reasoning)
        if not found:
            errors.append((field, "MISSING", ""))
        elif field != "Bước" and not content:
            # All fields except Bước must have content
            errors.append((field, "EMPTY_VALUE", f"{field}: (empty)"))
    
    # Special validation for "Bước" field
    buoc_match = re.search(r"Bước:[ \t]*(.*?)(?=\n|$)", reasoning)
    if buoc_match:
        buoc_value = buoc_match.group(1).strip()
        
        if buoc_value:  # If not empty, validate format
            # Valid formats:
            # 1. "N - description" where N is number
            # 2. "ngoại lệ - description"
            # 3. Empty (for không xác định/không liên quan)
            
            step_number_pattern = r"^\d+\s*-\s*.+"  # "1 - Xác thực thông tin"
            exception_pattern = r"^ngoại lệ\s*-\s*.+"  # "ngoại lệ - đơn hàng bị hủy"
            
            is_step_number = re.match(step_number_pattern, buoc_value, re.IGNORECASE)
            is_exception = re.match(exception_pattern, buoc_value, re.IGNORECASE)
            
            if not is_step_number and not is_exception:
                errors.append(("Bước", "INVALID_FORMAT", buoc_value))
    
    # Validate "Quy trình" special values
    quy_trinh_match = re.search(r"Quy trình:[ \t]*(.+?)(?=\n|$)", reasoning)
    if quy_trinh_match:
        quy_trinh_value = quy_trinh_match.group(1).strip().lower()
        buoc_match = re.search(r"Bước:[ \t]*(.*?)(?=\n|$)", reasoning)
        buoc_value = buoc_match.group(1).strip() if buoc_match else ""
        
        # If quy_trinh is "không xác định" or "không liên quan", Bước should be empty
        if quy_trinh_value in ["không xác định", "không liên quan"]:
            if buoc_value:
                errors.append(("Bước", "SHOULD_BE_EMPTY_FOR_UNDEFINED_PROCESS", buoc_value))
    
    # Check field order
    field_order = ["Tình huống", "Quy trình", "Bước", "Thông tin có", "Thông tin cần thêm", "Hành động"]
    positions = []
    for field in field_order:
        match = re.search(f"{field}:", reasoning)
        if match:
            positions.append((field, match.start()))
    
    # Check if fields are in correct order
    for i in range(len(positions) - 1):
        if positions[i][1] > positions[i + 1][1]:
            errors.append(("ORDER", "WRONG_FIELD_ORDER", f"{positions[i][0]} appears after {positions[i + 1][0]}"))
    
    return errors


def process_jsonl_file(file_path: str) -> List[ValidationError]:
    """Process a JSONL file and return all validation errors."""
    all_errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for conv_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(ValidationError(
                    file_path=file_path,
                    conversation_idx=conv_idx,
                    message_idx=-1,
                    field="JSON",
                    error_type="PARSE_ERROR",
                    actual_value=str(e),
                    reasoning_content=""
                ))
                continue
            
            messages = data.get('messages', [])
            
            for msg_idx, msg in enumerate(messages):
                if msg.get('role') != 'assistant':
                    continue
                
                reasoning_content = msg.get('reasoning_content', '')
                
                # Skip if no reasoning_content (some assistant messages may not have it)
                if not reasoning_content:
                    continue
                
                errors = validate_reasoning_content(reasoning_content)
                
                for field, error_type, actual_value in errors:
                    all_errors.append(ValidationError(
                        file_path=file_path,
                        conversation_idx=conv_idx,
                        message_idx=msg_idx,
                        field=field,
                        error_type=error_type,
                        actual_value=actual_value,
                        reasoning_content=reasoning_content
                    ))
    
    return all_errors


def print_errors(errors: List[ValidationError], verbose: bool = True):
    """Print validation errors in a readable format."""
    if not errors:
        print("✅ All reasoning_content fields are valid!")
        return
    
    print(f"\n❌ Found {len(errors)} validation error(s):\n")
    print("=" * 80)
    
    # Group errors by file and conversation
    grouped: Dict[str, Dict[int, List[ValidationError]]] = {}
    for error in errors:
        if error.file_path not in grouped:
            grouped[error.file_path] = {}
        if error.conversation_idx not in grouped[error.file_path]:
            grouped[error.file_path][error.conversation_idx] = []
        grouped[error.file_path][error.conversation_idx].append(error)
    
    for file_path, conversations in grouped.items():
        print(f"\n📁 File: {file_path}")
        print("-" * 80)
        
        for conv_idx, conv_errors in conversations.items():
            print(f"\n  📝 Conversation #{conv_idx + 1}")
            
            for error in conv_errors:
                print(f"    ├─ Message #{error.message_idx + 1}")
                print(f"    ├─ Field: {error.field}")
                print(f"    ├─ Error: {error.error_type}")
                if error.actual_value:
                    print(f"    ├─ Value: \"{error.actual_value[:100]}{'...' if len(error.actual_value) > 100 else ''}\"")
                
                if verbose and error.reasoning_content:
                    print(f"    └─ Full reasoning_content:")
                    for line in error.reasoning_content.split('\n'):
                        print(f"       │ {line}")
                print()
    
    # Summary
    print("=" * 80)
    print("\n📊 Summary by error type:")
    error_counts: Dict[str, int] = {}
    for error in errors:
        key = f"{error.field}: {error.error_type}"
        error_counts[key] = error_counts.get(key, 0) + 1
    
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  • {error_type}: {count}")


def main():
    """Main function to run validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate reasoning_content format in JSONL files')
    parser.add_argument('files', nargs='*', default=['data/train.jsonl', 'data/validation.jsonl'],
                        help='JSONL files to validate (default: data/train.jsonl data/validation.jsonl)')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Show full reasoning_content for errors')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Only show summary, not full content')
    
    args = parser.parse_args()
    
    all_errors = []
    
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"🔍 Checking {file_path}...")
        errors = process_jsonl_file(file_path)
        all_errors.extend(errors)
    
    print_errors(all_errors, verbose=not args.quiet)
    
    # Exit with error code if errors found
    if all_errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
