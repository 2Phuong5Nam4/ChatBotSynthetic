"""
Script to prepare training and validation datasets.
Reads synthetic_conversations.jsonl for train and refined_conversations.jsonl for validation.
Maps procedure_id to procedure details from extracted_procedure.json.
Keeps only 'messages' and mapped procedure fields.
"""

import json
from pathlib import Path


def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_procedures(file_path: Path) -> dict:
    """Load procedure definitions from extracted_procedure.json."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_keys_to_english(data):
    """Recursively convert Vietnamese keys to English."""
    if isinstance(data, dict):
        key_mapping = {
            'bước': 'step',
            'mô_tả': 'description',
            'chain_action': 'chain_action',
            'case': 'case',
            'điều_kiện': 'condition'
        }
        return {key_mapping.get(k, k): convert_keys_to_english(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_english(item) for item in data]
    else:
        return data


def map_procedure(procedure_id: str, procedures: dict) -> dict:
    """Map procedure_id to procedure details with English keys."""
    if procedure_id not in procedures:
        return {
            'name': None,
            'purpose': None,
            'main_flow': None,
            'edge_cases': None
        }

    proc = procedures[procedure_id]

    # Convert main_flow and edge_cases to English keys
    main_flow = proc.get('luồng_thực_thi_chung')
    edge_cases = proc.get('edge_cases')

    return {
        'name': proc.get('tên'),
        'purpose': proc.get('mục_tiêu'),
        'main_flow': convert_keys_to_english(main_flow) if main_flow else None,
        'edge_cases': convert_keys_to_english(edge_cases) if edge_cases else None
    }


def clean_messages(messages: list[dict]) -> list[dict]:
    """
    Clean messages to follow chat template format:
    - Extract <think>...</think> -> 'reasoning_content' field
    - Extract <tool_call>...</tool_call> -> 'tool_calls' array with {name, arguments}
    - Convert tool role messages with <tool_response> format
    """
    import re
    clean_messages = []

    for message in messages:
        clean_message = {}
        role = message.get('role', '').strip()
        clean_message['role'] = role
        content = message.get('content', '').strip()

        # Extract <think></think> -> 'reasoning_content' field
        reasoning_content = ""
        if '<think>' in content and '</think>' in content:
            start_idx = content.index('<think>') + len('<think>')
            end_idx = content.index('</think>')
            reasoning_content = content[start_idx:end_idx].strip()
            content = (content[:content.index('<think>')] +
                       content[end_idx + len('</think>'):]).strip()

        if reasoning_content:
            clean_message['reasoning_content'] = reasoning_content

        # Extract <tool_call>...</tool_call> -> 'tool_calls' array
        # Format in template: {"name": <function-name>, "arguments": <args-json-object>}
        if '<tool_call>' in content and '</tool_call>' in content:
            tool_calls = []
            # Find all tool_call blocks
            pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                tool_call_content = match.strip()
                # Parse tool call formats:
                # 1. tool_name({...}) - function call with JSON/dict
                # 2. tool_name{...} or tool_name:{...} - direct JSON
                # 3. tool_name(key=value, key2=value2) - keyword arguments
                # 4. {"name": "...", "arguments": {...}} - JSON formats

                # Try format 1: tool_name({...}) - function call style with JSON dict
                func_pattern = r'^(\w+)\s*\(\s*(\{.*\})\s*\)$'
                func_match = re.match(
                    func_pattern, tool_call_content, re.DOTALL)

                # Try format 2: tool_name{...} or tool_name:{...} or tool_name::{...}
                direct_pattern = r'^(\w+)\s*:*\s*(\{.*\})$'
                direct_match = re.match(
                    direct_pattern, tool_call_content, re.DOTALL)

                # Try format 3: tool_name(key=value, key2=value2) - keyword arguments
                kwargs_pattern = r'^(\w+)\s*\(([^{}]+)\)$'
                kwargs_match = re.match(
                    kwargs_pattern, tool_call_content, re.DOTALL)

                if func_match or direct_match:
                    matched = func_match or direct_match
                    tool_name = matched.group(1)
                    args_str = matched.group(2)
                    arguments = {}
                    # Try JSON first (double quotes)
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Try Python dict format (single quotes) using ast.literal_eval
                        import ast
                        try:
                            arguments = ast.literal_eval(args_str)
                        except (ValueError, SyntaxError):
                            arguments = {}  # Empty dict if unparseable
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": arguments
                    })
                elif kwargs_match:
                    # Parse keyword arguments: key=value, key2=value2
                    tool_name = kwargs_match.group(1)
                    kwargs_str = kwargs_match.group(2)
                    arguments = {}
                    # Split by comma and parse each key=value
                    for part in kwargs_str.split(','):
                        part = part.strip()
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            arguments[key] = value
                    tool_calls.append({
                        "name": tool_name,
                        "arguments": arguments
                    })
                else:
                    # Try JSON format: {"name": "...", "arguments": {...}} or {"tool": "...", "params": {...}}
                    parsed = None
                    try:
                        parsed = json.loads(tool_call_content)
                    except json.JSONDecodeError:
                        # Try Python dict format with ast.literal_eval
                        import ast
                        try:
                            parsed = ast.literal_eval(tool_call_content)
                        except (ValueError, SyntaxError):
                            pass

                    if isinstance(parsed, dict):
                        # Handle format: {"name": "...", "arguments": {...}}
                        if 'name' in parsed:
                            tool_calls.append({
                                "name": parsed.get("name"),
                                "arguments": parsed.get("arguments", {})
                            })
                        # Handle format: {"tool": "...", "params": {...}}
                        elif 'tool' in parsed:
                            tool_calls.append({
                                "name": parsed.get("tool"),
                                "arguments": parsed.get("params", {})
                            })
                        # Handle format: {"tool_name": "...", "params": {...}}
                        elif 'tool_name' in parsed:
                            tool_calls.append({
                                "name": parsed.get("tool_name"),
                                "arguments": parsed.get("params", {})
                            })
                        else:
                            tool_calls.append({
                                "name": "unknown",
                                "arguments": {"raw": tool_call_content}
                            })
                    else:
                        # Unknown format, store raw
                        tool_calls.append({
                            "name": "unknown",
                            "arguments": {"raw": tool_call_content}
                        })

            if tool_calls:
                clean_message['tool_calls'] = tool_calls

            # Remove all tool_call blocks from content
            content = re.sub(pattern, '', content, flags=re.DOTALL).strip()

        clean_message['content'] = content
        clean_messages.append(clean_message)

    return clean_messages


def filter_and_map_columns(records: list[dict], procedures: dict) -> list[dict]:
    """Keep only 'messages' and map procedure_id to procedure details."""
    filtered = []
    for record in records:
        procedure_id = record.get('procedure_id')
        procedure_details = map_procedure(procedure_id, procedures)

        filtered_record = {
            'procedure_id': procedure_id,
            'procedure_name': procedure_details['name'],
            'procedure_purpose': procedure_details['purpose'],
            'procedure_main_flow': procedure_details['main_flow'],
            'procedure_edge_cases': procedure_details['edge_cases'],
            'messages': clean_messages(record.get('messages'))
        }
        filtered.append(filtered_record)
    return filtered


def save_jsonl(records: list[dict], file_path: Path):
    """Save records to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(records)} records to {file_path}")


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data'

    synthetic_file = data_dir / 'synthetic_conversations.jsonl'
    refined_file = data_dir / 'refined_conversations.jsonl'
    procedures_file = data_dir / 'extracted_procedure.json'
    train_file = data_dir / 'train.jsonl'
    validation_file = data_dir / 'validation.jsonl'

    # Load procedure definitions
    print(f"Loading procedure definitions from {procedures_file}...")
    procedures = load_procedures(procedures_file)
    print(f"Loaded {len(procedures)} procedure definitions")

    # Prepare training data from synthetic_conversations.jsonl
    print(f"\nLoading synthetic conversations from {synthetic_file}...")
    train_records = load_jsonl(synthetic_file)
    print(f"Found {len(train_records)} training records")

    print("Mapping procedure_id to procedure details and filtering columns...")
    train_filtered = filter_and_map_columns(train_records, procedures)

    print(f"Saving training data to {train_file}...")
    save_jsonl(train_filtered, train_file)

    # Prepare validation data from refined_conversations.jsonl
    print(f"\nLoading refined conversations from {refined_file}...")
    val_records = load_jsonl(refined_file)
    print(f"Found {len(val_records)} validation records")

    print("Mapping procedure_id to procedure details and filtering columns...")
    val_filtered = filter_and_map_columns(val_records, procedures)

    print(f"Saving validation data to {validation_file}...")
    save_jsonl(val_filtered, validation_file)

    print("\n" + "="*50)
    print("Dataset preparation completed!")
    print(f"Training set: {len(train_filtered)} samples -> {train_file}")
    print(f"Validation set: {len(val_filtered)} samples -> {validation_file}")
    print("="*50)


if __name__ == "__main__":
    main()
