"""
Script to fix missing <tool_call> blocks in synthetic conversations.

Finds assistant messages that:
1. Contain <think> block mentioning a tool action
2. Are immediately followed by a tool message
3. But are missing the <tool_call> block

Uses LLM to generate the appropriate tool call based on context.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolCallGeneration(BaseModel):
    """Output model for generated tool call"""
    tool_name: str
    tool_args_json: str  # JSON string of arguments, e.g. '{"ma_cua_hang": "305789"}'


# =============================================================================
# DETECTION
# =============================================================================

def find_missing_tool_calls(messages: List[Dict]) -> List[int]:
    """
    Find indices of assistant messages that are missing <tool_call> blocks.
    
    Returns list of indices where:
    - Current message is assistant with <think> block
    - Next message is a tool response
    - Current message does NOT contain <tool_call>
    """
    missing_indices = []
    
    for i in range(len(messages) - 1):
        current_msg = messages[i]
        next_msg = messages[i + 1]
        
        # Check if current is assistant and next is tool
        if current_msg.get("role") != "assistant":
            continue
        if next_msg.get("role") != "tool":
            continue
            
        content = current_msg.get("content", "")
        
        # Check if has <think> but missing <tool_call>
        has_think = "<think>" in content and "</think>" in content
        has_tool_call = "<tool_call>" in content
        
        if has_think and not has_tool_call:
            missing_indices.append(i)
    
    return missing_indices


# =============================================================================
# LLM GENERATION
# =============================================================================

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích hội thoại CSKH Heineken Vietnam.

## NHIỆM VỤ
Dựa vào nội dung <think> block của assistant và kết quả tool response, hãy xác định:
1. Tên tool đã được gọi
2. Arguments đã truyền vào tool

## AVAILABLE TOOLS

1. **tra_cuu_thong_tin**: Tra cứu thông tin cửa hàng/NPP
   - Input: ma_cua_hang, sdt, ma_npp (tất cả optional, dùng 1 hoặc nhiều)

2. **kiem_tra_mqh**: Kiểm tra mối quan hệ Outlet-NPP/SubD
   - Input: outlet_id, npp_subd_id

3. **kiem_tra_don_hang**: Kiểm tra trạng thái đơn hàng
   - Input: ma_don_hang, kenh

4. **tao_ticket**: Tạo ticket chuyển tuyến
   - Input: team, noi_dung, du_lieu

5. **force_sync**: Force sync dữ liệu SEM
   - Input: outlet_id, npp_subd_id

6. **gui_huong_dan**: Gửi hướng dẫn SOP
   - Input: loai_huong_dan

## QUY TẮC
- Phân tích <think> block để biết ý định gọi tool nào
- Phân tích tool response để reverse-engineer arguments
- tool_args_json phải là JSON string hợp lệ, ví dụ: '{"ma_cua_hang": "305789"}'
"""


async def generate_tool_call(
    think_content: str,
    tool_response: str
) -> Optional[ToolCallGeneration]:
    """Generate tool call based on think block and tool response"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).with_structured_output(ToolCallGeneration)
    
    user_prompt = f"""## THINK BLOCK CỦA ASSISTANT:
{think_content}

## TOOL RESPONSE:
{tool_response}

Hãy xác định tool_name và tool_args_json (dạng JSON string) đã được sử dụng."""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        return response
    except Exception as e:
        print(f"Error generating tool call: {e}")
        return None


def format_tool_call(tool_name: str, tool_args_json: str) -> str:
    """Format tool call as XML block"""
    # Validate and normalize the JSON
    try:
        args_dict = json.loads(tool_args_json)
        args_str = json.dumps(args_dict, ensure_ascii=False)
    except json.JSONDecodeError:
        args_str = tool_args_json  # Use as-is if not valid JSON
    return f"<tool_call>\n{tool_name}({args_str})\n</tool_call>"


# =============================================================================
# FIX LOGIC
# =============================================================================

async def fix_conversation(messages: List[Dict]) -> List[Dict]:
    """Fix missing tool calls in a conversation"""
    
    missing_indices = find_missing_tool_calls(messages)
    
    if not missing_indices:
        return messages
    
    # Create a copy to modify
    fixed_messages = [msg.copy() for msg in messages]
    
    for idx in missing_indices:
        assistant_msg = fixed_messages[idx]
        tool_msg = fixed_messages[idx + 1]
        
        content = assistant_msg.get("content", "")
        tool_response = tool_msg.get("content", "")
        
        # Generate tool call
        result = await generate_tool_call(content, tool_response)
        
        if result:
            tool_call_block = format_tool_call(result.tool_name, result.tool_args_json)
            
            # Insert tool_call after </think>
            if "</think>" in content:
                new_content = content.replace(
                    "</think>",
                    f"</think>\n{tool_call_block}"
                )
                fixed_messages[idx]["content"] = new_content
                print(f"  Fixed message at index {idx}: added {result.tool_name} call")
            else:
                print(f"  Warning: Could not find </think> in message at index {idx}")
        else:
            print(f"  Warning: Could not generate tool call for message at index {idx}")
    
    return fixed_messages


async def fix_all_conversations(input_path: Path, output_path: Path):
    """Process all conversations and fix missing tool calls"""
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print("=" * 60)
    
    fixed_conversations = []
    total_fixes = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                conversation = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            
            messages = conversation.get("messages", [])
            missing_count = len(find_missing_tool_calls(messages))
            
            if missing_count > 0:
                print(f"\nConversation {line_num} (procedure_id={conversation.get('procedure_id', 'N/A')}): {missing_count} missing tool calls")
                fixed_messages = await fix_conversation(messages)
                conversation["messages"] = fixed_messages
                total_fixes += missing_count
            
            fixed_conversations.append(conversation)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in fixed_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: Fixed {total_fixes} missing tool calls")
    print(f"Output written to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    data_dir = Path(__file__).parent.parent / "data"
    input_path = data_dir / "synthetic_conversations.jsonl"
    output_path = data_dir / "fixed_syn_conversations.jsonl"
    
    await fix_all_conversations(input_path, output_path)


if __name__ == "__main__":
    asyncio.run(main())
