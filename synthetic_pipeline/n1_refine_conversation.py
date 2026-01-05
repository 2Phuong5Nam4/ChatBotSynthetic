"""
N1 Refine Conversation Pipeline
Sử dụng GPT-5 reasoning model như teacher LLM để:
1. Dạy LLM con cách think qua reasoning content
2. Dạy cách sử dụng tool qua tool calls
3. Viết lại conversation hoàn toàn theo đúng procedure
"""

import json
from typing import Dict, List, Optional
from openai import OpenAI

from synthetic_pipeline.state import State
from synthetic_pipeline.mock_tools import call_tool
from openai.types.responses.tool_param import ToolParam

def build_system_instructions(
    conversation_procedure: Dict,
    tools: List[Dict],
    raw_conversation: Dict
) -> str:
    """Build comprehensive instructions for teacher LLM"""
    instructions = f"""Bạn là TEACHER LLM - chuyên gia dịch vụ khách hàng Heineken Vietnam.

## NHIỆM VỤ
Viết lại hoàn toàn hội thoại khách hàng theo đúng procedure đã phân loại.
Mỗi turn bạn sẽ:
1. Suy nghĩ (reasoning) theo 4 dòng chuẩn
2. Gọi tools nếu procedure yêu cầu
3. Trả lời khách hàng
4. Tạo user message tiếp theo để tiếp tục hội thoại

## QUY TẮC QUAN TRỌNG
- GIỮ NGUYÊN văn phong lịch sự, thân thiện (anh/chị/em, ạ)
- GIỮ NGUYÊN vấn đề/issue của user từ hội thoại gốc
- VIẾT LẠI assistant response để TUÂN THỦ CHÍNH XÁC procedure
- MỖI bước trong procedure phải được thể hiện trong hội thoại

## PROCEDURE: {conversation_procedure['tên']}
Mục tiêu: {conversation_procedure['mục_tiêu']}

### Luồng thực thi chính:
"""

    for step in conversation_procedure['luồng_thực_thi_chung']:
        instructions += f"\n**Bước {step['bước']}**: {step['mô_tả']}\n"
        instructions += f"  Chain action: {step['chain_action']}\n"

    instructions += "\n### Edge cases (Trường hợp đặc biệt):\n"
    for edge_case in conversation_procedure.get('edge_cases', []):
        instructions += f"\n- **{edge_case['case']}**\n"
        instructions += f"  Điều kiện: {edge_case['điều_kiện']}\n"
        instructions += f"  Xử lý: {edge_case['chain_action']}\n"

    instructions += "\n\n## HỘI THOẠI GỐC (để tham khảo văn phong và issue):\n"
    instructions += f"- Loại: {raw_conversation.get('Sub_Category', 'unknown')}\n"
    instructions += f"- Đối tượng: {raw_conversation.get('Targeted_Customers', 'unknown')}\n"
    instructions += f"- Ý định: {raw_conversation.get('Intentions', 'unknown')}\n\n"
    instructions += "Nội dung:\n"
    
    for msg in raw_conversation.get('messages', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        instructions += f"[{role}]: {content}\n"

    instructions += """

## FORMAT REASONING (4 dòng bắt buộc - hãy suy nghĩ theo format này):
Nhận diện tình huống: [Phân tích tình huống hiện tại của KH]
Xác định quy trình áp dụng: [Tên procedure]
Xác định bước hiện tại trong quy trình: [Bước số mấy - mô tả]
Chuỗi hành động: [action A → action B → action C]

## CẤU TRÚC RESPONSE
Mỗi turn của bạn phải bao gồm:
1. Reasoning (model sẽ tự động tạo reasoning tokens khi suy nghĩ)
2. Tool calls (nếu cần) - gọi tools từ danh sách available tools
3. Assistant response (câu trả lời gửi cho user)
4. Để tiếp tục hội thoại: tạo user message tiếp theo ở cuối response với format:
   
   NEXT_USER_MESSAGE: [nội dung user reply]
   
5. Hoặc kết thúc hội thoại với:
   
   END_CONVERSATION: true

Hãy bắt đầu xử lý hội thoại!
"""

    return instructions


async def refine_conversation(state: State) -> Dict:
    """
    Refine conversation using GPT-5 reasoning model as teacher LLM.

    Flow:
    1. Build instructions với procedure + tools + raw conversation
    2. Teacher generate responses với:
       - reasoning (automatic reasoning tokens)
       - tool_calls (nếu cần)
       - assistant_response
       - next_user_message (để tiếp tục hội thoại)
    3. Execute tool calls và add results
    4. Repeat until END_CONVERSATION
    5. Save refined conversation
    """
    raw_conversation = state.raw_conversation
    procedure_id = state.procedure_id
    conversation_procedure = state.procedures.get(procedure_id, {})
    tools = state.tools

    # Initialize OpenAI client
    client = OpenAI()

    # Build instructions
    instructions = build_system_instructions(conversation_procedure, tools, raw_conversation)

    # Convert tools to OpenAI function format
    openai_tools: List[ToolParam] = []
    for tool in tools:
        # Convert parameters từ string description sang JSON Schema proper format
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Parse parameters từ description format
        if "parameters" in tool and isinstance(tool["parameters"], dict):
            for param_name, param_desc in tool["parameters"].items():
                # Parse description để extract type và required
                # Format: "type (optional/required) - description"
                param_type = "string"  # default
                is_required = True
                description = param_desc
                
                if isinstance(param_desc, str):
                    # Extract type from description
                    if "string" in param_desc.lower():
                        param_type = "string"
                    elif "boolean" in param_desc.lower():
                        param_type = "boolean"
                    elif "object" in param_desc.lower():
                        param_type = "object"
                    elif "number" in param_desc.lower():
                        param_type = "number"
                    
                    # Check if optional
                    if "(optional)" in param_desc or "optional" in param_desc.lower():
                        is_required = False
                    
                    # Extract description after "-"
                    if " - " in param_desc:
                        description = param_desc.split(" - ", 1)[1]
                    else:
                        description = param_desc
                
                parameters_schema["properties"][param_name] = {
                    "type": param_type,
                    "description": description
                }
                
                if is_required:
                    parameters_schema["required"].append(param_name)
        
        # Nếu không có parameters, vẫn phải có object rỗng
        if not parameters_schema["properties"]:
            parameters_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        else:
            parameters_schema["additionalProperties"] = False
        
        openai_tools.append({
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": parameters_schema,
            "strict": True  # Không bắt buộc strict với reasoning model
        })

    # Initialize conversation với user messages từ đầu
    origin_messages = raw_conversation.get('messages', [])
    conversation_input: List[Dict] = []

    # Add tất cả user messages liên tiếp từ đầu
    for msg in origin_messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'user':
            conversation_input.append({
                "role": "user",
                "content": content
            })
        elif role == 'assistant' and not conversation_input:
            # Skip assistant greeting nếu là message đầu tiên
            continue
        else:
            # Stop khi gặp assistant message sau user messages
            break

    # Nếu không có user message nào, tạo từ context
    if not conversation_input:
        first_user_content = "Tôi cần hỗ trợ"
        for msg in origin_messages:
            if msg.get('role') == 'user':
                first_user_content = msg.get('content', first_user_content)
                break
        conversation_input.append({
            "role": "user",
            "content": first_user_content
        })

    # Track refined messages cho output
    refined_messages: List[Dict] = []
    for msg in conversation_input:
        refined_messages.append(msg)

    # Main conversation loop
    turn_count = 0
    max_turns = 20  # Safety limit

    while turn_count < max_turns:
        turn_count += 1
        print(f"\n=== Turn {turn_count} ===")

        # Call GPT-5 với reasoning
        try:
            print("Calling GPT-5 with conversation input...", conversation_input)
            response = client.responses.create(
                model="gpt-5-mini",
                reasoning={
                    "effort": "medium",
                    "summary": "detailed"
                },
                instructions=instructions,
                input=conversation_input,
                tools=openai_tools,
                max_output_tokens=4000
            )
        except Exception as e:
            print(f"Error calling GPT-5 at turn {turn_count}: {e}")
            break

        # Check for incomplete response
        if response.status == "incomplete":
            print(f"Response incomplete: {response.incomplete_details}")
            break

        # Extract reasoning summary
        reasoning_summary = ""
        assistant_content = ""
        tool_calls_made = []
        
        # Parse output items
        for item in response.output:
            if item.type == "reasoning" and hasattr(item, 'summary'):
                # Extract reasoning summary
                for summary_item in item.summary:
                    if summary_item.type == "summary_text":
                        reasoning_summary = summary_item.text
                        
            elif item.type == "function_call":
                # Store tool call
                tool_calls_made.append({
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments
                })
                
            elif item.type == "message":
                # Extract assistant message
                for content_item in item.content:
                    if content_item.type == "output_text":
                        assistant_content += content_item.text

        # Build full assistant message với thinking tags
        assistant_parts = []
        if reasoning_summary:
            assistant_parts.append(f"<thinking>\n{reasoning_summary}\n</thinking>")

        # Execute tool calls nếu có
        if tool_calls_made:
            # Add tool calls to conversation input for next turn
            for tool_call in tool_calls_made:
                try:
                    # Parse arguments
                    args = json.loads(tool_call["arguments"])
                    
                    # Call tool
                    tool_result = call_tool(tool_name=tool_call["name"], **args)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)

                    # Add notation to assistant message
                    tool_notation = f"[Tool: {tool_call['name']}({tool_call['arguments']})]"
                    assistant_parts.append(tool_notation)
                    assistant_parts.append(f"[Tool Result: {tool_result_str}]")

                    # Add tool result to conversation input
                    conversation_input.append({
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": tool_result_str
                    })

                except Exception as e:
                    print(f"Error calling tool {tool_call['name']}: {e}")
                    assistant_parts.append(f"[Tool Error: {tool_call['name']} - {str(e)}]")

            # Nếu có tool calls, cần gọi lại model để lấy response sau tool
            # Keep reasoning items theo docs recommendation
            continue

        # Add assistant response
        if assistant_content:
            assistant_parts.append(assistant_content)

        # Combine assistant message
        full_assistant_message = "\n".join(assistant_parts)
        
        # Add to refined messages
        refined_messages.append({
            "role": "assistant",
            "content": full_assistant_message
        })

        # Add assistant message to conversation input
        conversation_input.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Parse assistant content để tìm next user message hoặc end signal
        if "END_CONVERSATION: true" in assistant_content or "END_CONVERSATION:true" in assistant_content:
            print("Conversation ended by assistant")
            break

        # Extract NEXT_USER_MESSAGE
        next_user_msg = None
        lines = assistant_content.split('\n')
        for line in lines:
            if line.startswith("NEXT_USER_MESSAGE:"):
                next_user_msg = line.replace("NEXT_USER_MESSAGE:", "").strip()
                break

        if next_user_msg:
            # Add next user message
            refined_messages.append({
                "role": "user",
                "content": next_user_msg
            })
            conversation_input.append({
                "role": "user",
                "content": next_user_msg
            })
        else:
            # No next message and no END signal -> end anyway
            print("No next user message found, ending conversation")
            break

    # Save refined conversation
    refined_data = {
        **raw_conversation,
        "refined_messages": refined_messages,
        "procedure_id": procedure_id,
        "total_turns": turn_count
    }

    with open("data/refined_conversations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(refined_data, ensure_ascii=False) + "\n")

    return {**state.__dict__, "refined_conversation": refined_data}
