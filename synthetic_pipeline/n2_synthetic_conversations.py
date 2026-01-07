"""
N2 Synthetic Conversations Pipeline
Generates multiple synthetic conversations from refined conversation + procedure:
1. One conversation following main flow exactly
2. One conversation for each edge case
3. One conversation where user asks irrelevant things and assistant refuses

Uses single agent to generate entire conversation at once.
"""

import json
import random
from typing import Dict, List, Literal

from synthetic_pipeline.state import State
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    


class SyntheticConversation(BaseModel):
    """Output model for generated conversation"""
    messages: List[Message]


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_conversation_generator_prompt(
    procedure: Dict,
    refined_messages: List[Dict],
    scenario: str,
    scenario_type: Literal["main_flow", "edge_case", "off_topic"]
) -> str:
    """Build system prompt for conversation generator"""

    prompt = f"""Bạn là chuyên gia tạo hội thoại tổng hợp cho hệ thống CSKH Heineken Vietnam.

## NHIỆM VỤ
Tạo một cuộc hội thoại HOÀN CHỈNH giữa khách hàng (user) và nhân viên CSKH (assistant).

## QUY TRÌNH: {procedure.get('tên', 'Unknown')}
Mục tiêu: {procedure.get('mục_tiêu', 'Unknown')}

### Luồng thực thi chính:
"""
    for step in procedure.get('luồng_thực_thi_chung', []):
        prompt += f"- Bước {step['bước']}: {step['mô_tả']} → {step['chain_action']}\n"

    prompt += "\n### Luồng ngoại lệ:\n"
    for edge_case in procedure.get('edge_cases', []):
        prompt += f"- {edge_case['case']}: {edge_case['chain_action']}\n"

    prompt += f"""

## HỘI THOẠI GỐC (tham khảo phong cách và thông tin):
"""
    for msg in refined_messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        prompt += f"[{role}]: {content}\n"

    prompt += """

## AVAILABLE TOOLS
Assistant có thể gọi các tools sau (output dạng <tool_call>...</tool_call>):

1. **tra_cuu_thong_tin**: Tra cứu thông tin cửa hàng/NPP
   - Input: ma_cua_hang, sdt, ma_npp
   - Output: {ma_cua_hang, ten_cua_hang, sdt_dang_ky, trang_thai, da_dang_ky_app}

2. **kiem_tra_mqh**: Kiểm tra mối quan hệ Outlet-NPP/SubD
   - Input: outlet_id, npp_subd_id
   - Output: {co_mqh, npp_subd_hien_tai, ten_npp_subd, trang_thai_mqh, tu_tao}

3. **kiem_tra_don_hang**: Kiểm tra trạng thái đơn hàng
   - Input: ma_don_hang, kenh
   - Output: {trang_thai, outlet_id, npp_subd, ngay_dat, loai_don, approved}

4. **tao_ticket**: Tạo ticket chuyển tuyến
   - Input: team, noi_dung, du_lieu
   - Output: {ticket_id, trang_thai}

5. **force_sync**: Force sync dữ liệu SEM
   - Input: outlet_id, npp_subd_id
   - Output: {thanh_cong}

6. **gui_huong_dan**: Gửi hướng dẫn SOP
   - Input: loai_huong_dan
   - Output: {da_gui}
"""

    # Scenario-specific instructions
    if scenario_type == "main_flow":
        prompt += f"""
## KỊCH BẢN: LUỒNG CHÍNH
- Tạo hội thoại theo đúng các bước trong luồng_thực_thi_chung
- Mọi thứ diễn ra SUÔN SẺ, không có edge case
- Tool results trả về kết quả thành công (trang_thai="active", co_mqh=true, etc.)
- Kịch bản: {scenario}
"""
    elif scenario_type == "edge_case":
        prompt += f"""
## KỊCH BẢN: Luồng ngoại lệ
- Edge case cần xử lý: **{scenario}**
- Bắt đầu theo luồng chính, sau đó GẶP edge case này
- Tool results PHẢI phản ánh edge case (VD: co_mqh=false, trang_thai="đóng", etc.)
- Xử lý theo chain_action của luồng ngoại lệ
"""
    else:  # off_topic
        prompt += f"""
## KỊCH BẢN: YÊU CẦU KHÔNG LIÊN QUAN
- Khách hàng hỏi: **{scenario}**
- Assistant lịch sự từ chối, giải thích đây là kênh hỗ trợ Heineken
- Không cần gọi tool
- Hội thoại ngắn (2-4 turns)
"""

    prompt += """

## FORMAT OUTPUT
Trả về list messages với format:
- role: "user" | "assistant" | "tool"
- content: nội dung tin nhắn

### Assistant message format:
```
<think>
Tình huống: [KH đang hỏi/yêu cầu gì?]
Quy trình: [Cần sử dụng quy trình nào?]
Bước: [Bước hiện tại trong quy trình]
Thông tin có: [Đã biết gì?]
Thông tin cần thêm: [Cần gì thêm?]
Hành động: [Trả lời / Gọi tool]
</think>
```
Lưu ý: 
- Quy trình có thể là "không xác định": chưa rõ ý định khách hàng, "không liên quan": khách hỏi linh tinh
- Bước: 1, 2, 3, ... - [mô tả bước] hoặc "ngoại lệ - [mô tả ngoại lệ]" hoặc bỏ trống nếu không xác định/không liên quan. Bước tối đa gồm 2 phần như ví dụ, không giải thích gì thêm. Không trộn bước 1,2 ,3 ... với ngoại lệ. ngoại lệ cần theo chuẩn "ngoại lệ - [mô tả ngoại lệ]"
##### Ví dụ:

```
<think>
Tình huống: KH quên mật khẩu đăng nhập app
Quy trình: Đăng ký/Quên mật khẩu
Bước: 2 - Xác thực thông tin KH
Thông tin có: KH cung cấp SĐT 0912345678
Thông tin cần thêm: Xác nhận OutletID
Hành động: Gọi tool tra_cuu_thong_tin để lấy thông tin đăng ký
</think>
```

```
<think>
Tình huống: KH chưa nêu rõ vấn đề
Quy trình: không xác định
Bước:
Thông tin có: Chưa có thông tin
Thông tin cần thêm: vấn đề KH đang gặp phải
Hành động: Hỏi KH vấn đề đang gặp phải
</think>
```

```
<think>
Tình huống: KH hỏi vấn đề không liên quan đến CSKH
Quy trình: không liên quan
Bước:
Thông tin có: không liên quan
Thông tin cần thêm: không cần thêm
Hành động: Thân thiện từ chối trả lời, hướng KH liên hệ kênh phù hợp
</think>
```

### Tool message format:
```json
{"key": "value", ...}
```

## QUY TẮC
1. Mỗi assistant turn CHỈ trả lời HOẶC gọi tool, không cả hai
2. Sau tool call PHẢI có tool result message
3. Văn phong lịch sự (anh/chị/em, ạ)
4. Dựa vào hội thoại gốc để lấy thông tin (OutletID, SĐT, tên cửa hàng)
5. Hội thoại kết thúc khi vấn đề được giải quyết
"""

    return prompt


# =============================================================================
# SINGLE CONVERSATION GENERATOR
# =============================================================================

async def generate_single_conversation(
    procedure: Dict,
    refined_messages: List[Dict],
    scenario: str,
    scenario_type: Literal["main_flow", "edge_case", "off_topic"]
) -> List[Dict]:
    """Generate a complete synthetic conversation for a given scenario"""

    system_prompt = build_conversation_generator_prompt(
        procedure=procedure,
        refined_messages=refined_messages,
        scenario=scenario,
        scenario_type=scenario_type
    )

    llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.7
    ).with_structured_output(SyntheticConversation)

    user_prompt = f"Hãy tạo cuộc hội thoại hoàn chỉnh cho kịch bản: {scenario}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # Convert to list of dicts
    return [{"role": msg.role, "content": msg.content} for msg in response.messages]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def synthetic_conversations(state: State) -> State:
    """
    Generate multiple synthetic conversations:
    1. One main flow conversation
    2. One conversation per edge case
    3. One off-topic rejection conversation
    """

    refined_messages = state.refined_messages
    if refined_messages is None:
        raise ValueError("refined_messages is None. Please run the refinement step first.")

    procedure_id = state.procedure_id
    procedure = state.procedures.get(procedure_id, {})

    procedure_name = procedure.get("tên", "unknown_procedure")
    edge_cases = procedure.get("edge_cases", [])

    all_conversations: List = []

    # =================================================================
    # 1. Generate main flow conversation
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Generating MAIN FLOW: {procedure_name}")
    print(f"{'='*60}")

    main_flow_scenario = f"Khách hàng thực hiện {procedure_name} theo đúng luồng chính"
    main_flow_messages = await generate_single_conversation(
        procedure=procedure,
        refined_messages=refined_messages,
        scenario=main_flow_scenario,
        scenario_type="main_flow"
    )
    all_conversations.append(main_flow_messages)
    print(f"Generated {len(main_flow_messages)} messages")

    # =================================================================
    # 2. Generate edge case conversations
    # =================================================================
    for edge_case in edge_cases:
        case_name = edge_case.get("case", "Unknown")
        case_condition = edge_case.get("điều_kiện", "")

        print(f"\n{'='*60}")
        print(f"Generating EDGE CASE: {case_name}")
        print(f"{'='*60}")

        edge_case_scenario = f"{case_name}: {case_condition}"
        edge_case_messages = await generate_single_conversation(
            procedure=procedure,
            refined_messages=refined_messages,
            scenario=edge_case_scenario,
            scenario_type="edge_case"
        )
        all_conversations.append(edge_case_messages)
        print(f"Generated {len(edge_case_messages)} messages")

    # =================================================================
    # 3. Generate off-topic rejection conversation
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Generating OFF-TOPIC rejection")
    print(f"{'='*60}")

    off_topic_scenarios = [
        "Hỏi về thời tiết hôm nay",
        "Yêu cầu đặt pizza",
        "Hỏi về kết quả bóng đá",
        "Yêu cầu tư vấn mua xe",
        "Hỏi về cách nấu phở"
    ]
    off_topic_scenario = random.choice(off_topic_scenarios)

    off_topic_messages = await generate_single_conversation(
        procedure=procedure,
        refined_messages=refined_messages,
        scenario=off_topic_scenario,
        scenario_type="off_topic"
    )
    all_conversations.append(off_topic_messages)
    print(f"Generated {len(off_topic_messages)} messages")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(all_conversations)} conversations")
    print(f"  - Main flow: 1")
    print(f"  - Edge cases: {len(edge_cases)}")
    print(f"  - Off-topic: 1")
    print(f"{'='*60}")
    state.synthetic_conversations = all_conversations
    return state
