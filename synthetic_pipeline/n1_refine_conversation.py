"""
N1 Refine Conversation Pipeline
Dual-Agent Architecture:
1. CSKH Agent: Chuyên gia CSKH Heineken, chỉ biết procedure + tools
2. User Agent: Simulate user behavior dựa trên raw_conversation, quyết định end

Flow: User msg → CSKH response/tool → User Agent generates next msg → loop
"""

import json
from typing import Dict, List, Optional, cast

from synthetic_pipeline.state import State
# Note: mock tools (call_tool) will be used in n3 for expansion
# In n1, we use Tool Result Agent for context-aware results
from langchain_openai import ChatOpenAI
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage, BaseMessage
from pydantic import BaseModel


# =============================================================================
# TOOL INSTRUCTIONS
# =============================================================================

def build_tool_instructions() -> str:
    """Build detailed tool instructions with examples from mock_tools"""
    return """
## AVAILABLE TOOLS

### 1. tra_cuu_thong_tin
**Mục đích:** Tra cứu thông tin cửa hàng/NPP/SubD trên hệ thống

**Khi nào dùng:**
- Xác minh thông tin KH (OutletID, SĐT, ma_npp)
- Kiểm tra KH đã đăng ký app HVN chưa
- Lấy SĐT đăng ký để gửi OTP

**Tham số:**
- ma_cua_hang (optional): Mã cửa hàng/OutletID 8 số
- sdt (optional): Số điện thoại 10 số
- ma_npp (optional): Mã NPP/SubD

**Ví dụ gọi:**
```json
{"name": "tra_cuu_thong_tin", "arguments": "{\\"ma_cua_hang\\": \\"63235514\\"}"}
```

**Kết quả trả về:**
```json
{
    "ma_cua_hang": "63235514",
    "ten_cua_hang": "Tạp Hóa Bảo Trân",
    "sdt_dang_ky": "0912345678",
    "trang_thai": "active",
    "da_dang_ky_app": true
}
```

**Cách xử lý kết quả:**
- Nếu `da_dang_ky_app=false`: Hướng dẫn KH đăng ký app
- Nếu `trang_thai="đóng"`: Thông báo cửa hàng đã đóng, hướng dẫn liên hệ SA
- Dùng `sdt_dang_ky` để xác nhận SĐT nhận OTP

---

### 2. kiem_tra_mqh
**Mục đích:** Kiểm tra mối quan hệ (MQH) giữa Outlet với NPP/SubD trên SEM

**Khi nào dùng:**
- KH không thấy NPP/SubD trên app
- Đơn hàng không về NPP
- Kiểm tra tính hợp lệ của MQH trước khi đặt hàng

**Tham số:**
- outlet_id (required): Mã cửa hàng 8 số
- npp_subd_id (optional): Mã NPP/SubD cần kiểm tra

**Ví dụ gọi:**
```json
{"name": "kiem_tra_mqh", "arguments": "{\\"outlet_id\\": \\"63235514\\", \\"npp_subd_id\\": \\"10375694\\"}"}
```

**Kết quả trả về:**
```json
{
    "co_mqh": true,
    "npp_subd_hien_tai": "10375694",
    "ten_npp_subd": "NPP QNI29",
    "trang_thai_mqh": "Active",
    "last_modified": "2024-01-15 10:30:00",
    "modified_by": "SA",
    "tu_tao": false
}
```

**Cách xử lý kết quả:**
- Nếu `co_mqh=false`: Thông báo chưa có MQH, hướng dẫn liên hệ SA tạo MQH
- Nếu `trang_thai_mqh="Inactive"`: MQH đã hết hiệu lực, cần SA kích hoạt lại
- Nếu `tu_tao=true`: MQH do user tự tạo, có thể cần SA xác nhận
- Kiểm tra `last_modified`: Nếu < 24h, nhắc KH chờ đồng bộ

---

### 3. kiem_tra_don_hang
**Mục đích:** Kiểm tra trạng thái đơn hàng trên hệ thống

**Khi nào dùng:**
- KH hỏi về trạng thái đơn đã đặt
- Kiểm tra đơn không về NPP/SubD
- Xác minh đơn Gratis đã được approve chưa

**Tham số:**
- ma_don_hang (required): Mã đơn hàng (VD: 2509076469100, CO251124-01481)
- kenh (required): Kênh đặt hàng - "SEM", "HVN", hoặc "DIS_Lite"

**Ví dụ gọi:**
```json
{"name": "kiem_tra_don_hang", "arguments": "{\\"ma_don_hang\\": \\"2509076469100\\", \\"kenh\\": \\"SEM\\"}"}
```

**Kết quả trả về:**
```json
{
    "trang_thai": "Đang xử lý",
    "outlet_id": "63235514",
    "npp_subd": "10375694",
    "ngay_dat": "2024-01-15",
    "loai_don": "Thường",
    "approved": null
}
```

**Cách xử lý kết quả:**
- `trang_thai="Chưa về NPP"`: Kiểm tra MQH bằng kiem_tra_mqh()
- `loai_don="Gratis"` và `approved=false`: Đơn chờ ASM approve
- `loai_don="Gratis"` qua SubD: Lỗi, Gratis chỉ qua NPP

---

### 4. tao_ticket
**Mục đích:** Tạo ticket chuyển tuyến cho team chuyên trách

**Khi nào dùng:**
- Vấn đề cần team khác xử lý (SEM/HVN/SA/CS/IT)
- Escalate case phức tạp

**Tham số:**
- team (required): Team xử lý - "SEM", "HVN", "SA", "CS", "IT"
- noi_dung (required): Mô tả vấn đề
- du_lieu (required): JSON string chứa dữ liệu liên quan

**Ví dụ gọi:**
```json
{"name": "tao_ticket", "arguments": "{\\"team\\": \\"SEM\\", \\"noi_dung\\": \\"Đơn không về NPP\\", \\"du_lieu\\": \\"{\\\\\\"outlet_id\\\\\\": \\\\\\"63235514\\\\\\", \\\\\\"ma_don\\\\\\": \\\\\\"2509076469100\\\\\\"}\\"}"}
```

**Kết quả trả về:**
```json
{
    "ticket_id": "TKT123456",
    "trang_thai": "Đã tạo thành công"
}
```

---

### 5. force_sync
**Mục đích:** Thực hiện Force Sync dữ liệu trên SEM

**Khi nào dùng:**
- Sau khi SA tạo/sửa MQH, cần sync ngay
- Dữ liệu không đồng bộ giữa các hệ thống

**Tham số:**
- outlet_id (required): Mã cửa hàng 8 số
- npp_subd_id (optional): Mã NPP/SubD

**Ví dụ gọi:**
```json
{"name": "force_sync", "arguments": "{\\"outlet_id\\": \\"63235514\\", \\"npp_subd_id\\": \\"10375694\\"}"}
```

**Kết quả trả về:**
```json
{
    "thanh_cong": true
}
```

---

### 6. gui_huong_dan
**Mục đích:** Gửi hướng dẫn/tài liệu SOP cho khách hàng

**Khi nào dùng:**
- KH cần hướng dẫn chi tiết về quy trình
- Gửi tài liệu hỗ trợ

**Tham số:**
- loai_huong_dan (required): Loại hướng dẫn - "xuat_gratis", "dang_nhap", "quen_mat_khau", "dat_hang", etc.

**Ví dụ gọi:**
```json
{"name": "gui_huong_dan", "arguments": "{\\"loai_huong_dan\\": \\"quen_mat_khau\\"}"}
```

**Kết quả trả về:**
```json
{
    "da_gui": true
}
```
"""


# =============================================================================
# CSKH AGENT - Customer Support Agent
# =============================================================================

def build_cskh_instructions(conversation_procedure: Dict) -> str:
    """Build instructions for CSKH Agent - only knows procedure and tools"""

    tool_instructions = build_tool_instructions()

    instructions = f"""Bạn là nhân viên CSKH (Chăm sóc khách hàng) của Heineken Vietnam.

## Bạn cần tuân thủ quy trình sau: {conversation_procedure.get('tên', 'Unknown')}
Mục tiêu: {conversation_procedure.get('mục_tiêu', 'Unknown')}

### Luồng thực thi chính:
"""
    for step in conversation_procedure.get('luồng_thực_thi_chung', []):
        instructions += f"- Bước {step['bước']}: {step['mô_tả']} → {step['chain_action']}\n"

    instructions += "\n### Các trường hợp ngoại lệ:\n"
    for edge_case in conversation_procedure.get('edge_cases', []):
        instructions += f"- {edge_case['case']}: {edge_case['chain_action']}\n"

    instructions += """

## QUY TẮC XỬ LÝ

### Xác định yêu cầu KH khi chưa rõ ý định khách hàng:
- Hỏi rõ vấn đề KH đang gặp phải

### Mỗi turn bạn CHỈ làm MỘT việc:
1. Trả lời khách hàng (assistant_response), HOẶC
2. Gọi tool để lấy thông tin (tool_call)

### KHÔNG ĐƯỢC:
- Xử lý nhiều bước cùng lúc

## FORMAT REASONING
```
Tình huống: [KH đang hỏi/yêu cầu gì?]
Quy trình: [Cần sử dụng quy trình nào?]
Bước: [Bước hiện tại trong quy trình]
Thông tin có: [Đã biết gì?]
Thông tin cần thêm: [Cần gì thêm?]
Hành động: [Trả lời / Gọi tool]
```
Lưu ý: 
- Quy trình có thể là "không xác định": chưa rõ ý định khách hàng, "không liên quan": khách hỏi linh tinh
- Bước: 1, 2, 3, ... - [mô tả bước] hoặc "ngoại lệ - [mô tả ngoại lệ]" hoặc bỏ trống nếu không xác định/không liên quan. Bước tối đa gồm 2 phần như ví dụ, không giải thích gì thêm. Không trộn bước 1,2 ,3 ... với ngoại lệ. ngoại lệ cần theo chuẩn "ngoại lệ - [mô tả ngoại lệ]"
##### Ví dụ:

```
Tình huống: KH quên mật khẩu đăng nhập app
Quy trình: Đăng ký/Quên mật khẩu
Bước: 2 - Xác thực thông tin KH
Thông tin có: KH cung cấp SĐT 0912345678
Thông tin cần thêm: Xác nhận OutletID
Hành động: Gọi tool tra_cuu_thong_tin để lấy thông tin đăng ký
```

```
Tình huống: KH chưa nêu rõ vấn đề
Quy trình: không xác định
Bước:
Thông tin có: Chưa có thông tin
Thông tin cần thêm: vấn đề KH đang gặp phải
Hành động: Hỏi KH vấn đề đang gặp phải
```

```
Tình huống: KH hỏi vấn đề không liên quan đến CSKH
Quy trình: không liên quan
Bước:
Thông tin có: không liên quan
Thông tin cần thêm: không cần thêm
Hành động: Thân thiện từ chối trả lời, hướng KH liên hệ kênh phù hợp
```

```
Tình huống: KH hỏi về trạng thái đơn hàng
Quy trình: Kiểm tra đơn hàng
Bước: ngoại lệ - đơn hàng bị hủy
Thông tin có: KH cung cấp mã đơn hàng 2509076469100
Thông tin cần thêm: Kiểm tra trạng thái đơn hàng
Hành động: Gọi tool kiem_tra_don_hang để kiểm tra trạng thái đơn
```


## VĂN PHONG
- Lịch sự, thân thiện (anh/chị/em, ạ)
- Ngắn gọn, đi thẳng vào vấn đề
"""

    instructions += tool_instructions

    return instructions


# =============================================================================
# USER AGENT - Simulates user behavior
# =============================================================================

def build_user_instructions(raw_conversation: Dict) -> str:
    """Build instructions for User Agent - knows the original conversation"""

    instructions = f"""Bạn đóng vai KHÁCH HÀNG trong cuộc hội thoại với CSKH Heineken Vietnam.

## NHIỆM VỤ
Dựa trên hội thoại gốc bên dưới, bạn sẽ simulate hành vi của khách hàng:
- Đặt câu hỏi tương tự như hội thoại gốc
- Cung cấp thông tin khi được hỏi
- Phản hồi tự nhiên như khách hàng thật

## HỘI THOẠI GỐC (tham khảo để biết user cần gì):
- Loại vấn đề: {raw_conversation.get('Sub_Category', 'unknown')}
- Đối tượng KH: {raw_conversation.get('Targeted_Customers', 'unknown')}
- Ý định: {raw_conversation.get('Intentions', 'unknown')}

### Nội dung hội thoại gốc:
"""

    for msg in raw_conversation.get('messages', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        instructions += f"[{role}]: {content}\n"

    instructions += """

## QUY TẮC

### Bạn phải:
1. Giữ nguyên VẤN ĐỀ CHÍNH của khách hàng trong hội thoại gốc
2. Cung cấp thông tin tương tự (OutletID, SĐT, tên cửa hàng, etc.)
3. Phản hồi tự nhiên, không quá formal
4. Quyết định khi nào kết thúc hội thoại (khi vấn đề đã được giải quyết)

### Văn phong khách hàng:
- Có thể viết tắt, không dấu
- Đôi khi thiếu kiên nhẫn nếu chờ lâu
- Cảm ơn khi được hỗ trợ tốt

## OUTPUT FORMAT
```json
{
  "reasoning": "Phân tích: CSKH vừa hỏi X, theo hội thoại gốc mình cần trả lời Y",
  "user_message": "Nội dung tin nhắn của khách hàng",
  "end_conversation": false
}
```

### Khi nào set end_conversation = true:
- CSKH đã giải quyết xong vấn đề
- Đã nhận được hướng dẫn/thông tin cần thiết
- Tương đương với kết thúc trong hội thoại gốc
"""

    return instructions


# =============================================================================
# TOOL RESULT AGENT - Generates context-aware tool results
# =============================================================================

def build_tool_result_instructions(raw_conversation: Dict) -> str:
    """Build instructions for Tool Result Agent - generates realistic tool results"""

    instructions = f"""Bạn là TOOL RESULT GENERATOR cho hệ thống CSKH Heineken Vietnam.

## NHIỆM VỤ
Khi CSKH gọi một tool, bạn sẽ tạo kết quả tool phù hợp với:
1. Luồng hội thoại gốc (để conversation đi đúng hướng)
2. Thông tin KH đã cung cấp trong conversation
3. Logic nghiệp vụ của tool

## HỘI THOẠI GỐC (tham khảo để biết kết quả mong đợi):
- Loại vấn đề: {raw_conversation.get('Sub_Category', 'unknown')}
- Đối tượng KH: {raw_conversation.get('Targeted_Customers', 'unknown')}
- Ý định: {raw_conversation.get('Intentions', 'unknown')}

### Nội dung hội thoại gốc:
"""

    for msg in raw_conversation.get('messages', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        instructions += f"[{role}]: {content}\n"

    instructions += """

## TOOL SCHEMAS

### tra_cuu_thong_tin
Output: {"ma_cua_hang": str, "ten_cua_hang": str, "sdt_dang_ky": str, "trang_thai": "active"|"đóng"|"inactive", "da_dang_ky_app": bool}

### kiem_tra_mqh
Output: {"co_mqh": bool, "npp_subd_hien_tai": str|null, "ten_npp_subd": str|null, "trang_thai_mqh": "Active"|"Inactive"|null, "last_modified": str|null, "modified_by": str|null, "tu_tao": bool}

### kiem_tra_don_hang
Output: {"trang_thai": str, "outlet_id": str, "npp_subd": str, "ngay_dat": str, "loai_don": "Thường"|"Gratis", "approved": bool|null}

### tao_ticket
Output: {"ticket_id": str, "trang_thai": "Đã tạo thành công"}

### force_sync
Output: {"thanh_cong": bool}

### gui_huong_dan
Output: {"da_gui": bool}

## QUY TẮC

### Bạn PHẢI:
1. Trả về kết quả JSON hợp lệ theo schema của tool
2. Dựa vào hội thoại gốc để biết kết quả nên là gì (VD: nếu conversation về login thành công → trang_thai="active")
3. Giữ nhất quán với thông tin KH đã cung cấp (OutletID, SĐT, etc.)
4. Tạo dữ liệu realistic (8 số cho OutletID, 10 số cho SĐT, etc.)

### Ví dụ:
- Nếu hội thoại gốc là về "quên mật khẩu" và CSKH gọi tra_cuu_thong_tin → trả về trang_thai="active", da_dang_ky_app=true
- Nếu hội thoại gốc có vấn đề về MQH → kiem_tra_mqh có thể trả về co_mqh=false hoặc trang_thai_mqh="Inactive"

## OUTPUT FORMAT
Trả về reasoning và tool_result (JSON string).
"""

    return instructions


class ToolResultResponse(BaseModel):
    """Response from Tool Result Agent"""
    reasoning: str
    tool_result: str  # JSON string of the tool result


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolCallRequest(BaseModel):
    name: str
    arguments: str


class CSKHResponse(BaseModel):
    """Response from CSKH Agent"""
    reasoning_content: str
    assistant_response: Optional[str] = None
    tool_call: Optional[ToolCallRequest] = None


class UserResponse(BaseModel):
    """Response from User Agent"""
    reasoning: str
    user_message: str
    end_conversation: bool = False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def refine_conversation(state: State) -> State:
    """
    Triple-agent conversation refinement:
    1. CSKH Agent: Responds to user, calls tools
    2. User Agent: Generates next user message based on raw_conversation
    3. Tool Result Agent: Generates context-aware tool results
    """

    raw_conversation = state.raw_conversation
    procedure_id = state.procedure_id
    conversation_procedure = state.procedures.get(procedure_id, {})

    # Build instructions for all agents
    cskh_instructions = build_cskh_instructions(conversation_procedure)
    user_instructions = build_user_instructions(raw_conversation)
    tool_result_instructions = build_tool_result_instructions(raw_conversation)

    # Initialize LLMs
    cskh_llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1
    ).with_structured_output(CSKHResponse)

    user_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2
    ).with_structured_output(UserResponse)

    tool_result_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1  # Lower temperature for more consistent tool results
    ).with_structured_output(ToolResultResponse)

    # Conversation histories (separate for each agent)
    cskh_history: List[BaseMessage] = [SystemMessage(content=cskh_instructions)]
    user_history: List[BaseMessage] = [SystemMessage(content=user_instructions)]
    tool_result_history: List[BaseMessage] = [SystemMessage(content=tool_result_instructions)]

    # Output: refined messages
    refined_messages: List[Dict] = []

    # Get first user message from raw_conversation
    first_user_msg = None
    for msg in raw_conversation.get('messages', []):
        if msg['role'] == 'user':
            first_user_msg = msg['content']
            break

    if not first_user_msg:
        first_user_msg = "Alo"

    # Add first user message
    cskh_history.append(HumanMessage(content=first_user_msg))
    refined_messages.append({"role": "user", "content": first_user_msg})

    print(f"Starting conversation refinement")
    print(f"First user message: {first_user_msg}")

    max_turns = 20  # Safety limit
    turn_count = 0

    while turn_count < max_turns:
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")

        # =================================================================
        # STEP 1: CSKH Agent responds
        # =================================================================
        cskh_response = cast(CSKHResponse, await cskh_llm.ainvoke(cskh_history))

        reasoning = cskh_response.reasoning_content
        assistant_response = cskh_response.assistant_response
        tool_call = cskh_response.tool_call

        # Build refined content with thinking
        refine_content = f"<think>{reasoning}</think>\n"

        if tool_call:
            # CSKH wants to call a tool
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool_id = str(uuid4())


            # Add to history
            tool_call_obj = ToolCall(name=tool_name, args=tool_args, id=tool_id, type="tool_call")
            ai_message = AIMessage(content="", tool_calls=[tool_call_obj])
            cskh_history.append(ai_message)

            # Save to refined messages
            refine_content += f"<tool_call>{{'name': '{tool_name}', 'arguments': {tool_args}}}</tool_call>"
            refined_messages.append({"role": "assistant", "content": refine_content})
            print(f"CSKH calls tool: {refine_content}")
            tool_context = f"""Tool được gọi: {tool_name}
Arguments: {json.dumps(tool_args, ensure_ascii=False)}

Conversation history hiện tại:
"""
            for msg in refined_messages:
                tool_context += f"[{msg['role']}]: {msg['content'][:200]}\n"

            tool_context += "\nHãy tạo tool result phù hợp với luồng hội thoại."

            tool_result_history.append(HumanMessage(content=tool_context))
            tool_result_response = cast(ToolResultResponse, await tool_result_llm.ainvoke(tool_result_history))


            tool_result_str = tool_result_response.tool_result
            print(f"+ Tool result: {tool_result_str}")

            # Update tool_result_history with the response for context
            tool_result_history.append(AIMessage(content=tool_result_str))

            # Add tool result to CSKH history
            tool_message = ToolMessage(content=tool_result_str, tool_call_id=tool_id)
            cskh_history.append(tool_message)
            refined_messages.append({"role": "tool", "content": tool_result_str})

            # Continue loop - CSKH will process tool result
            continue

        elif assistant_response:
            # CSKH responds to user

            refine_content += assistant_response
            cskh_history.append(AIMessage(content=assistant_response))
            refined_messages.append({"role": "assistant", "content": refine_content})
            print(f"CSKH response: {refine_content}")
            # Also add to user_history so User Agent sees the response
            user_history.append(AIMessage(content=assistant_response))

        else:
            raise ValueError("CSKH must provide either assistant_response or tool_call")

        # =================================================================
        # STEP 2: User Agent generates next message
        # =================================================================
        # Add context about what CSKH just said
        user_history.append(HumanMessage(content=f"CSKH vừa trả lời: {assistant_response}\n\nHãy tạo tin nhắn tiếp theo của khách hàng."))

        user_response = cast(UserResponse, await user_llm.ainvoke(user_history))

        if user_response.end_conversation:
            print("User Agent decided to end conversation.")
            # Optionally add a final thank you message
            if user_response.user_message:
                refined_messages.append({"role": "user", "content": user_response.user_message})
                cskh_history.append(HumanMessage(content=user_response.user_message))
            break

        # Add user message
        user_msg = user_response.user_message
        print(f"User message: <think>{user_response.reasoning}</think>\n {user_msg}")

        refined_messages.append({"role": "user", "content": user_msg})
        cskh_history.append(HumanMessage(content=user_msg))

        # Update user_history for next round
        user_history.append(AIMessage(content=user_msg))

    return {**state.__dict__, "refined_messages": refined_messages}
