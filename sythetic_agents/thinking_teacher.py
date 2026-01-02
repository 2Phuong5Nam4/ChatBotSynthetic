from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Load procedure definitions
with open("/home/namnp/ChatBotSynthetic/data/procedure.json", "r", encoding="utf-8") as f:
    procedure = json.load(f)
with open("/home/namnp/ChatBotSynthetic/data/synthetic_conversation.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)
    # # load 5 conversaiont with 5 different procedure_id for testing
    # current_ids = set()
    # filtered_conversations = []
    # for conv in conversations:
    #     pid = str(conv.get('procedure_id'))
    #     if pid not in current_ids and pid in procedure:
    #         filtered_conversations.append(conv)
    #         current_ids.add(pid)
    #     if len(current_ids) >= 5:
    #         break
    # conversations = filtered_conversations

class ThinkingTeacherResponse(BaseModel):
    """Response model for Thinking Teacher"""
    reasoning: str = Field(..., description="Phân tích ngắn gọn: Bước hiện tại trong procedure, tình huống KH, hành động cần thực hiện")
    

class ThinkingTeacher:
    """Agent that thinks step-by-step before answering"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

    def _build_system_prompt(self, procedure_name: str, procedure_detail: str) -> str:
        """Build system prompt for the Thinking Teacher"""
        prompt = f"""Bạn là nhân viên CSKH Heineken Vietnam, đang xử lý case cho khách hàng.

## QUY TRÌNH BẠN ĐÃ NẮM RÕ:
Tên quy trình: 
{procedure_name}
chi tiết quy trình:
{procedure_detail}

## NHIỆM VỤ:
Viết thinking - kế hoạch hành động ngắn gọn TRƯỚC KHI trả lời KH. Thinking này cần tuân thủ theo quy trinh đã cho để đảm bảo xử lý đúng và nhanh chóng.

FORMAT THINKING:
Nhận diện tình huống: ...
Xác định quy trình áp dụng: ...
Xác định bước hiện tại trong quy trình: ...
Xác định chuỗi hành động tiếp theo: action A → action B → action C ...

Dùng mũi tên (→) để chain các hành động, nhóm chi tiết vào ngoặc đơn ()

VÍ DỤ THINKING TỐT (ngắn gọn, theo đúng format):
✅
Nhận diện tình huống: KH chủ điểm bán, tài khoản bị khóa do nhập sai MK nhiều lần
Xác định quy trình áp dụng: Quy trình Quên/Đổi mật khẩu
Xác định bước hiện tại: Bước 1 - Thu thập thông tin và xác nhận tài khoản
Chuỗi hành động: xác nhận outlet + mã → xin ảnh lỗi → hướng dẫn Quên MK (nhập mã outlet → OTP → tạo MK mới 12 ký tự, không trùng cũ)

✅
Nhận diện tình huống: KH quên SĐT đăng ký, không nhận được OTP
Xác định quy trình áp dụng: Quy trình Quên/Đổi mật khẩu - Xử lý tình huống đặc biệt
Xác định bước hiện tại: Bước 5 - Xử lý tình huống không nhận OTP
Chuỗi hành động: yêu cầu thông tin (tên CH, mã CH, tên người liên hệ, SĐT mới) → chuyển bộ phận cập nhật

✅
Nhận diện tình huống: KH hỏi mã NV để cài app, có vẻ đã đăng ký
Xác định quy trình áp dụng: Quy trình Quên/Đổi mật khẩu - KH đã có tài khoản
Xác định bước hiện tại: Bước 1 - Xác định tài khoản
Chuỗi hành động: hỏi tên CH + SĐT → tra cứu tài khoản → hướng dẫn đăng nhập hoặc Quên MK

✅
Nhận diện tình huống: KH đã giải quyết xong vấn đề, cảm ơn
Xác định quy trình áp dụng: Quy trình Quên/Đổi mật khẩu
Xác định bước hiện tại: Bước 6 - Xác nhận hoàn tất và cung cấp kênh hỗ trợ
Chuỗi hành động: hỏi còn cần gì không → cung cấp kênh hỗ trợ (Zalo/hotline 1800234522)


QUY TẮC VIẾT:
0. Tuân thủ nghiêm ngặt quy trình đã cho
1. Format: Nhận diện tình huống: ... Xác định quy trình áp dụng: ... Xác định bước hiện tại trong quy trình: ... Xác định chuỗi hành động tiếp theo: action A → action B → action C ...
2. Sử dụng từ ngữ ngắn gọn, súc tích
3. Chi tiết phụ trong ngoặc đơn ()
"""
        return prompt
    async def process_conversation(self, conversation: Dict) -> Dict:
        """Process entire conversation with memory of previous turns"""
        # Get procedure and metadata
        procedure_id = str(conversation.get('procedure_id'))  # Default to procedure 2 (forget password)
        if not procedure_id or procedure_id not in procedure:
            raise ValueError(f"Invalid or missing procedure_id: {procedure_id}")
        procedure_detail = procedure.get(procedure_id, {}).get('detail_description', 'N/A')
        procedure_name = procedure.get(procedure_id, {}).get('name', 'N/A')
        # Build system prompt once
        system_prompt = self._build_system_prompt(procedure_name, procedure_detail)
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        ).with_structured_output(ThinkingTeacherResponse)
        # Initialize conversation memory
        memory: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        original_messages = conversation.get('messages', [])
        enhanced_messages = []

        # Process each turn
        i = 0
        while i < len(original_messages):
            current_msg = original_messages[i]

            # User message - just add to memory and output
            if current_msg.get('role') == 'user':
                enhanced_messages.append(current_msg)
                memory.append(HumanMessage(content=current_msg.get('content', '')))
                i += 1
                continue

            # Assistant message - need to evaluate and enhance
            if current_msg.get('role') == 'assistant':
                # Get corresponding user message (should be previous message)

                # Format current turn as if YOU (the agent) are handling this turn
                current_turn = f"""
TIN NHẮN bạn chuẩn bị trả lời cho KH:
{current_msg.get('content', '')}
Hãy viết thinking (kế hoạch hành động nội tâm) của bạn chuẩn bị cho TIN NHĂN này."""

                # Add to memory and get response
                memory.append(HumanMessage(content=current_turn))
                response = await llm.ainvoke(memory)
                if isinstance(response, dict):
                    response = ThinkingTeacherResponse(**response)

                # Create enhanced message with thinking
                enhanced_msg = {
                    'role': 'assistant',
                    'content': f"<thinking>{response.reasoning}</thinking>\n{current_msg.get('content', '')}",
                }
                enhanced_messages.append(enhanced_msg)

                # Update memory with the thinking and response for next turn
                # drop the last user message (current_turn)
                memory = memory[:-1]
                memory.append(AIMessage(content=enhanced_msg['content']))

                i += 1

        # Return enhanced conversation
        conversation['messages'] = enhanced_messages
        return conversation

    def _format_memory_history(self, messages: List[Dict]) -> str:
        """Format previous messages for context"""
        if not messages:
            return "(Chưa có lịch sử)"

        history_lines = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                history_lines.append(f"KH: {content}")
            elif role == 'assistant':
                # Show the final response (could be corrected or original)
                history_lines.append(f"Agent: {content}")

        return "\n".join(history_lines)
    


async def main():
    """Main function with semaphore for concurrent processing"""
    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(5)  # Adjust this number based on your API rate limits

    async def process_with_semaphore(conv: Dict, index: int) -> Dict:
        """Process conversation with semaphore control"""
        async with semaphore:
            print(f"Processing conversation {index + 1}/{len(conversations)}...")
            teacher = ThinkingTeacher(model_name="gpt-4.1")
            try:
                result = await teacher.process_conversation(conv)
                print(f"✓ Completed conversation {index + 1}")
                return result
            except Exception as e:
                print(f"✗ Error in conversation {index + 1}: {str(e)}")
                # Return original conversation if error occurs
                return conv

    # Process all conversations concurrently with semaphore
    tasks = [process_with_semaphore(conv, i) for i, conv in enumerate(conversations)]
    thinking_conversations = await asyncio.gather(*tasks)

    # Save results
    output_path = "/home/namnp/ChatBotSynthetic/data/thinking_teacher_conversations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(thinking_conversations, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully processed {len(thinking_conversations)} conversations")
    print(f"📝 Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())