from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Load procedure definitions
with open("/home/namnp/ChatBotSynthetic/prompts/procedure.json", "r", encoding="utf-8") as f:
    procedure = json.load(f)
with open("/home/namnp/ChatBotSynthetic/data/conversation_without_image.json", "r", encoding="utf-8") as f:
    conversations = json.load(f)[:1]


class ThinkingTeacherResponse(BaseModel):
    """Response model for Thinking Teacher"""
    reasoning: str = Field(..., description="Phân tích ngắn gọn: Bước hiện tại trong procedure, tình huống KH, hành động cần thực hiện")
    corrected_response: Optional[str] = Field(None, description="Câu trả lời đã được sửa theo procedure (nếu cần sửa). Để null nếu response gốc đã đúng")
    compliance_check: str = Field(..., description="OK nếu tuân thủ procedure, hoặc mô tả ngắn gọn vấn đề nếu không tuân thủ")
    

class ThinkingTeacher:
    """Agent that thinks step-by-step before answering"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        ).with_structured_output(ThinkingTeacherResponse)

    def _build_system_prompt(self, procedure_detail: str, category: str, sub_category: str, intentions: str) -> str:
        """Build system prompt for the Thinking Teacher"""
        prompt = f"""Bạn là một nhân viên CSKH Heineken Vietnam giàu kinh nghiệm, đã thuộc nằm lòng quy trình xử lý.

NGỮ CẢNH:
- Category: {category}
- Sub-Category: {sub_category}
- Intentions: {intentions}

QUY TRÌNH BẠN ĐÃ NẮM RÕ:
{procedure_detail}

VAI TRÒ CỦA BẠN:
Bạn đang review lại cách một đồng nghiệp mới xử lý case. Với mỗi câu trả lời của họ, bạn sẽ:
1. Nghĩ trong đầu về tình huống (như một nhân viên thật đang đọc tin nhắn KH)
2. Đánh giá xem đồng nghiệp có xử lý đúng không
3. Nếu sai, sửa lại cho họ theo cách bạn sẽ trả lời

CẦU THINKING (SỨY NGHĨ NỘI TÂM):
Viết như thể bạn đang tự nói chuyện với chính mình khi đọc tin nhắn:
- Tự nhiên, ngắn gọn, như suy nghĩ thật
- Không cần nói "Bước 1, Bước 2..." một cách cứng nhắc
- Tập trung vào: "KH đang cần gì?" → "Mình cần làm gì?" → "Có vấn đề gì không?"

VÍ DỤ THINKING TỐT (tự nhiên như người):
✅ "KH hỏi mã NV để cài app nhưng thực ra đã có tài khoản rồi. Cần hỏi thông tin để tra cứu xem đã đăng ký chưa, tránh tạo tài khoản trùng."
✅ "OK, KH quên mật khẩu. Cần hướng dẫn Quên MK đầy đủ: nhập mã → OTP → tạo MK mới 12 ký tự. Đồng nghiệp thiếu mất bước OTP."
✅ "Đơn giản, KH cảm ơn rồi. Chỉ cần hỏi thêm có cần gì không, nhắc hotline, xong."

VÍ DỤ THINKING TỆ (cứng nhắc, học thuộc):
❌ "Bước 1: Nhận diện nhu cầu và xác thực sơ bộ. Agent chưa xác định rõ nhu cầu KH và không hướng dẫn đúng quy trình đổi mật khẩu."
❌ "Bước 3: KH quên mật khẩu. Agent chưa hướng dẫn chi tiết cách đặt lại mật khẩu qua 'Quên mật khẩu'."
❌ "Trước tiên cần phân tích tình huống khách hàng đang gặp phải..."

YÊU CẦU KHI SỬA CÂU TRẢ LỜI:
- Giữ văn phong thân thiện, xưng hô tự nhiên (anh/chị/em)
- Ngắn gọn, đủ ý, không rườm rà
- Như cách bạn sẽ chat thật với KH

COMPLIANCE CHECK:
- Nếu OK → chỉ viết "OK"
- Nếu có vấn đề → viết ngắn gọn vấn đề gì (VD: "Thiếu bước OTP", "Chưa hỏi thông tin định danh")
"""
        return prompt
    async def process_conversation(self, conversation: Dict) -> Dict:
        """Process entire conversation with memory of previous turns"""
        # Get procedure and metadata
        procedure_id = str(conversation.get('procedure', '2'))  # Default to procedure 2 (forget password)
        procedure_detail = procedure.get(procedure_id, {}).get('detail_description', 'N/A')

        category = conversation.get('Category', 'N/A')
        sub_category = conversation.get('Sub_Category', 'N/A')
        intentions = conversation.get('Intentions', 'N/A')

        # Build system prompt once
        system_prompt = self._build_system_prompt(procedure_detail, category, sub_category, intentions)

        # Initialize conversation memory
        memory = [SystemMessage(content=system_prompt)]

        original_messages = conversation.get('messages', [])
        enhanced_messages = []

        # Process each turn
        i = 0
        while i < len(original_messages):
            current_msg = original_messages[i]

            # User message - just add to memory and output
            if current_msg.get('role') == 'user':
                enhanced_messages.append(current_msg)
                i += 1
                continue

            # Assistant message - need to evaluate and enhance
            if current_msg.get('role') == 'assistant':
                # Get corresponding user message (should be previous message)
                user_msg = original_messages[i-1] if i > 0 else {'role': 'user', 'content': ''}

                # Format current turn for evaluation
                current_turn = f"""LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
{self._format_memory_history(enhanced_messages)}

TURN HIỆN TẠI CẦN ĐÁNH GIÁ:
KH: {user_msg.get('content', '')}
Agent (gốc): {current_msg.get('content', '')}

Hãy đánh giá và sửa câu trả lời agent nếu cần."""

                # Add to memory and get response
                memory.append(HumanMessage(content=current_turn))
                response: ThinkingTeacherResponse = await self.llm.ainvoke(memory)

                # Determine final response (corrected or original)
                final_response = response.corrected_response if response.corrected_response else current_msg.get('content', '')

                # Create enhanced message with thinking
                enhanced_msg = {
                    'role': 'assistant',
                    'content': final_response,
                    'thinking': response.reasoning,
                    'compliance': response.compliance_check,
                    'original_content': current_msg.get('content', '') if response.corrected_response else None
                }
                enhanced_messages.append(enhanced_msg)

                # Update memory with the corrected version for next turn
                memory.append(AIMessage(content=f"[Thinking: {response.reasoning}]\n{final_response}"))

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
            teacher = ThinkingTeacher()
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