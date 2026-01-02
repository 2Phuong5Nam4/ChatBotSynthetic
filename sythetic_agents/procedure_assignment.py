from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
# Load procedure definitions
procedure = json.load(open("/home/namnp/ChatBotSynthetic/prompts/procedure.json", "r"))

class ProcedureClassification(BaseModel):
    """Classification result for a conversation"""
    reasoning: str = Field(..., description="Explanation for the classification")
    procedure_id: int = Field(..., description="Procedure ID (1-5)", ge=1, le=5)
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

class ConversationClassifier:
    """Classifier to assign conversations to procedures 1-5"""
    
    def __init__(self, model_name: str = "gpt-5.2", temperature: float = 0.1):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        ).with_structured_output(ProcedureClassification)
        
    def _build_system_prompt(self) -> str:
        """Build system prompt with all procedure descriptions"""
        prompt = """Bạn là chuyên gia phân loại hội thoại khách hàng vào các quy trình xử lý (procedure) của dịch vụ khách hàng Heineken.

Nhiệm vụ: Phân tích hội thoại và xác định procedure phù hợp nhất từ 1-5.

CÁC PROCEDURE:

"""
        for proc_id, proc_data in procedure.items():
            prompt += f"\n{'='*80}\n"
            prompt += f"PROCEDURE {proc_id}:\n"
            prompt += f"{proc_data['detail_description']}...\n"  # Truncate for token limit
            
        prompt += """

YÊU CẦU PHÂN LOẠI:
1. Đọc kỹ toàn bộ hội thoại
2. Xác định vấn đề chính của khách hàng
3. So khớp với mô tả các procedure
4. Trả về procedure_id phù hợp nhất (1-5)
5. Đưa ra confidence score (0-1)
6. Giải thích ngắn gọn lý do chọn procedure đó

CHÚ Ý:
- Procedure 1: Đăng nhập app HVN
- Procedure 2: Quên/đổi mật khẩu
- Procedure 3: Kiểm tra đơn hàng và mối quan hệ NPP/SubD
- Procedure 4: Check mối quan hệ Outlet-NPP/SubD
- Procedure 5: Hướng dẫn đặt hàng trên SEM
"""
        return prompt
    
    def _format_conversation(self, conversation: Dict) -> str:
        """Format conversation messages into readable text"""
        formatted = f"Category: {conversation.get('Category', 'N/A')}\n"
        formatted += f"Sub_Category: {conversation.get('Sub_Category', 'N/A')}\n"
        formatted += f"Intentions: {conversation.get('Intentions', 'N/A')}\n"
        formatted += f"Solutions: {conversation.get('Solutions', 'N/A')}\n\n"
        formatted += "Hội thoại:\n"
        
        for msg in conversation.get('messages', []):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted += f"{role.upper()}: {content}\n"
        
        return formatted
    
    def classify(self, conversation: Dict) -> ProcedureClassification:
        """Classify a single conversation"""
        system_prompt = self._build_system_prompt()
        user_prompt = self._format_conversation(conversation)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        result = self.llm.invoke(messages)
        return result
    


def main():
    """Example usage"""
    # Load conversations
    with open("/home/namnp/ChatBotSynthetic/data/raw_conversations.json", "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    # Initialize classifier
    classifier = ConversationClassifier()
    
    # Classify first 5 conversations as example
    print("Classifying conversations...")
    results = []
    
    for conv in conversations:
        classification = classifier.classify(conv)
        conv['procedure'] = classification.procedure_id
    
    with open("/home/namnp/ChatBotSynthetic/data/classified_conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to classified_conversations.json")


if __name__ == "__main__":
    main()