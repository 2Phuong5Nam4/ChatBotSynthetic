from synthetic_pipeline.state import State
from typing import Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from enum import Enum
import json
class ProcedureID(str, Enum):
    PROCEDURE_1 = "1"
    PROCEDURE_2 = "2"
    PROCEDURE_3 = "3"
    PROCEDURE_4 = "4"
    PROCEDURE_5 = "5"

class ProcedureClassification(BaseModel):
    procedure_id: ProcedureID = Field(..., description="Procedure ID (1-5)")

async def classify_conversation(state: State) -> Dict:
    # phân loại hội thoại theo procedure
    raw_conversation = state.raw_conversation
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1
        ).with_structured_output(ProcedureClassification)
    procedure = state.procedures
    prompt = f"""Bạn là chuyên gia phân loại hội thoại khách hàng vào các quy trình xử lý (procedure) của dịch vụ khách hàng Heineken.

Nhiệm vụ: Phân tích hội thoại và xác định procedure phù hợp nhất từ 1-5.

CÁC PROCEDURE:

"""
    for proc_id, proc_data in procedure.items():
        prompt += f"\n{'='*80}\n"
        prompt += f"## PROCEDURE {proc_id}:\n"
        prompt += f"- Tên procedure: {proc_data['tên']}\n- Mục tiêu của procedure: {proc_data['mục_tiêu']}\n"
        
    prompt += """
YÊU CẦU PHÂN LOẠI:
1. Đọc kỹ toàn bộ hội thoại
2. Xác định vấn đề chính của khách hàng
3. So khớp với mô tả các procedure
4. Trả về procedure_id phù hợp nhất (1-5)
"""
    system_message = SystemMessage(content=prompt)
    formatted_conversation = f"""
Hội thoại:
Loại hội thoại: {raw_conversation.get('Sub_Category', 'unknown')}
Đối tượng khách hàng: {raw_conversation.get('Targeted_Customers', 'unknown')}
Ý định: {raw_conversation.get('Intentions', 'unknown')}
Giải pháp đã cung cấp: {raw_conversation.get('Solutions', 'unknown')}
"""
    for msg in raw_conversation.get('messages', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_conversation += f"{role.capitalize()}: {content}\n"
    human_message = HumanMessage(content=formatted_conversation)
    response = await llm.ainvoke([system_message, human_message])
    procedure_id = response.procedure_id
    state.procedure_id = procedure_id
    # save raw_conversation with procedure_id  to classified_raw_conversations.json (by appending to the file if exists)
    classified_data = {**raw_conversation, "procedure_id": procedure_id}
    with open("data/classified_raw_conversations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(classified_data, ensure_ascii=False) + "\n")
    return {**state.__dict__}