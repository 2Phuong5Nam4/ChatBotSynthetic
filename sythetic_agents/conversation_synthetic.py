from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal, List
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from pathlib import Path
import json
from dotenv import load_dotenv
import asyncio

load_dotenv(Path(__file__).parent.parent / ".env")
PROCEDURE_JSON = json.load(
    open(Path(__file__).parent.parent / "data" / "procedure.json", "r")
)

raw_conversation_path = Path(__file__).parent.parent / "data" / "raw_conversation.json"
synthetic_conversation_path = (
    Path(__file__).parent.parent / "data" / "synthetic_conversation.json"
)

with open(raw_conversation_path, "r", encoding="utf-8") as f:
    RAW_CONVERSATIONS = json.load(f)


def conversation_to_prompt(conversation: dict) -> str:
    """Return a JSON string of a conversation without the Script field."""
    cleaned = {k: v for k, v in conversation.items() if k != "Script"}
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


class SyntheticConversations(BaseModel):
    class Conversation(BaseModel):
        class Message(BaseModel):
            role: Literal["user", "assistant"]
            content: str

        procedure_id: int
        messages: List[Message]

        @classmethod
        def validate(cls, value):
            # Call pydantic's built-in validation first
            instance = super().model_validate(value)
            messages = instance.messages
            if not messages or messages[0].role != "user":
                raise ValueError(
                    "The first message in the conversation must be from the user."
                )
            return instance

    synthetic_conversations: List[Conversation]

    @classmethod
    def validate(cls, value):
        instance = super().model_validate(value)
        conversations = instance.synthetic_conversations
        if len(conversations) != 3:
            raise ValueError("Must have exactly 3 synthetic conversations")
        return instance


SYSTEM_PROMPT = f"""
# Persona
Bạn là agent chuyên tạo sinh dữ liệu cho các hội thoại giữa khách hàng và call center chatbot dựa trên đoạn hội thoại gốc và quy trình xử lý cho sẵn.

# Goals
- Tạo đúng 3 đoạn hội thoại synthetic dựa trên hội thoại gốc, trong đó 1 conversation là perfect case, 2 conversation còn lại là edge case (khách hàng hỏi ngoài lề, tỏ ra khó chịu, viết tắt, không trả lời, ...).
- Trong mỗi turn của conversation, user và agent đều có thể hỏi/trả lời nhiều hơn 1 lần.
- Các cuộc hội thoại sinh ra phải giống thực tế nhất có thể.
- Một đoạn hội thoại được sinh ra chỉ liên quan tới duy nhất 1 procedure (tức là chỉ ứng với 1 procedure_id).
- Message đầu tiên của đoạn hội thoại luôn luôn là message từ user, message cuối cùng luôn luôn là message từ agent.

# Steps
Phân tích kĩ câu hỏi của user và so sánh với tổng quan các quy trình hiện tại để xác định quy trình nào phù hợp.

# Tổng quan các quy trình hiện tại
{PROCEDURE_JSON}
"""


@dataclass
class ConversationSyntheticAgent:
    def __post_init__(self):
        self.agent = create_agent(
            model="gpt-4.1",
            response_format=SyntheticConversations,
            system_prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    async def ainvoke(self, original_conversation: str) -> SyntheticConversations:
        res = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Hãy tạo các hội thoại synthetic dựa trên hội thoại gốc sau:

{original_conversation}
""",
                    }
                ]
            }
        )

        return SyntheticConversations.model_validate(res["structured_response"])


async def main():
    print("Starting synthetic conversation generation...")
    agent = ConversationSyntheticAgent()

    for i in range(18, len(RAW_CONVERSATIONS)):
        print(f"\nProcessing conversation {i + 1}/{len(RAW_CONVERSATIONS)}")
        conversation = RAW_CONVERSATIONS[i]
        all_convs = []
        if synthetic_conversation_path.exists():
            try:
                with open(synthetic_conversation_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_convs = json.loads(content)
                print(f"Loaded {len(all_convs)} existing synthetic conversations.")
            except json.JSONDecodeError:
                # Corrupted or partially written file; start fresh but don't crash
                print(
                    "Warning: Corrupted or partially written synthetic conversation file. Starting fresh."
                )
                all_convs = []

        prompt = conversation_to_prompt(conversation)
        print("Prompt to agent:\n", prompt)
        res = await agent.ainvoke(prompt)
        all_convs.extend([conv.model_dump() for conv in res.synthetic_conversations])

        print(
            f"Writing {len(all_convs)} synthetic conversations to {synthetic_conversation_path}"
        )
        with open(synthetic_conversation_path, "w", encoding="utf-8") as f:
            json.dump(all_convs, f, ensure_ascii=False, indent=2)
    print("Synthetic conversation generation completed.")


if __name__ == "__main__":
    asyncio.run(main())
