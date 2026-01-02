from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal, List
from dataclasses import dataclass
from tool import read_procedure_details, read_example_conversation
from langchain.agents import create_agent
from langchain.messages import SystemMessage
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

raw_conversation_path = Path(__file__).parent.parent / "data" / "raw_conversation.json"
RAW_CONVERSATIONS = [
    conversation["messages"]
    for conversation in json.load(open(raw_conversation_path, "r"))
]


class SyntheticConversations(BaseModel):
    class Conversation(BaseModel):
        class Message(BaseModel):
            role: Literal["user", "assistant"]
            content: str

        messages: List[Message]

    synthetic_conversations: List[Conversation]

    @classmethod
    def validate(cls, value):
        instance = super().model_validate(value)
        conversations = instance.synthetic_conversations
        if len(conversations) != 3:
            raise ValueError("Must have exactly 3 synthetic conversations")
        return instance


SYSTEM_PROMPT = """
# Persona
Bạn là agent chuyên tạo sinh dữ liệu cho các hội thoại giữa khách hàng và call center chatbot dựa trên đoạn hội thoại gốc.

# Goals
- Tạo đúng 3 đoạn hội thoại synthetic dựa trên hội thoại gốc, trong đó 1 conversation là perfect case, 2 conversation còn lại là edge case (khách hàng hỏi ngoài lề, tỏ ra khó chịu, viết tắt, không trả lời, ...).
- Trong mỗi conversation, user và agent đều có thể hỏi/trả lời nhiều lần.

# Steps
1. Phân tích kĩ từng câu hỏi của user và so sánh với tổng quan các quy trình hiện tại để xác định quy trình nào phù hợp.
2. Gọi tool read_procedure_details để đọc chi tiết quy trình cần thực hiện bất cứ khi nào cảm thấy cần thiết.
3. Gọi tool read_example_conversation để đọc ví dụ hội thoại của quy trình cần thực hiện (nếu cần thiết) nếu như thế câu hỏi của User có khả năng rơi vào các mẫu hội thoại được đề cập trong quy trình.

# Tool Call
Có thể call read_procedure_details và read_example_conversation nhiều lần hoặc cùng lúc để đọc chi tiết nhiều quy trình.

# Tổng quan các quy trình hiện tại
**Quy trình 1**
Quy trình L1 hỗ trợ Đăng nhập ứng dụng HVN Đặt Hàng: nhận diện các tình huống “không đăng nhập được”, “mật khẩu là gì”, “SĐT/mã không hợp lệ”, “chưa đăng ký app”, “cửa hàng bị đóng”, “cập nhật SĐT”. Kiểm tra/đối soát dữ liệu bắt buộc: Mã cửa hàng (OutletID), SĐT đã đăng ký, tên cửa hàng trên hệ thống, trạng thái cửa hàng. Hướng dẫn đúng phương thức đăng nhập (mã + mật khẩu, hoặc bằng SĐT), xử lý quên mật khẩu, chuẩn hóa SĐT (thêm số 0 đầu), nhận diện lỗi hệ thống phổ biến, và điều phối lên Sale/19001845 khi cần (cửa hàng đóng/chưa đăng ký/cập nhật SĐT).

**Quy trình 2**
Quy trình hỗ trợ KH không đăng nhập được/đổi mật khẩu HVN: nhận diện khi KH báo quên mật khẩu, nhập sai nhiều lần bị khóa, không nhớ tên đăng nhập (OutletID/NPP ID), cần OTP, muốn đăng nhập bằng SĐT. Kiểm tra dữ liệu: OutletID/NPP ID, tên cửa hàng/NPP, SĐT đăng ký, ảnh lỗi.

**Quy trình 3**
 Quy trình: Check đơn hàng & MQH NPP/SubD (SEM/HVN/DIS Lite) + Gratis. Nhận diện khi: (1) Đơn không về/không hiển thị ở NPP/SubD; (2) SubD không thấy đơn trên DIS Lite; (3) Yêu cầu gắn MQH/outlet–NPP; (4) Kiểm tra trạng thái đơn Gratis. Keyword: mã đơn hàng, OutletID, mã NPP/SubD, kênh đặt (SEM/HVN/DIS Lite), MQH, trạng thái (pending/delivered/approved), gratis chỉ qua NPP, log L2 push, route Team SEM/HVN, Zalo B2B/DIS Lite link, Sale Admin.

**Quy trình 4**
Quy trình check mối quan hệ Outlet – NPP/SubD trên SEM/DB. Áp dụng khi: cần xác minh outlet đang gán với NPP/SubD nào; nghi ngờ sai/mất mối quan hệ; SA báo đã gắn nhưng SEM vẫn hiển thị NPP/SubD khác; user tự gắn mqh; mqh Inactive; đặt đơn nhưng NPP không nhận; báo cáo có mqh nhưng SEM không có. Từ khóa nhận diện: OutletID, NPP/SubD, SEM, mqh Active/Inactive, SA, 24h đồng bộ, "tự tạo mqh".

**Quy trình 5**
 Quy trình hỗ trợ thao tác Order/Gratis trên SEM cho Sales/Sup/ASM. Nhận biết: yêu cầu hướng dẫn xuất đơn Gratis; xác nhận thời điểm ASM approve/raise Gratis (quy tắc 4 giờ); lỗi không hiển thị Sub/NPP trong Outlet Order; không thấy SKU trong bộ lọc nhưng có khi tìm tên; cần Force Sync đồng bộ dữ liệu. Dữ liệu cần thu: OutletID (mã outlet), NPP/SubD (mã code + mã ID), mã SKU, vai trò người dùng (SR/Sup/ASM), tình trạng thị trường (SR TBA/Sup cover), thời điểm thông báo hệ thống, ảnh màn hình.
"""


@dataclass
class ConversationSyntheticAgent:
    def __post_init__(self):
        self.agent = create_agent(
            model="gpt-4.1",
            tools=[read_procedure_details, read_example_conversation],
            response_format=SyntheticConversations,
            system_prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def invoke(self, original_conversation: str) -> SyntheticConversations:
        res = self.agent.invoke(
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


if __name__ == "__main__":
    agent = ConversationSyntheticAgent()
    conversation_1 = RAW_CONVERSATIONS[0]
    res = agent.invoke(conversation_1)
    print(f"output: {res}")
