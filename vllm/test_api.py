from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=""
)

# Định nghĩa tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "tra_cuu_thong_tin",
            "description": "Tra cứu thông tin cửa hàng theo số điện thoại, mã cửa hàng hoặc mật khẩu",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "Số điện thoại đăng ký"
                    },
                    "ma_cua_hang": {
                        "type": "string",
                        "description": "Mã cửa hàng (outlet code)"
                    },
                    "mat_khau": {
                        "type": "string",
                        "description": "Mật khẩu hiện tại (nếu có)"
                    }
                },
                "required": []
            }
        }
    }
]

resp = client.chat.completions.create(
    model="grpo_merged_model",
    messages=[
        {"role": "system", "content": "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."},
        {"role": "user", "content": "Alo"},
        {"role": "assistant", "content": "Dạ em chào anh/chị ạ. Anh/chị cần Heineken hỗ trợ vấn đề gì, để em kiểm tra giúp ạ?"},
        {"role": "user", "content": "tôi quên mật khẩu rồi"},
        {"role": "assistant", "content": "Dạ anh/chị vui lòng cung cấp số điện thoại hoặc mã cửa hàng đã đăng ký tài khoản để em kiểm tra và hỗ trợ đặt lại mật khẩu cho anh/chị ạ."},
        {"role": "user", "content": "số điện thoại của tôi là 0909123456"},
    ],
    tools=tools,  # ← THÊM DÒNG NÀY
    tool_choice="auto",  # hoặc "required" nếu bắt buộc phải gọi tool
    max_tokens=500
)

print(resp)
print("\n=== Tool Calls ===")
if resp.choices[0].message.tool_calls:
    for tool_call in resp.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")