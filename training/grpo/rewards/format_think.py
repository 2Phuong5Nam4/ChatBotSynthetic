import re
from typing import List
# Required fields trong thinking block




THINKING_FIELD_PATTERNS = {
    "Tình huống": r"Tình huống:[ \t]*([^\n]*)",
    "Quy trình": r"Quy trình:[ \t]*([^\n]*)",
    "Bước": r"Bước:[ \t]*([^\n]*)",
    "Thông tin có": r"Thông tin có:[ \t]*([^\n]*)",
    "Thông tin cần thêm": r"Thông tin cần thêm:[ \t]*([^\n]*)",
    "Hành động": r"Hành động:[ \t]*([^\n]*)"
}

def check_quy_trinh_valid(quy_trinh_content: str) -> bool:
    # Regex patterns để extract nội dung của từng field
    quy_trinh_list = ["Hỗ trợ đăng nhập", "Quên/Đổi mật khẩu", "Kiểm tra đơn hàng", "Kiểm tra MQH Outlet-NPP/SubD", "Hướng dẫn xử lý đơn hàng SEM", "không liên quan", "không xác định"]
    # Regex pattern để check quy trình hợp lệ (một trong các quy trình trong list)
    pattern = re.compile(r"(" + "|".join(re.escape(quy_trinh) for quy_trinh in quy_trinh_list) + r")")
    return bool(pattern.search(quy_trinh_content.strip()))

def check_buoc_valid(buoc_content: str) -> bool:
    """
    Validate format của Bước:
    - Format 1: "1, 2, 3, ... - [mô tả bước]" (các số bước + mô tả)
    - Format 2: "ngoại lệ - [mô tả ngoại lệ]"
    - Format 3: bỏ trống nếu không xác định/không liên quan
    - Không được trộn số bước với ngoại lệ
    """
    content = buoc_content.strip()
    
    # Format 3: Bỏ trống hoặc không xác định/không liên quan
    if not content:
        return True
    
    # Format 2: ngoại lệ - [mô tả ngoại lệ]
    ngoai_le_pattern = r"^ngoại lệ\s*-\s*.+$"
    if re.match(ngoai_le_pattern, content, re.IGNORECASE):
        # Đảm bảo không chứa số bước
        if not re.search(r"\b\d+\s*,", content):
            return True
        return False
    
    # Format 1: 1, 2, 3, ... - [mô tả bước]
    # Pattern: một hoặc nhiều số ngăn cách bởi dấu phẩy, theo sau là " - " và mô tả
    buoc_so_pattern = r"^\d+(?:\s*,\s*\d+)*\s*-\s*.+$"
    if re.match(buoc_so_pattern, content):
        # Đảm bảo không chứa "ngoại lệ"
        if "ngoại lệ" not in content.lower():
            return True
        return False
    
    return False

def evaluate_thinking_content(think_content: str) -> float:
    score = 0.15
    for field, pattern in THINKING_FIELD_PATTERNS.items():
        field_content = re.search(pattern, think_content, re.DOTALL)
        if field_content:
            score += 0.05
            field_content_str = field_content.group(1).strip()
            match field:
                case "Quy trình":
                    if check_quy_trinh_valid(field_content_str):
                        score += 0.05
                    
                case "Bước":
                    if check_buoc_valid(field_content_str):
                        score += 0.05
                    else:
                        print("Invalid Bước format:", field_content_str)
                
    # check Quy trình field
    
            
    # max 0.15 + 0.30 = 0.45
    return score +0.45

def format_thinking_reward(completions: List[List[dict]], **kwargs) -> List[float]:
    """
    Reward function cho việc validate format thinking.
    Scale: 0.0 - 1.0
    """
    rewards = []
    
    for completion in completions:
        content = completion[0].get("content", "")
        score = 0.0
        
        # 1. Check có <think> tag không (0.15 điểm)
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        
        if think_match is None:
            rewards.append(score)
            continue  # No think tag, reward 0.0
        think_block = think_match.group(1).strip()
        score += evaluate_thinking_content(think_block)
        rewards.append(score)  # Cap at 1.0
    
    return rewards
