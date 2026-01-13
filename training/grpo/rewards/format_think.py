import re
from typing import Any, List, Optional
# Required fields trong thinking block




THINKING_FIELD_PATTERNS = {
    "Tình huống": r"Tình huống:[ \t]*([^\n]*)",
    "Quy trình": r"Quy trình:[ \t]*([^\n]*)",
    "Bước": r"Bước:[ \t]*([^\n]*)",
    "Thông tin có": r"Thông tin có:[ \t]*([^\n]*)",
    "Thông tin cần thêm": r"Thông tin cần thêm:[ \t]*([^\n]*)",
    "Hành động": r"Hành động:[ \t]*([^\n]*)"
}

from collections import Counter

def calculate_token_ngram_similarity(candidate: str, reference: str, tokenizer: Optional[Any] = None, n: int = 2) -> float:
    """
    Tính độ tương đồng N-gram dựa trên Token IDs của chính Model.
    
    Args:
        candidate: Câu trả lời của AI.
        reference: Ground truth.
        tokenizer: Tokenizer của model (đã load từ transformers).
        n: Kích thước n-gram (2 = bigram).
    """
    if not candidate or not reference:
        return 0.0

    # 1. Chuyển text thành list các Token IDs (số nguyên)
    # add_special_tokens=False để tránh so sánh các token như <|im_start|>, <s>
    if tokenizer:
        cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
        ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    else:
        cand_ids = candidate.split()
        ref_ids = reference.split()

    if len(cand_ids) < n or len(ref_ids) < n:
        return 0.0

    # 2. Tạo n-grams từ các IDs (Thay vì từ string)
    # Ví dụ: [101, 204, 305] -> [(101, 204), (204, 305)]
    cand_ngrams = [tuple(cand_ids[i:i+n]) for i in range(len(cand_ids)-n+1)]
    ref_ngrams = [tuple(ref_ids[i:i+n]) for i in range(len(ref_ids)-n+1)]

    # 3. Đếm và tính toán (Giống logic cũ)
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    overlap = sum((cand_counts & ref_counts).values())
    total = sum(cand_counts.values()) + sum(ref_counts.values())

    if total == 0:
        return 0.0

    return 2.0 * overlap / total


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
    # Cho phép "ngoại lệ" xuất hiện trong mô tả (ví dụ: "3 - Xử lý (gặp ngoại lệ...)")
    buoc_so_pattern = r"^\d+(?:\s*,\s*\d+)*\s*-\s*.+$"
    if re.match(buoc_so_pattern, content):
        return True
    
    return False

def extract_buoc_steps(buoc_content: str) -> Optional[str]:
    """
    Extract phần bước từ content: số bước hoặc 'ngoại lệ' hoặc None nếu bỏ trống.
    """
    content = buoc_content.strip()
    
    if not content:
        return None
    
    # Format: "ngoại lệ - [mô tả]"
    if content.lower().startswith("ngoại lệ"):
        return "ngoại lệ"
    
    # Format: "1, 2, 3 - [mô tả]"
    match = re.match(r"^(\d+(?:\s*,\s*\d+)*)\s*-", content)
    if match:
        # Chuẩn hóa: loại bỏ space, sort số
        steps = [int(x.strip()) for x in match.group(1).split(",")]
        return ",".join(map(str, sorted(steps)))
    
    return None

def evaluate_thinking_content(think_content: str, ground_truth: str, tokenizer: Optional[Any] = None) -> float:
    # thưởng think tag
    score = 0.1
    for field, pattern in THINKING_FIELD_PATTERNS.items():
        field_content = re.search(pattern, think_content, re.DOTALL)
        ground_truth_content = re.search(pattern, ground_truth, re.DOTALL).group(1).strip()
        if field_content:
            # có field + 0.05, 6 filed tổng 0.3
            score += 0.05
            field_content_str = field_content.group(1).strip()
            match field:
                case "Quy trình":
                    if check_quy_trinh_valid(field_content_str):
                        score += 0.1
                case "Bước":
                    if check_buoc_valid(field_content_str):
                        score += 0.05
                        
                        # Check đúng bước: so sánh số bước hoặc "ngoại lệ"
                        gen_steps = extract_buoc_steps(field_content_str)
                        gt_steps = extract_buoc_steps(ground_truth_content)
                        if gen_steps == gt_steps:
                            score += 0.05  # Bonus cho đúng bước
                        
                case _:
                    # Both empty = perfect match, give full score
                    if not field_content_str and not ground_truth_content:
                        score += 0.1

                    elif len(field_content_str.split()) < 2 and len(ground_truth_content.split()) < 2:
                        # both too short, skip similarity check
                        if field_content_str == ground_truth_content:
                            score += 0.1
                    else:
                        sim = calculate_token_ngram_similarity(field_content_str, ground_truth_content, n=2, tokenizer=tokenizer)
                        score += sim * 0.1
                    
    # max 0.15 + 0.30 = 0.45
    return score

def format_thinking_reward(
    prompts: List[List[dict]],
    completions: List[List[dict]],
    answer: List[str],
    tokenizer: Optional[Any] = None,
    **kwargs,
) -> List[float]:
    """
    Reward function cho việc validate format thinking.
    Scale: 0.0 - 1.0
    """
    rewards = []
    
    for completion, ground_truth in zip(completions, answer):
        content = completion[0].get("content", "")
        score = 0.0
        
        # 1. Check có <think> tag không (0.15 điểm)
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        # remove think tag in ground truth
        ground_truth = re.search(r"<think>(.*?)</think>", ground_truth, re.DOTALL).group(1).strip()
        if think_match is None:
            rewards.append(score)
            continue  # No think tag, reward 0.0
        think_block = think_match.group(1).strip()
        score += evaluate_thinking_content(think_block, ground_truth, tokenizer=tokenizer)
        rewards.append(score) 
    
    return rewards
