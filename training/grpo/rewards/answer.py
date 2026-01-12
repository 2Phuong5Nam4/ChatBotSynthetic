import json
import re
from typing import List

from rouge_score import rouge_scorer

# Load ROUGE scorer once at module level for efficiency
_rouge_scorer = None

def get_rouge_scorer() -> rouge_scorer.RougeScorer:
    """Lazy load the ROUGE scorer."""
    global _rouge_scorer
    if _rouge_scorer is None:
        # Use ROUGE-L for longest common subsequence matching
        _rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    return _rouge_scorer
# Tool schema
AVAILABLE_TOOLS = {
    "tra_cuu_thong_tin": {
        "params": ["ma_cua_hang", "sdt", "ma_npp"],
        "required": []  # At least 1 param required
    },
    "kiem_tra_mqh": {
        "params": ["outlet_id", "npp_subd_id"],
        "required": ["outlet_id"]
    },
    "kiem_tra_don_hang": {
        "params": ["ma_don_hang", "kenh"],
        "required": ["ma_don_hang", "kenh"]
    },
    "tao_ticket": {
        "params": ["team", "noi_dung", "du_lieu"],
        "required": ["team", "noi_dung", "du_lieu"]
    },
    "force_sync": {
        "params": ["outlet_id", "npp_subd_id"],
        "required": ["outlet_id"]
    },
    "gui_huong_dan": {
        "params": ["loai_huong_dan"],
        "required": ["loai_huong_dan"]
    }
}
def parse_tool_call(content: str) -> dict | None:
    """Parse tool call từ content, return None nếu không parse được."""
    # Tìm <tool_call> hoặc <tool_calls>
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            # Handle escaped quotes
            json_str = json_str.replace('\\"', '"').strip('"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": match.group(1)}
    return None


def check_tool_call(
    response: str,
    ground_truth: str
) -> float:
    """Check tool call trong response so với ground truth."""
    tool_call = parse_tool_call(response)
    if tool_call is None:
        return 0.0 # No tool call found
    
    if tool_call.get("_parse_error"):
        return 0.2 # Parse error
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    
    # Handle arguments as string (some formats)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            return 0.3 # Argument parse error
    
    # Check tool exists
    if name not in AVAILABLE_TOOLS:
        return 0.3 # Unknown tool
    
    schema = AVAILABLE_TOOLS[name]
    
    # Check required params
    for req in schema["required"]:
        if req not in args:
            return 0.5 # Missing required param
    
    # Check no unknown params
    for param in args:
        if param not in schema["params"]:
            return 0.6 # Unknown param
            return 0.6 # Unknown param
        
    return 1.0 # Fully valid

def check_answer(
    response: str,
    ground_truth: str
) -> float:
    """Check similarity using ROUGE-L score."""
    if not response or not ground_truth:
        return 0.0

    scorer = get_rouge_scorer()

    # ROUGE-L returns precision, recall, fmeasure
    scores = scorer.score(ground_truth, response)
    
    # Return F1 score (fmeasure) - already in 0-1 range
    score = scores['rougeL'].fmeasure
    return score

def process_single_example(
    response: str,
    ground_truth: str
) -> float:
    if "<tool_call>" in ground_truth:
        tool_score = check_tool_call(response, ground_truth)
        return tool_score
    else:
        
        answer_score = check_answer(response, ground_truth)
        return answer_score

def answer_reward(
    prompts: List[List[dict]], 
    completions: List[List[dict]], 
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    Reward function cho việc validate answer/tool call.
    Scale: 0.0 - 1.0
    """
    rewards = []
    
    # split anwser into think and answer part
    for response, gt in zip(completions, answer):
        # Lấy phần content của assistant
        reponse = response[-1]["content"]
        if "</think>" not in reponse:
            # No think part, return 0.0
            rewards.append(0.0)
        else:
            score = process_single_example(
                response=response[-1]["content"].split("</think>")[-1].strip(),
                ground_truth=gt.split("</think>")[-1].strip()
            )
            rewards.append(score)
    
    return rewards