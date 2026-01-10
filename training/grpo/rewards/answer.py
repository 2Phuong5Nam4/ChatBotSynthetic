import json
import re
from typing import List
from difflib import SequenceMatcher
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
    patterns = [
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<tool_calls>\s*(.*?)\s*</tool_calls>",
        r"<tool_call>\s*(.*?)$",  # Unclosed tag
    ]
    
    for pattern in patterns:
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
def validate_tool_call(tool_call: dict) -> tuple[float, str]:
    """Validate tool call, return (score, reason)."""
    if tool_call is None:
        return 0.0, "no_tool_call"
    
    if tool_call.get("_parse_error"):
        return 0.2, "invalid_json"  # Partial credit for trying
    
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    
    # Handle arguments as string (some formats)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            return 0.3, "invalid_args_format"
    
    # Check tool exists
    if name not in AVAILABLE_TOOLS:
        return 0.3, "unknown_tool"
    
    schema = AVAILABLE_TOOLS[name]
    
    # Check required params
    for req in schema["required"]:
        if req not in args:
            return 0.5, f"missing_required_{req}"
    
    # Check no unknown params
    for param in args:
        if param not in schema["params"]:
            return 0.6, f"unknown_param_{param}"
    
    # Fully valid
    return 1.0, "valid"
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
    
    for i, completion in enumerate(completions):
        content = completion[0].get("content", "")
        ground_truth = answer[i] if i < len(answer) else ""
        
        score = 0.0
        
        # Parse both completion and ground truth
        completion_tool = parse_tool_call(content)
        gt_tool = parse_tool_call(ground_truth)
        
        # Case 1: Ground truth expects tool call
        if gt_tool is not None:
            if completion_tool is not None:
                # Validate tool call format (0.4 điểm)
                format_score, reason = validate_tool_call(completion_tool)
                score += format_score * 0.4
                
                # Compare with ground truth tool name (0.3 điểm)
                if completion_tool.get("name") == gt_tool.get("name"):
                    score += 0.3
                
                # Compare arguments (0.3 điểm)
                comp_args = completion_tool.get("arguments", {})
                gt_args = gt_tool.get("arguments", {})
                
                if isinstance(comp_args, str):
                    try: comp_args = json.loads(comp_args)
                    except: comp_args = {}
                if isinstance(gt_args, str):
                    try: gt_args = json.loads(gt_args)
                    except: gt_args = {}
                
                # Check key overlap
                if comp_args and gt_args:
                    common_keys = set(comp_args.keys()) & set(gt_args.keys())
                    matching_values = sum(
                        1 for k in common_keys 
                        if str(comp_args.get(k)) == str(gt_args.get(k))
                    )
                    if common_keys:
                        score += 0.3 * (matching_values / len(gt_args))
            else:
                # Ground truth expects tool but completion doesn't have one
                score = 0.1
        
        # Case 2: Ground truth expects text response (no tool)
        else:
            if completion_tool is None:
                # Extract response content (after </think>)
                response_match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
                if response_match:
                    response = response_match.group(1).strip()
                    gt_response_match = re.search(r"</think>\s*(.*)", ground_truth, re.DOTALL)
                    gt_response = gt_response_match.group(1).strip() if gt_response_match else ground_truth
                    
                    # Semantic similarity (simple: SequenceMatcher)
                    similarity = SequenceMatcher(None, response.lower(), gt_response.lower()).ratio()
                    score = similarity
                else:
                    score = 0.2
            else:
                # Expected text but got tool call
                score = 0.1
        
        rewards.append(min(score, 1.0))
    
    return rewards