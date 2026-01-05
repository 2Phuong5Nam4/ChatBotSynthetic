from typing import Callable, TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class State:
    raw_conversation: Dict # List of raw conversation data
    procedures : Dict[int, Dict]  # Mapping of procedure IDs to their descriptions
    tools: List[Any]  # List of tools available in the pipeline
    procedure_id: int  # Classified procedure ID (added after classification)
    call_tool: Callable[..., Any]  # Function to call tools (if needed in future steps)