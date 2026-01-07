import json
from synthetic_pipeline.n0_classify_conversation import classify_conversation
from synthetic_pipeline.n1_refine_conversation import refine_conversation
from synthetic_pipeline.n2_synthetic_conversations import synthetic_conversations
from synthetic_pipeline.state import State
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
load_dotenv()
extracted_procedures = json.load(open("data/extracted_procedure.json", "r", encoding="utf-8"))
tools = extracted_procedures.get("tools", [])
procedures = {k: v for k, v in extracted_procedures.items() if k != "tools"}
raw_conversations = json.load(open("data/raw_conversations.json", "r", encoding="utf-8"))[2:]


graph_builder = StateGraph(State)
graph_builder.add_node("classify_conversation",classify_conversation)
graph_builder.add_edge(START, "classify_conversation")
graph_builder.add_node("refine_conversation",refine_conversation)
graph_builder.add_edge("classify_conversation", "refine_conversation")
graph_builder.add_node("synthetic_conversations",synthetic_conversations)
graph_builder.add_edge("refine_conversation", "synthetic_conversations")
graph_builder.add_edge("synthetic_conversations", END)
graph = graph_builder.compile()
def append_jsonl(filepath: str, data: dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_processed_keys(filepath: str) -> set:
    """Đọc file jsonl và lấy danh sách (Sub_Category, Script_num) đã xử lý"""
    processed = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    key = (data.get("Sub_Category"), data.get("Script_num"))
                    processed.add(key)
    except FileNotFoundError:
        pass
    return processed

def make_key(conv: dict) -> tuple:
    return (conv.get("Sub_Category"), conv.get("Script_num"))

async def process_single(sem, idx: int, raw_conv: dict, total: int):
    async with sem:
        print(f"Processing conversation {idx+1}/{total}")
        state: State = State(
            procedure_id=0,
            raw_conversation=raw_conv,
            procedures=procedures,
            tools=tools,
            call_tool=lambda *args, **kwargs: None
        )
        try:
            result_state = await graph.ainvoke(state)
            return {"success": True, "result": result_state, "raw_conv": raw_conv}
        except Exception as e:
            print(f"[ERROR] Conversation {idx+1} failed: {e}")
            return {"success": False, "error": str(e), "raw_conv": raw_conv}

async def run_pipeline():
    import asyncio
    sem = asyncio.Semaphore(4)

    # Lấy danh sách đã xử lý
    processed = get_processed_keys("data/refined_conversations.jsonl")
    pending = [(idx, conv) for idx, conv in enumerate(raw_conversations) if make_key(conv) not in processed]
    print(f"Đã xử lý: {len(processed)}, còn lại: {len(pending)}")

    if not pending:
        print("Tất cả đã xử lý xong!")
        return

    tasks = [
        process_single(sem, idx, raw_conv, len(raw_conversations))
        for idx, raw_conv in pending
    ]

    for coro in asyncio.as_completed(tasks):
        output = await coro

        if not output["success"]:
            append_jsonl("data/failed_conversations.jsonl", {
                "Sub_Category": output["raw_conv"].get("Sub_Category"),
                "Script_num": output["raw_conv"].get("Script_num"),
                "error": output["error"]
            })
            continue

        result_state = output["result"]
        raw_conv = output["raw_conv"]

        raw_conv["messages"] = result_state["refined_messages"]
        raw_conv["procedure_id"] = result_state["procedure_id"]
        append_jsonl("data/refined_conversations.jsonl", raw_conv)

        synthetic_conversations = result_state["synthetic_conversations"]
        for conv in synthetic_conversations:
            new_conv = {}
            new_conv["procedure_id"] = result_state["procedure_id"]
            new_conv["messages"] = conv
            append_jsonl("data/synthetic_conversations.jsonl", new_conv)
                                    
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_pipeline())