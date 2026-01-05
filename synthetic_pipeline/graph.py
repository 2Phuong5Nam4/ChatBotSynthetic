import json
from synthetic_pipeline.n0_classify_conversation import classify_conversation
from synthetic_pipeline.n1_refine_conversation import refine_conversation
from synthetic_pipeline.state import State
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
load_dotenv()
extracted_procedures = json.load(open("data/extracted_procedure.json", "r", encoding="utf-8"))
tools = extracted_procedures.get("tools", [])
procedures = {k: v for k, v in extracted_procedures.items() if k != "tools"}
raw_conversations = json.load(open("data/raw_conversations.json", "r", encoding="utf-8"))[:1]


graph_builder = StateGraph(State)
graph_builder.add_node("classify_conversation",classify_conversation)
graph_builder.add_edge(START, "classify_conversation")
graph_builder.add_node("refine_conversation",refine_conversation)
graph_builder.add_edge("classify_conversation", "refine_conversation")
graph_builder.add_edge("refine_conversation", END)

graph = graph_builder.compile()
async def run_pipeline():
    for idx, raw_conv in enumerate(raw_conversations):
        print(f"Processing conversation {idx+1}/{len(raw_conversations)}")
        state: State = State(
            procedure_id=0,
            raw_conversation=raw_conv,
            procedures=procedures,
            tools=tools,
            call_tool=lambda *args, **kwargs: None
        )
        result_state = await graph.ainvoke(state)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_pipeline())