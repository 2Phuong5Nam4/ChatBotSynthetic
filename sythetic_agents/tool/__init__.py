from typing import Literal
from langchain.tools import tool
import json
import logging
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "data"
logger = logging.getLogger(__name__)
PROCEDURE_JSON = json.load(open(data_path / "procedure.json", "r"))


@tool(description="Create a list of tasks based on the current conversation context.")
async def read_procedure_details(procedure_id: Literal["1", "2", "3", "4", "5"]) -> str:
    logger.info(f"_read_procedure_details: {procedure_id}")
    return PROCEDURE_JSON[procedure_id]["detail_description"]


@tool(description="Read example conversation of a procedure.")
async def read_example_conversation(
    procedure_id: Literal["1", "2", "3", "4", "5"],
    example_conversation_id: Literal["1", "2", "3", "4", "5"],
) -> str:
    logger.info(
        f"_read_example_conversation: {procedure_id}, {example_conversation_id}"
    )
    return PROCEDURE_JSON[procedure_id]["example_conversation"][
        example_conversation_id
    ]["content"]
