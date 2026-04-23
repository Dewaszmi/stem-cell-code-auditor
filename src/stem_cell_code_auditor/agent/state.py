import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class StemState(TypedDict):
    repo_name: str
    messages: Annotated[List[BaseMessage], operator.add]
    specialization: str
    reasoning: str
