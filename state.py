from typing import List
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    memory: str
    context: List[str]
    answer: str