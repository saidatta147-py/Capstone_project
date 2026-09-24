"""FastAPI application with a three-node LangGraph policy router."""
from __future__ import annotations
import json, os
from typing import Literal, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from ingest import retrieve
from prompt import STRUCTURED_PROMPT

KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")
MOCK = os.getenv("MOCK_LLM", "1") != "0"
class AskRequest(BaseModel): query: str = Field(min_length=1)
class AnswerResponse(BaseModel): answer: str; sources: list[str]; confidence: float = Field(ge=0, le=1)
class State(TypedDict, total=False): query: str; intent: Literal["policy_question","general_question"]; chunks: list[dict]; response: AnswerResponse

def optional_llm(prompt: str) -> str:
    """Extension point: configure a provider here only with MOCK_LLM=0."""
    raise RuntimeError("Real LLM mode needs a provider implementation and API key.")
def validated_real_answer(prompt: str, sources: list[str]) -> AnswerResponse:
    for attempt in range(3):
        try: return AnswerResponse.model_validate_json(optional_llm(prompt + ("\nReturn valid schema JSON." if attempt else "")))
        except Exception as error: last = error
    return AnswerResponse(answer=f"ERROR: model output could not be validated: {last}", sources=sources, confidence=0.0)
def classify_intent(state: State) -> State:
    query = state["query"].lower()
    if MOCK: return {"intent": "policy_question" if any(word in query for word in KEYWORDS) else "general_question"}
    # A provider call can replace this conservative fallback in the optional path.
    return {"intent": "policy_question" if any(word in query for word in KEYWORDS) else "general_question"}
def retrieve_and_answer(state: State) -> State:
    chunks = retrieve(state["query"]); ids=[chunk["id"] for chunk in chunks]
    if MOCK: response=AnswerResponse(answer="Based on the retrieved context: " + chunks[0]["text"][:200], sources=ids, confidence=1.0)
    else: response=validated_real_answer(STRUCTURED_PROMPT.format(query=state["query"], context="\n".join(c["text"] for c in chunks)),ids)
    return {"chunks":chunks,"response":response}
def direct_answer(state: State) -> State:
    if MOCK: response=AnswerResponse(answer="I can only answer questions about Zepto policies right now.",sources=[],confidence=1.0)
    else: response=validated_real_answer(STRUCTURED_PROMPT.format(query=state["query"],context="No policy context was retrieved."),[])
    return {"response":response}
graph=StateGraph(State); graph.add_node("classify_intent",classify_intent); graph.add_node("retrieve_and_answer",retrieve_and_answer); graph.add_node("direct_answer",direct_answer); graph.set_entry_point("classify_intent"); graph.add_conditional_edges("classify_intent",lambda s:s["intent"],{"policy_question":"retrieve_and_answer","general_question":"direct_answer"}); graph.add_edge("retrieve_and_answer",END); graph.add_edge("direct_answer",END); app=FastAPI(title="Zepto Policy Assistant"); chain=graph.compile()
@app.post("/ask",response_model=AnswerResponse)
def ask(request: AskRequest): return chain.invoke({"query":request.query})["response"]
