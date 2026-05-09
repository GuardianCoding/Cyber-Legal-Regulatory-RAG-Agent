import hashlib
import json
from typing import Optional, TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from config import ANTHROPIC_API_KEY, LLM_DEV
from tools import hybrid_search

_SYSTEM_PROMPT = (
    "You are a legal research assistant specialising in Australian cybersecurity law, "
    "policy, and governance. You answer questions based ONLY on the provided source documents.\n\n"
    "Rules:\n"
    "1. Always cite your source: name the document and section/page for every claim.\n"
    "2. If the retrieved context does not contain enough information, say so explicitly.\n"
    "3. Never speculate about legal obligations — only state what the documents say.\n"
    "4. Use plain English but maintain legal precision.\n"
    "5. Structure your answer: direct answer first, then supporting detail with citations."
)


class AgentState(TypedDict):
    query: str
    sub_queries: list[str]
    doc_type_filter: Optional[str]
    chunks: list[dict]
    answer: str
    grounded: bool
    sources: list[str]


_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def query_decomposer(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        response = _client.messages.create(
            model=LLM_DEV,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Decompose the following query into 1-3 specific sub-queries "
                        "for searching a legal document database. "
                        "Return ONLY a JSON array of strings, nothing else.\n\n"
                        "Query: " + query
                    ),
                }
            ],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        sub_queries = json.loads(text)
        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError("not a list")
        sub_queries = [str(q) for q in sub_queries[:3]]
    except Exception:
        sub_queries = [query]
    return {**state, "sub_queries": sub_queries}


def retriever(state: AgentState) -> AgentState:
    seen: set[str] = set()
    unique_chunks: list[dict] = []
    doc_type_filter = state.get("doc_type_filter")

    for sub_query in state["sub_queries"]:
        for chunk in hybrid_search(sub_query, doc_type_filter=doc_type_filter):
            h = hashlib.md5(chunk["text"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_chunks.append(chunk)

    unique_chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    top_chunks = unique_chunks[:5]
    sources = list(dict.fromkeys(c["source"] for c in top_chunks if c.get("source")))
    return {**state, "chunks": top_chunks, "sources": sources}


def generator(state: AgentState) -> AgentState:
    chunks = state["chunks"]
    if not chunks:
        return {
            **state,
            "answer": "I couldn't find relevant information in the source documents for this query.",
            "grounded": True,
        }

    context = "\n\n".join(
        f"[{i}] Source: {c.get('source', '')} | Topic: {c.get('topic', '')} | Text: {c.get('text', '')}"
        for i, c in enumerate(chunks, 1)
    )
    response = _client.messages.create(
        model=LLM_DEV,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source documents:\n\n{context}\n\nQuestion: {state['query']}",
            }
        ],
    )
    return {**state, "answer": response.content[0].text.strip()}


def verifier(state: AgentState) -> AgentState:
    if not state["chunks"]:
        return {**state, "grounded": True}

    chunks_text = "\n\n".join(
        f"[{i}] {c.get('text', '')}" for i, c in enumerate(state["chunks"], 1)
    )
    prompt = (
        f"Given these source chunks:\n{chunks_text}\n\n"
        f"And this answer:\n{state['answer']}\n\n"
        "Does the answer follow only from the provided sources? "
        "Reply YES if fully grounded, or NO: <reason> if it contains unsupported claims."
    )
    try:
        response = _client.messages.create(
            model=LLM_DEV,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        grounded = response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        grounded = True
    return {**state, "grounded": grounded}


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("query_decomposer", query_decomposer)
    g.add_node("retriever", retriever)
    g.add_node("generator", generator)
    g.add_node("verifier", verifier)
    g.set_entry_point("query_decomposer")
    g.add_edge("query_decomposer", "retriever")
    g.add_edge("retriever", "generator")
    g.add_edge("generator", "verifier")
    g.add_edge("verifier", END)
    return g.compile()


_graph = _build_graph()


def run_agent(query: str, doc_type_filter: str | None = None) -> dict:
    initial_state: AgentState = {
        "query": query,
        "sub_queries": [],
        "doc_type_filter": doc_type_filter,
        "chunks": [],
        "answer": "",
        "grounded": False,
        "sources": [],
    }
    final = _graph.invoke(initial_state)
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "grounded": final["grounded"],
        "chunks": final["chunks"],
    }
