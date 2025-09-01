# langgraph_rag_with_memory_tavily.py
import os
import json
import time
from typing import Dict, Any, List

from langchain_community.llms import CTransformers
from langgraph.graph import StateGraph, END

# Optional LangChain pieces for local doc retrieval
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 🔹 Tavily client
from tavily import TavilyClient

# ----------------- CONFIG -----------------
MODEL_PATH = "./models/mistral-7b-instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MAX_NEW_TOKENS = 150
THREADS = 6
CONTEXT_LENGTH = 2048

MEMORY_FILE = "./chat_memory.json"
MAX_MEMORY_TURNS = 6
SEARCH_RESULTS_PER_QUERY = 3
RETRIEVE_K = 3

#TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
import os
print("Tavily API:", os.getenv("APIKEY"))
tavily = TavilyClient(api_key="APIKEY")

# ----------------- LLM (ctransformers) -----------------
print("Loading local LLM... (this may take a bit)")
LLM = CTransformers(
    model=MODEL_PATH,
    config={
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "context_length": CONTEXT_LENGTH,
        "threads": THREADS
    }
)
print("LLM loaded.")

def run_llm(prompt: str) -> str:
    start = time.time()
    print("\n[LLM] Running inference...")
    print(f"[LLM] Prompt length: {len(prompt)} chars (~{len(prompt.split())} tokens approx)")

    try:
        response = LLM(prompt)
        elapsed = time.time() - start
        print(f"[LLM] Response received in {elapsed:.2f}s")
        print(f"[LLM] Response length: {len(response)} chars (~{len(response.split())} tokens approx)")
        return response.strip()
    except Exception as e:
        print(f"[LLM] ERROR: {e}")
        return "LLM failed."

# ----------------- Persistent Chat Memory -----------------
def load_memory() -> List[Dict[str,str]]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(memory: List[Dict[str,str]]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def append_memory(user_msg: str, assistant_msg: str):
    mem = load_memory()
    mem.append({"user": user_msg, "assistant": assistant_msg, "ts": time.time()})
    if len(mem) > MAX_MEMORY_TURNS:
        mem = mem[-MAX_MEMORY_TURNS:]
    save_memory(mem)

def memory_to_prompt_prefix() -> str:
    mem = load_memory()
    if not mem:
        return ""
    parts = []
    for pair in mem:
        u = pair.get("user", "").strip()
        a = pair.get("assistant", "").strip()
        parts.append(f"User: {u}\nAssistant: {a}")
    return "Conversation history:\n" + "\n\n".join(parts) + "\n\n"

# ----------------- Local Document Vectorstore -----------------
def build_local_vectorstore(sample_texts: List[str]):
    docs = [Document(page_content=t) for t in sample_texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs_split, embedder)
    return vs

SAMPLE_TEXTS = [
    "LangGraph is a workflow/state graph library for building multi-agent systems.",
    "Mistral 7B is an open-weight model that can be quantized and run locally with ctransformers.",
    "HNSW or FAISS are commonly used for fast vector retrieval to support RAG systems."
]
vectorstore = build_local_vectorstore(SAMPLE_TEXTS)

# ----------------- Search helper (Tavily) -----------------
def web_search_snippets(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> List[str]:
    print(f"[SEARCH] Running Tavily search for: {query}")
    try:
        raw = tavily.search(query=query, search_depth="advanced", max_results=max_results)
        print(f"[SEARCH] Raw response keys: {list(raw.keys())}")
        snippets = []
        for r in raw.get("results", []):
            title = r.get("title", "").strip()
            content = r.get("content", "").strip()
            url = r.get("url", "").strip()
            if content:
                snippet = f"{title} - {content}\n(Source: {url})"
                snippets.append(snippet)
        print(f"[SEARCH] Got {len(snippets)} snippets for query: {query}")
        return snippets
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return [f"[Error querying Tavily: {e}]"]

# ----------------- Agent Nodes -----------------
def decompose_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] Decompose question")
    q = state["question"].strip()
    prompt = memory_to_prompt_prefix() + f"""
Decompose the user's question into up to 3 short web search queries (one per line). Keep them concise.

Question:
{q}

Subqueries:
"""
    out = run_llm(prompt).strip()
    lines = [line.strip("- \t") for line in out.splitlines() if line.strip()]
    if not lines:
        lines = [q]
    state["subqueries"] = lines[:3]
    print(f"[NODE] Subqueries: {state['subqueries']}")
    return state

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] Search")
    subs = state.get("subqueries", [])
    all_snippets = {}
    for sq in subs:
        snippets = web_search_snippets(sq, max_results=SEARCH_RESULTS_PER_QUERY)
        all_snippets[sq] = snippets
    state["search_snippets"] = all_snippets
    print(f"[NODE] Search complete. Got results for {len(all_snippets)} subqueries")
    return state

def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] Local retrieval")
    q = state["question"]
    results = vectorstore.similarity_search(q, k=RETRIEVE_K)
    state["local_retrievals"] = [r.page_content for r in results]
    print(f"[NODE] Retrieved {len(results)} local docs")
    return state

def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] Summarize")
    summaries = {}
    for sq, snippets in state.get("search_snippets", {}).items():
        context = "\n\n".join(snippets)[:1500]
        prompt = memory_to_prompt_prefix() + f"""
Summarize the following web snippets briefly (1-2 paragraphs). Strictly summarize the content without adding any new context.

Subquery: {sq}

Context:
{context}

Summary:
"""
        try:
            print(f"[DEBUG] Summarizing subquery: {sq}, snippets: {len(snippets)}")
            summaries[sq] = run_llm(prompt).strip()   
            print(f"[NODE] Summarized {sq}")
        except Exception as e:
            print(f"[ERROR] Failed to summarize {sq}: {e}")
            summaries[sq] = "LLM summarization failed."

    local_ctx = "\n\n".join(state.get("local_retrievals", []))[:1500]
    if local_ctx:
        prompt_local = memory_to_prompt_prefix() + f"""
Summarize the following local documents briefly:

Context:
{local_ctx}

Summary:
"""
        try:
            print(f"[DEBUG] Summarizing local docs, length={len(local_ctx)}")
            summaries["local"] = run_llm(prompt_local).strip()   
            print("[NODE] Summarized local docs")
        except Exception as e:
            print(f"[ERROR] Failed to summarize local docs: {e}")
            summaries["local"] = "LLM summarization failed."

    state["summaries"] = summaries
    return state


def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    print("[NODE] Synthesize final answer")
    question = state["question"]
    parts = []
    if "local" in state.get("summaries", {}):
        parts.append("Local docs summary:\n" + state["summaries"]["local"])
    for k, v in state.get("summaries", {}).items():
        if k == "local":
            continue
        parts.append(f"Subquery ({k}) summary:\n{v}")
    context = "\n\n".join(parts)[:2000]

    prompt = memory_to_prompt_prefix() + f"""
Using the following context, answer the user's question concisely and cite sources (web/local). If unsure, say "I don't know".

Question:
{question}

Context:
{context}

Answer:
"""
    answer = run_llm(prompt).strip()
    state["final_answer"] = answer
    print("[NODE] Final answer ready")
    try:
        append_memory(question, answer)
    except Exception as e:
        print("[MEMORY ERROR]", e)
    return state


# ----------------- LangGraph workflow -----------------
workflow = StateGraph(dict)
workflow.add_node("decompose", decompose_node)
workflow.add_node("search", search_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "search")
workflow.add_edge("search", "retrieve")
workflow.add_edge("retrieve", "summarize")
workflow.add_edge("summarize", "synthesize")
workflow.add_edge("synthesize", END)

graph = workflow.compile()

# ----------------- Chat loop -----------------
def chat_loop():
    print("RAG chat with Tavily (type 'exit' to quit). Memory file:", MEMORY_FILE)
    while True:
        user_q = input("\nYou: ").strip()
        if user_q.lower() in ("exit", "quit"):
            break
        start = time.time()
        state = {"question": user_q}
        try:
            print("\n[GRAPH] Starting graph execution...")
            result = graph.invoke(state)
            print("[GRAPH] Execution finished")
        except Exception as e: 
            print("Error running graph:", e)
            continue
        took = time.time() - start
        print("\n--- Final Answer ---")
        print(result.get("final_answer", ""))
        print(f"\n(Processed in {took:.2f}s)\n")

if __name__ == "__main__":
    chat_loop()
