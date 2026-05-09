import streamlit as st

from src.agent import run_agent

EXAMPLE_QUERIES = [
    "What are the NDB notification obligations?",
    "What does McClure v Medibank tell us about liability?",
    "What are the ASD Essential Eight maturity levels?",
]

DOC_TYPE_OPTIONS = ["All", "lecture_notes", "legislation", "framework", "case_law"]

# --- Sidebar ---
with st.sidebar:
    st.title("CYBR7003 Legal RAG Agent")
    selected_type = st.selectbox("Filter by source type", DOC_TYPE_OPTIONS)
    doc_type_filter = None if selected_type == "All" else selected_type

    st.markdown("---")
    st.markdown("**Example queries**")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True):
            st.session_state["pending_query"] = example

# --- Main area ---
query = st.chat_input("Ask a question about cybersecurity law, policy, or governance...")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.markdown(f"**Query:** {query}")

    with st.spinner("Searching and generating answer…"):
        result = run_agent(query, doc_type_filter=doc_type_filter)

    st.markdown(result["answer"])

    if not result["grounded"]:
        st.warning("⚠️ Answer may contain unsupported claims")

    with st.expander("Sources"):
        chunks = result.get("chunks", [])
        if not chunks:
            st.write("No source chunks retrieved.")
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            doc_type = chunk.get("document_type", "")
            topic = chunk.get("topic", "")
            preview = chunk.get("text", "")[:200]

            st.markdown(f"**{source}** &nbsp; `{doc_type}`")
            if topic:
                st.markdown(f"*{topic}*")
            st.markdown(f"> {preview}…")
            st.markdown("---")
