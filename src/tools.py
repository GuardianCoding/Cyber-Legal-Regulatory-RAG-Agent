from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from src.config import CHROMA_PATH, CHROMA_COLLECTION

# Load persisted index once at module level — never re-embeds
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_chroma_collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION)
_vector_store = ChromaVectorStore(chroma_collection=_chroma_collection)
_storage_context = StorageContext.from_defaults(vector_store=_vector_store)
_index = VectorStoreIndex.from_vector_store(
    _vector_store, storage_context=_storage_context
)

# BM25Retriever needs text nodes in memory; from_vector_store() doesn't populate
# the docstore, so we fetch all documents from ChromaDB directly.
_raw = _chroma_collection.get(include=["documents", "metadatas"])
_nodes = [
    TextNode(text=doc, metadata=meta or {})
    for doc, meta in zip(_raw["documents"], _raw["metadatas"])
]

_vector_retriever = VectorIndexRetriever(index=_index, similarity_top_k=10)
_bm25_retriever = BM25Retriever.from_defaults(nodes=_nodes, similarity_top_k=10)
_hybrid_retriever = QueryFusionRetriever(
    [_vector_retriever, _bm25_retriever],
    similarity_top_k=10,
    mode="reciprocal_rerank",
    use_async=False,
)

_reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)


def hybrid_search(query: str, doc_type_filter: str | None = None) -> list[dict]:
    nodes = _hybrid_retriever.retrieve(query)
    if doc_type_filter:
        nodes = [n for n in nodes if n.metadata.get("document_type") == doc_type_filter]
    nodes = _reranker.postprocess_nodes(nodes, query_str=query)
    return [
        {
            "text": n.get_content(),
            "source": n.metadata.get("source", ""),
            "document_type": n.metadata.get("document_type", ""),
            "topic": n.metadata.get("topic", ""),
            "score": float(n.score) if n.score is not None else 0.0,
        }
        for n in nodes
    ]
