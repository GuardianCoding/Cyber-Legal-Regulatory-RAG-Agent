"""
Unit and integration tests for the retrieval stack and agent.

Unit tests mock module-level objects (_hybrid_retriever, _reranker, Anthropic client)
so they run without network calls or a built ChromaDB index.

Integration tests hit the real index and are skipped if .chroma/ is absent.
Run integration tests with: pytest tests/ -m integration
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

CHROMA_EXISTS = (Path(__file__).parent.parent / ".chroma").exists()
integration = pytest.mark.skipif(
    not CHROMA_EXISTS,
    reason="ChromaDB index not built — run: python src/ingest.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_KEYS = {"text", "source", "document_type", "topic", "score"}


def _mock_node(
    text="Sample legal text.",
    source="privacy_act.pdf",
    doc_type="legislation",
    topic="APP 11",
    score=0.8,
):
    node = MagicMock()
    node.get_content.return_value = text
    node.metadata = {"source": source, "document_type": doc_type, "topic": topic}
    node.score = score
    return node


def _mock_message(text="YES"):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ---------------------------------------------------------------------------
# hybrid_search — unit tests
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def _call(self, nodes_in, nodes_out, query="test query", doc_type_filter=None):
        with patch("src.tools._hybrid_retriever") as mock_retr, \
             patch("src.tools._reranker") as mock_rerank:
            mock_retr.retrieve.return_value = nodes_in
            mock_rerank.postprocess_nodes.return_value = nodes_out
            from src.tools import hybrid_search
            results = hybrid_search(query, doc_type_filter=doc_type_filter)
            return results, mock_rerank

    def test_returns_list_of_dicts(self):
        results, _ = self._call([_mock_node()], [_mock_node()])
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_result_schema_keys(self):
        results, _ = self._call([_mock_node()], [_mock_node()])
        assert set(results[0].keys()) == CHUNK_KEYS

    def test_result_values_match_node(self):
        node = _mock_node(text="Act text", source="act.pdf", doc_type="legislation", topic="s 26WK", score=0.9)
        results, _ = self._call([node], [node])
        r = results[0]
        assert r["text"] == "Act text"
        assert r["source"] == "act.pdf"
        assert r["document_type"] == "legislation"
        assert r["topic"] == "s 26WK"
        assert r["score"] == pytest.approx(0.9)

    def test_score_is_always_float(self):
        nodes = [_mock_node(score=0.75), _mock_node(score=0.5)]
        results, _ = self._call(nodes, nodes)
        for r in results:
            assert isinstance(r["score"], float)

    def test_none_score_coerced_to_zero(self):
        node = _mock_node(score=None)
        results, _ = self._call([node], [node])
        assert results[0]["score"] == 0.0
        assert isinstance(results[0]["score"], float)

    def test_doc_type_filter_applied_before_rerank(self):
        nodes = [
            _mock_node(doc_type="legislation"),
            _mock_node(doc_type="lecture_notes"),
            _mock_node(doc_type="legislation"),
        ]
        legislation_nodes = [n for n in nodes if n.metadata["document_type"] == "legislation"]

        with patch("src.tools._hybrid_retriever") as mock_retr, \
             patch("src.tools._reranker") as mock_rerank:
            mock_retr.retrieve.return_value = nodes
            mock_rerank.postprocess_nodes.return_value = legislation_nodes

            from src.tools import hybrid_search
            hybrid_search("test", doc_type_filter="legislation")

            passed_to_reranker = mock_rerank.postprocess_nodes.call_args[0][0]
            assert all(n.metadata["document_type"] == "legislation" for n in passed_to_reranker)
            assert len(passed_to_reranker) == 2

    def test_empty_after_filter_returns_empty_list(self):
        nodes = [_mock_node(doc_type="lecture_notes")]
        with patch("src.tools._hybrid_retriever") as mock_retr, \
             patch("src.tools._reranker") as mock_rerank:
            mock_retr.retrieve.return_value = nodes
            mock_rerank.postprocess_nodes.return_value = []

            from src.tools import hybrid_search
            results = hybrid_search("test", doc_type_filter="case_law")

        assert results == []

    def test_no_filter_passes_all_nodes_to_reranker(self):
        nodes = [_mock_node(doc_type="legislation"), _mock_node(doc_type="case_law")]
        with patch("src.tools._hybrid_retriever") as mock_retr, \
             patch("src.tools._reranker") as mock_rerank:
            mock_retr.retrieve.return_value = nodes
            mock_rerank.postprocess_nodes.return_value = nodes

            from src.tools import hybrid_search
            hybrid_search("test", doc_type_filter=None)

            passed = mock_rerank.postprocess_nodes.call_args[0][0]
            assert len(passed) == 2


# ---------------------------------------------------------------------------
# run_agent — unit tests
#
# agent.py does `from src.tools import hybrid_search`, so the name is bound
# inside src.agent at import time. Patch src.agent.hybrid_search (not
# src.tools.hybrid_search) to intercept calls made by the agent.
# ---------------------------------------------------------------------------


class TestRunAgent:
    def _make_chunks(self, n=2):
        return [
            {
                "text": f"Chunk {i} text.",
                "source": f"source_{i}.pdf",
                "document_type": "legislation",
                "topic": "NDB",
                "score": 0.9 - i * 0.1,
            }
            for i in range(n)
        ]

    def _patch_agent(self, chunks, decompose_reply, generate_reply, verify_reply):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message(decompose_reply),
            _mock_message(generate_reply),
            _mock_message(verify_reply),
        ]
        return (
            patch("src.agent.hybrid_search", return_value=chunks),
            patch("src.agent._client", mock_client),
        )

    def test_return_schema(self):
        chunks = self._make_chunks()
        p1, p2 = self._patch_agent(chunks, '["NDB obligations"]', "Answer text.", "YES")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("What are NDB obligations?")
        assert set(result.keys()) == {"answer", "sources", "grounded", "chunks"}

    def test_answer_is_string(self):
        chunks = self._make_chunks()
        p1, p2 = self._patch_agent(chunks, '["NDB"]', "The answer.", "YES")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("NDB test")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_grounded_is_bool(self):
        chunks = self._make_chunks()
        p1, p2 = self._patch_agent(chunks, '["q"]', "Answer.", "YES")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("test")
        assert isinstance(result["grounded"], bool)

    def test_grounded_true_when_verifier_says_yes(self):
        chunks = self._make_chunks()
        p1, p2 = self._patch_agent(chunks, '["q"]', "Answer.", "YES this is grounded")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("test")
        assert result["grounded"] is True

    def test_grounded_false_when_verifier_says_no(self):
        chunks = self._make_chunks()
        p1, p2 = self._patch_agent(chunks, '["q"]', "Answer.", "NO: contains speculation")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("test")
        assert result["grounded"] is False

    def test_sources_are_deduplicated(self):
        chunks = [
            {"text": "a", "source": "act.pdf", "document_type": "legislation", "topic": "", "score": 0.9},
            {"text": "b", "source": "act.pdf", "document_type": "legislation", "topic": "", "score": 0.8},
            {"text": "c", "source": "e8.pdf", "document_type": "framework", "topic": "", "score": 0.7},
        ]
        p1, p2 = self._patch_agent(chunks, '["q"]', "Answer.", "YES")
        with p1, p2:
            from src.agent import run_agent
            result = run_agent("test")
        assert result["sources"].count("act.pdf") == 1

    def test_no_chunks_returns_graceful_answer(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message('["q"]')
        with patch("src.agent.hybrid_search", return_value=[]), \
             patch("src.agent._client", mock_client):
            from src.agent import run_agent
            result = run_agent("something obscure")
        assert "couldn't find" in result["answer"].lower()
        assert result["grounded"] is True
        assert result["chunks"] == []

    def test_decomposer_fallback_on_invalid_json(self):
        chunks = self._make_chunks(1)
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message("not valid json at all"),
            _mock_message("Answer."),
            _mock_message("YES"),
        ]
        with patch("src.agent.hybrid_search", return_value=chunks), \
             patch("src.agent._client", mock_client):
            from src.agent import run_agent
            result = run_agent("original query")
        assert isinstance(result["answer"], str)

    def test_doc_type_filter_passed_to_hybrid_search(self):
        chunks = self._make_chunks(1)
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message('["q"]'),
            _mock_message("Answer."),
            _mock_message("YES"),
        ]
        with patch("src.agent.hybrid_search", return_value=chunks) as mock_search, \
             patch("src.agent._client", mock_client):
            from src.agent import run_agent
            run_agent("test", doc_type_filter="legislation")
            call_kwargs = mock_search.call_args
        assert call_kwargs[1].get("doc_type_filter") == "legislation" or \
               (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "legislation")


# ---------------------------------------------------------------------------
# Integration tests — require built ChromaDB index
# ---------------------------------------------------------------------------


@integration
class TestHybridSearchIntegration:
    def test_ndb_query_returns_legislation_chunks(self):
        from src.tools import hybrid_search
        results = hybrid_search("NDB notification obligations")
        assert len(results) > 0
        assert all(set(r.keys()) == CHUNK_KEYS for r in results)
        assert any(r["document_type"] == "legislation" for r in results)

    def test_max_five_results_after_rerank(self):
        from src.tools import hybrid_search
        results = hybrid_search("Privacy Act personal information")
        assert len(results) <= 5

    def test_doc_type_filter_legislation_only(self):
        from src.tools import hybrid_search
        results = hybrid_search("data breach notification", doc_type_filter="legislation")
        assert all(r["document_type"] == "legislation" for r in results)

    def test_doc_type_filter_framework_only(self):
        from src.tools import hybrid_search
        results = hybrid_search("patch applications", doc_type_filter="framework")
        assert all(r["document_type"] == "framework" for r in results)

    def test_scores_ordered_descending(self):
        from src.tools import hybrid_search
        results = hybrid_search("Australian Privacy Principles")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True) or len(scores) <= 1

    def test_essential_eight_query_hits_framework(self):
        from src.tools import hybrid_search
        results = hybrid_search("Essential Eight maturity level")
        assert any("framework" in r["document_type"] or "essential" in r["text"].lower() for r in results)

    def test_medibank_query_hits_case_law(self):
        from src.tools import hybrid_search
        results = hybrid_search("McClure v Medibank")
        assert any(r["document_type"] == "case_law" for r in results)
