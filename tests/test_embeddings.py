"""
Unit tests for src/embeddings.py (TF-IDF + SVD dense embeddings).
"""

import numpy as np
import pytest

from retrieval.embeddings import LSAEmbeddings


@pytest.fixture(scope="module")
def fitted_embeddings():
    texts = [
        "quantization reduces model memory footprint",
        "reranking improves retrieval precision at the top",
        "hybrid retrieval combines dense and sparse search",
        "on-premise infrastructure requires GPU and storage planning",
    ]
    emb = LSAEmbeddings(n_components=2)
    emb.fit(texts)
    return emb


class TestLSAEmbeddings:
    def test_embed_query_returns_correct_dimensionality(self, fitted_embeddings):
        vec = fitted_embeddings.embed_query("quantization and memory")
        assert len(vec) == fitted_embeddings.svd.n_components

    def test_embed_documents_returns_one_vector_per_text(self, fitted_embeddings):
        vecs = fitted_embeddings.embed_documents(["a query", "another query", "third"])
        assert len(vecs) == 3

    def test_vectors_are_unit_normalized(self, fitted_embeddings):
        vec = fitted_embeddings.embed_query("hybrid retrieval")
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_scores_higher_than_dissimilar(self, fitted_embeddings):
        query_vec = np.array(fitted_embeddings.embed_query("quantization memory reduction"))
        similar_vec = np.array(fitted_embeddings.embed_query("quantization reduces memory footprint"))
        dissimilar_vec = np.array(fitted_embeddings.embed_query("on-premise GPU storage planning"))

        sim_similar = np.dot(query_vec, similar_vec)
        sim_dissimilar = np.dot(query_vec, dissimilar_vec)
        assert sim_similar > sim_dissimilar

    def test_save_and_load_roundtrip(self, fitted_embeddings, tmp_path):
        path = str(tmp_path / "test_embeddings.pkl")
        fitted_embeddings.save(path)
        reloaded = LSAEmbeddings.load(path)

        original_vec = fitted_embeddings.embed_query("test query")
        reloaded_vec = reloaded.embed_query("test query")
        assert np.allclose(original_vec, reloaded_vec)

    def test_n_components_capped_by_corpus_size(self):
        # requesting more components than samples/features allows should
        # not crash -- it should cap gracefully (see fit()'s min() call)
        emb = LSAEmbeddings(n_components=999)
        emb.fit(["one short document", "another short document"])
        assert emb.svd.n_components < 999
