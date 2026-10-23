from agentrag.bench.systems import CrossEncoderReranker, HybridRRFRetriever, Retriever, TfidfRetriever
from agentrag.fusion import rrf_fuse


def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == 1 / 61
    assert fused["b"] == 1 / 62 + 1 / 61
    assert fused["c"] == 1 / 62


def test_rrf_rewards_agreement_over_a_single_top_rank():
    fused = [d for d, _ in rrf_fuse([["x", "a", "b"], ["y", "a", "c"]])]
    assert fused[0] == "a"


def test_rrf_respects_limit_and_is_deterministic():
    lists = [["a", "b", "c"], ["c", "b", "a"]]
    assert rrf_fuse(lists, limit=2) == rrf_fuse(lists, limit=2)
    assert len(rrf_fuse(lists, limit=2)) == 2


class _Fixed(Retriever):
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, corpus):
        self.indexed = True

    def search(self, query, k=100):
        return [(d, 1.0) for d in self.ranking[:k]]


def test_hybrid_indexes_each_part_once_and_fuses():
    shared = _Fixed(["a", "b"])
    hybrid = HybridRRFRetriever([shared, _Fixed(["b", "c"])])
    hybrid.index({"a": "", "b": "", "c": ""})
    assert shared.indexed
    assert hybrid.search("q", k=3)[0][0] == "b"


def test_default_search_batch_matches_sequential_search():
    retriever = _Fixed(["a", "b", "c"])
    assert retriever.search_batch(["q1", "q2"], k=2) == [retriever.search("q1", k=2), retriever.search("q2", k=2)]


def test_hybrid_search_batch_matches_per_query_search():
    hybrid = HybridRRFRetriever([_Fixed(["a", "b", "c"]), _Fixed(["c", "b", "d"])])
    hybrid.index({})
    queries = ["q1", "q2", "q3"]
    assert hybrid.search_batch(queries, k=3) == [hybrid.search(q, k=3) for q in queries]


class _FakeCrossEncoder:
    """Scores a (query, text) pair by text length, so reranking has a known effect."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [float(len(text)) for _, text in pairs]


def test_cross_encoder_batch_matches_single_and_only_reorders_the_head():
    corpus = {"a": "x", "b": "xxx", "c": "xx", "d": "xxxx"}
    first = _Fixed(["a", "b", "c", "d"])
    reranker = CrossEncoderReranker(first, rerank_depth=3)
    reranker.index(corpus)
    reranker._model = _FakeCrossEncoder()

    single = reranker.search("q", k=4)
    assert [d for d, _ in single] == ["b", "c", "a", "d"]  # head sorted by length, tail 'd' untouched
    assert reranker.search_batch(["q", "q"], k=4) == [single, single]


def test_tfidf_retriever_returns_doc_ids():
    retriever = TfidfRetriever()
    retriever.index({"d1": "retrieval augmented generation with vector stores", "d2": "cooking pasta recipes"})
    assert retriever.search("vector retrieval", k=2)[0][0] == "d1"
