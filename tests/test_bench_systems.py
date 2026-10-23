from agentrag.bench.systems import HybridRRFRetriever, Retriever, TfidfRetriever, rrf_fuse


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


def test_tfidf_retriever_returns_doc_ids():
    retriever = TfidfRetriever()
    retriever.index({"d1": "retrieval augmented generation with vector stores", "d2": "cooking pasta recipes"})
    assert retriever.search("vector retrieval", k=2)[0][0] == "d1"
