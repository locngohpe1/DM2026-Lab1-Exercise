import math
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_fixture():
    # Instantiate session once for this test
    session = SessionState()

    # Load premade load_dataset_tool to load sample fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first to populate session.artifacts['count_vectorizer']
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer 16 assertions:
    # 1. Doc 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert len(res_0_1["shared_terms_sample"]) <= 10

    # 2. Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)
    assert res_0_3["shared_terms_sample"] == ["always"]

    # 3. Doc 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # 4. Doc 0 vs 7 (catA vs empty text row): exactly 0.0 and NOT NaN
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    sim_0_7 = res_0_7["cosine_similarity"]
    assert not math.isnan(sim_0_7)
    assert sim_0_7 == 0.0

    # 5. Out-of-range index handling (index 99)
    err_res = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 99})
    assert "error" in err_res


def test_feature_correlation_matrix_tool_fixture():
    # Fresh session for feature_correlation_matrix_tool
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    result = feature_corr_tool.invoke({})

    # Known Answer 19: All 5 terms ordered by variance descending
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert result["terms"] == expected_terms
    assert result["n_terms"] == 5

    terms = result["terms"]
    matrix = result["correlation_matrix"]

    # Helper function to look up correlation by term names
    def get_corr(t1: str, t2: str) -> float:
        i1 = terms.index(t1)
        i2 = terms.index(t2)
        return matrix[i1][i2]

    # Verify symmetry and diagonal
    for i in range(len(terms)):
        assert matrix[i][i] == pytest.approx(1.0000, abs=1e-3)
        for j in range(len(terms)):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-3)

    # Known Answer 19 pairwise values:
    # alpha vs gamma: -0.5718
    assert get_corr("alpha", "gamma") == pytest.approx(-0.5718, abs=1e-3)
    # alpha vs delta: -0.6882
    assert get_corr("alpha", "delta") == pytest.approx(-0.6882, abs=1e-3)
    # alpha vs beta: 0.8885
    assert get_corr("alpha", "beta") == pytest.approx(0.8885, abs=1e-3)
    # alpha vs always: 0.2601
    assert get_corr("alpha", "always") == pytest.approx(0.2601, abs=1e-3)
    # gamma vs delta: 0.8307
    assert get_corr("gamma", "delta") == pytest.approx(0.8307, abs=1e-3)
    # gamma vs beta: -0.6435
    assert get_corr("gamma", "beta") == pytest.approx(-0.6435, abs=1e-3)
    # gamma vs always: 0.3140
    assert get_corr("gamma", "always") == pytest.approx(0.3140, abs=1e-3)
    # delta vs beta: -0.7746
    assert get_corr("delta", "beta") == pytest.approx(-0.7746, abs=1e-3)
    # delta vs always: 0.3780
    assert get_corr("delta", "always") == pytest.approx(0.3780, abs=1e-3)
    # beta vs always: 0.2928
    assert get_corr("beta", "always") == pytest.approx(0.2928, abs=1e-3)

    # Plot contract assertion: session.pending_figure must be populated
    assert session.pending_figure is not None
