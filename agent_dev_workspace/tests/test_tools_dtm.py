import pytest
import scipy.sparse as sp
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
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

    # Instantiate student's tools using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default settings
    result = build_dtm_tool.invoke({})

    # Known Answer 4: 5 terms ['alpha', 'always', 'beta', 'delta', 'gamma']
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert result["n_terms"] == 5
    assert result["n_documents"] == 8

    # Sparsity on 8x5 matrix: non_zero=21, total_elements=40, sparsity_pct=47.5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Matrix must remain sparse in session
    assert sp.issparse(session.feature_matrix)
    assert session.feature_matrix.shape == (8, 5)

    # Vectorizer artifact stored under 'count_vectorizer'
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None

    # Labels must hold the string category names ('catA', 'catB'), not integer codes
    assert session.labels is not None
    assert set(session.labels) == {"catA", "catB"}
    assert list(session.labels) == ["catA", "catA", "catA", "catB", "catB", "catB", "catB", "catA"]
    assert session.categories == ["catA", "catB"]


def test_term_frequency_tool_fixture():
    # Fresh session for term_frequency_tool
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Default invocation (top_n=20)
    result = term_frequency_tool.invoke({})

    # Known Answer 14:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert result["total_terms"] == 5
    assert result["total_token_occurrences"] == 27

    # Verify full_report from session.results_store
    result_id = result["result_id"]
    stored_entry = session.results_store[result_id]
    full_report_df = stored_entry["full_report"]

    actual_counts_dict = dict(zip(full_report_df["term"], full_report_df["frequency"]))
    assert actual_counts_dict == expected_counts

    # 2. Call again with top_n=2
    result_top2 = term_frequency_tool.invoke({"top_n": 2})
    top2_records = result_top2["top_terms"]

    # Summary has exactly 2 records: always (7) then gamma (7) by alphabetical tie-break
    assert len(top2_records) == 2
    assert top2_records == [
        {"term": "always", "count": 7},
        {"term": "gamma", "count": 7},
    ]

    # full_report still retains all 5 terms
    result_id_top2 = result_top2["result_id"]
    full_report_df_top2 = session.results_store[result_id_top2]["full_report"]
    assert len(full_report_df_top2) == 5


def test_dtm_heatmap_tool_fixture():
    # Fresh session for dtm_heatmap_tool
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Defaults (n_terms=20, n_documents=20) -> full 8x5 matrix per Known Answer 21
    res1 = dtm_heatmap_tool.invoke({})
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert res1["matrix"] == expected_full_matrix
    assert res1["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res1["actual_n_documents"] == 8
    assert res1["actual_n_terms"] == 5
    assert session.pending_figure is not None

    # 2. Slicing check: n_terms=2, n_documents=3 -> top-left 3x2 corner per Known Answer 21
    res2 = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res2["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert res2["terms"] == ["alpha", "always"]
    assert res2["actual_n_documents"] == 3
    assert res2["actual_n_terms"] == 2

    # 3. Clamping check: n_terms=500, n_documents=500 reports requested 500 and actual bounds 8x5
    res3 = dtm_heatmap_tool.invoke({"n_terms": 500, "n_documents": 500})
    assert res3["requested_n_documents"] == 500
    assert res3["requested_n_terms"] == 500
    assert res3["actual_n_documents"] == 8
    assert res3["actual_n_terms"] == 5
    assert res3["matrix"] == expected_full_matrix
