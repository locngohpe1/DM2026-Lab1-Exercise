import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_fixture():
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

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate filtering tools using the same session
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 1. Call variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Known Answer 5:
    # alpha: 1.1875, always: 0.109375, beta: 0.234375, delta: 0.25, gamma: 1.109375
    # threshold=0.15 keeps alpha, beta, delta, gamma and removes always
    assert result["threshold"] == 0.15
    assert result["total_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1

    # Verify session.artifacts['variance_kept_terms'] matches {alpha, beta, delta, gamma}
    assert "variance_kept_terms" in session.artifacts
    assert set(session.artifacts["variance_kept_terms"]) == {"alpha", "beta", "delta", "gamma"}

    # Verify full_report from session.results_store
    result_id = result["result_id"]
    stored_entry = session.results_store[result_id]
    full_report_df = stored_entry["full_report"]

    # First column must hold terms, second column must hold variance
    assert list(full_report_df.columns)[:2] == ["term", "variance"]

    # Verify numerical variances from full_report against Known Answer 5
    report_dict = dict(zip(full_report_df["term"], full_report_df["variance"]))
    assert report_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert report_dict["always"] == pytest.approx(0.109375, abs=1e-3)
    assert report_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert report_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert report_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)

    # 2. Call again with top_n=2 and verify preview capping
    result_top2 = variance_filter_tool.invoke({"threshold": 0.15, "top_n": 2})
    assert len(result_top2["kept_terms"]) == 2
