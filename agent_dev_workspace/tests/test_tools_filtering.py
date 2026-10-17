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


def test_pearson_filter_tool_fixture():
    # Fresh session for pearson_filter_tool
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 1. Invoke with target_class="catB"
    result = pearson_filter_tool.invoke({"target_class": "catB"})

    # Verify artifact session.artifacts['pearson_catB']
    assert "pearson_catB" in session.artifacts
    artifact_df = session.artifacts["pearson_catB"]

    # Check first column is the term
    assert artifact_df.columns[0] == "term"
    assert artifact_df.columns[1] == "pearson_r"

    # Known Answer 6:
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    r_dict = dict(zip(artifact_df["term"], artifact_df["pearson_r"]))
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # 2. Check unknown target_class returns error
    err_res = pearson_filter_tool.invoke({"target_class": "catC"})
    assert "error" in err_res

    # 3. Call with top_n=1: top_positive has only delta, top_negative has only beta
    result_top1 = pearson_filter_tool.invoke({"target_class": "catB", "top_n": 1})
    assert len(result_top1["top_positive"]) == 1
    assert result_top1["top_positive"][0]["term"] == "delta"
    assert result_top1["top_positive"][0]["pearson_r"] == pytest.approx(1.0, abs=1e-3)

    assert len(result_top1["top_negative"]) == 1
    assert result_top1["top_negative"][0]["term"] == "beta"
    assert result_top1["top_negative"][0]["pearson_r"] == pytest.approx(-0.7746, abs=1e-3)


def test_spearman_filter_tool_fixture():
    # Fresh session for spearman_filter_tool
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Invoke with target_class="catB"
    result = spearman_filter_tool.invoke({"target_class": "catB"})

    assert result["total_terms"] == 5
    assert result["n_evaluated_terms"] == 5

    # Verify artifact session.artifacts['spearman_catB']
    assert "spearman_catB" in session.artifacts
    artifact_df = session.artifacts["spearman_catB"]

    # Check first column is term, second is spearman_r
    assert artifact_df.columns[0] == "term"
    assert artifact_df.columns[1] == "spearman_r"

    # Known Answer 6 (Spearman r):
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    r_dict = dict(zip(artifact_df["term"], artifact_df["spearman_r"]))
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # 2. Check top_n=1 preview capping
    res_top1 = spearman_filter_tool.invoke({"target_class": "catB", "top_n": 1})
    assert len(res_top1["top_positive"]) <= 1
    assert len(res_top1["top_negative"]) <= 1

    # 3. Check max_terms=2 restricts evaluation to top 2 terms by variance (alpha and gamma)
    res_max2 = spearman_filter_tool.invoke({"target_class": "catB", "max_terms": 2})
    assert res_max2["total_terms"] == 5
    assert res_max2["n_evaluated_terms"] == 2
    artifact_max2 = session.artifacts["spearman_catB"]
    assert set(artifact_max2["term"]) == {"alpha", "gamma"}
