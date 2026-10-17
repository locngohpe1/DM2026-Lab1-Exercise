import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_fixture():
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
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Invoke check_missing_tool
    result = check_missing_tool.invoke({})

    # Known Answer 2: exactly 1 row has missing/empty text (the last row, index 7)
    assert result["total_documents"] == 8
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]
    assert "result_id" in result


def test_check_duplicates_tool_inspect():
    # Fresh session so data mutations don't leak between tests
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Known Answer 3 (drop=False): exactly 1 duplicate (row 6 is a copy of row 5)
    result = check_duplicates_tool.invoke({"drop": False})

    assert result["n_duplicates"] == 1
    assert result["duplicate_indices"] == [6]
    assert result["n_rows_before"] == 8
    assert result["n_rows_after"] == 8
    assert result["dropped"] is False
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_drop():
    # Fresh session so drop=True starts from clean 8-row fixture
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Known Answer 3 (drop=True, keep=False): drops both copies (rows 5 and 6), leaving 6 rows
    result = check_duplicates_tool.invoke({"drop": True})

    assert result["n_duplicates"] == 1
    assert result["n_rows_before"] == 8
    assert result["n_rows_after"] == 6
    assert result["dropped"] is True
    assert len(session.dataframe) == 6
    # Index is reset and contiguous from 0 to 5
    assert list(session.dataframe.index) == list(range(6))


def test_sample_data_tool_without_replace():
    # Fresh session: sample without mutating working dataframe
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Known Answer 13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    result = sample_data_tool.invoke({
        "n": 4,
        "random_state": 42,
        "replace_working_data": False,
    })

    assert result["n_sampled"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["replace_working_data"] is False
    assert len(session.dataframe) == 8  # working dataframe unchanged
    assert "sample_df" in session.artifacts
    assert len(session.artifacts["sample_df"]) == 4


def test_sample_data_tool_with_replace():
    # Fresh session: sample with replacement of working dataframe
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Known Answer 13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    result = sample_data_tool.invoke({
        "n": 4,
        "random_state": 42,
        "replace_working_data": True,
    })

    assert result["n_sampled"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["replace_working_data"] is True
    assert len(session.dataframe) == 4  # working dataframe overwritten
    assert list(session.dataframe.index) == list(range(4))  # index reset contiguous 0..3


def test_describe_data_tool_fixture():
    # Fresh session for describe_data_tool
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})

    # Known Answer 18:
    # Overall: count=8, min=0, max=30, mean=19.875, std=9.4330, 50%=20.5
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["min"] == 0
    assert overall["max"] == 30
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)

    # catA: mean=17.25, std=12.5
    cat_a = result["by_category"]["catA"]
    assert cat_a["mean"] == pytest.approx(17.25, abs=1e-3)
    assert cat_a["std"] == pytest.approx(12.5, abs=1e-3)

    # catB: mean=22.5, std=5.7446
    cat_b = result["by_category"]["catB"]
    assert cat_b["mean"] == pytest.approx(22.5, abs=1e-3)
    assert cat_b["std"] == pytest.approx(5.7446, abs=1e-3)

    # Plot contract: session.pending_figure must be populated
    assert session.pending_figure is not None


def test_inspect_data_tool():
    # Fresh session for inspect_data_tool test suite
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Defaults: total_documents=8, 5 rows (indices 0..4), text_lengths match Known Answer 18 [29, 23, 17, 30, 24]
    res1 = inspect_data_tool.invoke({})
    assert res1["total_documents"] == 8
    assert len(res1["rows"]) == 5
    assert [r["index"] for r in res1["rows"]] == [0, 1, 2, 3, 4]
    assert [r["text_length"] for r in res1["rows"]] == [29, 23, 17, 30, 24]

    # 2. n_rows=20 clamped to 8 rows; row 7 has text_length 0 and text ""
    res2 = inspect_data_tool.invoke({"n_rows": 20})
    assert len(res2["rows"]) == 8
    row7 = next(r for r in res2["rows"] if r["index"] == 7)
    assert row7["text_length"] == 0
    assert row7["text"] == ""

    # 3. n_rows=0 clamped up to 1 row
    res3 = inspect_data_tool.invoke({"n_rows": 0})
    assert len(res3["rows"]) == 1
    assert res3["rows"][0]["index"] == 0

    # 4. max_chars=10 truncates row 0 to 'always alp...' while text_length remains 29
    res4 = inspect_data_tool.invoke({"n_rows": 1, "max_chars": 10})
    assert res4["rows"][0]["text"] == "always alp..."
    assert res4["rows"][0]["text_length"] == 29

    # 5. random_sample=True, n_rows=4, random_state=42 returns indices [1, 5, 0, 7] (matching Known Answer 13)
    res5 = inspect_data_tool.invoke({"random_sample": True, "n_rows": 4, "random_state": 42})
    assert [r["index"] for r in res5["rows"]] == [1, 5, 0, 7]


def test_tokenize_tool_fixture():
    # Fresh session for tokenize_tool test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Known Answer 10: token lists for all 8 fixture rows
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert "unigrams" in session.dataframe.columns
    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens

    # Verify summary diagnostics: 27 total tokens, 5 unique vocab terms
    assert result["total_documents"] == 8
    assert result["total_tokens"] == 27
    assert result["vocab_size"] == 5
    assert result["sample_first_doc_tokens"] == ["always", "alpha", "alpha", "alpha", "beta"]


def test_list_files_tool():
    # Fresh session for list_files_tool
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer 11: subdirectory="newdataset" returns directories: [], files: ["Reddit-stock-sentiment.csv"]
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["truncated"] is False

    # Path traversal protection: ".." must return an error dict rather than a listing
    err_result = list_files_tool.invoke({"subdirectory": ".."})
    assert "error" in err_result
