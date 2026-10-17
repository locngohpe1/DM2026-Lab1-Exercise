import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session_with_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


@pytest.mark.parametrize("category", ["catA", "catB"])
@pytest.mark.parametrize("filtering_method", ["variance", "term_frequency"])
def test_mine_patterns_zero_terms_survived(category, filtering_method):
    # Known Answer 17: variance and term_frequency keep zero terms on both categories
    session, mine_patterns_tool = _setup_session_with_fixture()

    result = mine_patterns_tool.invoke({
        "category_name": category,
        "filtering_method": filtering_method,
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert "error" not in result
    assert result["n_surviving_terms"] == 0
    assert result["n_transactions"] == 0
    assert result["n_patterns"] == 0
    assert result["patterns"] == []
    assert result["note"] == "no terms survived filtering"


def test_mine_patterns_tfidf_fpgrowth_catA():
    # Known Answer 17: catA keeps only alpha (support 3), 0 patterns at min_sup=4
    session, mine_patterns_tool = _setup_session_with_fixture()

    for sup in [1, 2, 3]:
        res = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": sup,
        })
        assert res["n_surviving_terms"] == 1
        assert res["n_patterns"] == 1
        assert res["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    # At min_sup=4, support threshold exceeds alpha's support (3)
    res_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_sup4["n_patterns"] == 0
    assert res_sup4["patterns"] == []


def test_mine_patterns_tfidf_fpgrowth_catB():
    # Known Answer 17: catB keeps only gamma (support 4), found at min_sup=1..4
    session, mine_patterns_tool = _setup_session_with_fixture()

    for sup in [1, 2, 3, 4]:
        res = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": sup,
        })
        assert res["n_surviving_terms"] == 1
        assert res["n_patterns"] == 1
        assert res["patterns"] == [{"pattern": ["gamma"], "support": 4}]


@pytest.mark.parametrize("category, expected_term, expected_sup", [
    ("catA", "alpha", 3),
    ("catB", "gamma", 4),
])
@pytest.mark.parametrize("k_val", [1, 2, 3])
def test_mine_patterns_topk(category, expected_term, expected_sup, k_val):
    # Known Answer 20: topk at k=1, 2, 3 returns exactly 1 pattern for both categories
    session, mine_patterns_tool = _setup_session_with_fixture()

    result = mine_patterns_tool.invoke({
        "category_name": category,
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": k_val,
    })

    assert result["n_surviving_terms"] == 1
    assert result["n_patterns"] == 1
    assert result["patterns"] == [{"pattern": [expected_term], "support": expected_sup}]


@pytest.mark.parametrize("category", ["catA", "catB"])
@pytest.mark.parametrize("sup_val", [1, 4])
def test_mine_patterns_maxfpgrowth_zero_patterns(category, sup_val):
    # Known Answer 20: maxfpgrowth returns 0 patterns for both catA and catB at min_sup=1 and min_sup=4
    session, mine_patterns_tool = _setup_session_with_fixture()

    result = mine_patterns_tool.invoke({
        "category_name": category,
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": sup_val,
    })

    assert "error" not in result
    assert result["n_surviving_terms"] == 1
    assert result["n_patterns"] == 0
    assert result["patterns"] == []
