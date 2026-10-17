import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_tool_pca_fixture():
    # Fresh session for PCA test
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

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_tool.invoke({"method": "pca"})

    # Verify session.artifacts['reduced_pca']
    assert "reduced_pca" in session.artifacts
    coords = session.artifacts["reduced_pca"]

    # Known Answer 7:
    # Doc 0: [2.3669, 0.8871]
    assert coords[0][0] == pytest.approx(2.3669, abs=1e-3)
    assert coords[0][1] == pytest.approx(0.8871, abs=1e-3)

    # Doc 3: [-2.0759, 1.0129]
    assert coords[3][0] == pytest.approx(-2.0759, abs=1e-3)
    assert coords[3][1] == pytest.approx(1.0129, abs=1e-3)

    # Doc 7: [0.0894, -1.4822]
    assert coords[7][0] == pytest.approx(0.0894, abs=1e-3)
    assert coords[7][1] == pytest.approx(-1.4822, abs=1e-3)

    # explained_variance_ratio: [0.7532, 0.1965]
    assert result["explained_variance_ratio"][0] == pytest.approx(0.7532, abs=1e-3)
    assert result["explained_variance_ratio"][1] == pytest.approx(0.1965, abs=1e-3)

    # Plot contract assertion: session.pending_figure must be populated
    assert session.pending_figure is not None


def test_reduce_dimensions_tool_tsne_properties():
    # Fresh session for t-SNE property-based test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run 1: method="tsne", perplexity=2, random_state=42
    reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42,
    })

    assert "reduced_tsne" in session.artifacts
    coords1 = np.asarray(session.artifacts["reduced_tsne"])

    # 1. Output shape is (8, 2) and all values are finite
    assert coords1.shape == (8, 2)
    assert np.all(np.isfinite(coords1))

    # 2. Determinism: identical random_state=42 produces identical coordinates
    reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42,
    })
    coords2 = np.asarray(session.artifacts["reduced_tsne"])
    assert np.allclose(coords1, coords2)

    # 3. Exact duplicates overlap: doc 5 and doc 6 ('always gamma delta') map to the same point
    assert np.allclose(coords1[5], coords1[6], atol=1e-3)

    # 4. Cluster separability: average within-group distance < average between-group distance
    # catA non-empty docs: 0, 1, 2; catB docs: 3, 4, 5, 6
    cat_a_indices = [0, 1, 2]
    cat_b_indices = [3, 4, 5, 6]

    def avg_pairwise_dist(idx_list1, idx_list2):
        dists = []
        for i in idx_list1:
            for j in idx_list2:
                if i != j:
                    dists.append(np.linalg.norm(coords1[i] - coords1[j]))
        return np.mean(dists)

    dist_within_a = avg_pairwise_dist(cat_a_indices, cat_a_indices)
    dist_within_b = avg_pairwise_dist(cat_b_indices, cat_b_indices)
    dist_between = avg_pairwise_dist(cat_a_indices, cat_b_indices)

    avg_within = (dist_within_a + dist_within_b) / 2.0
    assert avg_within < dist_between

    # 5. Plot contract: session.pending_figure must be populated
    assert session.pending_figure is not None


def test_reduce_dimensions_tool_umap_fixture():
    # Fresh session for UMAP test
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

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer 9: method="umap", n_neighbors=3, random_state=42
    reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })

    # Verify session.artifacts['reduced_umap']
    assert "reduced_umap" in session.artifacts
    coords = session.artifacts["reduced_umap"]

    # Doc 0: [6.6811, -6.8218] using absolute tolerance 1e-3
    assert coords[0][0] == pytest.approx(6.6811, abs=1e-3)
    assert coords[0][1] == pytest.approx(-6.8218, abs=1e-3)

    # Doc 3: [10.2228, -6.2310] using absolute tolerance 1e-3
    assert coords[3][0] == pytest.approx(10.2228, abs=1e-3)
    assert coords[3][1] == pytest.approx(-6.2310, abs=1e-3)

    # Plot contract assertion: session.pending_figure must be populated
    assert session.pending_figure is not None


def test_binarize_labels_tool_fixture():
    # Fresh session for binarize_labels_tool
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_tool.invoke({})

    # 1. Summary checks: 8 documents, 2 classes, class order ['catA', 'catB']
    assert result["n_documents"] == 8
    assert result["n_classes"] == 2
    assert result["classes"] == ["catA", "catB"]

    # 2. Artifact checks: matrix has shape (8, 2)
    assert "binarized_labels" in session.artifacts
    artifact_data = session.artifacts["binarized_labels"]
    matrix = artifact_data["matrix"]
    assert matrix.shape == (8, 2)
    assert artifact_data["classes"] == ["catA", "catB"]

    # 3. Row 0 (catA) is [1, 0]
    assert list(matrix[0]) == [1, 0]

    # 4. Row 3 (catB) is [0, 1]
    assert list(matrix[3]) == [0, 1]

    # 5. Every row sums to exactly 1 (one-hot invariant)
    assert np.all(matrix.sum(axis=1) == 1)
