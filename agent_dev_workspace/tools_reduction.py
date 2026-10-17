from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for dimensionality reduction and label encoding."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduce document-term matrix dimensions to 2D and generate a scatter plot.

        Projects session.feature_matrix to 2 components using PCA, t-SNE, or UMAP.
        - For PCA and t-SNE: converts matrix to dense array to ensure full-SVD centering
          and deterministic PCA initialization.
        - For UMAP: operates directly on the sparse CSR matrix for efficiency.
        For every method, builds a 2D scatter plot colored by document category labels,
        assigns it to session.pending_figure, and closes the figure. Stores the full
        coordinates array in session.artifacts['reduced_<method>'].

        Args:
            method: Reduction algorithm ('pca', 'tsne', or 'umap'). Defaults to 'pca'.
            random_state: Seed for reproducible reductions across PCA, t-SNE, and UMAP.
                Defaults to 42.
            perplexity: Perplexity parameter for t-SNE. Defaults to 30.0.
            n_neighbors: Number of nearest neighbors for UMAP. Defaults to 15.

        Returns:
            A summary dictionary containing result_id, method, n_points,
            explained_variance_ratio (for PCA only), plot_attached flag,
            and coordinates_sample (first 20 rows of coordinates rounded to 4 decimals).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No document labels found in session. Please run build_dtm_tool first."}

        n_samples, n_features = session.feature_matrix.shape
        if n_samples == 0 or n_features == 0:
            return {"error": "Document-term matrix is empty."}

        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            # Densify for exact mean-centering matching Master; pass random_state for randomized solver on large data
            dense_x = session.feature_matrix.toarray()
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(dense_x)
            explained_variance_ratio = [
                round(float(v), 4) for v in reducer.explained_variance_ratio_
            ]

        elif method == "tsne":
            # Validate perplexity against sample count
            if perplexity >= n_samples:
                return {
                    "error": f"perplexity ({perplexity}) must be less than n_samples ({n_samples}). "
                    f"Please specify a smaller perplexity value."
                }
            # Densify to preserve deterministic init='pca' initialization
            dense_x = session.feature_matrix.toarray()
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(dense_x)

        elif method == "umap":
            # Validate n_neighbors against sample count
            if n_neighbors >= n_samples:
                return {
                    "error": f"n_neighbors ({n_neighbors}) must be less than n_samples ({n_samples}). "
                    f"Please specify a smaller n_neighbors value."
                }
            # Pass sparse matrix directly to UMAP for memory efficiency
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(session.feature_matrix)

        else:
            return {"error": f"Unsupported reduction method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store complete coordinates in session.artifacts
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts[f"reduced_{method}"] = coords

        # Build 2D scatter plot colored by document labels
        labels_list = [str(lbl) for lbl in session.labels]
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels_list,
            style=labels_list,
            palette="tab10",
            s=60,
            ax=ax,
        )
        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        # Set figure on session and close
        session.pending_figure = fig
        plt.close(fig)

        # Build compact sample of coordinates (up to first 20 rows)
        coords_sample = [
            [round(float(row[0]), 4), round(float(row[1]), 4)]
            for row in coords[:20]
        ]

        result_id = session.next_result_id(f"reduce_{method}")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "method": method,
            "n_points": int(n_samples),
            "plot_attached": True,
            "coordinates_sample": coords_sample,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode document category labels into a binary indicator matrix.

        Encodes labels from session.dataframe['category_name']. Guarantees an explicit
        column for every class by expanding 2-class binary outputs from LabelBinarizer
        into a 2-column matrix [1 - y, y] aligned with classes[0] and classes[1].
        Stores the binarized matrix and class ordering in session.artifacts['binarized_labels'].

        Returns:
            A summary dictionary containing result_id, n_documents, n_classes,
            classes (in column order), and sample_rows (first 3 rows with index,
            category_name, and vector of plain ints).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "category_name" not in df.columns:
            return {"error": "Column 'category_name' not found in session.dataframe."}

        labels = df["category_name"].fillna("unknown").astype(str).tolist()
        n_docs = len(labels)

        if n_docs == 0:
            return {"error": "Working dataset is empty."}

        lb = LabelBinarizer()
        raw_binarized = lb.fit_transform(labels)
        classes_list = [str(c) for c in lb.classes_]
        n_classes = len(classes_list)

        # Ensure one column per class for binary (2-class) datasets
        if n_classes == 2 and raw_binarized.shape[1] == 1:
            y = raw_binarized[:, 0]
            # Column 0 corresponds to classes[0], Column 1 to classes[1]
            binarized_matrix = np.column_stack([1 - y, y]).astype(int)
        elif n_classes == 1:
            # Single class edge case: all 1s
            binarized_matrix = np.ones((n_docs, 1), dtype=int)
        else:
            binarized_matrix = raw_binarized.astype(int)

        # Store in session.artifacts
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts["binarized_labels"] = {
            "matrix": binarized_matrix,
            "classes": classes_list,
        }

        # Build sample_rows for the first 3 documents
        sample_n = min(3, n_docs)
        sample_rows: List[Dict[str, Any]] = []
        for i in range(sample_n):
            row_idx = df.index[i]
            sample_rows.append({
                "index": int(row_idx) if isinstance(row_idx, (int, float, np.integer)) else str(row_idx),
                "category_name": str(labels[i]),
                "vector": [int(val) for val in binarized_matrix[i]],
            })

        result_id = session.next_result_id("binarize")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_documents": int(n_docs),
            "n_classes": int(n_classes),
            "classes": classes_list,
            "sample_rows": sample_rows,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
