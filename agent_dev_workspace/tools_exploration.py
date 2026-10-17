from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for data exploration and pairwise comparisons."""

    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Compute cosine similarity between two documents from the working dataset.

        Retrieves the two documents by positional index from session.dataframe
        and transforms their text using the CountVectorizer stored in
        session.artifacts['count_vectorizer']. Computes cosine similarity over
        raw counts. If either document has zero vector norm (e.g. empty text),
        returns a similarity of exactly 0.0.

        Args:
            doc_index_1: Positional row index of the first document (0-based).
            doc_index_2: Positional row index of the second document (0-based).

        Returns:
            A summary dictionary containing result_id, document indices,
            category labels, 80-character text previews, cosine_similarity
            rounded to 4 decimal places, total number of shared terms,
            and up to 10 shared term names.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        total_docs = len(df)

        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        # Validate vectorizer artifact
        if session.artifacts is None or "count_vectorizer" not in session.artifacts:
            return {
                "error": "No count vectorizer found in session.artifacts. "
                "Please run build_dtm_tool first."
            }
        vectorizer = session.artifacts["count_vectorizer"]

        # Validate positional indices
        if not (0 <= doc_index_1 < total_docs):
            return {"error": f"doc_index_1={doc_index_1} is out of range [0, {total_docs - 1}]."}
        if not (0 <= doc_index_2 < total_docs):
            return {"error": f"doc_index_2={doc_index_2} is out of range [0, {total_docs - 1}]."}

        row1 = df.iloc[doc_index_1]
        row2 = df.iloc[doc_index_2]

        raw_text1 = row1.get("text")
        raw_text2 = row2.get("text")

        text1 = "" if pd.isna(raw_text1) else str(raw_text1)
        text2 = "" if pd.isna(raw_text2) else str(raw_text2)

        # Vectorize both documents using the fitted CountVectorizer
        vectors = vectorizer.transform([text1, text2])
        v1 = vectors[0]
        v2 = vectors[1]

        # Compute L2 norms directly from sparse dot products
        norm1_sq = float(v1.dot(v1.T).toarray()[0, 0])
        norm2_sq = float(v2.dot(v2.T).toarray()[0, 0])

        norm1 = np.sqrt(norm1_sq)
        norm2 = np.sqrt(norm2_sq)

        # Handle zero-norm edge case (e.g. empty text document against any document)
        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
            shared_terms: List[str] = []
        else:
            dot_prod = float(v1.dot(v2.T).toarray()[0, 0])
            similarity = float(np.clip(dot_prod / (norm1 * norm2), 0.0, 1.0))

            # Identify shared terms (both counts > 0)
            feature_names = vectorizer.get_feature_names_out()
            v1_dense = v1.toarray().ravel()
            v2_dense = v2.toarray().ravel()
            shared_mask = (v1_dense > 0) & (v2_dense > 0)
            shared_terms = [str(term) for term in feature_names[shared_mask]]

        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        label1 = str(row1[cat_col]) if cat_col and pd.notna(row1[cat_col]) else "unknown"
        label2 = str(row2[cat_col]) if cat_col and pd.notna(row2[cat_col]) else "unknown"

        preview1 = text1[:80] + ("..." if len(text1) > 80 else "")
        preview2 = text2[:80] + ("..." if len(text2) > 80 else "")

        result_id = session.next_result_id("cosine")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "doc1": {
                "index": int(doc_index_1),
                "category_name": label1,
                "preview": preview1,
            },
            "doc2": {
                "index": int(doc_index_2),
                "category_name": label2,
                "preview": preview2,
            },
            "cosine_similarity": round(similarity, 4),
            "n_shared_terms": len(shared_terms),
            "shared_terms_sample": shared_terms[:10],
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(n_terms: int = 20) -> Dict[str, Any]:
        """Compute and plot a feature-vs-feature Pearson correlation matrix for top-variance terms.

        Identifies the top terms by population variance across the global DTM
        (capped at 20 terms per spec contract), computes their pairwise Pearson
        correlation matrix, renders an annotated heatmap assigned to
        session.pending_figure, and returns the matrix and term labels in the summary.

        Args:
            n_terms: Number of top-variance terms to correlate (clamped between 2 and 20).
                Defaults to 20.

        Returns:
            A summary dictionary containing result_id, n_terms, terms (ordered by
            variance descending), correlation_matrix (nested list of floats rounded
            to 4 decimals), and plot_attached flag.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape
        if total_docs == 0 or total_terms == 0:
            return {"error": "Document-term matrix is empty."}

        feature_matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute population variance (ddof=0) directly on sparse CSR matrix
        # Var(X) = E[X^2] - (E[X])^2
        mean_x = np.asarray(feature_matrix.mean(axis=0)).ravel()
        mean_x_sq = np.asarray(feature_matrix.power(2).mean(axis=0)).ravel()
        variances = np.maximum(mean_x_sq - (mean_x ** 2), 0.0)

        # Clamp n_terms between 2 and 20 (and bounded by total available vocabulary)
        k = max(2, min(n_terms, 20, total_terms))

        # Select top k terms by variance descending, tie-breaking by original index
        top_indices = np.argsort(-variances, kind="stable")[:k]
        top_terms = [str(feature_names[i]) for i in top_indices]

        # Extract only the top-variance columns and convert only that submatrix to dense
        submatrix = feature_matrix[:, top_indices].toarray()

        # Compute Pearson correlation matrix across columns (rowvar=False)
        corr_matrix = np.corrcoef(submatrix, rowvar=False)

        # Handle 1x1 or scalar return from np.corrcoef edge case
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])
        elif corr_matrix.ndim == 1:
            corr_matrix = corr_matrix.reshape(1, 1)

        # Replace any NaNs (from constant columns) with 0.0 for JSON serializability
        clean_corr = np.nan_to_num(corr_matrix, nan=0.0)

        # Build heatmap figure using explicit object-oriented API
        fig_size = max(6, min(14, k * 0.7))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(
            clean_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {k} Terms by Variance)")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        # Set figure on session and close
        session.pending_figure = fig
        plt.close(fig)

        # Convert matrix to nested list of rounded floats (strictly JSON-safe)
        matrix_list = [
            [round(float(val), 4) for val in row]
            for row in clean_corr
        ]

        result_id = session.next_result_id("feature_corr")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_terms": k,
            "terms": top_terms,
            "correlation_matrix": matrix_list,
            "plot_attached": True,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"n_terms": n_terms},
            summary,
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
