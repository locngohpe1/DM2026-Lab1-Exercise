from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for Document-Term Matrix (DTM) construction and analysis."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None,
        max_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the global document-term matrix from session.dataframe.

        Fits CountVectorizer on the 'text' column of session.dataframe and stores:
        - session.feature_matrix: the sparse CSR matrix (kept sparse for scalability).
        - session.feature_names: list of vocabulary terms aligned with columns.
        - session.artifacts['count_vectorizer']: the fitted CountVectorizer instance.
        - session.labels and session.categories: updated via session.set_labels(category_name).

        Args:
            ngram_range: Lower and upper boundary of n-gram range as [min_n, max_n].
                Defaults to [1, 1] (unigrams).
            stop_words: Stopword list or language string (e.g. 'english').
                Defaults to None (retaining all terms).
            max_features: Maximum vocabulary size by term frequency. Defaults to None.

        Returns:
            A summary dictionary containing result_id, n_documents, n_terms,
            non_zero count, total_elements, sparsity_pct, and feature_names_sample
            (up to 20 terms).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}
        if "category_name" not in df.columns:
            return {"error": "Column 'category_name' not found in session.dataframe."}

        # Convert backend-safe List[int] to tuple for CountVectorizer
        if len(ngram_range) != 2 or ngram_range[0] > ngram_range[1] or ngram_range[0] < 1:
            return {"error": f"Invalid ngram_range {ngram_range}; must be [min_n, max_n] with 1 <= min_n <= max_n."}
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        # Clean text series (handling missing/empty values gracefully)
        text_series = df["text"].fillna("").astype(str)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            stop_words=stop_words,
            max_features=max_features,
        )

        try:
            feature_matrix = vectorizer.fit_transform(text_series)
        except ValueError as e:
            # Handles edge cases such as empty vocabulary
            return {"error": f"CountVectorizer failed to build vocabulary: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()
        n_rows, n_cols = feature_matrix.shape

        # Keep feature_matrix strictly sparse
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        # Store fitted vectorizer in artifacts per spec contract
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts["count_vectorizer"] = vectorizer

        # Synchronize labels and categories using category_name
        session.set_labels(df["category_name"].to_numpy())

        # Compute exact sparsity metrics matching Master's specification
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_documents": int(n_rows),
            "n_terms": int(n_cols),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names_sample": feature_names[:20],
        }

        session.store_result(
            "build_dtm_tool",
            {"ngram_range": ngram_range, "stop_words": stop_words, "max_features": max_features},
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregate total term frequencies across the document-term matrix.

        Sums counts for each term across all documents using the sparse matrix
        directly without densification. Sorts terms descending by total frequency
        (with alphabetical tie-breaking) and passes the complete report as a
        DataFrame to session.store_result for visualization compatibility.

        Args:
            top_n: Number of top-frequency terms to return in the summary
                (clamped between 1 and 100). Defaults to 20.

        Returns:
            A summary dictionary containing result_id, total_terms,
            total_token_occurrences, and top_terms (a list of {term, count} records).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        # Sum along columns directly on the sparse matrix (shape: 1 x n_terms)
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        feature_names = session.feature_names

        # Build full report DataFrame: term in col 1, frequency in col 2
        report_df = pd.DataFrame({
            "term": feature_names,
            "frequency": [int(x) for x in col_sums],
        })

        # Sort by frequency descending, tie-break alphabetically ascending
        report_df = report_df.sort_values(
            by=["frequency", "term"],
            ascending=[False, True],
        ).reset_index(drop=True)

        clamped_top_n = max(1, min(top_n, 100, len(report_df)))
        top_slice = report_df.head(clamped_top_n)

        top_terms_list: List[Dict[str, Any]] = [
            {"term": str(row["term"]), "count": int(row["frequency"])}
            for _, row in top_slice.iterrows()
        ]

        total_occurrences = int(col_sums.sum())
        total_terms = len(feature_names)

        result_id = session.next_result_id("term_freq")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "total_terms": total_terms,
            "total_token_occurrences": total_occurrences,
            "top_terms": top_terms_list,
        }

        # Pass full_report with term in col 1, frequency in col 2, sorted
        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Render a heatmap of a positional slice of the raw document-term matrix.

        Takes a direct positional slice of session.feature_matrix[0:n_documents, 0:n_terms]
        without sorting or filtering, converts only that submatrix to dense, and generates
        an annotated heatmap set on session.pending_figure.

        Args:
            n_terms: Number of vocabulary columns to include from the start of
                session.feature_names (clamped between 1 and 50). Defaults to 20.
            n_documents: Number of document rows to include from the start of
                session.feature_matrix (clamped between 1 and 50). Defaults to 20.

        Returns:
            A summary dictionary containing result_id, requested and actual slice
            dimensions, terms included, document indices, and the sliced count matrix
            as a nested list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape

        if total_docs == 0 or total_terms == 0:
            return {"error": "Document-term matrix is empty."}

        # Clamp slice bounds between 1 and 50, and not exceeding matrix dimensions
        actual_docs = max(1, min(n_documents, 50, total_docs))
        actual_terms = max(1, min(n_terms, 50, total_terms))

        # Convert only the slice to dense array
        dense_slice = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        terms_included = [str(t) for t in session.feature_names[:actual_terms]]
        doc_indices = list(range(actual_docs))
        doc_labels = [f"Doc {i}" for i in doc_indices]

        # Build heatmap figure using explicit object-oriented API
        fig_width = max(6, min(14, actual_terms * 0.8))
        fig_height = max(4, min(12, actual_docs * 0.5))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_included,
            yticklabels=doc_labels,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap Slice ({actual_docs} docs x {actual_terms} terms)")
        ax.set_xlabel("Vocabulary Terms")
        ax.set_ylabel("Documents")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        # Set figure on session and close
        session.pending_figure = fig
        plt.close(fig)

        # Convert dense slice to nested list of standard Python ints
        matrix_list = [[int(val) for val in row] for row in dense_slice]

        result_id = session.next_result_id("dtm_heatmap")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "requested_n_documents": n_documents,
            "requested_n_terms": n_terms,
            "actual_n_documents": actual_docs,
            "actual_n_terms": actual_terms,
            "terms": terms_included,
            "document_indices": doc_indices,
            "matrix": matrix_list,
            "plot_attached": True,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
