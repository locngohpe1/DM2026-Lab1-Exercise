from typing import Any, Dict, List
import numpy as np
import pandas as pd
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for vocabulary filtering and feature selection."""

    @tool
    def variance_filter_tool(threshold: float = 0.0, top_n: int = 10) -> Dict[str, Any]:
        """Flag near-constant terms by variance for feature selection.

        Computes the population variance (ddof=0) for each vocabulary term directly
        on session.feature_matrix without densification. Keeps terms whose variance
        is strictly greater than threshold. Does not mutate session.feature_matrix,
        preserving vocabulary alignment for downstream tools and multiple threshold
        evaluations. Stores the list of kept term names in
        session.artifacts['variance_kept_terms'] and passes the complete sorted report
        to session.store_result for visualization.

        Args:
            threshold: Minimum variance cutoff. Terms with variance > threshold
                are kept. Defaults to 0.0 (removes strictly constant terms).
            top_n: Number of terms to preview in kept_terms and removed_terms
                summaries (clamped between 1 and 50). Defaults to 10.

        Returns:
            A summary dictionary containing result_id, threshold, total_terms,
            n_kept, n_removed, and previews of kept_terms (highest variance) and
            removed_terms (lowest variance) with variances rounded to 6 decimal places.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        feature_matrix = session.feature_matrix
        feature_names = session.feature_names

        # Compute population variance (ddof=0) directly on sparse CSR matrix
        # Var(X) = E[X^2] - (E[X])^2
        mean = np.asarray(feature_matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(feature_matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # Clip tiny negative numbers arising from floating-point inaccuracies
        variances = np.maximum(variances, 0.0)

        # Build full report DataFrame: col 1 = term, col 2 = variance
        report_df = pd.DataFrame({
            "term": [str(t) for t in feature_names],
            "variance": [float(v) for v in variances],
        })

        # Sort descending by variance, tie-break alphabetically by term
        report_df = report_df.sort_values(
            by=["variance", "term"],
            ascending=[False, True],
        ).reset_index(drop=True)

        # Strictly greater than threshold per VarianceThreshold contract
        mask_kept = report_df["variance"] > threshold
        df_kept = report_df[mask_kept]
        df_removed = report_df[~mask_kept]

        kept_terms_list = df_kept["term"].tolist()

        # Store kept terms in artifacts for downstream reference
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts["variance_kept_terms"] = kept_terms_list

        clamped_top_n = max(1, min(top_n, 50))

        # Preview: top_n highest variance among kept terms
        kept_preview = [
            {"term": row["term"], "variance": round(float(row["variance"]), 6)}
            for _, row in df_kept.head(clamped_top_n).iterrows()
        ]

        # Preview: top_n lowest variance among removed terms (sorted ascending by variance)
        removed_sorted_asc = df_removed.sort_values(
            by=["variance", "term"],
            ascending=[True, True],
        )
        removed_preview = [
            {"term": row["term"], "variance": round(float(row["variance"]), 6)}
            for _, row in removed_sorted_asc.head(clamped_top_n).iterrows()
        ]

        result_id = session.next_result_id("variance")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "threshold": float(threshold),
            "total_terms": len(feature_names),
            "n_kept": len(df_kept),
            "n_removed": len(df_removed),
            "kept_terms": kept_preview,
            "removed_terms": removed_preview,
        }

        # Pass full_report with term as col 1 and variance as col 2, sorted
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold, "top_n": top_n},
            summary,
            full_report=report_df,
        )
        return summary

    return [variance_filter_tool]
