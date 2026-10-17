from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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

    @tool
    def pearson_filter_tool(target_class: str, top_n: int = 10) -> Dict[str, Any]:
        """Compute Pearson correlation between term frequencies and a target class.

        Constructs a one-vs-rest binary target from session.labels and calculates
        Pearson r for each vocabulary term directly on session.feature_matrix
        using centered sparse matrix-vector products. For constant terms (zero variance),
        r is set to 0.0. Stores the sorted report in session.artifacts['pearson_{target_class}']
        and passes full_report (term as col 1, pearson_r as col 2, sorted by absolute r)
        to session.store_result for visualization.

        Args:
            target_class: The label to correlate against (e.g. 'catB').
            top_n: Number of top positive and negative terms to preview
                (clamped between 1 and 50). Defaults to 10.

        Returns:
            A summary dictionary containing result_id, target_class, total_terms,
            n_constant_terms, and top_positive and top_negative previews
            rounded to 4 decimal places.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No document labels found in session. Please run build_dtm_tool first."}

        labels = np.asarray(session.labels)
        available_categories = list(session.categories) if session.categories else sorted(list(set(labels)))

        if target_class not in available_categories:
            return {
                "error": f"Target class '{target_class}' not found in session labels. "
                f"Available classes: {available_categories}"
            }

        # Build one-vs-rest binary target vector
        y = (labels == target_class).astype(float)
        n_pos = int(np.sum(y))
        n_total = len(y)

        # Check for constant target (all documents or no documents belong to target_class)
        if n_pos == 0 or n_pos == n_total:
            return {
                "error": f"Target class '{target_class}' contains all documents or no documents "
                f"in session.labels; correlation is undefined for a constant target."
            }

        # Centered target and sum of squared deviations for y
        y_mean = np.mean(y)
        y_centered = y - y_mean
        ss_y = np.sum(y_centered ** 2)

        # Compute terms mean and sum of squared deviations on sparse feature_matrix
        feature_matrix = session.feature_matrix
        feature_names = session.feature_names

        # Numerator: sum (X_ij - X_bar_j) * (y_i - y_bar) = X^T * y_centered
        # Using sparse matrix-vector multiplication (feature_matrix is shape N x M)
        numerator = np.asarray(feature_matrix.T.dot(y_centered)).ravel()

        # Sum of squared deviations for each term: SS_X = sum (X_ij - X_bar_j)^2 = N * Var_pop(X)
        mean_x = np.asarray(feature_matrix.mean(axis=0)).ravel()
        mean_x_sq = np.asarray(feature_matrix.power(2).mean(axis=0)).ravel()
        var_x = np.maximum(mean_x_sq - (mean_x ** 2), 0.0)
        ss_x = n_total * var_x

        # Denominator: sqrt(ss_x * ss_y)
        denom = np.sqrt(ss_x * ss_y)

        # Compute r, assigning 0.0 to constant columns (where denom == 0)
        constant_mask = denom <= 1e-12
        n_constant_terms = int(np.sum(constant_mask))

        r_values = np.zeros_like(numerator)
        valid_mask = ~constant_mask
        r_values[valid_mask] = numerator[valid_mask] / denom[valid_mask]

        # Clip numerical inaccuracies to [-1.0, 1.0]
        r_values = np.clip(r_values, -1.0, 1.0)

        # Build full report DataFrame: col 1 = term, col 2 = pearson_r
        report_df = pd.DataFrame({
            "term": [str(t) for t in feature_names],
            "pearson_r": [float(r) for r in r_values],
            "abs_r": [float(abs(r)) for r in r_values],
        })

        # Sort by |r| descending, signed pearson_r descending, term ascending
        report_df = report_df.sort_values(
            by=["abs_r", "pearson_r", "term"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        # Drop helper column abs_r from the public full_report DataFrame
        public_report_df = report_df[["term", "pearson_r"]].copy()

        # Store full report in artifacts under pearson_{target_class}
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts[f"pearson_{target_class}"] = public_report_df

        clamped_top_n = max(1, min(top_n, 50))

        # Top positive terms (filtered for r > 0, sorted descending by r)
        pos_df = report_df[report_df["pearson_r"] > 0].sort_values(
            by=["pearson_r", "term"], ascending=[False, True]
        )
        top_positive = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 4)}
            for _, row in pos_df.head(clamped_top_n).iterrows()
        ]

        # Top negative terms (filtered for r < 0, sorted ascending by r / most negative first)
        neg_df = report_df[report_df["pearson_r"] < 0].sort_values(
            by=["pearson_r", "term"], ascending=[True, True]
        )
        top_negative = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 4)}
            for _, row in neg_df.head(clamped_top_n).iterrows()
        ]

        result_id = session.next_result_id("pearson")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": len(feature_names),
            "n_constant_terms": n_constant_terms,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

        # Store result with public_report_df (term col 1, pearson_r col 2)
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "top_n": top_n},
            summary,
            full_report=public_report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        top_n: int = 10,
        max_terms: Optional[int] = 2000,
    ) -> Dict[str, Any]:
        """Compute Spearman rank correlation between term frequencies and a target class.

        Constructs a one-vs-rest binary target from session.labels and calculates
        Spearman rank correlation using average ranks for ties. To optimize performance
        on large vocabularies, if max_terms is specified and smaller than the total
        vocabulary, evaluation is restricted to the top max_terms terms by population
        variance. Slicing uses CSC representation for fast column access without
        materializing a dense matrix. Constant columns are assigned rho = 0.0 and
        retained in the report (matching pearson_filter_tool). Stores the report in
        session.artifacts['spearman_{target_class}'] and passes full_report (term as
        col 1, spearman_r as col 2, sorted by |spearman_r| descending) to
        session.store_result for visualization.

        Args:
            target_class: The label to correlate against (e.g. 'catB').
            top_n: Number of top positive and negative terms to preview
                (clamped between 1 and 50). Defaults to 10.
            max_terms: Maximum number of highest-variance terms to evaluate.
                If None or greater than vocabulary size, evaluates all terms.
                Defaults to 2000.

        Returns:
            A summary dictionary containing result_id, target_class, total_terms
            (global vocabulary size), n_evaluated_terms (number of terms scored),
            n_constant_terms, and top_positive and top_negative previews
            rounded to 4 decimal places.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No document labels found in session. Please run build_dtm_tool first."}

        labels = np.asarray(session.labels)
        available_categories = list(session.categories) if session.categories else sorted(list(set(labels)))

        if target_class not in available_categories:
            return {
                "error": f"Target class '{target_class}' not found in session labels. "
                f"Available classes: {available_categories}"
            }

        # Build one-vs-rest binary target vector
        y = (labels == target_class).astype(float)
        n_pos = int(np.sum(y))
        n_total = len(y)

        # Check for constant target
        if n_pos == 0 or n_pos == n_total:
            return {
                "error": f"Target class '{target_class}' contains all documents or no documents "
                f"in session.labels; correlation is undefined for a constant target."
            }

        feature_matrix = session.feature_matrix
        feature_names = session.feature_names
        total_vocab_size = len(feature_names)

        # Compute term population variances using sparse matrix operations
        mean_x = np.asarray(feature_matrix.mean(axis=0)).ravel()
        mean_x_sq = np.asarray(feature_matrix.power(2).mean(axis=0)).ravel()
        variances = np.maximum(mean_x_sq - (mean_x ** 2), 0.0)

        # Determine indices of terms to evaluate based on max_terms
        if max_terms is not None and 0 < max_terms < total_vocab_size:
            # Select top max_terms by variance descending
            top_indices = np.argsort(-variances)[:max_terms]
            # Keep indices in sorted order for clean slicing
            eval_indices = np.sort(top_indices)
        else:
            eval_indices = np.arange(total_vocab_size)

        n_evaluated = len(eval_indices)
        eval_terms = [feature_names[i] for i in eval_indices]

        # Convert to CSC once for fast column slicing without densification
        csc_matrix = feature_matrix[:, eval_indices].tocsc()

        # Compute average ranks for target vector y
        rank_y = rankdata(y, method="average")
        rank_y_mean = np.mean(rank_y)
        rank_y_centered = rank_y - rank_y_mean
        ss_rank_y = np.sum(rank_y_centered ** 2)

        rho_values: List[float] = []
        n_constant_terms = 0

        for col_idx in range(n_evaluated):
            # Extract 1D column slice
            col = csc_matrix[:, col_idx].toarray().ravel()

            # Check for constant column (matches pearson_filter_tool: r=0.0 and kept in report)
            if np.all(col == col[0]):
                rho_values.append(0.0)
                n_constant_terms += 1
                continue

            rank_x = rankdata(col, method="average")
            rank_x_centered = rank_x - np.mean(rank_x)
            ss_rank_x = np.sum(rank_x_centered ** 2)

            denom = np.sqrt(ss_rank_x * ss_rank_y)
            if denom <= 1e-12:
                rho_values.append(0.0)
                n_constant_terms += 1
            else:
                rho = np.sum(rank_x_centered * rank_y_centered) / denom
                rho_clipped = float(np.clip(rho, -1.0, 1.0))
                rho_values.append(rho_clipped)

        # Build full report DataFrame: col 1 = term, col 2 = spearman_r
        report_df = pd.DataFrame({
            "term": [str(t) for t in eval_terms],
            "spearman_r": [float(r) for r in rho_values],
            "abs_r": [float(abs(r)) for r in rho_values],
        })

        # Sort by |r| descending, signed spearman_r descending, term ascending
        report_df = report_df.sort_values(
            by=["abs_r", "spearman_r", "term"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        public_report_df = report_df[["term", "spearman_r"]].copy()

        # Store full report in artifacts under spearman_{target_class}
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts[f"spearman_{target_class}"] = public_report_df

        clamped_top_n = max(1, min(top_n, 50))

        # Top positive terms (r > 0, sorted descending by r)
        pos_df = report_df[report_df["spearman_r"] > 0].sort_values(
            by=["spearman_r", "term"], ascending=[False, True]
        )
        top_positive = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 4)}
            for _, row in pos_df.head(clamped_top_n).iterrows()
        ]

        # Top negative terms (r < 0, sorted ascending by r / most negative first)
        neg_df = report_df[report_df["spearman_r"] < 0].sort_values(
            by=["spearman_r", "term"], ascending=[True, True]
        )
        top_negative = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 4)}
            for _, row in neg_df.head(clamped_top_n).iterrows()
        ]

        result_id = session.next_result_id("spearman")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": total_vocab_size,
            "n_evaluated_terms": n_evaluated,
            "n_constant_terms": n_constant_terms,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

        # Store result with public_report_df (term col 1, spearman_r col 2)
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "top_n": top_n, "max_terms": max_terms},
            summary,
            full_report=public_report_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
