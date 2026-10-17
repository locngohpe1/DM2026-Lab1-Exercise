from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for data inspection and preprocessing."""

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing, empty, or whitespace-only text values in the working dataset.

        Inspects the 'text' column of session.dataframe for true nulls (NaN/None)
        as well as blank or whitespace-only strings. Does not drop any rows.

        Returns:
            A summary dictionary containing the result_id, total document count,
            number of missing documents, and a list of up to the first 20 row
            indices that contain missing text. If no dataset is loaded or the
            'text' column is missing, returns a dictionary with an 'error' key.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        # Flag true nulls, empty strings, and whitespace-only text
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_indices = df.index[is_missing].tolist()

        result_id = session.next_result_id("missing")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "total_documents": len(df),
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices[:20],
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for and optionally drop duplicate documents in session.dataframe.

        Compares duplicates based on 'text' and 'category_name' columns.
        When drop=True, this tool mutates session.dataframe in place using
        drop_duplicates(keep=False) to discard every copy of any duplicated
        document, and resets the DataFrame index. If a document-term matrix or
        count vectorizer was previously built, they are invalidated and cleared
        from session state to prevent stale downstream matrix dimensions.

        Args:
            drop: If True, removes all occurrences of duplicate rows from
                session.dataframe. Defaults to False (inspection only).

        Returns:
            A summary dictionary containing result_id, n_duplicates, n_rows_before,
            n_rows_after, at most the first 20 duplicate row indices, and whether
            downstream DTM state was cleared.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        subset_cols = [c for c in ["text", "category_name"] if c in df.columns]
        if not subset_cols:
            return {"error": "Neither 'text' nor 'category_name' column found in session.dataframe."}

        dup_mask = df.duplicated(subset=subset_cols, keep="first")
        duplicate_indices = df.index[dup_mask].tolist()
        n_duplicates = len(duplicate_indices)
        n_rows_before = len(df)
        dtm_cleared = False

        if drop:
            # Drop all copies of duplicated rows per Master's keep=False contract
            df_cleaned = df.drop_duplicates(subset=subset_cols, keep=False).reset_index(drop=True)
            session.dataframe = df_cleaned
            n_rows_after = len(df_cleaned)

            # Invalidate downstream DTM state if any was populated
            if (
                session.feature_matrix is not None
                or session.feature_names is not None
                or (session.artifacts and "count_vectorizer" in session.artifacts)
            ):
                session.feature_matrix = None
                session.feature_names = None
                session.labels = None
                if session.artifacts and "count_vectorizer" in session.artifacts:
                    del session.artifacts["count_vectorizer"]
                dtm_cleared = True
        else:
            n_rows_after = n_rows_before

        result_id = session.next_result_id("duplicates")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "n_rows_before": n_rows_before,
            "n_rows_after": n_rows_after,
            "duplicate_indices": duplicate_indices[:20],
            "dropped": drop,
            "dtm_cleared": dtm_cleared,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace_working_data: bool = False,
    ) -> Dict[str, Any]:
        """Subsample documents from session.dataframe.

        Draws a random sample of documents by count (n) or fraction (frac).
        By default (replace_working_data=False), session.dataframe is untouched
        and the sampled DataFrame is saved into session.artifacts['sample_df'].
        WARNING: If replace_working_data=True, session.dataframe is overwritten
        with the sample (index reset to 0..n-1) and any existing document-term
        matrix (DTM), feature names, labels, and count vectorizer are cleared
        to prevent downstream dimensional mismatches.

        Args:
            n: Number of documents to sample. Exactly one of n or frac must be given.
            frac: Fraction of documents to sample (0.0 to 1.0). Exactly one of
                n or frac must be given.
            random_state: Seed for reproducible random sampling.
            replace_working_data: If True, replaces session.dataframe with the
                sample, resets index, and clears existing DTM state. Defaults to False.

        Returns:
            A summary dictionary containing result_id, n_sampled, total_documents,
            original sampled row indices (capped at 20), category counts before
            and after sampling, replace_working_data flag, and whether DTM state
            was cleared.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe

        if (n is None and frac is None) or (n is not None and frac is not None):
            return {"error": "Must provide exactly one of 'n' or 'frac'."}

        if n is not None:
            if n < 0:
                return {"error": f"Sample size 'n' must be non-negative, got {n}."}
            if n > len(df):
                return {"error": f"Sample size n={n} is larger than dataset size ({len(df)} rows)."}

        if frac is not None:
            if frac < 0.0 or frac > 1.0:
                return {"error": f"Fraction 'frac' must be between 0.0 and 1.0, got {frac}."}

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Compute per-category distribution before and after
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        counts_before: Dict[str, int] = df[cat_col].value_counts().to_dict() if cat_col else {}
        counts_after: Dict[str, int] = sampled_df[cat_col].value_counts().to_dict() if cat_col else {}

        dtm_cleared = False
        if replace_working_data:
            session.dataframe = sampled_df.reset_index(drop=True)
            if (
                session.feature_matrix is not None
                or session.feature_names is not None
                or (session.artifacts and "count_vectorizer" in session.artifacts)
            ):
                session.feature_matrix = None
                session.feature_names = None
                session.labels = None
                if session.artifacts and "count_vectorizer" in session.artifacts:
                    del session.artifacts["count_vectorizer"]
                dtm_cleared = True
        else:
            if session.artifacts is None:
                session.artifacts = {}
            session.artifacts["sample_df"] = sampled_df

        result_id = session.next_result_id("sample")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_sampled": len(sampled_df),
            "total_documents": len(df),
            "sampled_indices": sampled_indices[:20],
            "counts_before": counts_before,
            "counts_after": counts_after,
            "replace_working_data": replace_working_data,
            "dtm_cleared": dtm_cleared,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state, "replace_working_data": replace_working_data},
            summary,
        )
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute text length statistics and generate a category box plot.

        Calculates document text length in characters (df['text'].apply(len))
        on a local copy of the working DataFrame without modifying session.dataframe.
        Produces summary statistics (.describe()) overall and per category_name.
        Generates a seaborn box plot of text_length by category and assigns it
        to session.pending_figure for notebook rendering.

        Returns:
            A summary dictionary containing result_id, overall descriptive statistics,
            and per-category descriptive statistics (count, mean, std, min, 25%,
            50%, 75%, max) rounded to 4 decimal places.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        # Work on a local series/copy without mutating session.dataframe
        text_series = df["text"].fillna("").astype(str)
        text_length = text_series.apply(len)

        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        def clean_stats(s: pd.Series) -> Dict[str, Optional[float]]:
            desc = s.describe()
            cleaned: Dict[str, Optional[float]] = {}
            for stat, val in desc.items():
                if pd.isna(val) or (isinstance(val, (int, float)) and math.isnan(val)):
                    cleaned[stat] = None
                else:
                    cleaned[stat] = round(float(val), 4)
            return cleaned

        overall_stats = clean_stats(text_length)

        by_category_stats: Dict[str, Dict[str, Optional[float]]] = {}
        if cat_col:
            for cat_val, group_indices in df.groupby(cat_col).groups.items():
                cat_name = str(cat_val)
                cat_lengths = text_length.loc[group_indices]
                by_category_stats[cat_name] = clean_stats(cat_lengths)

        # Build box plot on a new figure
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = pd.DataFrame({
            "text_length": text_length,
            "category": df[cat_col].astype(str) if cat_col else "all",
        })
        sns.boxplot(data=plot_df, x="category", y="text_length", ax=ax)
        ax.set_title("Document Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")

        # Set figure on session without calling savefig or show
        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "total_documents": len(df),
            "overall": overall_stats,
            "by_category": by_category_stats,
            "plot_attached": True,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        random_sample: bool = False,
        random_state: Optional[int] = None,
        max_chars: int = 200,
    ) -> Dict[str, Any]:
        """Inspect a sample of rows from the working dataset.

        Returns a concise view of rows without modifying session.dataframe,
        including row index, category label, full text character length,
        and truncated text preview.

        Args:
            n_rows: Number of rows to inspect (clamped between 1 and 20). Defaults to 5.
            random_sample: If True, draws random rows rather than the first rows. Defaults to False.
            random_state: Seed for reproducible random sampling when random_sample=True.
            max_chars: Maximum characters to display for text before truncating with '...'. Defaults to 200.

        Returns:
            A summary dictionary containing result_id, total_documents, column_names,
            and a list of inspected row dicts (index, category_name, text_length, text).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        total_docs = len(df)
        col_names = list(df.columns)

        if total_docs == 0:
            result_id = session.next_result_id("inspect")
            summary = {
                "result_id": result_id,
                "total_documents": 0,
                "column_names": col_names,
                "rows": [],
            }
            session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
            return summary

        # Clamp n_rows between 1 and 20, and never exceed total_docs
        clamped_n = max(1, min(n_rows, 20, total_docs))

        if random_sample:
            view_df = df.sample(n=clamped_n, random_state=random_state)
        else:
            view_df = df.head(clamped_n)

        rows: List[Dict[str, Any]] = []
        for idx, row in view_df.iterrows():
            # Fall back to category if category_name is absent
            cat_val = row.get("category_name")
            if cat_val is None or (isinstance(cat_val, float) and math.isnan(cat_val)):
                cat_val = row.get("category")
            category_str = "" if pd.isna(cat_val) else str(cat_val)

            raw_text = row.get("text")
            text_str = "" if pd.isna(raw_text) else str(raw_text)
            full_length = len(text_str)

            if full_length > max_chars:
                preview_text = text_str[:max_chars] + "..."
            else:
                preview_text = text_str

            rows.append({
                "index": int(idx) if isinstance(idx, (int, float)) else idx,
                "category_name": category_str,
                "text_length": full_length,
                "text": preview_text,
            })

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_documents": total_docs,
            "column_names": col_names,
            "rows": rows,
        }

        session.store_result(
            "inspect_data_tool",
            {"n_rows": n_rows, "random_sample": random_sample, "random_state": random_state, "max_chars": max_chars},
            summary,
        )
        return summary

    @tool
    def tokenize_tool(lowercase: bool = True) -> Dict[str, Any]:
        """Tokenize document text into unigrams using nltk.word_tokenize.

        Processes the 'text' column of session.dataframe and stores the resulting
        token lists into session.dataframe['unigrams']. Handles missing or empty
        text as an empty token list.

        Args:
            lowercase: If True, converts text to lowercase before tokenization.
                Defaults to True.

        Returns:
            A summary dictionary containing result_id, total_documents, total_tokens,
            vocab_size, and a sample of the first 10 tokens from the first document.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        # Check NLTK models once before the document processing loop
        try:
            nltk.word_tokenize("test")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.word_tokenize("test")
            except Exception:
                return {
                    "error": (
                        "NLTK punkt models not found and could not be downloaded automatically. "
                        "Please run nltk.download('punkt_tab') or ensure internet access."
                    )
                }

        token_lists: List[List[str]] = []
        unique_tokens = set()
        total_tokens = 0

        for text_val in df["text"]:
            if pd.isna(text_val):
                tokens: List[str] = []
            else:
                s = str(text_val)
                if not s.strip():
                    tokens = []
                else:
                    if lowercase:
                        s = s.lower()
                    tokens = nltk.word_tokenize(s)

            token_lists.append(tokens)
            total_tokens += len(tokens)
            unique_tokens.update(tokens)

        # Store only in session.dataframe['unigrams'] per spec
        session.dataframe["unigrams"] = token_lists

        sample_tokens = token_lists[0][:10] if token_lists else []

        result_id = session.next_result_id("tokenize")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "total_documents": len(df),
            "total_tokens": total_tokens,
            "vocab_size": len(unique_tokens),
            "sample_first_doc_tokens": sample_tokens,
        }

        session.store_result("tokenize_tool", {"lowercase": lowercase}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List directories and files in the repository under a given subdirectory.

        Guards against directory traversal (e.g., '../') by ensuring all resolved
        paths reside strictly within the repository root. Ignores hidden entries
        (starting with '.').

        Args:
            subdirectory: Path relative to repository root. Defaults to "" (root).

        Returns:
            A summary dictionary containing result_id, subdirectory, sorted lists
            of directories and files (each capped at 100 entries), and a truncated
            boolean flag.
        """
        repo_root = Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        # Enforce path containment within repo_root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {"error": f"Access denied: path '{subdirectory}' resolves outside repository root."}

        if not target_path.exists():
            return {"error": f"Path '{subdirectory}' does not exist."}
        if not target_path.is_dir():
            return {"error": f"Path '{subdirectory}' is a file, not a directory."}

        dirs: List[str] = []
        files: List[str] = []

        try:
            for item in target_path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    dirs.append(item.name)
                elif item.is_file():
                    files.append(item.name)
        except PermissionError:
            return {"error": f"Permission denied accessing directory '{subdirectory}'."}

        dirs.sort()
        files.sort()

        truncated = len(dirs) > 100 or len(files) > 100
        dirs_capped = dirs[:100]
        files_capped = files[:100]

        result_id = session.next_result_id("list_files")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": dirs_capped,
            "files": files_capped,
            "total_directories": len(dirs),
            "total_files": len(files),
            "truncated": truncated,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool,
        tokenize_tool,
        list_files_tool,
    ]
