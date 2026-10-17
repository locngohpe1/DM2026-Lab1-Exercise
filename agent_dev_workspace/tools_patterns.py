import locale
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth


def make_tools(session: Any) -> List[Any]:
    """Factory creating tools for frequent pattern mining."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns from a single category's documents.

        Constructs a fresh category-local document-term matrix with English stopwords
        removed. Filters vocabulary using the specified filtering_method:
        - 'variance': keeps the middle 90% (>= 5th and <= 95th percentile) of term variances.
        - 'term_frequency': keeps the middle 90% (>= 5th and <= 95th percentile) of total term occurrences.
        - 'tfidf': keeps the top 80% (>= 20th percentile) by mean TF-IDF score.
        Converts surviving terms into a transactional database via DF2DB and mines
        patterns using PAMI (FPGrowth, FAE, or MaxFPGrowth). Handles 0 surviving terms
        and 0 mined patterns cleanly without exceptions.

        Args:
            category_name: Target category label to mine.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of highest-support patterns to keep (required for 'topk').
            top_n: Number of mined patterns to return in the summary (clamped between 1 and 50).
                Defaults to 20.

        Returns:
            A summary dictionary containing result_id, category_name, filtering_method,
            algorithm, n_surviving_terms, n_transactions, n_patterns, patterns preview,
            and an optional note.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if not cat_col:
            return {"error": "No category column found in session.dataframe."}

        # Filter documents belonging to category_name
        cat_mask = df[cat_col].astype(str) == str(category_name)
        cat_df = df[cat_mask]
        if len(cat_df) == 0:
            available = sorted(list(df[cat_col].astype(str).unique()))
            return {
                "error": f"Category '{category_name}' not found in session.dataframe. "
                f"Available categories: {available}"
            }

        # Validate algorithm-specific parameters
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup < 1:
                return {"error": f"Parameter 'min_sup' (positive int) is required for algorithm '{algorithm}'."}
        elif algorithm == "topk":
            if k is None or k < 1:
                return {"error": "Parameter 'k' (positive int) is required for algorithm 'topk'."}
        else:
            return {"error": f"Unsupported algorithm '{algorithm}'."}

        # Step 1: Fresh category-local CountVectorizer with English stopwords removed
        cat_texts = cat_df["text"].fillna("").astype(str)
        local_vec = CountVectorizer(stop_words="english")
        try:
            local_dtm = local_vec.fit_transform(cat_texts)
        except ValueError:
            # Empty vocabulary after stopword removal
            local_dtm = None
            local_features = []
        else:
            local_features = local_vec.get_feature_names_out().tolist()

        if local_dtm is None or len(local_features) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_transactions": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        # Step 2: Apply the chosen filtering_method
        dense_local = local_dtm.toarray()
        surviving_terms: List[str] = []

        if filtering_method == "variance":
            # Population variance per column
            mean_c = dense_local.mean(axis=0)
            mean_sq = (dense_local ** 2).mean(axis=0)
            variances = np.maximum(mean_sq - (mean_c ** 2), 0.0)

            # Middle 90%: >= 5th and <= 95th percentile (matching text_filtering_helpers)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            keep_mask = (variances >= p5) & (variances <= p95)
            surviving_terms = [local_features[i] for i, keep in enumerate(keep_mask) if keep]

        elif filtering_method == "term_frequency":
            # Column sums
            col_sums = dense_local.sum(axis=0)
            p5 = np.percentile(col_sums, 5)
            p95 = np.percentile(col_sums, 95)
            keep_mask = (col_sums >= p5) & (col_sums <= p95)
            surviving_terms = [local_features[i] for i, keep in enumerate(keep_mask) if keep]

        elif filtering_method == "tfidf":
            # TfidfTransformer mean score per column
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(local_dtm).toarray()
            mean_tfidf = tfidf_matrix.mean(axis=0)

            # Top 80%: at or above 20th percentile
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20
            surviving_terms = [local_features[i] for i, keep in enumerate(keep_mask) if keep]

        else:
            return {"error": f"Unsupported filtering_method '{filtering_method}'."}

        # Step 3: Legitimate zero-terms-survived handling
        n_surviving = len(surviving_terms)
        if n_surviving == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": 0,
                "n_transactions": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        # Step 4: Build transactional database with DF2DB and mine with PAMI
        surviving_indices = [local_features.index(t) for t in surviving_terms]
        surviving_matrix = dense_local[:, surviving_indices]

        # Filter out rows with zero surviving items (empty transactions) before passing to DF2DB
        row_has_items = surviving_matrix.sum(axis=1) > 0
        filtered_surviving_matrix = surviving_matrix[row_has_items]
        n_transactions = int(len(filtered_surviving_matrix))

        if n_transactions == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_surviving_terms": n_surviving,
                "n_transactions": 0,
                "n_patterns": 0,
                "patterns": [],
                "note": "no non-empty transactions after filtering",
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        tx_df = pd.DataFrame(filtered_surviving_matrix, columns=surviving_terms)

        temp_dir = tempfile.mkdtemp(prefix="pami_tx_")
        tx_file_path = Path(temp_dir) / "transactions.tsv"
        tx_file_str = str(tx_file_path.resolve())

        try:
            # Match Master's DF2DB call exactly
            converter = DF2DB(tx_df)
            converter.convert2TransactionalDatabase(tx_file_str, ">=", 1)

            # Ensure UTF-8 encoding on Windows to prevent UnicodeDecodeError
            raw_bytes = tx_file_path.read_bytes()
            try:
                raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                system_enc = locale.getpreferredencoding(False)
                decoded_text = raw_bytes.decode(system_enc, errors="replace")
                tx_file_path.write_text(decoded_text, encoding="utf-8")

            # Run chosen PAMI miner with .mine()
            if algorithm == "fpgrowth":
                miner = FPGrowth(iFile=tx_file_str, minSup=min_sup, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()

            elif algorithm == "topk":
                miner = FAE(iFile=tx_file_str, k=k, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()

            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth(iFile=tx_file_str, minSup=min_sup, sep="\t")
                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()

            patterns_list: List[Dict[str, Any]] = []
            if patterns_df is not None and not patterns_df.empty:
                # PAMI returns columns like ['Patterns', 'Support']
                pat_col = [c for c in patterns_df.columns if "pattern" in c.lower()][0]
                sup_col = [c for c in patterns_df.columns if "sup" in c.lower()][0]

                for _, row in patterns_df.iterrows():
                    pat_val = str(row[pat_col]).strip()
                    # Split pattern string into a clean list of term tokens
                    pat_tokens = pat_val.split("\t") if "\t" in pat_val else pat_val.split()
                    patterns_list.append({
                        "pattern": pat_tokens,
                        "support": int(row[sup_col]),
                    })

                # Sort patterns descending by support, tie-breaking by pattern length and terms
                patterns_list.sort(key=lambda x: (-x["support"], len(x["pattern"]), x["pattern"]))

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Store in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts[artifact_key] = patterns_list

        clamped_top_n = max(1, min(top_n, 50))
        n_patterns = len(patterns_list)
        note = "no patterns found matching support criteria" if n_patterns == 0 else None

        result_id = session.next_result_id("patterns")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_surviving_terms": n_surviving,
            "n_transactions": n_transactions,
            "n_patterns": n_patterns,
            "patterns": patterns_list[:clamped_top_n],
        }
        if note:
            summary["note"] = note

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
                "top_n": top_n,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
