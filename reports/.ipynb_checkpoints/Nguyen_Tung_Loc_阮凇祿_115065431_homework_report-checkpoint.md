# Homework Report: Reddit Stock Sentiment
Full Name: Nguyen Tung Loc 阮凇祿
Student ID: 1150650431

## 1. Data and cleaning

Running `load_dataset_1` returned 847 posts across three sentiment labels:
* Neutral (0.0): 423 posts
* Negative (-1.0): 315 posts
* Positive (1.0): 109 posts (about 13% of the dataset)

Inspecting the raw text (`inspect_2`) revealed short, informal posts, several of which were not about stocks at all. It also showed a long promotional post at index 3 labelled as positive, which suggests that the dataset labels are fairly noisy. 

Quality checks via `missing_3` found no empty posts. However, `duplicates_4` reported 23 duplicate entries. Running `duplicates_5` with `keep=False` removed all 34 rows involved, which corresponded to 11 distinct texts that appeared more than once, leaving 813 posts in total. I noticed here that my tool's summary counts only the extra copies while the filter itself drops every copy, which is an inconsistency I should fix in a future version. All of the removed rows belonged to either the neutral class (dropping from 423 to 391) or the positive class (dropping from 109 to 107).

The length statistics in `describe_6` show a median post length of just 80 characters, compared with 1,235 characters on 20-newsgroups, with a minimum of 1 character and a maximum of 2,753 characters. Broken down by class, the median lengths are 96 characters for negative, 68 for neutral, and 81 for positive posts.

![Text length by label](../plots/Nguyen_Tung_Loc_阮凇祿_homework_plot_describe_6.png)

## 2. Matrix, frequency and variance

The document-term matrix from `dtm_7` has dimensions of 813 by 4,313 and a sparsity of 99.525%, which is almost identical to the 99.547% sparsity observed on 20-newsgroups. Although the individual posts are much shorter here, containing about 20.5 distinct terms per post instead of 162, the overall vocabulary shrank proportionally from 35,788 terms down to 4,313. As a result, each row still fills roughly 0.47% of the columns.

In `term_freq_8`, all 20 of the most frequent terms are stop words, and "the" alone accounts for 874 out of the 20,315 total tokens. Because Reddit posts do not contain email headers, the header leakage seen in 20-newsgroups is completely gone, yet no sentiment words appear in the top frequency list either. 

At a threshold of 0.01, `variance_9` kept only 490 of the 4,313 terms, or about 11%. Across 813 posts, a 0.01 variance cutoff means a word must appear in roughly 9 posts to survive. Because the positive class contains only 107 posts, that threshold would strip out most genuine positive sentiment words, so I chose not to rely on this filter when looking for sentiment features.

## 3. Correlation filters

For the positive class, `pearson_10` tops out at `r` = 0.156, which is far below the 0.28 correlation reached on `comp.graphics`. The tokens `huge`, `main`, and `tldr` all share the exact same score of 0.1563. That specific value corresponds to a word that appears in exactly 3 posts in the entire dataset, all of which happen to be positive. Consequently, the top positive features are simply rare words drawn from a handful of posts, likely the promotional ones, rather than generalizable sentiment markers.

For the negative class, `pearson_11` is led by `he` (0.176), `his`, `trump` (0.151), `china`, `economy`, and `trade`, alongside profanity. This indicates that the negative label largely tracks discussions of Trump and the trade war rather than pure sentiment vocabulary. Meanwhile, `x200b`, which is Reddit's zero-width-space character code, appears on the opposite side of the ranking as a formatting artifact.

The Spearman results in `spearman_12` are almost identical to `pearson_10`, producing the exact same tied values. On 20-newsgroups, the two correlation measures disagreed because long posts repeated certain words many times. Here, because the posts are so short, word counts are almost entirely 0 or 1, causing both Pearson and Spearman to reduce to a binary presence or absence correlation. With such a small sample size, neither list provides a reliable set of sentiment features.

![Pearson correlation with the negative class](../plots/Nguyen_Tung_Loc_阮凇祿_homework_plot_turn76.png)

## 4. Patterns

When mining frequent patterns on the positive class using TF-IDF filtering and Top-K (`patterns_13`), every single returned pattern was a single word, and the 20th pattern had a support of only 4 out of 107 posts. To maintain the same relative support threshold across the 315 negative posts, I scaled the minimum support by 315 divided by 107, which gave `min_sup` = 12 (`patterns_14`). That run also returned only single words, led by `trump` (45), `just` (42), and `china` (33).

Comparing the two outputs shows that exactly 10 of the top 20 words are shared between the positive and negative classes, including `market`, `economy`, and `tariffs`. This confirms that shared stock-market and macroeconomic topics dominate both groups. A few meaningful differences still emerge:
* The word `good` appears in 11.2% of positive posts compared with 4.1% of negative posts.
* Topic words like `trump` (14.3%) and `china` (10.5%) appear only in the negative top list.
* Contraction fragments such as `don` and `doesn` show up as markers of negation.

## 5. Reduction, similarity and conclusions

In `reduce_pca_15`, the first two principal components explain 26.7% and 5.1% of the variance. Although this is much lower than the 65.9% captured by the first component on 20-newsgroups, the first component here still tracks post length: a 16-character post sits at minus 1.78, whereas the 1,079-character promotional post sits at plus 9.67, leaving the sentiment labels heavily overlapped.

In `reduce_umap_16`, the three classes look noticeably more separated. Because no individual word correlates with any label above 0.18, the most plausible explanation is that UMAP's local neighborhoods group posts by narrow sub-topics (such as the trade war) and text formats (such as long promotional posts) that happen to align with the sentiment labels. This remains a plausible hypothesis, though I did not test it directly.

![PCA](../plots/Nguyen_Tung_Loc_阮凇祿_homework_plot_reduce_pca_15.png)
![UMAP](../plots/Nguyen_Tung_Loc_阮凇祿_homework_plot_reduce_umap_16.png)

Running `binarize_17` generated three one-hot indicator columns. However, because sentiment is naturally ordinal from negative to neutral to positive, one-hot encoding discards that ordering information. 

The cosine similarity checks showed the same stop-word dominance seen earlier. Two neutral posts scored 0.077 (`cosine_18`), sharing only the words "it" and "like", while a neutral and negative pair scored 0.086 (`cosine_19`). In short, shared stop words rather than shared sentiment drive the similarity scores.

Overall, while my tools ran without code changes on this dataset, default settings that worked well for long newsgroup posts (such as the 0.01 variance threshold, an unscaled absolute `min_sup`, and keeping stop words) do not transfer well to short social media text. To improve this pipeline for sentiment analysis, I would remove stop words, add bigrams to capture negations like "not good", apply TF-IDF weighting, and collect more positive examples to balance the classes. Finally, as in the previous run, I audited the agent's interpretations and corrected three specific errors: its claim that this matrix would be much sparser than 20-newsgroups, its statement that "more than half" of the top patterns were shared when the exact count was 10 out of 20, and its claim that the word `good` carries no sentiment signal.