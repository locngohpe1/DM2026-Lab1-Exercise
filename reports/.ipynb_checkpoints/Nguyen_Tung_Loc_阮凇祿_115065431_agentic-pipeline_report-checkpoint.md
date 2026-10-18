# Agentic Pipeline Report: 20 Newsgroups, four categories
Full Name: Nguyen Tung Loc 阮凇祿
Student ID: 1150650431

## 1. What I did

In this notebook, I used the 18 tools I built in the AgentDev notebook alongside the two provided tools to guide the agent through the same workflow as the Master notebook. The pipeline covered loading the data, checking its quality, building the document-term matrix, filtering features, mining frequent patterns, reducing dimensions, and measuring document similarity. 

Rather than accepting the agent's summaries at face value, I checked every number it gave me against the raw tool outputs. Whenever possible, I also compared my results with the Master notebook to see whether my tools reproduced the baseline. Every result discussed below is cited by its `result_id` so that it can be traced directly in the session log.

## 2. Data and data quality

Running `load_dataset_1` returned 2,257 posts across the four categories:
* `soc.religion.christian`: 599 posts
* `sci.med`: 594 posts
* `comp.graphics`: 584 posts
* `alt.atheism`: 480 posts

These counts match the Master notebook exactly, ensuring that every later comparison with Master rests on the same underlying data. Quality checks with `check_missing_tool` (`missing_3`) found no empty or blank posts, and `check_duplicates_tool` (`duplicates_4`) found no duplicates, so I did not remove any records.

The length statistics in `describe_5` shaped how I interpreted many of the later results. Post lengths are strongly skewed to the right. The median post has 1,235 characters and 75% of the posts have at most 2,059 characters, but the longest post, located in `comp.graphics`, reaches 60,713 characters, which is almost 30 times the 75th percentile. The categories also differ noticeably in typical length. `comp.graphics` has the shortest posts (median 883 characters), whereas `soc.religion.christian` has the longest posts on average (mean 2,263, median 1,649). Following the Master notebook, I kept the long posts and treated this length skew as a structural limitation of the data.

A random sample of 1,000 posts drawn via `sample_6` (`random_state 1`) preserved the class balance well. The largest shifts were `comp.graphics` moving from 25.9% to 27.8% and `soc.religion.christian` moving from 26.5% to 24.8%, meaning no category lost much of its share.

When I inspected the first few posts (`inspect_2`), I noticed that every text still begins with its email headers, such as `From`, `Subject`, `Organization`, `Lines`, and occasionally an `NNTP` posting host line. One `soc.religion.christian` post also quotes an address at `athos.rutgers.edu`, the server through which that moderated newsgroup was posted. I raised this with the agent early on as a potential source of header leakage, where a model learns to recognize the mail server or header format instead of the actual topic. This concern turned out to matter across several later steps.

![Text length by category](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_describe_5.png)

## 3. Document term matrix

Running `build_dtm_tool` (`dtm_7`) produced a matrix of 2,257 posts by 35,788 terms, with 365,886 nonzero cells out of 80,773,516 total cells, corresponding to a sparsity of 99.547%. 

My tool keeps English stop words, whereas Master removed them and arrived at 35,482 terms, leaving my vocabulary with 306 extra columns. Those few hundred columns change the frequency profile considerably. In `term_freq_8`, 18 of the 20 most frequent terms are stop words, led by "the" with 29,825 occurrences. The remaining two are "from", which serves as both a stop word and the `From` header field, and "edu", which comes from university email addresses. Not one of the top 20 terms says anything about an actual topic.

The heatmap of the first 20 posts and the first 20 columns (`dtm_heatmap_9`) contains only 2 nonzero cells out of 400, matching the overall sparsity. Because the vocabulary is sorted alphabetically and digits precede letters, those initial columns represent number tokens such as `00` and `000`. As a result, this slice displays rare noise rather than meaningful words.

![Slice of the document term matrix](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_dtm_heatmap_9.png)

## 4. Feature selection

### Variance filter

Applying `variance_filter_tool` across the 35,788 terms kept 5,354 terms at a threshold of 0.01 (`variance_10`), and only 1,710 terms at a threshold of 0.05 (`variance_11`). The lowest variance among the removed terms was 0.000443, which is the exact variance of a word that appears once in a single post. Consequently, the removed group consists mostly of one-off words, typos, and numbers, including the number tokens seen in `dtm_heatmap_9`. Removing them is safe because a word seen in only one or two posts cannot help describe an entire category.

I chose the 0.01 threshold. By my own rough calculation, a word that appears once in each of `k` posts has a variance of about `k` divided by 2,257. A threshold of 0.01 therefore keeps words used in roughly 23 posts or more, while 0.05 requires roughly 119 posts. Because each category contains only 480 to 599 posts, many words that are typical of a single category fall between those two values and would be lost at 0.05. 

I also noticed that neither threshold removes a single stop word. Words like "the" (variance 690.27), "of", and "to" have by far the highest variance simply because longer posts contain more of them. The agent confirmed from the tool's docstring that this filter only records the kept terms and does not alter the matrix itself, so the correlation filters in the next step still ran on all 35,788 terms.

![Top terms by variance](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_turn80.png)

### Pearson and Spearman for comp.graphics

The two correlation filters returned very different top 10 lists for `comp.graphics`, sharing only a single word, "thanks". 

Pearson (`pearson_12`) was led by `thanks` (0.278), `keywords` (0.233), `looking` (0.214), and `vga` (0.211). Out of the top 10, only `vga`, `video`, `card`, and `algorithm` are genuine graphics words. Terms like `keywords`, `host`, and `nntp` come from header lines, while `thanks`, `hi`, and `looking` reflect the help-request style of this newsgroup. On the negative side, `re` and `writes` (0.257 and 0.234 in absolute value) mark replies and quoted text, which are common in long religious debates. The token `rutgers` also appeared on the negative side, providing direct evidence of the header leakage I noted at the start. 

By contrast, Spearman (`spearman_13`) returned a much more topical list led by `graphics` (0.521), `files` (0.317), `image` (0.309), and `3d` (0.306).

My first explanation was that Spearman only evaluates the 2,000 terms with the highest variance, meaning it might never see words like `vga` or `keywords`. I tested this hypothesis by running Spearman again on 5,354 terms (`spearman_14`), matching the count kept by `variance_10`. The top 10 list did not change at all, showing that my initial explanation was wrong and that the difference comes from the correlation measures themselves. 

Because the matrix is so sparse, most values are zero and tied, causing Spearman to behave almost like a correlation of presence or absence. It rewards words that appear across many graphics posts, even if they occur only once per post. Pearson instead rewards words with steady counts. I also had to correct the agent at this point because it claimed that a word repeated many times in a few posts would be pushed up by Pearson. In fact, for a word that appears in only one graphics post, Pearson `r` stays around 0.036 regardless of how many times it is repeated, because a large count raises the standard deviation just as much as the covariance. Words like `thanks`, `host`, and `nntp`, which typically appear about once per post, fit the Pearson list much better.

The two charts below compare the top 15 terms from each method side by side.

![Pearson top terms for comp.graphics](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_turn110.png)
![Spearman top terms for comp.graphics](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_turn106.png)

### Correlation between terms

Running `feature_correlation_matrix_tool` (`feature_corr_15`) computed the pairwise correlations among the 20 terms with the highest variance. Eighteen of them are stop words, and the other two are `jpeg` and `edu`. 

Every pair is positively correlated. The strongest pairs are `the` and `of` (0.944), `of` and `in` (0.911), and `the` and `to` (0.902). The weakest pairs both involve `edu`, scoring 0.092 against `not` and 0.094 against `that`. This pattern has a straightforward explanation. A longer post contains more of every stop word, so their counts rise together, whereas `edu` comes from the header and appears roughly the same number of times regardless of post length. In other words, post length acts as a hidden common cause behind these correlations. 

The term `jpeg` is also worth noting. It has the 11th highest variance in the entire matrix, yet it does not appear in the `comp.graphics` top 10 under either Pearson or Spearman, suggesting that its counts come from heavy repetition in a few very long posts rather than broad usage across many posts.

![Correlation between the top 20 terms by variance](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_feature_corr_15.png)

## 5. Frequent patterns

I first ran `mine_patterns_tool` with the same settings as Master, using term-frequency filtering, `FPGrowth`, and `min_sup` 9. It reproduced Master's numbers exactly in both cases I checked, returning 5,422 patterns for `sci.med` (`patterns_16`) and 16,971 patterns for `comp.graphics` (`patterns_19`). For me, this was the most reassuring result of the run because it confirmed that my tool follows the exact same steps as the reference notebook.

The `sci.med` patterns, however, are not medical. The top entries are generic single words such as `advance`, `easy`, and `mass`, each with a support of only 23 out of 594 posts, or about 3.9%. The reason lies in how the percentile filter works on sparse counts. Because most words in a category occur only once, the 5th percentile of term counts is 1, meaning the lower bound removes nothing. Consequently, the filter only drops the top 5% most frequent terms, which is precisely where core medical words like `patients` or `disease` sit. Running `MaxFPGrowth` with the same settings (`patterns_17`) kept 859 maximal patterns, or about 16% of the frequent set. This made the definition of a maximal pattern easy to see in action. Terms like `guest` and `internal` appeared in the top 20 of `patterns_16` but vanished from the maximal list because each of them is part of a larger frequent pattern.

The `topk` run on `comp.graphics` with `tfidf` filtering (`patterns_18`) produced the clearest example of header leakage in the entire session. All 20 top patterns consisted entirely of header tokens: `lines` and `subject` appeared in all 584 posts, `organization` in 560, `edu` in 320, `posting` in 288, and `host` in 276. At first this looked like a structural difference between categories, but that run had changed both the category and the filter at once. To compare them fairly, I ran `comp.graphics` again using the same term-frequency filter as `sci.med` (`patterns_19`). Its top patterns turned out to be just as generic, featuring words like `personally`, `recently`, `wondering`, and `appreciate`, which showed that the filter choice rather than the topic explained most of the contrast.

At the same time, `patterns_19` surfaced tokens that clearly belong to signatures and quoted lines, such as `disclaimer`, `opinions`, `tdawson`, and `1993apr19`. My hypothesis was that a few users repeat the exact same signature across many posts, causing every combination of its words to become a frequent pattern. Running `MaxFPGrowth` on `comp.graphics` (`patterns_20`) compressed the 16,971 frequent patterns down to only 517 maximal ones, which is 3% compared with 16% for `sci.med`. In addition, `tdawson`, `personally`, `rumours`, and `disclaimer` dropped out of the maximal list, confirming that each belongs to a larger pattern. Even so, I consider this only partial support for the hypothesis because the tool sorts patterns by support and I never saw the long patterns themselves. The agent initially called this strong support, and I asked it to correct that claim.

## 6. Dimensionality reduction, labels and similarity

In `PCA` (`reduce_pca_21`), the first component captured 65.94% of the variance and the second captured only 4.33%. The sample coordinates track post length almost perfectly:
* 609-character post: minus 16.2
* 686-character post: minus 17.0
* 966-character post: minus 8.9
* 1,004-character post: minus 6.8
* 3,200-character post: plus 15.8

Because the primary axis of the data reflects post length rather than topic, all four categories overlap heavily in the scatter plot. `UMAP` (`reduce_umap_22`, `random_state 42`) showed the exact same effect at the level of individual posts. The `comp.graphics` post of 686 characters sits right next to a `soc.religion.christian` post of 609 characters at a distance of about 0.9, whereas the two `comp.graphics` posts of 686 and 1,004 characters sit about 3.7 apart.

![PCA projection](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_reduce_pca_21.png)
![UMAP projection](../plots/Nguyen_Tung_Loc_阮凇祿_agentic-pipeline_plot_reduce_umap_22.png)

Running `binarize_labels_tool` (`binarize_23`) generated four binary indicator columns, one per category. For distance-based models such as KNN, this encoding is preferable to integer codes because integers artificially make certain categories appear closer to one another for no real reason.

The cosine similarity results told the same story. Two `comp.graphics` posts scored 0.353 (`cosine_24`, 17 shared terms), while a `comp.graphics` post compared against a `soc.religion.christian` post scored 0.340 (`cosine_25`, 21 shared terms). In both pairs, the shared terms consisted mostly of stop words and header words such as `from`, `lines`, and `organization`. When calculated on raw counts, cosine similarity mainly measures shared grammar and email formatting rather than shared subject matter.

## 7. What I learned and the limitations

The main lesson of this run is that raw counts containing stop words and email headers primarily measure two things unrelated to topic: how long a post is and what its header looks like. Document length drove the stop-word variances, the term-to-term correlations, the first PCA component, the UMAP layout, and the cosine similarity scores. Header leakage surfaced in Pearson through `host`, `nntp`, and `rutgers`, and appeared most starkly in `patterns_18`. Spearman was the one method that consistently identified topical words because it ignores how many times a word is repeated within a post.

If I were to run this analysis again, I would remove email headers and signatures before building the matrix, strip out stop words as Master did, and apply `tfidf` weighting with normalized rows, or cosine distance in UMAP, so that topic rather than length drives the results. I would also add an option to `mine_patterns_tool` to sort patterns by length so that the signature hypothesis could be checked directly.

Finally, I did not take the agent's explanations at face value. I corrected the agent on five specific points during the session:
1. When it claimed header words would get a `tfidf` score close to zero, overlooking that with `sklearn`'s smooth idf the minimum possible idf is 1.
2. When it listed "a" among the most frequent words, forgetting that single letters are not treated as tokens by default.
3. When it stated Master's sparsity and top words from memory without computing them.
4. When it tried to explain Pearson rankings through bursts of repeated counts in a few posts.
5. When it described a PCA point at index 12 that the tool never returned.

My own initial hypothesis about the Spearman variance cutoff also turned out to be wrong, and I kept it in this report because the test that disproved it was an important part of the analysis.