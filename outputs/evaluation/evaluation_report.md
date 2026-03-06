# Recommendation Models Evaluation Report

## 1. Overview
This report provides a comprehensive comparison of all recommendation models evaluated in the pipeline.

### Executive Summary
- **Best overall ranking model (NDCG@50):** `two_stage_gbm` with a score of 0.1296.
- **Best recall model (Recall@50):** `two_stage_gbm` with a score of 0.3554.

## 2. Quantitative Comparison Table
The following table presents the full metrics across different cutoff points (K).

| model                |   recall@10 |   ndcg@10 |    mrr@10 |   coverage@10 |   recall@20 |   ndcg@20 |    mrr@20 |   coverage@20 |   recall@50 |   ndcg@50 |    mrr@50 |   coverage@50 |   recall@100 |   ndcg@100 |   mrr@100 |   coverage@100 |
|:---------------------|------------:|----------:|----------:|--------------:|------------:|----------:|----------:|--------------:|------------:|----------:|----------:|--------------:|-------------:|-----------:|----------:|---------------:|
| svd                  |   0.109502  | 0.0565249 | 0.0406475 |           856 |   0.174207  | 0.0728117 | 0.0450784 |          1106 |    0.302597 | 0.0981766 | 0.0491059 |          1617 |     0.427669 |   0.118424 | 0.0508778 |           2232 |
| two_stage_gbm        |   0.152192  | 0.0854811 | 0.065219  |          2023 |   0.224491  | 0.103708  | 0.0701944 |          2625 |    0.35537  | 0.129587  | 0.0743132 |          3628 |     0.47757  |   0.149429 | 0.0760658 |           4386 |
| popularity           |   0.0490715 | 0.0264258 | 0.0195266 |            41 |   0.0833387 | 0.0348682 | 0.0217286 |            65 |    0.135792 | 0.0453519 | 0.0234432 |           118 |     0.205284 |   0.056504 | 0.0243895 |            195 |
| two_stage_gbm_simple |   0.136047  | 0.0765238 | 0.058557  |          2241 |   0.199094  | 0.0923933 | 0.0628781 |          2897 |    0.323974 | 0.117038  | 0.0667765 |          3826 |     0.44962  |   0.137369 | 0.068553  |           4579 |
| als                  |   0.0857635 | 0.0418972 | 0.0287482 |          3102 |   0.149129  | 0.057771  | 0.0330276 |          3512 |    0.28473  | 0.0845697 | 0.037285  |          4123 |     0.425244 |   0.10732  | 0.0392766 |           4651 |
| itemcf               |   0.107843  | 0.0556645 | 0.0400135 |          2127 |   0.168975  | 0.0710435 | 0.0441936 |          2909 |    0.284347 | 0.0938309 | 0.0478081 |          4226 |     0.406292 |   0.113571 | 0.0495345 |           5341 |

## 3. Visual Analysis
Please refer to the following generated charts in this directory:
- `model_comparison_bar_plot.png`: Direct comparison of top models at K=50.
- `model_comparison_radar.png`: **[NEW]** Multidimensional polygon comparison across Normalized Recall, NDCG, MRR, and Coverage.
- `accuracy_vs_diversity_bubble.png`: **[NEW]** Scatter plot demonstrating the trade-off between recommending popular relevant items and diverse niche items.
- `recall_trend_plot.png`: Line chart showing how Recall scales as K increases (10 to 100).
- `ndcg_trend_plot.png`: Line chart showing how ranking quality (NDCG) behaves across different K values.

## 4. Relative Improvement over Baseline
Comparing all models against the fundamental Popularity baseline (percentage improvement):

| model                | recall@10   | ndcg@10   | mrr@10   | coverage@10   | recall@20   | ndcg@20   | mrr@20   | coverage@20   | recall@50   | ndcg@50   | mrr@50   | coverage@50   | recall@100   | ndcg@100   | mrr@100   | coverage@100   |
|:---------------------|:------------|:----------|:---------|:--------------|:------------|:----------|:---------|:--------------|:------------|:----------|:---------|:--------------|:-------------|:-----------|:----------|:---------------|
| svd                  | +123.15%    | +113.90%  | +108.17% | +1987.80%     | +109.04%    | +108.82%  | +107.46% | +1601.54%     | +122.84%    | +116.48%  | +109.47% | +1270.34%     | +108.33%     | +109.59%   | +108.61%  | +1044.62%      |
| two_stage_gbm        | +210.14%    | +223.48%  | +234.00% | +4834.15%     | +169.37%    | +197.43%  | +223.05% | +3938.46%     | +161.70%    | +185.74%  | +216.99% | +2974.58%     | +132.64%     | +164.46%   | +211.88%  | +2149.23%      |
| popularity           | +0.00%      | +0.00%    | +0.00%   | +0.00%        | +0.00%      | +0.00%    | +0.00%   | +0.00%        | +0.00%      | +0.00%    | +0.00%   | +0.00%        | +0.00%       | +0.00%     | +0.00%    | +0.00%         |
| two_stage_gbm_simple | +177.24%    | +189.58%  | +199.88% | +5365.85%     | +138.90%    | +164.98%  | +189.38% | +4356.92%     | +138.58%    | +158.07%  | +184.84% | +3142.37%     | +119.02%     | +143.11%   | +181.08%  | +2248.21%      |
| als                  | +74.77%     | +58.55%   | +47.23%  | +7465.85%     | +78.94%     | +65.68%   | +52.00%  | +5303.08%     | +109.68%    | +86.47%   | +59.04%  | +3394.07%     | +107.15%     | +89.93%    | +61.04%   | +2285.13%      |
| itemcf               | +119.77%    | +110.64%  | +104.92% | +5087.80%     | +102.76%    | +103.75%  | +103.39% | +4375.38%     | +109.40%    | +106.90%  | +103.93% | +3481.36%     | +97.92%      | +101.00%   | +103.10%  | +2638.97%      |

## 5. Conclusion & Insights
1. **Tree-based re-ranking** typically provides the highest performance improvements due to complex feature interactions.
2. **Matrix Factorization** (ALS/SVD) provides strong collaborative signals but differs widely in item coverage compared to ItemCF.
3. **Accuracy vs. Diversity Trade-off**: High scoring models might occasionally suffer from low coverage (popularity bias). Using the bubble chart helps determine the most balanced model.
