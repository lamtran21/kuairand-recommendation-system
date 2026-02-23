# Recommendation System Design Suggestions (KuaiRand)

This document summarizes a practical roadmap to turn your current EDA outputs into a complete recommendation-system project.

## 1. Define the Task and Labels

- Task: **Top-N implicit recommendation** (recommend videos for each user).
- Start with `is_click` as the primary label.
- Add comparison targets: `long_view` and `is_like`.
- Use `log_standard_*` for training.
- Use `log_random_*` mainly for more robust/fair evaluation and debiasing discussion.

## 2. Build Three Baselines First

These are essential for project credibility and model comparison:

- **Popularity baseline**: globally popular videos (e.g., by click rate or long-view rate).
- **UserCF**: user-user collaborative filtering.
- **ItemCF**: item-item collaborative filtering (co-occurrence or cosine similarity).

These establish a clear "minimum bar" before advanced models.

## 3. Add a Matrix-Factorization CF Model (Core Model)

- Use **implicit ALS** or **BPR**.
- Build a sparse user-item interaction matrix.
- Suggested weighted signal:
  - `w = 1*is_click + 2*long_view + 3*is_like`
- This model usually outperforms simple CF baselines on sparse implicit-feedback data.

## 4. Use Side Features for a Hybrid Model

You already have good side information:

- User features: `user_active_degree`, `follow_user_num`, `fans_user_num`, `register_days`, `onehot_feat*`.
- Video features: `video_type`, `upload_type`, `video_duration`, and aggregated `video_features_statistic` metrics.

A good next step is **LightFM** (CF + user/item features), which is very suitable for course projects.

## 5. Evaluation Design (Most Important Part)

- Use a **time-based split** (train on earlier interactions, test on later interactions) to avoid leakage.
- Report ranking metrics:
  - `Recall@K`
  - `NDCG@K`
  - `HitRate@K`
  - `MAP@K`
- Report **cold-start subsets** separately:
  - new/low-activity users
  - new/low-exposure videos
- Add a "random-traffic" evaluation on `log_random` and discuss exposure bias.

## 6. Project Storyline for Final Report

You can structure your final narrative like this:

- EDA shows strong sparsity and long-tail behavior, motivating CF/matrix factorization.
- `is_click` is much denser than `is_like`, motivating multi-signal weighting.
- KuaiRand's random exposure log enables a stronger debiasing/evaluation discussion than many public datasets.

## 7. Suggested Implementation Order

1. Popularity baseline
2. ItemCF baseline
3. ALS (implicit feedback)
4. Hybrid model (LightFM)
5. Unified evaluation table + ablation study (single signal vs weighted multi-signal)

---

If needed, the next practical step is to implement one script that runs:

- `Popularity + ItemCF + ALS`
- time-based split
- `Recall@10` and `NDCG@10` comparison table

This will give you a strong, reproducible baseline package for your final project.
