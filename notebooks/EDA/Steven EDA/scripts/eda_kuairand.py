"""
Exploratory Data Analysis for the KuaiRand dataset.

This script is designed for early-stage recommendation system projects.
It creates data quality reports, correlation analyses, and a set of plots
that are useful before modeling.

Usage:
    py scripts/eda_kuairand.py
or:
    python scripts/eda_kuairand.py --data-dir Data/data --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns


# Keep plotting style consistent across all exported figures.
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120


def resolve_data_dir(explicit_dir: str | None = None) -> Path:
    """
    Resolve dataset directory robustly because the folder name can appear as
    "Data/data" or "data/data" on different machines.
    """
    script_dir = Path(__file__).resolve().parent
    project_like_roots = [
        Path("."),
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        script_dir.parent.parent.parent,
        script_dir.parent.parent.parent.parent,
    ]

    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))

    for root in project_like_roots:
        candidates.extend(
            [
                root / "Data/data",
                root / "data/data",
                root / "Data",
                root / "data",
            ]
        )

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(
        "Could not locate dataset directory. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output subdirectories for figures and tabular reports."""
    figures = output_dir / "figures"
    reports = output_dir / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, "reports": reports}


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all core KuaiRand tables.

    Dtype hints are added for large log tables to reduce memory footprint and
    speed up parsing. These choices are conservative for EDA tasks.
    """
    log_cols_int8 = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
        "tab",
    ]
    log_dtype = {
        "user_id": "int32",
        "video_id": "int32",
        "date": "int32",
        "hourmin": "int32",
        "time_ms": "int64",
        "play_time_ms": "float32",
        "duration_ms": "float32",
        "profile_stay_time": "float32",
        "comment_stay_time": "float32",
    }
    for c in log_cols_int8:
        log_dtype[c] = "int8"

    logs = {
        "log_standard_4_08_to_4_21": pd.read_csv(
            data_dir / "log_standard_4_08_to_4_21_pure.csv", dtype=log_dtype
        ),
        "log_standard_4_22_to_5_08": pd.read_csv(
            data_dir / "log_standard_4_22_to_5_08_pure.csv", dtype=log_dtype
        ),
        "log_random_4_22_to_5_08": pd.read_csv(
            data_dir / "log_random_4_22_to_5_08_pure.csv", dtype=log_dtype
        ),
    }

    for name, df in logs.items():
        df["source_table"] = name

    # Union all interaction logs into one table for global behavioral analysis.
    log_all = pd.concat(logs.values(), axis=0, ignore_index=True)

    user_features = pd.read_csv(data_dir / "user_features_pure.csv")
    video_basic = pd.read_csv(data_dir / "video_features_basic_pure.csv")
    video_stats = pd.read_csv(data_dir / "video_features_statistic_pure.csv")

    return {
        "log_all": log_all,
        "user_features": user_features,
        "video_basic": video_basic,
        "video_stats": video_stats,
    }


def dataset_overview(dfs: Dict[str, pd.DataFrame], reports_dir: Path) -> pd.DataFrame:
    """Create a high-level table with rows, columns, and approximate memory."""
    rows = []
    for name, df in dfs.items():
        rows.append(
            {
                "table": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            }
        )
    out = pd.DataFrame(rows).sort_values("rows", ascending=False)
    out.to_csv(reports_dir / "dataset_overview.csv", index=False)
    return out


def data_quality_report(df: pd.DataFrame, table_name: str, reports_dir: Path) -> pd.DataFrame:
    """
    Build a column-level quality report with:
    - data type
    - missing count and rate
    - unique count
    - simple numeric summary
    """
    q = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    q["min"] = np.nan
    q["max"] = np.nan
    q["mean"] = np.nan
    q["std"] = np.nan

    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().T
        for c in numeric_cols:
            q.loc[q["column"] == c, "min"] = desc.loc[c, "min"]
            q.loc[q["column"] == c, "max"] = desc.loc[c, "max"]
            q.loc[q["column"] == c, "mean"] = desc.loc[c, "mean"]
            q.loc[q["column"] == c, "std"] = desc.loc[c, "std"]

    q.sort_values(["missing_rate", "n_unique"], ascending=[False, False], inplace=True)
    q.to_csv(reports_dir / f"quality_{table_name}.csv", index=False)

    summary = {
        "table": table_name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_row_rate": float(df.duplicated().mean()),
    }
    pd.DataFrame([summary]).to_csv(reports_dir / f"quality_summary_{table_name}.csv", index=False)

    return q


def parse_time_columns(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date/hour columns to rich timestamp fields for temporal analysis.

    The raw dataset provides integer date and hour-minute fields.
    """
    out = log_df.copy()
    out["date_dt"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
    out["hour"] = (out["hourmin"] // 100).astype("Int64")

    # Build a rough event timestamp by combining date and hour-minute.
    hhmm = out["hourmin"].astype(str).str.zfill(4)
    out["event_dt"] = pd.to_datetime(
        out["date"].astype(str) + hhmm,
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    return out


def analyze_log_table(log_df: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    """Generate behavior, temporal, and engagement plots from interaction logs."""
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    action_cols = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
    ]

    # 1) Global action rate distribution.
    action_rate = (
        log_df[action_cols].mean().sort_values(ascending=False).rename("rate").reset_index()
    )
    action_rate.columns = ["action", "rate"]
    action_rate.to_csv(rpt_dir / "log_action_rates.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=action_rate, x="action", y="rate", hue="action", legend=False, palette="viridis")
    plt.title("Global Positive Action Rate")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean (Probability)")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_action_rate_bar.png")
    plt.close()

    # 2) Daily exposure/click volume trend.
    daily = (
        log_df.groupby("date_dt", dropna=True)
        .agg(
            impressions=("video_id", "size"),
            clicks=("is_click", "sum"),
            likes=("is_like", "sum"),
            long_views=("long_view", "sum"),
        )
        .reset_index()
        .sort_values("date_dt")
    )
    daily["ctr"] = daily["clicks"] / daily["impressions"].clip(lower=1)
    daily.to_csv(rpt_dir / "log_daily_metrics.csv", index=False)

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=daily, x="date_dt", y="impressions", label="impressions")
    sns.lineplot(data=daily, x="date_dt", y="clicks", label="clicks")
    sns.lineplot(data=daily, x="date_dt", y="likes", label="likes")
    plt.title("Daily Interaction Volume")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_daily_volume.png")
    plt.close()

    plt.figure(figsize=(13, 5))
    sns.lineplot(data=daily, x="date_dt", y="ctr", marker="o", color="#d95f02")
    plt.title("Daily Click-Through Rate (CTR)")
    plt.xlabel("Date")
    plt.ylabel("CTR")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_daily_ctr.png")
    plt.close()

    # 3) Hour-level activity and CTR.
    hourly = (
        log_df.groupby("hour", dropna=True)
        .agg(impressions=("video_id", "size"), clicks=("is_click", "sum"))
        .reset_index()
        .sort_values("hour")
    )
    hourly["ctr"] = hourly["clicks"] / hourly["impressions"].clip(lower=1)
    hourly.to_csv(rpt_dir / "log_hourly_metrics.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=hourly, x="hour", y="impressions", color="#80b1d3", ax=ax1)
    ax1.set_ylabel("Impressions")
    ax1.set_xlabel("Hour of Day")

    ax2 = ax1.twinx()
    sns.lineplot(data=hourly, x="hour", y="ctr", marker="o", color="#fb8072", ax=ax2)
    ax2.set_ylabel("CTR")
    plt.title("Hourly Impressions and CTR")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_hourly_impressions_ctr.png")
    plt.close()

    # 4) Relationship between play time and video duration.
    sampled = log_df[["play_time_ms", "duration_ms", "is_click", "long_view"]].dropna()
    if len(sampled) > 300_000:
        sampled = sampled.sample(300_000, random_state=42)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=sampled,
        x="duration_ms",
        y="play_time_ms",
        hue="is_click",
        size="long_view",
        alpha=0.2,
        palette="Set1",
        sizes=(10, 30),
    )
    plt.title("Play Time vs Duration")
    plt.xlabel("Video Duration (ms)")
    plt.ylabel("Play Time (ms)")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_playtime_vs_duration_scatter.png")
    plt.close()

    # 5) Correlation among key numeric behavior features.
    corr_cols = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
    ]
    corr = log_df[corr_cols].corr(numeric_only=True)
    corr.to_csv(rpt_dir / "log_correlation_matrix.csv")

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Log Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_correlation_heatmap.png")
    plt.close()

    # 6) User and item interaction count distribution (sparsity/cold-start signal).
    user_interactions = log_df.groupby("user_id").size().rename("n_interactions")
    item_interactions = log_df.groupby("video_id").size().rename("n_interactions")

    pd.DataFrame({"user_id": user_interactions.index, "n_interactions": user_interactions.values}).to_csv(
        rpt_dir / "user_interaction_counts.csv", index=False
    )
    pd.DataFrame({"video_id": item_interactions.index, "n_interactions": item_interactions.values}).to_csv(
        rpt_dir / "item_interaction_counts.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(np.log1p(user_interactions.values), bins=60, ax=axes[0], color="#66c2a5")
    axes[0].set_title("User Interaction Count (log1p)")
    axes[0].set_xlabel("log1p(count)")

    sns.histplot(np.log1p(item_interactions.values), bins=60, ax=axes[1], color="#fc8d62")
    axes[1].set_title("Video Interaction Count (log1p)")
    axes[1].set_xlabel("log1p(count)")

    plt.tight_layout()
    plt.savefig(fig_dir / "log_user_item_sparsity_hist.png")
    plt.close()

    # 7) Missing-data matrix for a sampled subset.
    miss_sample = log_df.sample(n=min(5_000, len(log_df)), random_state=42)
    plt.figure(figsize=(14, 5))
    msno.matrix(miss_sample, sparkline=False)
    plt.title("Missingness Matrix (Log Sample)")
    plt.tight_layout()
    plt.savefig(fig_dir / "log_missingness_matrix.png")
    plt.close()


def analyze_log_source_comparison(log_df: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    """
    Compare the three log datasets directly:
    - random intervention log
    - standard recommendation logs in two time windows
    """
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    sources = sorted(log_df["source_table"].dropna().unique().tolist())
    if len(sources) < 2:
        return

    # 1) Basic dataset summary per source.
    summary = (
        log_df.groupby("source_table")
        .agg(
            rows=("video_id", "size"),
            unique_users=("user_id", "nunique"),
            unique_items=("video_id", "nunique"),
            ctr=("is_click", "mean"),
            ltr=("is_like", "mean"),
            lvtr=("long_view", "mean"),
            follow_rate=("is_follow", "mean"),
            comment_rate=("is_comment", "mean"),
            forward_rate=("is_forward", "mean"),
            hate_rate=("is_hate", "mean"),
            avg_play_time_ms=("play_time_ms", "mean"),
            avg_duration_ms=("duration_ms", "mean"),
        )
        .reset_index()
    )
    summary["rows_per_user"] = summary["rows"] / summary["unique_users"].clip(lower=1)
    summary["rows_per_item"] = summary["rows"] / summary["unique_items"].clip(lower=1)
    summary.to_csv(rpt_dir / "compare_logs_summary.csv", index=False)

    # 2) Action rate heatmap by source.
    rate_cols = ["ctr", "ltr", "lvtr", "follow_rate", "comment_rate", "forward_rate", "hate_rate"]
    rate_df = summary[["source_table"] + rate_cols].set_index("source_table")
    plt.figure(figsize=(10, 5))
    sns.heatmap(rate_df, annot=True, fmt=".4f", cmap="YlOrRd")
    plt.title("Action Rates by Log Source")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_action_rate_heatmap.png")
    plt.close()

    # 3) Daily metrics by source.
    daily = (
        log_df.groupby(["source_table", "date_dt"], dropna=True)
        .agg(
            impressions=("video_id", "size"),
            ctr=("is_click", "mean"),
            ltr=("is_like", "mean"),
            lvtr=("long_view", "mean"),
        )
        .reset_index()
        .sort_values(["source_table", "date_dt"])
    )
    daily.to_csv(rpt_dir / "compare_logs_daily_metrics.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    sns.lineplot(data=daily, x="date_dt", y="ctr", hue="source_table", marker="o", ax=axes[0])
    axes[0].set_title("Daily CTR by Log Source")
    axes[0].set_ylabel("CTR")
    sns.lineplot(data=daily, x="date_dt", y="lvtr", hue="source_table", marker="o", ax=axes[1])
    axes[1].set_title("Daily Long View Rate by Log Source")
    axes[1].set_ylabel("Long View Rate")
    sns.lineplot(data=daily, x="date_dt", y="ltr", hue="source_table", marker="o", ax=axes[2])
    axes[2].set_title("Daily Like Rate by Log Source")
    axes[2].set_ylabel("Like Rate")
    axes[2].set_xlabel("Date")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_daily_rate_lines.png")
    plt.close()

    # 4) Hourly CTR by source.
    hourly = (
        log_df.groupby(["source_table", "hour"], dropna=True)
        .agg(impressions=("video_id", "size"), ctr=("is_click", "mean"), lvtr=("long_view", "mean"))
        .reset_index()
        .sort_values(["source_table", "hour"])
    )
    hourly.to_csv(rpt_dir / "compare_logs_hourly_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    sns.lineplot(data=hourly, x="hour", y="ctr", hue="source_table", marker="o", ax=axes[0])
    axes[0].set_title("Hourly CTR by Log Source")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("CTR")
    sns.lineplot(data=hourly, x="hour", y="lvtr", hue="source_table", marker="o", ax=axes[1])
    axes[1].set_title("Hourly Long View Rate by Log Source")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Long View Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_hourly_rate_lines.png")
    plt.close()

    # 5) Tab distribution and tab-level CTR by source.
    tab_stats = (
        log_df.groupby(["source_table", "tab"])
        .agg(impressions=("video_id", "size"), ctr=("is_click", "mean"), lvtr=("long_view", "mean"))
        .reset_index()
    )
    tab_stats.to_csv(rpt_dir / "compare_logs_tab_metrics.csv", index=False)

    tab_pivot = tab_stats.pivot(index="source_table", columns="tab", values="impressions").fillna(0)
    tab_share = tab_pivot.div(tab_pivot.sum(axis=1), axis=0)
    tab_share.to_csv(rpt_dir / "compare_logs_tab_share_matrix.csv")

    plt.figure(figsize=(12, 5))
    sns.heatmap(tab_share, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Tab Share by Log Source")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_tab_share_heatmap.png")
    plt.close()

    tab_ctr = tab_stats.pivot(index="source_table", columns="tab", values="ctr")
    tab_ctr.to_csv(rpt_dir / "compare_logs_tab_ctr_matrix.csv")
    plt.figure(figsize=(12, 5))
    sns.heatmap(tab_ctr, annot=True, fmt=".3f", cmap="RdYlGn")
    plt.title("Tab CTR by Log Source")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_tab_ctr_heatmap.png")
    plt.close()

    # 6) User and item overlap across logs (count + Jaccard).
    user_sets = {s: set(log_df.loc[log_df["source_table"] == s, "user_id"].unique()) for s in sources}
    item_sets = {s: set(log_df.loc[log_df["source_table"] == s, "video_id"].unique()) for s in sources}

    overlap_rows = []
    for a in sources:
        for b in sources:
            u_int = len(user_sets[a].intersection(user_sets[b]))
            u_uni = len(user_sets[a].union(user_sets[b]))
            i_int = len(item_sets[a].intersection(item_sets[b]))
            i_uni = len(item_sets[a].union(item_sets[b]))
            overlap_rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "user_overlap_count": u_int,
                    "user_jaccard": (u_int / u_uni) if u_uni else np.nan,
                    "item_overlap_count": i_int,
                    "item_jaccard": (i_int / i_uni) if i_uni else np.nan,
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(rpt_dir / "compare_logs_overlap_matrix.csv", index=False)

    user_j = overlap.pivot(index="source_a", columns="source_b", values="user_jaccard")
    item_j = overlap.pivot(index="source_a", columns="source_b", values="item_jaccard")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(user_j, annot=True, fmt=".3f", cmap="Purples", ax=axes[0])
    axes[0].set_title("User Jaccard Overlap")
    sns.heatmap(item_j, annot=True, fmt=".3f", cmap="Greens", ax=axes[1])
    axes[1].set_title("Item Jaccard Overlap")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_overlap_heatmaps.png")
    plt.close()

    # 7) Popularity rank correlation across logs.
    pop = (
        log_df.groupby(["source_table", "video_id"])
        .size()
        .rename("impressions")
        .reset_index()
    )
    pop_wide = pop.pivot(index="video_id", columns="source_table", values="impressions").fillna(0)
    rankcorr = pop_wide.corr(method="spearman")
    rankcorr.to_csv(rpt_dir / "compare_logs_item_pop_rankcorr.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(rankcorr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Item Popularity Rank Correlation (Spearman)")
    plt.tight_layout()
    plt.savefig(fig_dir / "compare_logs_item_rankcorr_heatmap.png")
    plt.close()


def analyze_user_features(user_df: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    """Analyze user profile table and create useful visual summaries."""
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    # Distribution of user activity labels.
    if "user_active_degree" in user_df.columns:
        deg = (
            user_df["user_active_degree"].value_counts(dropna=False).rename_axis("user_active_degree").reset_index(name="count")
        )
        deg.to_csv(rpt_dir / "user_active_degree_counts.csv", index=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=deg, x="user_active_degree", y="count", hue="user_active_degree", legend=False, palette="mako")
        plt.title("User Active Degree Distribution")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "user_active_degree_distribution.png")
        plt.close()

    # Correlation among numeric user features.
    user_num = user_df.select_dtypes(include=[np.number])
    if user_num.shape[1] >= 2:
        corr = user_num.corr(numeric_only=True)
        corr.to_csv(rpt_dir / "user_numeric_correlation_matrix.csv")

        # Show only strongest relationships for readability in large one-hot spaces.
        plt.figure(figsize=(13, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("User Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(fig_dir / "user_numeric_correlation_heatmap.png")
        plt.close()


def analyze_video_features(video_basic: pd.DataFrame, video_stats: pd.DataFrame, dirs: Dict[str, Path]) -> None:
    """Analyze both basic and aggregate-statistics video tables."""
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    # Basic categorical distributions to understand content inventory composition.
    for col in ["video_type", "upload_type"]:
        if col in video_basic.columns:
            vc = video_basic[col].value_counts(dropna=False).head(20).rename_axis(col).reset_index(name="count")
            vc.to_csv(rpt_dir / f"video_basic_{col}_counts.csv", index=False)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=vc, x=col, y="count", hue=col, legend=False, palette="crest")
            plt.title(f"Top {col} Categories")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / f"video_basic_{col}_distribution.png")
            plt.close()

    # Numeric distributions in basic table.
    for col in ["video_duration", "server_width", "server_height"]:
        if col in video_basic.columns:
            s = pd.to_numeric(video_basic[col], errors="coerce").dropna()
            plt.figure(figsize=(10, 5))
            sns.histplot(np.log1p(s), bins=60, color="#8da0cb")
            plt.title(f"{col} Distribution (log1p scale)")
            plt.xlabel(f"log1p({col})")
            plt.tight_layout()
            plt.savefig(fig_dir / f"video_basic_{col}_hist.png")
            plt.close()

    # Correlation of core aggregated video statistics.
    core_cols = [
        "show_cnt",
        "play_cnt",
        "play_user_num",
        "complete_play_cnt",
        "valid_play_cnt",
        "long_time_play_cnt",
        "short_time_play_cnt",
        "play_progress",
        "like_cnt",
        "comment_cnt",
        "follow_cnt",
        "share_cnt",
        "collect_cnt",
    ]
    core_cols = [c for c in core_cols if c in video_stats.columns]

    if len(core_cols) >= 2:
        corr = video_stats[core_cols].corr(numeric_only=True)
        corr.to_csv(rpt_dir / "video_stats_core_correlation_matrix.csv")

        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
        plt.title("Video Statistics Correlation (Core Metrics)")
        plt.tight_layout()
        plt.savefig(fig_dir / "video_stats_core_correlation_heatmap.png")
        plt.close()


def analyze_joined_table(
    log_df: pd.DataFrame,
    user_df: pd.DataFrame,
    video_basic: pd.DataFrame,
    video_stats: pd.DataFrame,
    dirs: Dict[str, Path],
) -> None:
    """
    Join interaction logs with user/video features for feature-target inspection.

    This is especially useful for recommendation model design because it lets us
    quickly inspect which side features are associated with click/like/long_view.
    """
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    # Keep join size manageable for EDA speed and memory safety.
    sample_n = min(1_000_000, len(log_df))
    sampled_log = log_df.sample(n=sample_n, random_state=42)

    merged = sampled_log.merge(user_df, on="user_id", how="left")
    merged = merged.merge(video_basic[["video_id", "video_type", "video_duration"]], on="video_id", how="left")
    merged = merged.merge(
        video_stats[["video_id", "show_cnt", "play_cnt", "play_progress", "like_cnt", "comment_cnt"]],
        on="video_id",
        how="left",
    )

    merged.to_csv(rpt_dir / "joined_sample_for_modeling.csv", index=False)

    # Numeric correlation against key labels.
    target_cols = ["is_click", "is_like", "long_view"]
    numeric = merged.select_dtypes(include=[np.number]).copy()
    useful = [c for c in numeric.columns if c in target_cols or c in ["play_time_ms", "duration_ms", "video_duration", "show_cnt", "play_cnt", "play_progress", "like_cnt", "comment_cnt", "register_days", "follow_user_num", "fans_user_num"]]
    useful = [c for c in useful if c in numeric.columns]

    if len(useful) >= 4:
        corr = numeric[useful].corr(numeric_only=True)
        corr.to_csv(rpt_dir / "joined_selected_numeric_correlation.csv")

        plt.figure(figsize=(11, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Joined Table: Selected Numeric Correlations")
        plt.tight_layout()
        plt.savefig(fig_dir / "joined_selected_numeric_correlation_heatmap.png")
        plt.close()

    if "video_type" in merged.columns:
        vt = (
            merged.groupby("video_type", dropna=False)
            .agg(impressions=("video_id", "size"), ctr=("is_click", "mean"), ltr=("is_like", "mean"))
            .reset_index()
            .sort_values("impressions", ascending=False)
            .head(20)
        )
        vt.to_csv(rpt_dir / "joined_video_type_behavior.csv", index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=vt, x="video_type", y="ctr", color="#1f78b4", label="CTR")
        sns.scatterplot(data=vt, x="video_type", y="ltr", color="#e31a1c", s=80, label="Like Rate")
        plt.xticks(rotation=35, ha="right")
        plt.title("Behavior by Video Type")
        plt.ylabel("Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "joined_video_type_ctr_ltr.png")
        plt.close()


def analyze_recsys_readiness(
    log_df: pd.DataFrame,
    user_df: pd.DataFrame,
    video_basic: pd.DataFrame,
    dirs: Dict[str, Path],
) -> None:
    """
    Produce additional EDA artifacts that are directly useful for recommender design.

    Focus areas:
    - time-based split sanity checks
    - cold-start rates in the test window
    - popularity bias and long-tail concentration
    - label co-occurrence structure
    - behavior drift across days and traffic sources
    """
    fig_dir = dirs["figures"]
    rpt_dir = dirs["reports"]

    action_cols = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view"]
    action_cols = [c for c in action_cols if c in log_df.columns]

    # 1) Time-based split (80/20 by date) and cold-start diagnostics.
    unique_dates = sorted(log_df["date_dt"].dropna().unique())
    if len(unique_dates) >= 2:
        split_idx = max(1, int(len(unique_dates) * 0.8))
        split_date = unique_dates[split_idx - 1]
    else:
        split_date = log_df["date_dt"].dropna().max()

    train = log_df[log_df["date_dt"] <= split_date].copy()
    test = log_df[log_df["date_dt"] > split_date].copy()
    if test.empty:
        # Fall back to row-wise split if dates are degenerate.
        cut = int(len(log_df) * 0.8)
        train = log_df.iloc[:cut].copy()
        test = log_df.iloc[cut:].copy()

    train_users = set(train["user_id"].unique())
    train_items = set(train["video_id"].unique())

    test = test.copy()
    test["is_new_user"] = ~test["user_id"].isin(train_users)
    test["is_new_item"] = ~test["video_id"].isin(train_items)
    test["is_new_user_or_item"] = test["is_new_user"] | test["is_new_item"]

    split_summary = pd.DataFrame(
        [
            {"metric": "train_rows", "value": int(len(train))},
            {"metric": "test_rows", "value": int(len(test))},
            {"metric": "train_users", "value": int(train["user_id"].nunique())},
            {"metric": "test_users", "value": int(test["user_id"].nunique())},
            {"metric": "train_items", "value": int(train["video_id"].nunique())},
            {"metric": "test_items", "value": int(test["video_id"].nunique())},
            {"metric": "test_new_user_rate", "value": float(test["is_new_user"].mean())},
            {"metric": "test_new_item_rate", "value": float(test["is_new_item"].mean())},
            {"metric": "test_new_user_or_item_rate", "value": float(test["is_new_user_or_item"].mean())},
        ]
    )
    split_summary.to_csv(rpt_dir / "recsys_split_coldstart_summary.csv", index=False)

    # 2) User/item long-tail concentration (Lorenz-like cumulative coverage).
    user_cnt = log_df.groupby("user_id").size().sort_values(ascending=False)
    item_cnt = log_df.groupby("video_id").size().sort_values(ascending=False)

    def top_share(series: pd.Series, top_frac: float) -> float:
        n = max(1, int(len(series) * top_frac))
        return float(series.iloc[:n].sum() / series.sum())

    longtail = pd.DataFrame(
        [
            {"entity": "user", "top_fraction": 0.01, "interaction_share": top_share(user_cnt, 0.01)},
            {"entity": "user", "top_fraction": 0.05, "interaction_share": top_share(user_cnt, 0.05)},
            {"entity": "user", "top_fraction": 0.10, "interaction_share": top_share(user_cnt, 0.10)},
            {"entity": "item", "top_fraction": 0.01, "interaction_share": top_share(item_cnt, 0.01)},
            {"entity": "item", "top_fraction": 0.05, "interaction_share": top_share(item_cnt, 0.05)},
            {"entity": "item", "top_fraction": 0.10, "interaction_share": top_share(item_cnt, 0.10)},
        ]
    )
    longtail.to_csv(rpt_dir / "recsys_longtail_concentration.csv", index=False)

    def lorenz_curve(series: pd.Series) -> pd.DataFrame:
        s = series.sort_values(ascending=True).to_numpy(dtype=float)
        cum_pop = np.arange(1, len(s) + 1) / len(s)
        cum_inter = np.cumsum(s) / np.sum(s)
        return pd.DataFrame({"cum_population": cum_pop, "cum_interactions": cum_inter})

    user_lorenz = lorenz_curve(user_cnt)
    item_lorenz = lorenz_curve(item_cnt)
    user_lorenz.to_csv(rpt_dir / "recsys_user_lorenz.csv", index=False)
    item_lorenz.to_csv(rpt_dir / "recsys_item_lorenz.csv", index=False)

    plt.figure(figsize=(9, 7))
    plt.plot(user_lorenz["cum_population"], user_lorenz["cum_interactions"], label="users")
    plt.plot(item_lorenz["cum_population"], item_lorenz["cum_interactions"], label="items")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect equality")
    plt.xlabel("Cumulative population share")
    plt.ylabel("Cumulative interaction share")
    plt.title("Long-Tail Concentration (Lorenz Curves)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "recsys_longtail_lorenz.png")
    plt.close()

    # 3) Label co-occurrence matrix for multi-objective design.
    if action_cols:
        arr = log_df[action_cols].astype(int).to_numpy()
        cooc = (arr.T @ arr).astype(float)
        cooc_df = pd.DataFrame(cooc, index=action_cols, columns=action_cols)
        cooc_df.to_csv(rpt_dir / "recsys_action_cooccurrence_counts.csv")

        # Normalize by diagonal to read conditional rates better.
        denom = np.diag(cooc).copy()
        denom[denom == 0] = 1.0
        cond = cooc / denom[:, None]
        cond_df = pd.DataFrame(cond, index=action_cols, columns=action_cols)
        cond_df.to_csv(rpt_dir / "recsys_action_cooccurrence_conditional.csv")

        plt.figure(figsize=(10, 8))
        sns.heatmap(cond_df, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1)
        plt.title("Action Co-occurrence (Row-normalized)")
        plt.tight_layout()
        plt.savefig(fig_dir / "recsys_action_cooccurrence_heatmap.png")
        plt.close()

    # 4) Source bias: compare standard and random traffic.
    source_metrics = (
        log_df.groupby("source_table")
        .agg(
            impressions=("video_id", "size"),
            ctr=("is_click", "mean"),
            ltr=("is_like", "mean"),
            lvtr=("long_view", "mean"),
        )
        .reset_index()
    )
    source_metrics.to_csv(rpt_dir / "recsys_source_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    melted = source_metrics.melt(
        id_vars=["source_table"],
        value_vars=["ctr", "ltr", "lvtr"],
        var_name="metric",
        value_name="rate",
    )
    sns.barplot(data=melted, x="source_table", y="rate", hue="metric")
    plt.title("Metric Shift Across Traffic Sources")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "recsys_source_metric_shift.png")
    plt.close()

    # 5) Train-item popularity deciles vs test behavior (popularity bias check).
    item_train_cnt = train.groupby("video_id").size().rename("train_impressions")
    test_item = test.groupby("video_id").agg(
        test_impressions=("video_id", "size"),
        test_ctr=("is_click", "mean"),
        test_ltr=("is_like", "mean"),
        test_lvtr=("long_view", "mean"),
    )
    pop = test_item.join(item_train_cnt, how="left").fillna({"train_impressions": 0})
    pop["train_impressions"] = pop["train_impressions"].astype(float)

    try:
        pop["pop_decile"] = pd.qcut(pop["train_impressions"].rank(method="first"), 10, labels=False) + 1
    except ValueError:
        pop["pop_decile"] = 1

    pop_decile = (
        pop.groupby("pop_decile")
        .agg(
            n_items=("test_impressions", "size"),
            avg_test_impressions=("test_impressions", "mean"),
            avg_test_ctr=("test_ctr", "mean"),
            avg_test_ltr=("test_ltr", "mean"),
            avg_test_lvtr=("test_lvtr", "mean"),
        )
        .reset_index()
        .sort_values("pop_decile")
    )
    pop_decile.to_csv(rpt_dir / "recsys_popularity_decile_behavior.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    sns.barplot(data=pop_decile, x="pop_decile", y="avg_test_impressions", color="#9ecae1", ax=ax1)
    ax1.set_xlabel("Train Popularity Decile (1=tail, 10=head)")
    ax1.set_ylabel("Avg Test Impressions per Item")
    ax2 = ax1.twinx()
    sns.lineplot(data=pop_decile, x="pop_decile", y="avg_test_ctr", marker="o", color="#ef3b2c", ax=ax2, label="CTR")
    sns.lineplot(data=pop_decile, x="pop_decile", y="avg_test_lvtr", marker="o", color="#31a354", ax=ax2, label="LongView")
    ax2.set_ylabel("Rate")
    plt.title("Popularity Bias: Exposure and Quality by Popularity Decile")
    plt.tight_layout()
    plt.savefig(fig_dir / "recsys_popularity_bias_decile.png")
    plt.close()

    # 6) User activity buckets (from train) and downstream quality in test.
    user_train_cnt = train.groupby("user_id").size().rename("train_user_interactions")
    user_test = test.groupby("user_id").agg(
        test_impressions=("video_id", "size"),
        test_ctr=("is_click", "mean"),
        test_ltr=("is_like", "mean"),
        test_lvtr=("long_view", "mean"),
    )
    ub = user_test.join(user_train_cnt, how="left").fillna({"train_user_interactions": 0})

    bins = [-0.1, 0.1, 5, 20, 50, np.inf]
    labels = ["cold(0)", "1-5", "6-20", "21-50", "50+"]
    ub["activity_bucket"] = pd.cut(ub["train_user_interactions"], bins=bins, labels=labels)
    user_bucket = (
        ub.groupby("activity_bucket", observed=False)
        .agg(
            n_users=("test_impressions", "size"),
            avg_test_impressions=("test_impressions", "mean"),
            avg_test_ctr=("test_ctr", "mean"),
            avg_test_ltr=("test_ltr", "mean"),
            avg_test_lvtr=("test_lvtr", "mean"),
        )
        .reset_index()
    )
    user_bucket.to_csv(rpt_dir / "recsys_user_bucket_behavior.csv", index=False)

    plt.figure(figsize=(10, 6))
    melted = user_bucket.melt(
        id_vars=["activity_bucket"],
        value_vars=["avg_test_ctr", "avg_test_ltr", "avg_test_lvtr"],
        var_name="metric",
        value_name="rate",
    )
    sns.lineplot(data=melted, x="activity_bucket", y="rate", hue="metric", marker="o")
    plt.title("Test Behavior by User Activity Bucket (Train-defined)")
    plt.xlabel("User Activity Bucket")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "recsys_user_bucket_behavior.png")
    plt.close()

    # 7) Day-by-day label drift (for split stability checks).
    daily_rate = (
        log_df.groupby("date_dt")
        .agg(
            impressions=("video_id", "size"),
            ctr=("is_click", "mean"),
            ltr=("is_like", "mean"),
            lvtr=("long_view", "mean"),
            ftr=("is_follow", "mean"),
        )
        .reset_index()
        .sort_values("date_dt")
    )
    daily_rate.to_csv(rpt_dir / "recsys_daily_label_drift.csv", index=False)

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=daily_rate, x="date_dt", y="ctr", marker="o", label="CTR")
    sns.lineplot(data=daily_rate, x="date_dt", y="lvtr", marker="o", label="Long View Rate")
    sns.lineplot(data=daily_rate, x="date_dt", y="ltr", marker="o", label="Like Rate")
    plt.title("Daily Label Drift")
    plt.xlabel("Date")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "recsys_daily_label_drift.png")
    plt.close()

    # 8) Item metadata alignment with engagement (video duration bucket).
    if "video_duration" in video_basic.columns:
        duration = pd.to_numeric(video_basic["video_duration"], errors="coerce")
        vb = video_basic[["video_id"]].copy()
        vb["video_duration"] = duration
        joined = log_df[["video_id", "is_click", "is_like", "long_view"]].merge(vb, on="video_id", how="left")
        joined = joined.dropna(subset=["video_duration"])
        joined["duration_bucket_sec"] = pd.cut(
            joined["video_duration"] / 1000.0,
            bins=[0, 10, 20, 30, 60, 120, np.inf],
            labels=["0-10s", "10-20s", "20-30s", "30-60s", "60-120s", "120s+"],
            include_lowest=True,
        )
        dur = (
            joined.groupby("duration_bucket_sec", observed=False)
            .agg(
                impressions=("video_id", "size"),
                ctr=("is_click", "mean"),
                ltr=("is_like", "mean"),
                lvtr=("long_view", "mean"),
            )
            .reset_index()
        )
        dur.to_csv(rpt_dir / "recsys_duration_bucket_behavior.csv", index=False)

        fig, ax1 = plt.subplots(figsize=(11, 6))
        sns.barplot(data=dur, x="duration_bucket_sec", y="impressions", color="#c7e9c0", ax=ax1)
        ax1.set_ylabel("Impressions")
        ax1.set_xlabel("Video Duration Bucket")
        ax2 = ax1.twinx()
        sns.lineplot(data=dur, x="duration_bucket_sec", y="ctr", marker="o", color="#08519c", ax=ax2, label="CTR")
        sns.lineplot(data=dur, x="duration_bucket_sec", y="lvtr", marker="o", color="#cb181d", ax=ax2, label="LongView")
        ax2.set_ylabel("Rate")
        plt.title("Behavior by Video Duration Bucket")
        plt.tight_layout()
        plt.savefig(fig_dir / "recsys_duration_bucket_behavior.png")
        plt.close()


def write_metadata_notes(reports_dir: Path) -> None:
    """Save concise metadata notes captured from the official KuaiRand page."""
    notes = """# KuaiRand Metadata Notes (from official repository)

- Official source: https://github.com/Kuairand/KuaiRand
- Log tables contain interaction events with labels such as `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, and `long_view`.
- Core log time fields: `date` (YYYYMMDD), `hourmin` (HHMM), and `time_ms` (Unix timestamp in milliseconds).
- User table contains profile/behavioral attributes and anonymized one-hot features (`onehot_feat0` ... `onehot_feat17`).
- Video basic table contains static metadata (`video_type`, `upload_type`, `video_duration`, resolution, music info, tag).
- Video statistics table contains aggregate behavior metrics (show/play/like/comment/follow/share/collect style counts and user counts).

These notes are intended as a practical summary for EDA and feature engineering.
"""
    (reports_dir / "kuairand_metadata_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KuaiRand EDA pipeline and export figures/reports.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to folder containing KuaiRand CSV files.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to EDA output directory.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.output_dir)
    dirs = ensure_dirs(out_dir)

    dfs = load_data(data_dir)

    # Parse time fields once for repeated temporal analyses.
    dfs["log_all"] = parse_time_columns(dfs["log_all"])

    overview = dataset_overview(dfs, dirs["reports"])
    print("\nDataset overview:\n", overview)

    # Create table-specific quality reports.
    for name, df in dfs.items():
        data_quality_report(df, name, dirs["reports"])

    analyze_log_table(dfs["log_all"], dirs)
    analyze_log_source_comparison(dfs["log_all"], dirs)
    analyze_user_features(dfs["user_features"], dirs)
    analyze_video_features(dfs["video_basic"], dfs["video_stats"], dirs)
    analyze_joined_table(
        dfs["log_all"],
        dfs["user_features"],
        dfs["video_basic"],
        dfs["video_stats"],
        dirs,
    )
    analyze_recsys_readiness(
        dfs["log_all"],
        dfs["user_features"],
        dfs["video_basic"],
        dirs,
    )
    write_metadata_notes(dirs["reports"])

    print("\nEDA completed.")
    print(f"Figures saved to: {dirs['figures']}")
    print(f"Reports saved to: {dirs['reports']}")


if __name__ == "__main__":
    main()

