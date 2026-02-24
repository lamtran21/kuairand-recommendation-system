from __future__ import annotations

from pathlib import Path
import html
import pandas as pd


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def df_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        shown = df.head(max_rows).copy()
        note = f"<p class='note'>Showing first {max_rows:,} rows out of {len(df):,} rows.</p>"
    else:
        shown = df
        note = ""
    table_html = shown.to_html(index=False, classes="data-table", border=0, escape=False)
    return note + table_html


def section(title: str, body: str, level: int = 2) -> str:
    return f"<section><h{level}>{html.escape(title)}</h{level}>{body}</section>"


def img_block(title: str, rel_path: str, caption: str) -> str:
    return (
        "<div class='figure-card'>"
        f"<h4>{html.escape(title)}</h4>"
        f"<img src='{html.escape(rel_path)}' alt='{html.escape(title)}' loading='lazy'/>"
        f"<p>{html.escape(caption)}</p>"
        "</div>"
    )


def main() -> None:
    base = Path("outputs")
    reports = base / "reports"
    figures = base / "figures"
    out_file = base / "kuairand_eda_full_report.html"

    if not reports.exists() or not figures.exists():
        raise FileNotFoundError("Expected outputs/reports and outputs/figures to exist.")

    parts: list[str] = []

    intro = """
    <p>This report consolidates KuaiRand EDA outputs into one readable document.</p>
    <ul>
      <li>Variable dictionary (official definitions + local dataset stats) is placed first for quick lookup.</li>
      <li>Then: dataset overview, data quality, behavior metrics, correlations, and visual diagnostics.</li>
      <li>All statistics are computed from your local files under <code>Data/data</code> (or <code>data/data</code>).</li>
    </ul>
    """
    parts.append(section("Report Introduction", intro))

    # 1) Variable dictionary at top
    var_file = reports / "kuairand_variable_dictionary.csv"
    if var_file.exists():
        var_df = read_csv_safe(var_file)
        summary = (
            var_df.groupby("table", as_index=False)["column"]
            .count()
            .rename(columns={"column": "n_variables"})
        )
        body = ""
        body += "<p>Official definition source: <a href='https://github.com/Kuairand/KuaiRand' target='_blank'>KuaiRand official repository</a>.</p>"
        body += "<h3>Variable Counts by Table</h3>"
        body += df_to_html_table(summary)
        body += "<h3>Full Variable Dictionary</h3>"
        body += df_to_html_table(var_df)
        parts.append(section("Variable Introduction", body))

    meta_file = reports / "kuairand_metadata_notes.md"
    if meta_file.exists():
        txt = meta_file.read_text(encoding="utf-8")
        body = "<pre class='pre-block'>" + html.escape(txt) + "</pre>"
        parts.append(section("Official Metadata Notes", body))

    # 2) Overview + quality tables
    ordered_csv = [
        "dataset_overview.csv",
        "quality_summary_log_all.csv",
        "quality_summary_user_features.csv",
        "quality_summary_video_basic.csv",
        "quality_summary_video_stats.csv",
        "quality_log_all.csv",
        "quality_user_features.csv",
        "quality_video_basic.csv",
        "quality_video_stats.csv",
        "log_action_rates.csv",
        "log_daily_metrics.csv",
        "log_hourly_metrics.csv",
        "joined_video_type_behavior.csv",
        "joined_selected_numeric_correlation.csv",
        "log_correlation_matrix.csv",
        "user_numeric_correlation_matrix.csv",
        "video_stats_core_correlation_matrix.csv",
        "video_basic_video_type_counts.csv",
        "video_basic_upload_type_counts.csv",
        "user_active_degree_counts.csv",
        "compare_logs_summary.csv",
        "compare_logs_daily_metrics.csv",
        "compare_logs_hourly_metrics.csv",
        "compare_logs_tab_metrics.csv",
        "compare_logs_overlap_matrix.csv",
        "compare_logs_item_pop_rankcorr.csv",
        "compare_logs_tab_share_matrix.csv",
        "compare_logs_tab_ctr_matrix.csv",
        "recsys_split_coldstart_summary.csv",
        "recsys_longtail_concentration.csv",
        "recsys_source_metrics.csv",
        "recsys_popularity_decile_behavior.csv",
        "recsys_user_bucket_behavior.csv",
        "recsys_duration_bucket_behavior.csv",
        "recsys_daily_label_drift.csv",
        "recsys_action_cooccurrence_counts.csv",
        "recsys_action_cooccurrence_conditional.csv",
    ]

    body = ""
    for name in ordered_csv:
        p = reports / name
        if not p.exists():
            continue
        df = read_csv_safe(p)
        max_rows = None
        if name in {"user_numeric_correlation_matrix.csv", "video_stats_core_correlation_matrix.csv", "log_correlation_matrix.csv"}:
            max_rows = 200
        body += f"<h3>{html.escape(name)}</h3>"
        body += df_to_html_table(df, max_rows=max_rows)
    parts.append(section("Summary CSV Tables", body))

    # Large CSVs: keep as note + samples
    extra_large = [
        "user_interaction_counts.csv",
        "item_interaction_counts.csv",
        "joined_sample_for_modeling.csv",
    ]
    body = ""
    for name in extra_large:
        p = reports / name
        if not p.exists():
            continue
        body += f"<h3>{html.escape(name)}</h3>"
        if name == "joined_sample_for_modeling.csv":
            body += (
                "<p class='note'>This file is very large. Embedded view is intentionally skipped to keep the HTML responsive.</p>"
            )
            body += f"<p><code>{html.escape(str(p))}</code></p>"
        else:
            df = read_csv_safe(p)
            body += df_to_html_table(df, max_rows=200)
    parts.append(section("Large Outputs (Sampled View)", body))

    # 3) Figures in logical order
    figure_order = [
        ("Global Action Rates", "log_action_rate_bar.png", "Overall probability of each positive/negative action."),
        ("Daily Interaction Volume", "log_daily_volume.png", "Daily impressions, clicks, and likes over time."),
        ("Daily CTR", "log_daily_ctr.png", "Daily click-through rate trend."),
        ("Hourly Impressions & CTR", "log_hourly_impressions_ctr.png", "Within-day traffic and CTR pattern by hour."),
        ("Play Time vs Duration", "log_playtime_vs_duration_scatter.png", "Relationship between consumed watch time and raw duration."),
        ("Log Feature Correlation", "log_correlation_heatmap.png", "Correlation map among key log features and labels."),
        ("Log Missingness Matrix", "log_missingness_matrix.png", "Missing-value structure in sampled interaction records."),
        ("User/Item Sparsity", "log_user_item_sparsity_hist.png", "Long-tail interaction-count distributions for users and items."),
        ("User Active Degree", "user_active_degree_distribution.png", "User activity-level composition."),
        ("User Feature Correlation", "user_numeric_correlation_heatmap.png", "Numeric user-feature relationships."),
        ("Video Type Distribution", "video_basic_video_type_distribution.png", "Inventory share by video type."),
        ("Upload Type Distribution", "video_basic_upload_type_distribution.png", "Inventory share by upload pipeline type."),
        ("Video Duration Distribution", "video_basic_video_duration_hist.png", "Duration distribution in log scale."),
        ("Server Width Distribution", "video_basic_server_width_hist.png", "Distribution of stored video width."),
        ("Server Height Distribution", "video_basic_server_height_hist.png", "Distribution of stored video height."),
        ("Video Stats Correlation", "video_stats_core_correlation_heatmap.png", "Correlation among aggregate item-level performance metrics."),
        ("Joined Numeric Correlation", "joined_selected_numeric_correlation_heatmap.png", "Feature-target relationships after joining logs with side features."),
        ("Behavior by Video Type", "joined_video_type_ctr_ltr.png", "CTR and like-rate comparison by video type."),
        ("Source Action Rate Heatmap", "compare_logs_action_rate_heatmap.png", "Direct rate comparison across random and standard logs."),
        ("Source Daily Rate Curves", "compare_logs_daily_rate_lines.png", "Daily CTR/LVTR/LTR by source."),
        ("Source Hourly Rate Curves", "compare_logs_hourly_rate_lines.png", "Hourly CTR and long-view rate by source."),
        ("Tab Share by Source", "compare_logs_tab_share_heatmap.png", "Scenario (tab) traffic composition across logs."),
        ("Tab CTR by Source", "compare_logs_tab_ctr_heatmap.png", "CTR shift by tab and source."),
        ("User/Item Overlap", "compare_logs_overlap_heatmaps.png", "Jaccard overlap among datasets for users and items."),
        ("Item Popularity Rank Correlation", "compare_logs_item_rankcorr_heatmap.png", "Spearman correlation of item popularity ranks across logs."),
        ("Long-Tail Lorenz Curves", "recsys_longtail_lorenz.png", "Concentration curves for user/item interaction distributions."),
        ("Action Co-occurrence", "recsys_action_cooccurrence_heatmap.png", "Row-normalized co-occurrence among implicit feedback labels."),
        ("Source Metric Shift", "recsys_source_metric_shift.png", "CTR/LTR/LVTR differences between standard and random traffic."),
        ("Popularity Bias by Decile", "recsys_popularity_bias_decile.png", "Exposure and quality trends from item tail to head."),
        ("User Bucket Behavior", "recsys_user_bucket_behavior.png", "Test behavior stratified by train-time user activity buckets."),
        ("Daily Label Drift", "recsys_daily_label_drift.png", "Temporal movement of key targets (CTR/like/long-view rates)."),
        ("Duration Bucket Behavior", "recsys_duration_bucket_behavior.png", "Engagement and exposure by video duration buckets."),
    ]

    fig_body = "<div class='figure-grid'>"
    for title, filename, caption in figure_order:
        p = figures / filename
        if p.exists():
            rel = f"figures/{filename}"
            fig_body += img_block(title, rel, caption)
    fig_body += "</div>"
    parts.append(section("Figure Gallery", fig_body))

    html_doc = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>KuaiRand EDA Consolidated Report</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #1c2430;
      --muted: #5e6a7a;
      --line: #d8e0ea;
      --accent: #1f77b4;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      background: linear-gradient(120deg, #e8f1ff, #f7fbff);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 22px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
    .subtitle {{ margin: 0; color: var(--muted); }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
      overflow-x: auto;
    }}
    h2 {{ margin-top: 0; color: #17324d; }}
    h3 {{ margin-bottom: 8px; margin-top: 20px; color: #244565; }}
    h4 {{ margin: 0 0 8px 0; color: #214667; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-bottom: 12px;
    }}
    .data-table th, .data-table td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      background: #f0f5fb;
      z-index: 1;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      margin: 6px 0;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fcfeff;
      padding: 10px;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #e4ebf4;
      background: #fff;
    }}
    .figure-card p {{
      margin: 8px 0 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .pre-block {{
      white-space: pre-wrap;
      font-size: 12px;
      background: #f7fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>KuaiRand EDA Consolidated Report</h1>
      <p class="subtitle">Generated from local outputs in <code>outputs/reports</code> and <code>outputs/figures</code>.</p>
    </header>
    {''.join(parts)}
  </div>
</body>
</html>
"""

    out_file.write_text(html_doc, encoding="utf-8")
    print(f"Generated: {out_file}")


if __name__ == "__main__":
    main()
