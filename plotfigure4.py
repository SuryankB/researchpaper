
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "results/gemma_qwen_comparison_per_example.csv"
OUTPUT_PNG = "figure4_per_article_wins.png"
OUTPUT_PDF = "figure4_per_article_wins.pdf"

df = pd.read_csv(INPUT_FILE)

# Metric columns: (display label, Gemma column, Qwen column)
metrics = [
    ("ROUGE-1", "gemma_rouge1_f1", "qwen_rouge1_f1"),
    ("ROUGE-2", "gemma_rouge2_f1", "qwen_rouge2_f1"),
    ("ROUGE-L", "gemma_rougeL_f1", "qwen_rougeL_f1"),
    ("BERTScore F1", "gemma_bertscore_f1", "qwen_bertscore_f1"),
]

languages = [
    ("Bengali", "Bn"),
    ("Hindi", "Hi"),
]

# Colors similar to your example
QWEN_COLOR = "#e67836"
GEMMA_COLOR = "#3976b9"
TIE_COLOR = "#bdbdbd"
DISAGREEMENT_COLOR = "#6b55c9"

# --------------------------------------------------
# Calculate wins and ties for each metric/language
# --------------------------------------------------
win_rows = []

for language, short_name in languages:
    sub = df[df["language"].str.lower() == language.lower()]

    for metric, gemma_col, qwen_col in metrics:
        valid = sub[[gemma_col, qwen_col]].dropna()
        qwen = valid[qwen_col].to_numpy()
        gemma = valid[gemma_col].to_numpy()

        win_rows.append({
            "language": language,
            "short_name": short_name,
            "metric": metric,
            "qwen_higher": int((qwen > gemma).sum()),
            "gemma_higher": int((gemma > qwen).sum()),
            "tie": int((gemma == qwen).sum()),
            "n": len(valid),
        })

wins = pd.DataFrame(win_rows)
wins.to_csv("figure4_win_counts.csv", index=False)

# --------------------------------------------------
# Calculate ROUGE-1 / BERTScore F1 disagreements
# --------------------------------------------------
disagreement_rows = []

for language, _ in languages:
    sub = df[df["language"].str.lower() == language.lower()]

    cols = [
        "gemma_rouge1_f1", "qwen_rouge1_f1",
        "gemma_bertscore_f1", "qwen_bertscore_f1",
    ]
    valid = sub[cols].dropna()

    rouge_direction = np.sign(
        valid["qwen_rouge1_f1"] - valid["gemma_rouge1_f1"]
    )
    bert_direction = np.sign(
        valid["qwen_bertscore_f1"] - valid["gemma_bertscore_f1"]
    )

    # Count only cases where both metrics select a model
    # and the selected models differ.
    eligible = (rouge_direction != 0) & (bert_direction != 0)
    disagreement = (
        eligible & (rouge_direction != bert_direction)
    )

    n_eligible = int(eligible.sum())
    n_disagreement = int(disagreement.sum())

    disagreement_rows.append({
        "language": language,
        "disagreements": n_disagreement,
        "eligible_articles": n_eligible,
        "percent": (
            100 * n_disagreement / n_eligible
            if n_eligible else np.nan
        ),
    })

disagreements = pd.DataFrame(disagreement_rows)
disagreements.to_csv(
    "figure4_metric_disagreements.csv", index=False
)

# --------------------------------------------------
# Plot: panel (a) wins; panel (b) disagreements
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5.0),
    gridspec_kw={"width_ratios": [2.2, 1]}
)

# Panel (a): stacked horizontal bars
plot_rows = []

for language, short_name in languages:
    for metric, _, _ in metrics:
        row = wins[
            (wins["language"] == language)
            & (wins["metric"] == metric)
        ].iloc[0]

        plot_rows.append({
            "label": f"{short_name}  {metric}",
            "qwen": row["qwen_higher"],
            "gemma": row["gemma_higher"],
            "tie": row["tie"],
        })

plot_df = pd.DataFrame(plot_rows)
y = np.arange(len(plot_df))

ax1.barh(y, plot_df["qwen"], color=QWEN_COLOR, label="Qwen higher")
ax1.barh(
    y, plot_df["gemma"],
    left=plot_df["qwen"],
    color=GEMMA_COLOR, label="Gemma higher"
)
ax1.barh(
    y, plot_df["tie"],
    left=plot_df["qwen"] + plot_df["gemma"],
    color=TIE_COLOR, label="Tie"
)

# Label segments only when there is enough room
for i, row in plot_df.iterrows():
    if row["qwen"] >= 25:
        ax1.text(
            row["qwen"] / 2, i, f'{int(row["qwen"])}',
            ha="center", va="center", color="white", fontsize=9
        )
    if row["gemma"] >= 25:
        ax1.text(
            row["qwen"] + row["gemma"] / 2, i,
            f'{int(row["gemma"])}',
            ha="center", va="center", color="white", fontsize=9
        )
    if row["tie"] >= 15:
        ax1.text(
            row["qwen"] + row["gemma"] + row["tie"] / 2,
            i, f'{int(row["tie"])}',
            ha="center", va="center", color="black", fontsize=9
        )

ax1.set_yticks(y)
ax1.set_yticklabels(plot_df["label"])
ax1.invert_yaxis()
ax1.set_xlabel("Articles")
ax1.set_title("(a) Per-article wins")
ax1.set_xlim(0, 200)
ax1.set_xticks(np.arange(0, 201, 25))
ax1.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.20),
    ncol=3, frameon=False, fontsize=8
)

# Panel (b): disagreement percentages
x = np.arange(len(disagreements))
bars = ax2.bar(
    x, disagreements["percent"],
    color=DISAGREEMENT_COLOR, width=0.58
)

for bar, row in zip(bars, disagreements.itertuples()):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{row.percent:.1f}%\n"
        f"({row.disagreements} of {row.eligible_articles})",
        ha="center", va="bottom", fontsize=8
    )

ax2.set_xticks(x)
ax2.set_xticklabels(disagreements["language"])
ax2.set_ylabel("% of articles")
ax2.set_title("(b) ROUGE-1 vs BERTScore F1\nfavour different models")
ax2.set_ylim(
    0,
    max(40, disagreements["percent"].max() + 10)
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.show()

print("Saved:", OUTPUT_PNG)
print("Saved:", OUTPUT_PDF)
print("\nWin counts:")
print(wins.to_string(index=False))
print("\nMetric disagreements:")
print(disagreements.to_string(index=False))
