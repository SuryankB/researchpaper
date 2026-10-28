
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_FILE = "results/gemma_qwen_comparison_per_example.csv"
OUTPUT_PNG = "results/figure3_paired_bootstrap_differences.png"
OUTPUT_PDF = "results/figure3_paired_bootstrap_differences.pdf"

N_BOOTSTRAPS = 10_000
SEED = 42

# Each pair is (Gemma score column, Qwen score column).
METRICS = [
    ("ROUGE-1", "gemma_rouge1_f1", "qwen_rouge1_f1"),
    ("ROUGE-2", "gemma_rouge2_f1", "qwen_rouge2_f1"),
    ("ROUGE-L", "gemma_rougeL_f1", "qwen_rougeL_f1"),
    ("BERTScore P", "gemma_bertscore_precision",
     "qwen_bertscore_precision"),
    ("BERTScore R", "gemma_bertscore_recall",
     "qwen_bertscore_recall"),
    ("BERTScore F1", "gemma_bertscore_f1",
     "qwen_bertscore_f1"),
]

# --------------------------------------------------
# Load and validate data
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

required = ["language"] + [
    column
    for _, gemma_col, qwen_col in METRICS
    for column in (gemma_col, qwen_col)
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

languages = ["Bengali", "Hindi"]
rng = np.random.default_rng(SEED)
results = []

# --------------------------------------------------
# Paired bootstrap: Qwen minus Gemma
# --------------------------------------------------
for language in languages:
    subset = df[df["language"].str.lower() == language.lower()]

    for metric, gemma_col, qwen_col in METRICS:
        paired = subset[[gemma_col, qwen_col]].dropna()

        gemma = paired[gemma_col].to_numpy(dtype=float)
        qwen = paired[qwen_col].to_numpy(dtype=float)

        if len(paired) == 0:
            raise ValueError(f"No valid data for {language}, {metric}")

        differences = qwen - gemma
        mean_difference = differences.mean()
        n = len(differences)

        # Resample paired article-level differences.
        bootstrap_means = np.empty(N_BOOTSTRAPS)
        for b in range(N_BOOTSTRAPS):
            sample_indices = rng.integers(0, n, size=n)
            bootstrap_means[b] = differences[sample_indices].mean()

        lower, upper = np.percentile(
            bootstrap_means, [2.5, 97.5]
        )

        results.append({
            "language": language,
            "metric": metric,
            "n": n,
            "mean_difference": mean_difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "excludes_zero": lower > 0 or upper < 0,
        })

results_df = pd.DataFrame(results)
results_df.to_csv("figure3_bootstrap_results.csv", index=False)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(10.5, 4.0),
    sharey=True
)

# Screenshot-like palette
positive_color = "#e67836"
negative_color = "#3976b9"

y_positions = np.arange(len(METRICS))[::-1]

for ax, language in zip(axes, languages):
    lang_results = (
        results_df[results_df["language"] == language]
        .set_index("metric")
        .loc[[m[0] for m in METRICS]]
        .reset_index()
    )

    for y, row in zip(y_positions, lang_results.itertuples()):
        mean = row.mean_difference
        lower = row.ci_lower
        upper = row.ci_upper

        color = positive_color if mean >= 0 else negative_color
        filled = row.excludes_zero

        # Confidence interval
        ax.plot(
            [lower, upper], [y, y],
            color=color,
            linewidth=1.5,
            zorder=2
        )

        # End caps
        ax.plot(
            [lower, lower], [y - 0.08, y + 0.08],
            color=color, linewidth=1
        )
        ax.plot(
            [upper, upper], [y - 0.08, y + 0.08],
            color=color, linewidth=1
        )

        # Filled if the 95% interval excludes zero;
        # hollow if it crosses zero.
        ax.scatter(
            mean, y,
            s=65,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.5,
            zorder=3
        )

        # Label the mean difference
        offset = 0.0008
        if mean >= 0:
            label_x = upper + offset
            ha = "left"
        else:
            label_x = lower - offset
            ha = "right"

        ax.text(
            label_x, y,
            f"{mean:+.4f}",
            va="center",
            ha=ha,
            fontsize=8
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(language, fontsize=12)
    ax.set_xlabel("Qwen minus Gemma (mean difference)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in METRICS])
    ax.grid(False)

    # Shared scale for direct language comparison
    ax.set_xlim(-0.025, 0.055)

axes[0].set_ylabel("")
fig.text(
    0.5, 0.015,
    "Filled marker: 95% paired-bootstrap interval excludes zero; "
    "hollow marker: interval crosses zero (unadjusted, exploratory).",
    ha="center",
    fontsize=8
)

fig.tight_layout(rect=[0, 0.06, 1, 1])

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.show()

print("\nSaved:", OUTPUT_PNG)
print("Saved:", OUTPUT_PDF)
print("\nBootstrap results:")
print(results_df.round(5).to_string(index=False))
