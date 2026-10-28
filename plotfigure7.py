
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/consistent_word_counts_per_example.csv")

colors = {
    "Reference": "#bdbdbd",
    "Gemma": "#7fa2d6",
    "Qwen": "#f19a6b",
}

columns = {
    "Reference": "reference_words",
    "Gemma": "gemma_words",
    "Qwen": "qwen_words",
}

# Shared bins for a fair visual comparison
bins = np.arange(0, 131, 3)

fig, axes = plt.subplots(
    1, 2, figsize=(12, 4.5), sharey=True
)

for ax, language in zip(axes, ["Bengali", "Hindi"]):
    subset = df[df["language"].str.lower() == language.lower()]

    for label, column in columns.items():
        values = subset[column].dropna()
        sd = values.std(ddof=1)

        ax.hist(
            values,
            bins=bins,
            color=colors[label],
            alpha=0.9,
            edgecolor="none",
            label=f"{label} (SD {sd:.1f})",
        )

    ax.set_title(language, fontsize=13)
    ax.set_xlabel("Words per summary", fontsize=11)
    ax.set_xlim(-5, 130)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(False)

axes[0].set_ylabel("Number of articles", fontsize=11)

fig.tight_layout()

fig.savefig(
    "figure7_summary_length_histograms.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    "figure7_summary_length_histograms.pdf",
    bbox_inches="tight",
)

plt.show()

print("Saved Figure 7 as PNG and PDF.")
