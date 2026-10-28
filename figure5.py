
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "results/consistent_word_counts_per_example.csv"

df = pd.read_csv(INPUT_FILE)

columns = [
    "reference_words",
    "gemma_words",
    "qwen_words",
]

labels = ["Reference", "Gemma", "Qwen"]
languages = ["Bengali", "Hindi"]

colors = ["#bdbdbd", "#4f7fbd", "#e98b50"]

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 4.3),
    sharey=True
)

for ax, language in zip(axes, languages):
    subset = df[
        df["language"].str.lower() == language.lower()
    ]

    data = [
        subset[col].dropna().values
        for col in columns
    ]

    boxplot = ax.boxplot(
        data,
        label=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        flierprops={
            "marker": ".",
            "markerfacecolor": "gray",
            "markeredgecolor": "gray",
            "markersize": 3,
            "alpha": 0.8,
        },
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(1)

    ax.set_title(language)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=9)

axes[0].set_ylabel("Words per summary (whitespace-separated)")

fig.tight_layout()

fig.savefig(
    "figure5_summary_length_boxplots.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    "figure5_summary_length_boxplots.pdf",
    bbox_inches="tight",
)

plt.show()

print("Saved Figure 5 as PNG and PDF.")
