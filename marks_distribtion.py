import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bloom_results.csv")

# Create 6 × 4 Revised Bloom's Matrix
matrix = pd.crosstab(
    df["Cognitive_Process"],
    df["Knowledge_Type"]
)

# Ensure all 24 cells are present
cognitive_order = [
    "Remember", "Understand", "Apply",
    "Analyze", "Evaluate", "Create"
]

knowledge_order = [
    "Factual", "Conceptual",
    "Procedural", "Metacognitive"
]

matrix = matrix.reindex(
    index=cognitive_order,
    columns=knowledge_order,
    fill_value=0
)

# Plot heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    cbar_kws={"label": "Number of Questions"}
)

plt.xlabel("Knowledge Type")
plt.ylabel("Cognitive Process")
plt.title("Distribution of Questions Across the Revised Bloom's Taxonomy Matrix")

plt.tight_layout()
plt.savefig(
    "bloom_matrix_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

######################################################################

import pandas as pd

df = pd.read_csv("bloom_results.csv")

df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

marks = df.pivot_table(
    index="Year",
    columns="Knowledge_Type",
    values="Marks",
    aggfunc="sum",
    fill_value=0
)

weightage = marks.div(marks.sum(axis=1), axis=0) * 100

order = [
    "Factual",
    "Conceptual",
    "Procedural",
    "Metacognitive"
]

weightage = weightage.reindex(columns=order, fill_value=0)

print(weightage.round(2))

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bloom_results.csv")

df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

df["Thinking_Level"] = df["Cognitive_Process"].map({
    "Remember": "LOTS",
    "Understand": "LOTS",
    "Apply": "LOTS",
    "Analyze": "HOTS",
    "Evaluate": "HOTS",
    "Create": "HOTS"
})

# Mark weightage by year
marks = pd.pivot_table(
    df,
    index="Year",
    columns="Thinking_Level",
    values="Marks",
    aggfunc="sum",
    fill_value=0
)

weightage = marks.div(marks.sum(axis=1), axis=0) * 100
weightage = weightage[["LOTS", "HOTS"]]

# Plot
fig, ax = plt.subplots(figsize=(8.5, 5.2))

weightage.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.72
)

# Formatting
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Share of Total Marks (%)", fontsize=11)

ax.set_ylim(0, 100)

ax.set_xticklabels(
    weightage.index,
    rotation=0,
    fontsize=9
)

ax.tick_params(axis="y", labelsize=9)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Subtle horizontal grid
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

ax.set_axisbelow(True)

# Legend
ax.legend(
    title="Cognitive Level",
    frameon=False,
    loc="upper right"
)

# Add percentage labels
for container in ax.containers:
    labels = [
        f"{bar.get_height():.0f}%"
        if bar.get_height() >= 5 else ""
        for bar in container
    ]

    ax.bar_label(
        container,
        labels=labels,
        label_type="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "LOTS_HOTS_mark_weightage_research.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
