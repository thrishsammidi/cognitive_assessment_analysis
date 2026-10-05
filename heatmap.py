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
