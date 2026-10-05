import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association

df = pd.read_csv("bloom_results.csv")

# -------------------------
# PRE vs POST NEP
# -------------------------
df["Period"] = df["Year"].apply(
    lambda x: "Pre-NEP" if x < 2020 else "Post-NEP"
)

table_nep = pd.crosstab(
    df["Period"],
    df["Cognitive_Process"]
)

chi2, p, dof, expected = chi2_contingency(table_nep)
cramers_v = association(table_nep, method="cramer")

print("PRE vs POST NEP")
print(table_nep)
print(f"Chi-square = {chi2:.3f}")
print(f"df = {dof}")
print(f"p-value = {p:.4f}")
print(f"Cramer's V = {cramers_v:.3f}")


# -------------------------
# SUBJECT COMPARISON
# -------------------------
subjects = ["Physics", "Chemistry", "Biology"]

subject_df = df[df["Subject"].isin(subjects)]

table_subject = pd.crosstab(
    subject_df["Subject"],
    subject_df["Cognitive_Process"]
)

chi2, p, dof, expected = chi2_contingency(table_subject)
cramers_v = association(table_subject, method="cramer")

print("\nSUBJECT COMPARISON")
print(table_subject)
print(f"Chi-square = {chi2:.3f}")
print(f"df = {dof}")
print(f"p-value = {p:.4f}")
print(f"Cramer's V = {cramers_v:.3f}")
