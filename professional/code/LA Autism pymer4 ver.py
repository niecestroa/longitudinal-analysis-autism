# Last Editted: December 12, 2025
# Longitudinal autism analysis – Option B:
# pymer4 (lme4 syntax) + pandas + seaborn + tableone

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pymer4.models import Lmer
from tableone import TableOne
import statsmodels.api as sm

sns.set(style="whitegrid")


# ============================
# 1. LOAD + CLEAN DATA
# ============================

autism = (
    pd.read_csv(
        r"~/BIST0650 Applied Longitudinal Data Analysis/BIST0650 Final Project/BIST050_Project_Data/autism.csv"
    )
)

# Convert to categorical (matches R)
autism["sicdegp"] = pd.Categorical(autism["sicdegp"], categories=["low", "med", "high"], ordered=True)
autism["bestest2"] = autism["bestest2"].astype("category")
autism["gender"] = autism["gender"].astype("category")
autism["race"] = autism["race"].astype("category")
autism["childid"] = autism["childid"].astype("category")

print(autism.info())
print(autism.head())


# ============================
# 2. SUMMARY STATISTICS
# ============================

print("\n=== Summary ===")
print(autism.describe(include="all"))

print("\n=== Unique subjects ===")
print(autism["childid"].nunique())

for col in ["childid", "age", "age2", "vsae", "obs", "gender", "race", "sicdegp", "bestest2"]:
    print(f"\n=== Count: {col} ===")
    print(autism[col].value_counts(dropna=False))

print("\n=== Mean VSAE ===")
print(autism["vsae"].mean())


# ============================
# 3. TABLEONE SUMMARIES
# ============================

myVars = ["age", "bestest2", "gender", "race", "sicdegp"]
catVars = ["age", "age2", "bestest2", "childid", "gender", "obs", "race", "sicdegp", "vsae"]

def print_tableone(groupby):
    print(f"\n=== TableOne stratified by {groupby} ===")
    t1 = TableOne(
        autism,
        columns=myVars,
        categorical=catVars,
        groupby=groupby,
        pval=False
    )
    print(t1)

for g in ["bestest2", "gender", "race", "sicdegp", "obs"]:
    print_tableone(g)


# ============================
# 4. VISUALIZATIONS
# ============================

def boxplot_var(var):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=autism, x=var, y="vsae")
    plt.title(f"VSAE by {var}")
    plt.tight_layout()
    plt.show()

for v in ["sicdegp", "age", "age2", "gender", "race", "bestest2", "obs"]:
    boxplot_var(v)

# Spaghetti plot
plt.figure(figsize=(8, 5))
for cid, df_sub in autism.groupby("childid"):
    plt.plot(df_sub["age"], df_sub["vsae"], alpha=0.3)
plt.xlabel("Age (years)")
plt.ylabel("VSAE")
plt.title("VSAE Score Over Time by Child")
plt.tight_layout()
plt.show()

# Stratified spaghetti plots
def spaghetti_facet(by):
    g = sns.FacetGrid(autism, col=by, col_wrap=3, sharey=True, sharex=True, height=3)
    g.map_dataframe(
        sns.lineplot,
        x="age",
        y="vsae",
        hue="childid",
        legend=False,
        alpha=0.3
    )
    g.set_axis_labels("Age (years)", "VSAE")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"VSAE over Age stratified by {by}")
    plt.show()

for by in ["bestest2", "race", "gender", "sicdegp"]:
    spaghetti_facet(by)


# ============================
# 5. MIXED MODELS (pymer4)
# ============================

def fit_lmer(formula, data=autism):
    print(f"\n=== Lmer Model: {formula} ===")
    model = Lmer(formula, data=data)
    result = model.fit()
    print(result)
    return model, result

def compare_lmer(form1, form2, data=autism):
    print(f"\n=== Comparing Models ===")
    print(f"Model 1: {form1}")
    print(f"Model 2: {form2}")

    m1, r1 = fit_lmer(form1, data)
    m2, r2 = fit_lmer(form2, data)

    print("\n=== Likelihood Ratio Test ===")
    print(m1.compare(m2))

    return m1, m2


# ============================
# 6. MIXED MODELS – QUESTIONS
# ============================

# Question A: Language
qa_m1, qa_m2 = compare_lmer(
    "vsae ~ age + sicdegp + (age | childid)",
    "vsae ~ age * sicdegp + (age | childid)"
)

# Question B: Diagnosis
qb_m1, qb_m2 = compare_lmer(
    "vsae ~ age + bestest2 + (age | childid)",
    "vsae ~ age * bestest2 + (age | childid)"
)

# Question C: Gender
qc_m1, qc_m2 = compare_lmer(
    "vsae ~ age + gender + (age | childid)",
    "vsae ~ age * gender + (age | childid)"
)

# Question C: Race
qd_m1, qd_m2 = compare_lmer(
    "vsae ~ age + race + (age | childid)",
    "vsae ~ age * race + (age | childid)"
)

# Gender × Race
qe_m1, qe_m2 = compare_lmer(
    "vsae ~ age + gender + race + (age | childid)",
    "vsae ~ age + gender + race + gender:race + (age | childid)"
)


# ============================
# 7. FINAL MODELS
# ============================

# Final model using age2
final_age2_model, final_age2_result = fit_lmer(
    "vsae ~ age2 * sicdegp + age2 * bestest2 + age2 * race + (age2 | childid)"
)

# Final model using age
final_age_model, final_age_result = fit_lmer(
    "vsae ~ age * sicdegp + age * bestest2 + age * race + (age | childid)"
)


# ============================
# 8. DIAGNOSTICS
# ============================

# Choose final model for diagnostics
model = final_age_model
autism["fitted"] = model.predict()
autism["resid"] = autism["vsae"] - autism["fitted"]
autism["abs_resid"] = autism["resid"].abs()

# 1. Residuals vs Fitted
plt.figure(figsize=(6, 4))
sns.scatterplot(data=autism, x="fitted", y="resid", alpha=0.5)
plt.axhline(0, color="red")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()

# 2. Normal Q-Q plot
sm.qqplot(autism["resid"], line="45")
plt.title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()

# 3. Scale–Location plot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=autism, x="fitted", y=np.sqrt(autism["abs_resid"]), alpha=0.5)
sns.regplot(
    data=autism,
    x="fitted",
    y=np.sqrt(autism["abs_resid"]),
    scatter=False,
    lowess=True,
    color="blue"
)
plt.title("Scale–Location Plot")
plt.ylabel("√|Residuals|")
plt.tight_layout()
plt.show()

# 4. Residual trajectories by subject
plt.figure(figsize=(8, 5))
for cid, df_sub in autism.groupby("childid"):
    plt.plot(df_sub["fitted"], df_sub["resid"], alpha=0.3)
plt.title("Residual Trajectories by Subject")
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# 5. Random effects diagnostics
re_df = model.ranef.reset_index().rename(columns={"index": "childid"})

plt.figure(figsize=(6, 4))
sns.histplot(re_df["(Intercept)"], bins=30, kde=True)
plt.title("Distribution of Random Intercepts")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(re_df["age"], bins=30, kde=True)
plt.title("Distribution of Random Slopes (Age)")
plt.tight_layout()
plt.show()

sm.qqplot(re_df["(Intercept)"], line="45")
plt.title("Q-Q Plot: Random Intercepts")
plt.tight_layout()
plt.show()

sm.qqplot(re_df["age"], line="45")
plt.title("Q-Q Plot: Random Slopes (Age)")
plt.tight_layout()
plt.show()

# 6. Observed vs Fitted
plt.figure(figsize=(6, 4))
sns.scatterplot(data=autism, x="fitted", y="vsae", alpha=0.5)
sns.regplot(data=autism, x="fitted", y="vsae", scatter=False, lowess=True, color="blue")
plt.title("Observed vs Fitted VSAE")
plt.ylabel("Observed VSAE")
plt.tight_layout()
plt.show()
