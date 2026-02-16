# Last Editted on December 10, 2025
# Longitudinal autism analysis – Option A:
# pandas + seaborn + matplotlib + statsmodels + tableone
# This version mirrors your R script but uses standard scientific Python tools.


# ============================
# IMPORTS
# ============================

import pandas as pd              # pandas = Python's data frame library (similar to dplyr/tidyverse)
import numpy as np               # numpy = numerical computing (R uses base numeric types)
import seaborn as sns            # seaborn = high-level plotting (similar to ggplot2 themes)
import matplotlib.pyplot as plt  # matplotlib = low-level plotting backend (ggplot2 equivalent)

from tableone import TableOne    # Python version of R's tableone package
from statsmodels.regression.mixed_linear_model import MixedLM  # Mixed-effects models
import statsmodels.api as sm     # for QQ plots and diagnostics

sns.set(style="whitegrid")       # sets a ggplot-like theme


# ============================
# 1. LOAD + CLEAN DATA
# ============================

# In R: read_csv("path") %>% mutate(...)
# In Python: pd.read_csv() + .assign() or direct column assignment
autism = (
    pd.read_csv(
        r"~/BIST0650 Applied Longitudinal Data Analysis/BIST0650 Final Project/BIST050_Project_Data/autism.csv"
    )
)

# Convert variables to categorical (R uses factor(); Python uses .astype("category"))
autism["sicdegp"] = pd.Categorical(autism["sicdegp"], categories=["low", "med", "high"], ordered=True)
autism["bestest2"] = autism["bestest2"].astype("category")
autism["gender"] = autism["gender"].astype("category")
autism["race"] = autism["race"].astype("category")
autism["childid"] = autism["childid"].astype("category")

print(autism.info())   # R equivalent: str(autism)
print(autism.head())   # R equivalent: head(autism)


# ============================
# 2. SUMMARY STATISTICS
# ============================

# R: summary(autism)
print("\n=== Summary ===")
print(autism.describe(include="all"))

# R: n_distinct(childid)
print("\n=== Unique subjects ===")
print(autism["childid"].nunique())

# R: table(variable)
for col in ["childid", "age", "age2", "vsae", "obs", "gender", "race", "sicdegp", "bestest2"]:
    print(f"\n=== Count: {col} ===")
    print(autism[col].value_counts(dropna=False))

# R: mean(vsae)
print("\n=== Mean VSAE ===")
print(autism["vsae"].mean())


# ============================
# 3. TABLEONE SUMMARIES
# ============================

# Same variables as R
myVars = ["age", "bestest2", "gender", "race", "sicdegp"]
catVars = ["age", "age2", "bestest2", "childid", "gender", "obs", "race", "sicdegp", "vsae"]

# Python TableOne is nearly identical to R's version
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

# R: ggplot(autism, aes(...)) + geom_boxplot()
# Python: seaborn.boxplot()
def boxplot_var(var):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=autism, x=var, y="vsae")
    plt.title(f"VSAE by {var}")
    plt.tight_layout()
    plt.show()

for v in ["sicdegp", "age", "age2", "gender", "race", "bestest2", "obs"]:
    boxplot_var(v)

# R: ggplot spaghetti plot with geom_line(aes(group=childid))
# Python: must loop manually unless using plotnine (Option C)
plt.figure(figsize=(8, 5))
for cid, df_sub in autism.groupby("childid"):
    plt.plot(df_sub["age"], df_sub["vsae"], alpha=0.3)
plt.xlabel("Age (years)")
plt.ylabel("VSAE")
plt.title("VSAE Score Over Time by Child")
plt.tight_layout()
plt.show()

# Faceted spaghetti plots (similar to facet_grid in ggplot2)
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
# 5. MIXED MODELS (statsmodels)
# ============================

# R: lmer(vsae ~ age * sicdegp + (age | childid))
# Python: MixedLM.from_formula("vsae ~ age * sicdegp", groups="childid", re_formula="~ age")

def fit_mixed(formula, re_formula="~ age", data=autism):
    md = MixedLM.from_formula(
        formula,
        groups="childid",
        re_formula=re_formula,
        data=data
    )
    m = md.fit(method="lbfgs", maxiter=200)
    print(f"\n=== Mixed Model: {formula} ===")
    print(m.summary())
    return m

# R: anova(m1, m2)
# Python: statsmodels does NOT support LRT for mixed models → compare AIC/BIC manually
def compare_models(form1, form2, data=autism):
    m1 = fit_mixed(form1, data=data)
    m2 = fit_mixed(form2, data=data)

    print("\n=== Model Comparison (AIC/BIC/LL) ===")
    print("Model 1:", form1)
    print("  AIC:", m1.aic, "BIC:", m1.bic, "LL:", m1.llf)
    print("Model 2:", form2)
    print("  AIC:", m2.aic, "BIC:", m2.bic, "LL:", m2.llf)

    return m1, m2


# ============================
# 6. MIXED MODELS – QUESTIONS
# ============================

# Question A: Language
qa_m1, qa_m2 = compare_models(
    "vsae ~ age + sicdegp",
    "vsae ~ age * sicdegp"
)

# Question B: Diagnosis
qb_m1, qb_m2 = compare_models(
    "vsae ~ age + bestest2",
    "vsae ~ age * bestest2"
)

# Question C: Gender
qc_m1, qc_m2 = compare_models(
    "vsae ~ age + gender",
    "vsae ~ age * gender"
)

# Question C: Race
qd_m1, qd_m2 = compare_models(
    "vsae ~ age + race",
    "vsae ~ age * race"
)

# Gender × Race
qe_m1, qe_m2 = compare_models(
    "vsae ~ age + gender + race",
    "vsae ~ age + gender + race + gender:race"
)


# ============================
# 7. FINAL MODELS
# ============================

# Final model using age2
final_model_age2 = fit_mixed(
    "vsae ~ age2 * sicdegp + age2 * bestest2 + age2 * race",
    re_formula="~ age2",
    data=autism
)

# Final model using age
final_model_age = fit_mixed(
    "vsae ~ age * sicdegp + age * bestest2 + age * race",
    re_formula="~ age",
    data=autism
)


# ============================
# 8. DIAGNOSTICS
# ============================

# Choose final model for diagnostics
m = final_model_age

# R: fitted(m), resid(m)
autism["fitted"] = m.fittedvalues
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
re_dict = m.random_effects
re_df = (
    pd.DataFrame.from_dict(re_dict, orient="index")
    .reset_index()
    .rename(columns={"index": "childid", 0: "intercept", 1: "slope"})
)

plt.figure(figsize=(6, 4))
sns.histplot(re_df["intercept"], bins=30, kde=True)
plt.title("Distribution of Random Intercepts")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(re_df["slope"], bins=30, kde=True)
plt.title("Distribution of Random Slopes (Age)")
plt.tight_layout()
plt.show()

sm.qqplot(re_df["intercept"], line="45")
plt.title("Q-Q Plot: Random Intercepts")
plt.tight_layout()
plt.show()

sm.qqplot(re_df["slope"], line="45")
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
