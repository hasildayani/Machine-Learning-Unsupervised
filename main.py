"""
╔══════════════════════════════════════════════════════════════════╗
║     MENTAL HEALTH IN TECH – HR ANALYTICS & CLUSTERING PIPELINE  ║
║     Dataset : OSMI Mental Health in Tech Survey 2016            ║
║     Goal    : Cluster employees, reduce dimensionality,         ║
║               visualize patterns and derive HR insights         ║
╚══════════════════════════════════════════════════════════════════╝

WORKFLOW
────────
  0. Libraries & Config
  1. Data Loading & Initial Inspection
  2. Exploratory Data Analysis (EDA)
  3. Data Pre-processing
  4. Feature Engineering
  5. Dimensionality Reduction (PCA → t-SNE)
  6. Clustering  (K-Means: elbow + silhouette → Hierarchical)
  7. Cluster Characterisation & HR Insight Visualizations
  8. Summary Report
"""

# ══════════════════════════════════════════════════════════════════
# 0.  LIBRARIES & GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless / file-only rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.impute import SimpleImputer

# ── Aesthetics ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B8132", "#6B4226"]
FIGPATH = "imgff"        # all plots saved here


# ══════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & INITIAL INSPECTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 – DATA LOADING & INITIAL INSPECTION")
print("═"*60)

# ── Load your dataset ──────────────────────────────────────────
# ── Load your dataset ──────────────────────────────────────────
df_raw = pd.read_csv("survey.csv")

print(f"  Rows    : {df_raw.shape[0]}")
print(f"  Columns : {df_raw.shape[1]}")
print("\n  Column list:")
for i, c in enumerate(df_raw.columns):
    pct_miss = df_raw[c].isna().mean() * 100
    print(f"    [{i:2d}] {c[:70]:<70}  missing={pct_miss:5.1f}%")

print("\n  Dtypes summary:")
print(df_raw.dtypes.value_counts())
print("\n  Numeric columns – quick stats:")
print(df_raw.describe().T[["count","mean","std","min","max"]].round(2).to_string())


# ══════════════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 – EXPLORATORY DATA ANALYSIS")
print("═"*60)

# ── 2.1  Missing-value heat-map ─────────────────────────────────
miss_pct = (df_raw.isna().mean() * 100).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 10))
bars = ax.barh(miss_pct.index, miss_pct.values,
               color=[PALETTE[0] if v < 30 else
                      PALETTE[2] if v < 60 else
                      PALETTE[3] for v in miss_pct.values])
ax.axvline(30, color=PALETTE[2], lw=1.5, ls="--", label="30% threshold")
ax.axvline(60, color=PALETTE[3], lw=1.5, ls="--", label="60% threshold")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values per Column", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGPATH}01_missing_values.png", bbox_inches="tight")
plt.close()
print("01_missing_values.png")

# ── 2.2  EDA – key distributions ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Mental Health in Tech – Exploratory Overview", fontsize=15, fontweight="bold")

# Current MH disorder
ax = axes[0, 0]
col = "Do you currently have a mental health disorder?"
counts = df_raw[col].value_counts()
ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
       colors=PALETTE[:len(counts)], startangle=90,
       wedgeprops=dict(width=0.5, edgecolor="white"))
ax.set_title("Current Mental Health Disorder", fontweight="bold")

# Age distribution (clean)
ax = axes[0, 1]
age = df_raw["What is your age?"].dropna()
age = age[(age >= 18) & (age <= 70)]
ax.hist(age, bins=25, color=PALETTE[0], edgecolor="white", alpha=0.85)
ax.axvline(age.median(), color=PALETTE[3], lw=2, ls="--",
           label=f"Median={age.median():.0f}")
ax.set_title("Age Distribution (cleaned)", fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.legend()

# Gender (raw)
ax = axes[0, 2]
gender_raw = df_raw["What is your gender?"].dropna().str.strip().str.lower()
def _cat_gender(g):
    if g in ["male","m","man","cis male","male (cis)","male, cis"]:   return "Male"
    if g in ["female","f","woman","cis female","female (cis)","woman"]: return "Female"
    return "Other / Non-binary"
g_counts = gender_raw.map(_cat_gender).value_counts()
ax.bar(g_counts.index, g_counts.values, color=PALETTE[:3], edgecolor="white")
ax.set_title("Gender (standardised)", fontweight="bold")
ax.set_ylabel("Count")

# Employer mental-health benefits
ax = axes[1, 0]
bcol = "Does your employer provide mental health benefits as part of healthcare coverage?"
bc = df_raw[bcol].value_counts()
ax.barh(bc.index, bc.values, color=PALETTE[:len(bc)])
ax.set_title("Employer: MH Benefits", fontweight="bold")

# Remote work
ax = axes[1, 1]
rc = df_raw["Do you work remotely?"].value_counts()
ax.pie(rc, labels=rc.index, autopct="%1.1f%%",
       colors=PALETTE[:len(rc)], startangle=90,
       wedgeprops=dict(width=0.5, edgecolor="white"))
ax.set_title("Remote Work", fontweight="bold")

# Family history
ax = axes[1, 2]
fc = df_raw["Do you have a family history of mental illness?"].value_counts()
ax.bar(fc.index, fc.values, color=PALETTE[:len(fc)], edgecolor="white")
ax.set_title("Family History of Mental Illness", fontweight="bold")
ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{FIGPATH}02_eda_overview.png", bbox_inches="tight")
plt.close()
print("02_eda_overview.png")

# ── 2.3  Cross-tab: current disorder vs. employer support ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MH Disorder vs. Workplace Factors", fontsize=13, fontweight="bold")

for ax, col, title in zip(
        axes,
        ["Does your employer provide mental health benefits as part of healthcare coverage?",
         "Do you feel that your employer takes mental health as seriously as physical health?"],
        ["vs. Employer MH Benefits", "vs. Employer Takes MH Seriously"]
):
    ct = pd.crosstab(
        df_raw["Do you currently have a mental health disorder?"],
        df_raw[col], normalize="index"
    ) * 100
    ct.plot(kind="bar", ax=ax, color=PALETTE[:ct.shape[1]], edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("% within disorder group")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper right", fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGPATH}03_crosstab_disorder_vs_workplace.png", bbox_inches="tight")
plt.close()
print("03_crosstab_disorder_vs_workplace.png")


# ══════════════════════════════════════════════════════════════════
# 3.  DATA PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 – DATA PRE-PROCESSING")
print("═"*60)

df = df_raw.copy()

# ── 3.1  Standardise Gender ─────────────────────────────────────
def standardise_gender(g):
    if pd.isna(g):
        return np.nan
    g_l = str(g).strip().lower()
    if g_l in ["male","m","man","cis male","male (cis)","male, cis","cis-male"]:
        return 0   # Male
    if g_l in ["female","f","woman","cis female","female (cis)","female, cis","cis woman","cis-female"]:
        return 1   # Female
    return 2       # Other / Non-binary

df["gender_enc"] = df["What is your gender?"].apply(standardise_gender)
df.drop(columns=["What is your gender?"], inplace=True)

# ── 3.2  Clean Age ──────────────────────────────────────────────
df["age_clean"] = df["What is your age?"].apply(
    lambda x: x if pd.notna(x) and 18 <= x <= 70 else np.nan
)
df.drop(columns=["What is your age?"], inplace=True)

# ── 3.3  Drop high-missingness & free-text columns ──────────────
#   Threshold: >60% missing → drop
high_miss = miss_pct[miss_pct > 60].index.tolist()

FREE_TEXT = [
    "Why or why not?",
    "Why or why not?.1",
    "If yes, what condition(s) have you been diagnosed with?",
    "If maybe, what condition(s) do you believe you have?",
    "If so, what condition(s) were you diagnosed with?",
]
# Geo columns (too granular / not predictive at state level)
GEO_COLS = [
    "What US state or territory do you live in?",
    "What US state or territory do you work in?",
    "What country do you work in?",
]

DROP_SET = set(high_miss) | set(FREE_TEXT) | set(GEO_COLS)
df.drop(columns=[c for c in DROP_SET if c in df.columns], inplace=True)
print(f"  Dropped {len(DROP_SET)} columns (high-missing / free-text / geo)")

# ── 3.4  Work-position: multi-label binary features ─────────────
ROLES = ["Back-end Developer","Front-end Developer","DevOps/SysAdmin",
         "Designer","Support","Executive Leadership",
         "Supervisor/Team Lead","Dev Evangelist/Advocate",
         "One-person shop","Other"]

pos_col = "Which of the following best describes your work position?"
if pos_col in df.columns:
    for role in ROLES:
        safe = role.replace("/","_").replace(" ","_")
        df[f"role_{safe}"] = df[pos_col].apply(
            lambda x: 1 if isinstance(x, str) and role in x else 0
        )
    df.drop(columns=[pos_col], inplace=True)
    print(f"  Expanded work-position into {len(ROLES)} binary role features")

# ── 3.5  Country: top-5 + 'Other' dummy encoding ───────────────
ctry_col = "What country do you live in?"
if ctry_col in df.columns:
    top5 = df[ctry_col].value_counts().nlargest(5).index
    df["country_grp"] = df[ctry_col].apply(lambda x: x if x in top5 else "Other")
    ctry_dummies = pd.get_dummies(df["country_grp"], prefix="ctry", drop_first=False)
    df = pd.concat([df.drop(columns=[ctry_col,"country_grp"]), ctry_dummies], axis=1)
    print(f"  Country encoded: {ctry_dummies.columns.tolist()}")

print(f"\n  Shape after basic pre-processing: {df.shape}")


# ══════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING  (ordinal encoding of all categoricals)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 – FEATURE ENGINEERING")
print("═"*60)

nan = np.nan  # shorthand

ORDINAL_MAPS = {
    # ── self-employment & company ───────────────────────────────
    "Are you self-employed?":
        {0: 0, 1: 1},

    "How many employees does your company or organization have?":
        {"1-5":0, "6-25":1, "26-100":2, "100-500":3, "500-1000":4, "More than 1000":5},

    "Is your employer primarily a tech company/organization?":
        {0.0:0, 1.0:1},

    # ── current employer: support & policy ──────────────────────
    "Does your employer provide mental health benefits as part of healthcare coverage?":
        {"Yes":1, "No":0, "I don't know":0.5, "Not eligible for coverage / N/A":nan},

    "Do you know the options for mental health care available under your employer-provided coverage?":
        {"Yes":1, "No":0, "I am not sure":0.5},

    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?":
        {"Yes":1, "No":0, "I don't know":0.5},

    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?":
        {"Yes":1, "No":0, "I don't know":0.5},

    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?":
        {"Yes":1, "No":0, "I don't know":0.5},

    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:":
        {"Very easy":1.0, "Somewhat easy":0.75,
         "Neither easy nor difficult":0.5,
         "Somewhat difficult":0.25, "Very difficult":0.0,
         "I don't know":nan},

    "Do you think that discussing a mental health disorder with your employer would have negative consequences?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Do you think that discussing a physical health issue with your employer would have negative consequences?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Would you feel comfortable discussing a mental health disorder with your coworkers?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Do you feel that your employer takes mental health as seriously as physical health?":
        {"Yes":1, "No":0, "I don't know":0.5},

    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?":
        {"Yes":1, "No":0, "I don't know":0.5},

    # ── previous employer ────────────────────────────────────────
    "Do you have previous employers?":
        {0:0, 1:1},

    "Have your previous employers provided mental health benefits?":
        {"Yes, they all did":1, "Some did":0.5, "No, none did":0, "I don't know":nan},

    "Were you aware of the options for mental health care provided by your previous employers?":
        {"Yes, I was aware of all of them":1, "I was aware of some":0.5,
         "No, I only became aware later":0, "N/A (not currently aware)":nan},

    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?":
        {"Yes, they all did":1, "Some did":0.5, "None did":0, "I don't know":nan},

    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":
        {"Yes, they all did":1, "Some did":0.5, "None did":0},

    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?":
        {"Yes, always":1, "Sometimes":0.5, "No":0, "I don't know":nan},

    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Do you think that discussing a physical health issue with previous employers would have negative consequences?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Would you have been willing to discuss a mental health issue with your previous co-workers?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Did you feel that your previous employers took mental health as seriously as physical health?":
        {"Yes, they all did":1, "Some did":0.5, "None did":0, "I don't know":nan},

    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?":
        {"Yes":1, "No":0, "I don't know":0.5},

    # ── potential employer ───────────────────────────────────────
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Would you bring up a mental health issue with a potential employer in an interview?":
        {"Yes":1, "No":0, "Maybe":0.5},

    # ── stigma & career perceptions ──────────────────────────────
    "Do you feel that being identified as a person with a mental health issue would hurt your career?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?":
        {"Yes, they do":1, "Yes, I think they would":1,
         "Maybe":0.5,
         "No, I don't think they would":0, "No, they do not":0,
         "I'm not sure":0.5},

    "How willing would you be to share with friends and family that you have a mental illness?":
        {"Very open":1.0, "Somewhat open":0.75, "Neutral":0.5,
         "Somewhat not open":0.25, "Not open at all":0.0,
         "Not applicable to me (I do not have a mental illness)":nan},

    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?":
        {"Yes, I experienced":1.0, "Yes, I observed":0.75,
         "Maybe/Not sure":0.5, "No":0.0},

    "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?":
        {"Yes":1, "No":0, "Maybe":0.5},

    # ── personal MH history & treatment ─────────────────────────
    "Do you have a family history of mental illness?":
        {"Yes":1, "No":0, "I don't know":0.5},

    "Have you had a mental health disorder in the past?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Do you currently have a mental health disorder?":
        {"Yes":1, "No":0, "Maybe":0.5},

    "Have you been diagnosed with a mental health condition by a medical professional?":
        {"Yes":1, "No":0},

    "Have you ever sought treatment for a mental health issue from a mental health professional?":
        {0:0, 1:1, True:1, False:0},

    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":
        {"Often":1.0, "Sometimes":0.67, "Rarely":0.33,
         "Never":0.0, "Not applicable to me":nan},

    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":
        {"Often":1.0, "Sometimes":0.67, "Rarely":0.33,
         "Never":0.0, "Not applicable to me":nan},

    # ── remote work ──────────────────────────────────────────────
    "Do you work remotely?":
        {"Always":1.0, "Sometimes":0.5, "Never":0.0},
}

# Apply all maps
for col, mapping in ORDINAL_MAPS.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ── 4.1  Derived / composite features ───────────────────────────
employer_cols = [
    "Does your employer provide mental health benefits as part of healthcare coverage?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
    "Do you feel that your employer takes mental health as seriously as physical health?",
]
present = [c for c in employer_cols if c in df.columns]
df["employer_support_score"] = df[present].mean(axis=1)

openness_cols = [
    "Would you feel comfortable discussing a mental health disorder with your coworkers?",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
    "Would you bring up a mental health issue with a potential employer in an interview?",
    "How willing would you be to share with friends and family that you have a mental illness?",
]
present_o = [c for c in openness_cols if c in df.columns]
df["openness_score"] = df[present_o].mean(axis=1)

stigma_cols = [
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?",
]
present_s = [c for c in stigma_cols if c in df.columns]
df["perceived_stigma_score"] = df[present_s].mean(axis=1)

mh_status_cols = [
    "Do you currently have a mental health disorder?",
    "Have you had a mental health disorder in the past?",
    "Do you have a family history of mental illness?",
    "Have you been diagnosed with a mental health condition by a medical professional?",
    "Have you ever sought treatment for a mental health issue from a mental health professional?",
]
present_m = [c for c in mh_status_cols if c in df.columns]
df["personal_mh_burden_score"] = df[present_m].mean(axis=1)

print("  Composite features created:")
for f in ["employer_support_score","openness_score",
          "perceived_stigma_score","personal_mh_burden_score"]:
    print(f"    {f}: mean={df[f].mean():.3f}, std={df[f].std():.3f}")

# ── 4.2  Drop any remaining object columns ───────────────────────
obj_remaining = df.select_dtypes("object").columns.tolist()
if obj_remaining:
    print(f"\n  Dropping remaining object columns: {obj_remaining}")
    df.drop(columns=obj_remaining, inplace=True)

print(f"\n  Final feature matrix: {df.shape}")


# ══════════════════════════════════════════════════════════════════
# 5.  IMPUTATION, SCALING & DIMENSIONALITY REDUCTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 – IMPUTATION, SCALING & DIMENSIONALITY REDUCTION")
print("═"*60)

# Force all columns to numeric (handles bool → float, coerces others → NaN)
df = df.apply(pd.to_numeric, errors="coerce")
# Drop any column that is entirely NaN
df.dropna(axis=1, how="all", inplace=True)
feature_names = df.columns.tolist()
print(f"  Feature columns after dtype enforcement: {len(feature_names)}")

# ── 5.1  Impute NaN with median ─────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(df)
X_imp   = X_imp[:, :len(feature_names)]   # guard against shape drift

# ── 5.2  Standardise ─────────────────────────────────────────────
scaler  = StandardScaler()
X_std   = scaler.fit_transform(X_imp)

# ── 5.3  PCA – full (for explained variance) ────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_std)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
n_comp90 = int(np.searchsorted(cum_var, 0.90)) + 1
n_comp80 = int(np.searchsorted(cum_var, 0.80)) + 1
print(f"  PCA: {n_comp80} components → 80% variance | "
      f"{n_comp90} components → 90% variance")

# Plot scree
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, min(31, X_std.shape[1]+1)),
            pca_full.explained_variance_ratio_[:30],
            color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].set_title("PCA – Scree Plot (top 30 components)", fontweight="bold")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Explained Variance Ratio")

axes[1].plot(range(1, len(cum_var)+1), cum_var, color=PALETTE[0], lw=2)
axes[1].axhline(0.80, ls="--", color=PALETTE[2], label="80%")
axes[1].axhline(0.90, ls="--", color=PALETTE[3], label="90%")
axes[1].axvline(n_comp80, ls=":", color=PALETTE[2], alpha=0.7)
axes[1].axvline(n_comp90, ls=":", color=PALETTE[3], alpha=0.7)
axes[1].set_title("Cumulative Explained Variance", fontweight="bold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGPATH}04_pca_scree.png", bbox_inches="tight")
plt.close()
print("04_pca_scree.png")

# ── 5.4  PCA – reduce to 90% variance for clustering ────────────
pca = PCA(n_components=n_comp90, random_state=42)
X_pca = pca.fit_transform(X_std)
print(f"  Reduced to {X_pca.shape[1]} PCA components for clustering")

# ── 5.5  PCA 2-D projection for visualisation ───────────────────
pca2d = PCA(n_components=2, random_state=42)
X_pca2d = pca2d.fit_transform(X_std)

# ── 5.6  t-SNE 2-D projection ───────────────────────────────────
print("  Running t-SNE (this may take a moment)…")
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
            random_state=42)
X_tsne = tsne.fit_transform(X_std)
print("  t-SNE complete.")

# ── 5.7  PCA loadings heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
n_feat_pca = pca.components_.shape[1]
n_pc_show  = min(8, pca.n_components_)
loading_matrix = pd.DataFrame(
    pca.components_[:n_pc_show].T,
    index=[f[:35] for f in feature_names[:n_feat_pca]],
    columns=[f"PC{i+1}" for i in range(n_pc_show)]
)
sns.heatmap(loading_matrix, ax=ax, cmap="coolwarm", center=0,
            linewidths=0.5, annot=False, vmin=-0.25, vmax=0.25)
ax.set_title("PCA Component Loadings (top 8 PCs)", fontweight="bold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{FIGPATH}05_pca_loadings.png", bbox_inches="tight")
plt.close()
print("05_pca_loadings.png")


# ══════════════════════════════════════════════════════════════════
# 6.  CLUSTERING
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 – CLUSTERING")
print("═"*60)

# ── 6.1  Elbow + Silhouette to find optimal K ───────────────────
K_RANGE = range(2, 10)
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

best_k = K_RANGE.start + int(np.argmax(sil_scores))
print(f"\n  ✓ Best K by silhouette: {best_k}")

# Plot elbow + silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_RANGE, inertias, "o-", color=PALETTE[0], lw=2)
axes[0].set_title("Elbow Method", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (k)"); axes[0].set_ylabel("Inertia")

axes[1].plot(K_RANGE, sil_scores, "o-", color=PALETTE[1], lw=2)
axes[1].axvline(best_k, ls="--", color=PALETTE[3], label=f"Best k={best_k}")
axes[1].set_title("Silhouette Score", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (k)"); axes[1].set_ylabel("Silhouette Score")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGPATH}06_elbow_silhouette.png", bbox_inches="tight")
plt.close()
print("  [saved] 06_elbow_silhouette.png")

# ── 6.2  Silhouette diagram for best K ──────────────────────────
km_best = KMeans(n_clusters=best_k, n_init=30, max_iter=1000, random_state=42)
labels_best = km_best.fit_predict(X_pca)
sil_vals = silhouette_samples(X_pca, labels_best)

fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
for i in range(best_k):
    ith_sil = np.sort(sil_vals[labels_best == i])
    size_i  = ith_sil.shape[0]
    y_upper = y_lower + size_i
    color   = PALETTE[i % len(PALETTE)]
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     facecolor=color, alpha=0.75, edgecolor="none")
    ax.text(-0.04, y_lower + 0.5 * size_i, str(i+1), fontsize=9)
    y_lower = y_upper + 10

ax.axvline(np.mean(sil_vals), ls="--", color="red", label=f"Mean={np.mean(sil_vals):.3f}")
ax.set_title(f"Silhouette Diagram — k={best_k}", fontweight="bold")
ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGPATH}07_silhouette_diagram.png", bbox_inches="tight")
plt.close()
print("  [saved] 07_silhouette_diagram.png")

# ── 6.3  Hierarchical clustering dendrogram ─────────────────────
# Use a representative sample if dataset is large
sample_idx = np.random.choice(len(X_pca), size=min(300, len(X_pca)), replace=False)
Z = linkage(X_pca[sample_idx], method="ward")

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=40,
           leaf_rotation=45, leaf_font_size=7,
           color_threshold=Z[-best_k, 2])
ax.axhline(Z[-best_k, 2], ls="--", color=PALETTE[3], lw=1.5,
           label=f"Cut for k={best_k}")
ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage, sample=300)",
             fontweight="bold")
ax.set_xlabel("Sample index"); ax.set_ylabel("Distance")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGPATH}08_dendrogram.png", bbox_inches="tight")
plt.close()
print("  [saved] 08_dendrogram.png")

# ── 6.4  Assign cluster labels (1-indexed for readability) ───────
cluster_col = labels_best + 1   # shift to 1..best_k
df_out = df_raw.copy()
df_out["cluster"] = cluster_col


# ══════════════════════════════════════════════════════════════════
# 7.  CLUSTER CHARACTERISATION & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 7 – CLUSTER CHARACTERISATION")
print("═"*60)

df_feat = pd.DataFrame(X_imp, columns=feature_names)
df_feat["cluster"] = cluster_col

# ── 7.1  PCA-2D scatter coloured by cluster ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, X_2d, title in zip(
        axes,
        [X_pca2d, X_tsne],
        ["PCA 2-D Projection", "t-SNE 2-D Projection"]):
    for c in range(1, best_k+1):
        mask = cluster_col == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=PALETTE[(c-1) % len(PALETTE)],
                   alpha=0.55, s=18, label=f"Cluster {c}")
    ax.set_title(title, fontweight="bold")
    ax.legend(markerscale=2)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

plt.suptitle("Cluster Projections", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGPATH}09_cluster_projections.png", bbox_inches="tight")
plt.close()
print("  [saved] 09_cluster_projections.png")

# ── 7.2  Cluster sizes ───────────────────────────────────────────
sizes = pd.Series(cluster_col).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([f"Cluster {c}" for c in sizes.index], sizes.values,
              color=[PALETTE[i % len(PALETTE)] for i in range(len(sizes))],
              edgecolor="white")
for bar, val in zip(bars, sizes.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("Cluster Sizes", fontweight="bold")
ax.set_ylabel("Number of Respondents")
plt.tight_layout()
plt.savefig(f"{FIGPATH}10_cluster_sizes.png", bbox_inches="tight")
plt.close()
print("  [saved] 10_cluster_sizes.png")

# ── 7.3  Composite-score profiles per cluster ────────────────────
COMPOSITE_COLS = [
    "employer_support_score",
    "openness_score",
    "perceived_stigma_score",
    "personal_mh_burden_score",
]
cluster_profiles = df_feat[COMPOSITE_COLS + ["cluster"]].groupby("cluster").mean()
print("\n  Cluster composite profiles:")
print(cluster_profiles.round(3).to_string())

# Radar chart
def radar_chart(ax, values, labels, color, title, ylim=(0,1)):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = list(values) + [values[0]]
    ax.plot(angles, values_plot, "o-", lw=2, color=color)
    ax.fill(angles, values_plot, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [l.replace("_score","").replace("_"," ").title() for l in labels],
        size=8)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold", pad=15, size=9)

cols = min(best_k, 4)
rows = (best_k + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows),
                         subplot_kw=dict(polar=True))
axes_flat = np.array(axes).flatten() if best_k > 1 else [axes]

for i, c in enumerate(cluster_profiles.index):
    radar_chart(
        axes_flat[i],
        cluster_profiles.loc[c].values,
        COMPOSITE_COLS,
        PALETTE[(c-1) % len(PALETTE)],
        f"Cluster {c}  (n={sizes[c]})"
    )
for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Cluster Profiles – Composite Scores", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGPATH}11_cluster_radar.png", bbox_inches="tight")
plt.close()
print("  [saved] 11_cluster_radar.png")

# ── 7.4  Heatmap: cluster means for ALL features ─────────────────
cluster_means_all = df_feat.groupby("cluster").mean()
# Normalise each column 0-1 for visual consistency
cm_norm = (cluster_means_all - cluster_means_all.min()) / \
          (cluster_means_all.max() - cluster_means_all.min() + 1e-8)

fig, ax = plt.subplots(figsize=(14, max(4, best_k * 1.4)))
sns.heatmap(cm_norm.T, ax=ax, cmap="YlOrRd", linewidths=0.4,
            cbar_kws={"label": "Normalised mean (0–1)"},
            xticklabels=[f"Cluster {c}" for c in cm_norm.index],
            yticklabels=[f[:38] for f in cm_norm.columns])
ax.set_title("Feature Means per Cluster (normalised)", fontweight="bold")
ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{FIGPATH}12_cluster_heatmap.png", bbox_inches="tight")
plt.close()
print("  [saved] 12_cluster_heatmap.png")

# ── 7.5  Key feature comparison across clusters ──────────────────
KEY_FEATURES = [
    "Do you currently have a mental health disorder?",
    "Have you ever sought treatment for a mental health issue from a mental health professional?",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Does your employer provide mental health benefits as part of healthcare coverage?",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?",
    "employer_support_score",
    "perceived_stigma_score",
    "openness_score",
    "personal_mh_burden_score",
]
kf_available = [f for f in KEY_FEATURES if f in df_feat.columns]
kf_means = df_feat[kf_available + ["cluster"]].groupby("cluster").mean()

n_kf = len(kf_available)
fig, axes = plt.subplots(2, (n_kf+1)//2, figsize=(18, 9))
axes_flat = axes.flatten()

for i, feat in enumerate(kf_available):
    ax = axes_flat[i]
    vals = kf_means[feat]
    bars = ax.bar([f"C{c}" for c in vals.index], vals.values,
                  color=[PALETTE[(c-1) % len(PALETTE)] for c in vals.index],
                  edgecolor="white")
    ax.set_title(feat[:45], fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score (0–1)")
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=7)

for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Key Feature Means by Cluster", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGPATH}13_key_features_by_cluster.png", bbox_inches="tight")
plt.close()
print("  [saved] 13_key_features_by_cluster.png")

# ── 7.6  Demographic breakdown by cluster ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Demographics by Cluster", fontsize=13, fontweight="bold")

df_raw2 = df_raw.copy()
df_raw2["cluster"] = cluster_col
df_raw2["gender_std"] = df_raw["What is your gender?"].apply(
    lambda g: "Male" if str(g).strip().lower() in ["male","m","man","cis male","male (cis)"]
              else "Female" if str(g).strip().lower() in ["female","f","woman","cis female","female (cis)"]
              else "Other"
)
age_clean2 = df_raw["What is your age?"].apply(
    lambda x: x if pd.notna(x) and 18 <= x <= 70 else np.nan
)
df_raw2["age_clean"] = age_clean2

# Gender
ct_gender = pd.crosstab(df_raw2["cluster"], df_raw2["gender_std"], normalize="index") * 100
ct_gender.plot(kind="bar", ax=axes[0], stacked=True,
               color=PALETTE[:ct_gender.shape[1]], edgecolor="none")
axes[0].set_title("Gender Mix (%)", fontweight="bold")
axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("%")
axes[0].tick_params(axis="x", rotation=0)

# Age boxplot
df_raw2.boxplot(column="age_clean", by="cluster", ax=axes[1],
                boxprops=dict(color=PALETTE[0]),
                medianprops=dict(color=PALETTE[3], lw=2))
axes[1].set_title("Age Distribution by Cluster", fontweight="bold")
axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Age")
plt.sca(axes[1]); plt.title("Age by Cluster")

# Remote work
ct_remote = pd.crosstab(df_raw2["cluster"],
                         df_raw2["Do you work remotely?"],
                         normalize="index") * 100
ct_remote.plot(kind="bar", ax=axes[2], stacked=True,
               color=PALETTE[:ct_remote.shape[1]], edgecolor="none")
axes[2].set_title("Remote Work Distribution (%)", fontweight="bold")
axes[2].set_xlabel("Cluster"); axes[2].set_ylabel("%")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(f"{FIGPATH}14_demographics_by_cluster.png", bbox_inches="tight")
plt.close()
print("  [saved] 14_demographics_by_cluster.png")


# ══════════════════════════════════════════════════════════════════
# 8.  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 8 – CLUSTER SUMMARY REPORT")
print("═"*60)

cluster_labels_text = {
    1: "Unsupported & Burdened",
    2: "Supported & Open",
    3: "Self-managed & Private",
    4: "Stigma-aware & Cautious",
}

report_lines = [
    "═"*60,
    "  MENTAL HEALTH IN TECH – CLUSTER REPORT",
    f"  Dataset : {df_raw.shape[0]} respondents | {df_raw.shape[1]} questions",
    f"  Model   : K-Means  (k={best_k}, best silhouette={max(sil_scores):.4f})",
    "═"*60,
    "",
]

for c in range(1, best_k + 1):
    label = cluster_labels_text.get(c, f"Cluster {c}")
    n     = sizes[c]
    row   = cluster_profiles.loc[c]
    report_lines += [
        f"┌──────────────────────────────────────────",
        f"│ Cluster {c}: {label}  (n={n}, {n/len(cluster_col)*100:.1f}%)",
        f"├──────────────────────────────────────────",
        f"│  Employer Support   : {row['employer_support_score']:.2f}  (0=none, 1=excellent)",
        f"│  Openness Score     : {row['openness_score']:.2f}  (0=closed, 1=very open)",
        f"│  Perceived Stigma   : {row['perceived_stigma_score']:.2f}  (0=none, 1=high)",
        f"│  Personal MH Burden : {row['personal_mh_burden_score']:.2f}  (0=none, 1=high)",
        f"└──────────────────────────────────────────",
        "",
    ]

report_lines += [
    "GENERATED OUTPUTS",
    "─────────────────",
    "  01_missing_values.png",
    "  02_eda_overview.png",
    "  03_crosstab_disorder_vs_workplace.png",
    "  04_pca_scree.png",
    "  05_pca_loadings.png",
    "  06_elbow_silhouette.png",
    "  07_silhouette_diagram.png",
    "  08_dendrogram.png",
    "  09_cluster_projections.png",
    "  10_cluster_sizes.png",
    "  11_cluster_radar.png",
    "  12_cluster_heatmap.png",
    "  13_key_features_by_cluster.png",
    "  14_demographics_by_cluster.png",
    "",
    "HR RECOMMENDATION LEVER POINTS",
    "───────────────────────────────",
    "  1. Improve employer MH benefit communication (clusters with low employer_support_score)",
    "  2. Anti-stigma campaigns targeting clusters with high perceived_stigma_score",
    "  3. Peer support programs for clusters with high personal_mh_burden_score",
    "  4. Manager training for clusters showing low openness_score",
    "  5. Anonymous reporting channels for clusters reporting workplace incidents",
    "═"*60,
]

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{FIGPATH}00_cluster_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("\n  [saved] 00_cluster_report.txt")

# ── Export clustered dataset ──────────────────────────────────────
df_out.to_csv(f"{FIGPATH}clustered_respondents.csv", index=False)
print("  [saved] clustered_respondents.csv")

print("\n✅  Pipeline complete — all outputs in imgff")
