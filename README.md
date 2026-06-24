# Machine-Learning-Unsupervised

## Overview

This project analyzes the **OSMI Mental Health in Tech Survey 2016** dataset to identify employee mental health patterns using unsupervised machine learning techniques.

The pipeline performs:

- Data quality assessment
- Exploratory Data Analysis (EDA)
- Feature engineering
- Missing value treatment
- Dimensionality reduction (PCA & t-SNE)
- K-Means clustering
- Hierarchical clustering
- Cluster profiling
- HR-focused insight generation
- Automated visualization and reporting

---

## Dataset

Dataset expected:

```
survey.csv
```

Source:

- OSMI (Open Sourcing Mental Illness)
- Mental Health in Tech Survey 2016

Place `survey.csv` in the same directory as the script.

---

## Project Workflow

### 1. Data Loading & Inspection

- Dataset shape
- Missing values
- Data types
- Summary statistics

### 2. Exploratory Data Analysis

Generates:

- Missing value visualization
- Mental health disorder distribution
- Age distribution
- Gender distribution
- Employer support analysis
- Remote work analysis
- Family history analysis

### 3. Data Preprocessing

- Gender standardization
- Age cleaning
- High-missing-column removal
- Free-text feature removal
- Geographic feature removal
- Role expansion into binary indicators
- Country encoding

### 4. Feature Engineering

Ordinal encoding of survey responses.

Creates composite HR indicators:

- employer_support_score
- openness_score
- perceived_stigma_score
- personal_mh_burden_score

### 5. Dimensionality Reduction

- Median imputation
- Feature scaling
- PCA
- t-SNE visualization

### 6. Clustering

Uses:

- K-Means
- Silhouette analysis
- Elbow method
- Hierarchical clustering (Ward linkage)

### 7. Cluster Characterization

Produces:

- Cluster projections
- Cluster sizes
- Radar profiles
- Feature heatmaps
- Demographic breakdowns

### 8. Summary Report

Generates:

- Cluster descriptions
- HR recommendations
- Clustered respondent dataset

---

## Generated Outputs

All outputs are saved inside:

```
imgff/
```

Generated files:

```
00_cluster_report.txt
01_missing_values.png
02_eda_overview.png
03_crosstab_disorder_vs_workplace.png
04_pca_scree.png
05_pca_loadings.png
06_elbow_silhouette.png
07_silhouette_diagram.png
08_dendrogram.png
09_cluster_projections.png
10_cluster_sizes.png
11_cluster_radar.png
12_cluster_heatmap.png
13_key_features_by_cluster.png
14_demographics_by_cluster.png
clustered_respondents.csv
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd mental-health-tech-clustering
```

### Create Virtual Environment

```bash
python -m venv venv
```

Linux/Mac:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run:

```bash
python main.py
```

Ensure:

```text
survey.csv
main.py
```

exist in the same folder.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

---

## Machine Learning Methods

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Clustering

- K-Means
- Agglomerative Hierarchical Clustering

### Evaluation

- Elbow Method
- Silhouette Score
- Silhouette Diagram

---

## HR Insights Produced

The clustering framework helps identify employee groups such as:

- Supported & Open Employees
- Unsupported & Burdened Employees
- Stigma-Aware Employees
- Private/Self-Managed Employees

Potential HR interventions:

1. Improve mental-health benefit communication
2. Launch anti-stigma campaigns
3. Introduce peer-support programs
4. Train managers in mental-health awareness
5. Strengthen anonymous support channels

---

## License

This project is intended for educational and research purposes.
