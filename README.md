# 📊 Digital Media Analytics with Intelligent Decision Support System

![NED University](https://img.shields.io/badge/NED%20University-Engineering%20%26%20Technology-blue?style=for-the-badge)
![Course](https://img.shields.io/badge/Course-CT--471%20DAV-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red?style=for-the-badge)

**Department of Computer Science & IT (Specialization in Data Science)**  
**Course:** CT-471 – Data Analytics and Visualization (DAV)  
**Supervisor:** Dr. Muhammad Umer Farooque

</div>

---

##  Overview

This project presents a complete end-to-end **Digital Media Analytics pipeline** for four leading Pakistani YouTube channels — **Ajj TV**, **Hum TV**, **Raftaar TV**, and **Studio One**. The system continuously monitors, evaluates, and derives strategic insights from multi-channel engagement data.

The pipeline addresses **14 distinct analytical tasks** spanning:

- Synthetic data generation at scale (2–5 GB)
- Hierarchical data modeling (nested JSON/relational)
- Advanced preprocessing & imputation
- Statistical & exploratory analysis
- NLP with transformer-based sentiment analysis
- Text augmentation (BERT, WordNet, back-translation)
- High-dimensional feature engineering
- Predictive modeling (XGBoost, LSTM, TFT)
- Unsupervised learning (K-Means, DBSCAN, Agglomerative)
- Graph construction & analytics (GCN, TGN, GNNExplainer)
- Temporal dynamics & burst detection
- Visualization & interpretability dashboards
- Strategic decision support framework

---

##  Architecture & Technical Stack

### Channel Parameters (Synthetic Dataset)

| Channel | Views μ (log) | Views σ | Like Rate | Comment Rate | Content |
|---|---|---|---|---|---|
| Ajj TV | 13.5 | 1.8 | 4.5% | 0.8% | News, Current Affairs |
| Hum TV | 14.2 | 1.5 | 6.2% | 1.2% | Drama, Entertainment |
| Raftaar TV | 12.8 | 2.0 | 3.8% | 0.6% | Mixed, Music, Sports |
| Studio One | 13.0 | 1.7 | 5.0% | 0.9% | Talk Shows, Lifestyle |

### Key Libraries

| Library | Role |
|---|---|
| PyTorch ≥ 2.0 | LSTM, TFT-Lite, GCN, TGN, GNNExplainer |
| HuggingFace Transformers ≥ 4.x | DistilBERT fill-mask, sentiment analysis |
| sentence-transformers ≥ 2.x | SBERT all-MiniLM-L6-v2 (384-D embeddings) |
| XGBoost ≥ 1.7 | Gradient-boosted regression & classification |
| scikit-learn ≥ 1.2 | Preprocessing, clustering, evaluation |
| NetworkX ≥ 3.x | Heterogeneous graph construction & analytics |
| BERTopic | UMAP + HDBSCAN + c-TF-IDF topic modeling |
| spaCy + NLTK | Linguistic preprocessing, lemmatization |
| statsmodels | Seasonal decomposition (STL-style) |
| JAX / jaxlib | TPU device enumeration, XLA pipeline |

---

##  Task Breakdown (Q1–Q14)

### Q1 — Synthetic Data Generation
- **500,000 video records** across 4 channels, serialized in Apache Parquet (Snappy compression)
- Log-normal view distributions validated via **Kolmogorov-Smirnov (KS) test**
- Beta-distributed like/comment rates reflecting real-world engagement sparsity
- **BERT-based title augmentation** using DistilBERT MLM (20% mask ratio)
- Chunked generation (50K records/chunk) for memory efficiency

### Q2 — Hierarchical Data Modeling
- Recursive JSON comment threads (up to depth 4) mimicking real YouTube reply trees
- Flattened via depth-first traversal into relational `df_comments` table
- Referential integrity validated: zero orphan references, zero duplicates

### Q3 — Advanced Preprocessing Pipeline
- **MCAR + MAR missingness injection** (3–7% random + view-dependent)
- **3-strategy imputation:** Group-aware Median → KNN (k=7) → IterativeImputer (RandomForest)
- **Consensus outlier detection** across Z-Score, IQR, and Isolation Forest (≥2/3 agreement)
- ~237,000 clean records after 5% outlier removal
- Weekly & monthly aggregations for temporal analysis

### Q4 — Statistical & Exploratory Analysis
- Dual Pearson + Spearman correlation matrices on 7 engagement metrics
- Distribution fitting with **AIC model comparison** (Log-Normal wins across all channels)
- STL-style seasonal decomposition with trend slope computation
- Isolation Forest anomaly detection (3% contamination → 7,014 consensus anomalies)

### Q5 — Advanced NLP Processing
- spaCy `en_core_web_sm` lemmatization pipeline (context-aware, avoids over-stemming)
- **SBERT** (all-MiniLM-L6-v2) → 5,000 × 384 embedding matrix
- Discourse-level sentiment metrics: coherence, sentiment arc, controversy index
- **BERTopic** (UMAP 384D→5D + HDBSCAN + c-TF-IDF) → 22 topics discovered
- K-Means on PCA-reduced SBERT space (best K=12, silhouette=0.533)

### Q6 — Text Augmentation & Expansion
- POS-aware WordNet synonym replacement (20% ratio)
- Back-translation simulation (double-pass: 25% → 15%)
- BERT contextual augmentation (DistilBERT MLM)
- All methods pass semantic preservation threshold (cosine similarity ≥ 0.80)

### Q7 — High-Dimensional Feature Engineering
- **13 temporal derivative features:** lags, rolling means, growth rates, velocity, acceleration, EMAs
- **Exponential decay curve fitting** (V₀·exp(−λt)) per channel via Levenberg-Marquardt
- **14 interaction-based metrics:** norm_engagement_idx, like_view_ratio, watch_efficiency, engagement_momentum, etc.
- **IncrementalPCA** (mini-batch, 5K rows) for 237K × 23 → 50D representation

### Q8 — Predictive Modeling

| Model | Task | Performance |
|---|---|---|
| Random Forest | Regression (engagement_ratio) | R² ≈ 0.72 |
| **XGBoost** | Regression + Classification | R² ≈ 0.78 / AUC ≈ 0.998 |
| LSTM (per-channel) | Weekly view forecasting | R² varies by channel |
| TFT-Lite (4-head attention) | Weekly view forecasting | Matches/exceeds LSTM |

Top predictors: `engagement_momentum`, `views_ema_7d`, `norm_engagement_idx`

### Q9 — Unsupervised Learning

| Method | Silhouette | Notes |
|---|---|---|
| K-Means (K=3) | 0.329 | Most scalable; best for full 237K dataset |
| Agglomerative (Ward, K=3) | 0.149 | Dendrogram reveals content hierarchy |
| DBSCAN (eps=3.89) | 0.413 | 4 clusters + 4.2% noise identification |

### Q10 — Graph Construction
- **Heterogeneous graph:** 6,016 nodes, 11,233 edges (Channels, Videos, Users, Topics)
- Edge types: `posted_on`, `commented`, `has_topic`, `similar_to` (cosine sim > 0.65)
- Saved as GraphML (2.3 MB, Gephi-compatible); 100% node coverage in single component

### Q11 — Graph Analytics
- **PageRank (α=0.85):** Channel nodes are top hubs; Raftaar TV highest PageRank
- **Betweenness centrality (k=800):** Ajj TV highest broker influence
- **Louvain community detection:** 106 communities, modularity Q=0.682
- **Link prediction:** Adamic-Adar index outperforms Jaccard & Preferential Attachment
- **GCN (2-layer):** Channel classification accuracy 0.77 vs 0.25 random baseline (+0.52 improvement)

### Q12 — Temporal Dynamics Analysis
- STL decomposition (period=13 weeks) isolating trend, seasonal, residual per channel
- **Burst detection** via rolling z-score (|z| > 2.5, window=8 weeks): 39 total bursts across channels
- **TGN** (Time2Vec → GRU memory → GraphSAGE → MLP): Test AUC-ROC = 0.9903, AP = 0.9908

### Q13 — Visualization & Interpretability
- 6-panel analytics dashboard (KPI heatmap, temporal trajectories, topic mix, network, regression fit, clustering)
- **Permutation importance** (model-agnostic, avoids high-cardinality bias)
- **Partial Dependence Plots** (non-linear engagement_momentum → diminishing returns at 1.5×)
- **GNNExplainer** (fidelity + entropy + size regularization): per-node edge masks up to 0.967

### Q14 — Decision Support Framework
- 8-KPI composite scorecard (avg_views, engagement_ratio, anomaly_rate, trend_slope, burst_count, PageRank, LSTM R², TFT R²)
- Evidence-backed programmatic recommendations per channel
- **Composite ranking:** Studio One (0.715) > Hum TV (0.650) > Ajj TV (0.436) > Raftaar TV (0.316)

---

##  Repository Structure

```
digital-media-analytics/
│
├── data/
│   ├── synthetic/               # Generated Parquet files (chunked)
│   ├── processed/               # df_clean, df_weekly, df_monthly
│   └── graph/                   # q10_hetero_graph.graphml
│
├── notebooks/
│   ├── Q1_Data_Generation.ipynb
│   ├── Q2_Hierarchical_Modeling.ipynb
│   ├── Q3_Preprocessing.ipynb
│   ├── Q4_Statistical_Analysis.ipynb
│   ├── Q5_NLP_Processing.ipynb
│   ├── Q6_Text_Augmentation.ipynb
│   ├── Q7_Feature_Engineering.ipynb
│   ├── Q8_Predictive_Modeling.ipynb
│   ├── Q9_Unsupervised_Learning.ipynb
│   ├── Q10_Graph_Construction.ipynb
│   ├── Q11_Graph_Analytics.ipynb
│   ├── Q12_Temporal_Dynamics.ipynb
│   ├── Q13_Visualization.ipynb
│   └── Q14_Decision_Support.ipynb
│
├── src/
│   ├── data_generation/
│   │   ├── synthetic_generator.py
│   │   └── bert_augmentation.py
│   ├── preprocessing/
│   │   ├── imputation.py
│   │   └── outlier_detection.py
│   ├── nlp/
│   │   ├── sbert_embeddings.py
│   │   ├── sentiment_discourse.py
│   │   └── topic_modeling.py
│   ├── models/
│   │   ├── xgboost_models.py
│   │   ├── lstm_forecaster.py
│   │   ├── tft_lite.py
│   │   ├── gcn_classifier.py
│   │   └── tgn_link_predictor.py
│   ├── graph/
│   │   ├── graph_builder.py
│   │   ├── centrality.py
│   │   └── gnn_explainer.py
│   └── decision_support/
│       ├── kpi_scorecard.py
│       └── recommendations.py
│
├── outputs/
│   ├── embeddings/              # q5_embeddings.npy
│   ├── models/                  # Saved model checkpoints
│   ├── figures/                 # All visualization outputs
│   └── reports/                 # q14_scorecard.csv, q14_recommendations.csv
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

##  Setup & Installation

### Prerequisites
- Python 3.10+
- Google Colab (recommended) with v5e-1 TPU or GPU runtime
- ~8 GB RAM minimum; 16 GB recommended for full dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/digital-media-analytics.git
cd digital-media-analytics

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### requirements.txt (key dependencies)

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
xgboost>=1.7.0
scikit-learn>=1.2.0
networkx>=3.0
bertopic
spacy>=3.5.0
nltk
statsmodels
pandas>=2.0.0
numpy>=1.24.0
pyarrow
fastparquet
scipy
matplotlib
seaborn
jax
jaxlib
```

### Running the Pipeline

```bash
# Run all tasks sequentially (recommended on Colab with GPU/TPU)
jupyter nbconvert --to notebook --execute notebooks/Q1_Data_Generation.ipynb

# Or run individual task notebooks
jupyter lab notebooks/Q8_Predictive_Modeling.ipynb
```

---

##  Key Results Summary

| Task | Method | Key Metric |
|---|---|---|
| Data Generation | Log-Normal + DistilBERT MLM | KS p-value > 0.60 for views/likes |
| Imputation | IterativeImputer (RF) | 0 remaining nulls |
| Outlier Detection | Consensus voting (3 methods) | 5% removed, ~237K clean records |
| Distribution Fitting | Log-Normal (AIC) | Best fit across all channels |
| Sentiment | DistilBERT SST-2 | Discourse coherence median = 0.42 |
| Topic Modeling | BERTopic | 22 topics, silhouette K=12 → 0.533 |
| XGBoost Regression | Gradient Boosting | R² = 0.98, AUC = 0.998 |
| LSTM Forecasting | 1-layer, hidden=32 | Per-channel R² varies |
| TFT-Lite | 4-head self-attention | Matches/exceeds LSTM |
| Clustering | K-Means (K=3) | Silhouette = 0.329 |
| Graph Community | Louvain | Q = 0.682, 106 communities |
| GCN Classification | 2-layer GCN | Val Acc = 0.77 (+0.52 vs random) |
| TGN Link Prediction | Time2Vec + GRU + GraphSAGE | AUC-ROC = 0.9903 |
| Decision Support | 8-KPI composite scorecard | Studio One leads at 0.715 |

---

##  Design Decisions

**Why Consensus Outlier Detection?**  
Each method (Z-Score, IQR, Isolation Forest) has different failure modes. Requiring ≥2/3 agreement reduces false positives — a genuinely viral video correctly triggers Z-score but is retained by Isolation Forest.

**Why IncrementalPCA over standard PCA?**  
Processing 237K × 23 features in mini-batches of 5,000 rows avoids loading the full matrix into RAM simultaneously — a critical requirement for TPU/large-dataset pipelines.

**Why Adamic-Adar over Jaccard for link prediction?**  
Adamic-Adar rewards shared neighbors with low degree — two users commenting on the same obscure video signals stronger affinity than both commenting on a viral video with millions of engagements.

**Why TFT-Lite over LSTM?**  
Multi-head self-attention allows each timestep to directly attend to any other position, avoiding the vanishing gradient problem in long LSTM chains. Burst patterns 10 weeks prior can directly influence current predictions.

---

##  Strategic Recommendations (Q14)

| Channel | Composite Score | Primary Action |
|---|---|---|
| **Studio One** | **0.715** | Lead-channel brand strategy; replicate 11 viral burst formats |
| Hum TV | 0.650 | Forward-sell ad inventory (predictable LSTM/TFT); quality-control review |
| Ajj TV | 0.436 | Audience-activation campaign (engagement below median); content decay monitoring |
| Raftaar TV | 0.316 | Programming refresh (A/B test); cross-promotion with high-PageRank channels |

---

##  References

- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
- Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecasting*.
- Rossi et al. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *ICML Workshop*.
- Ying et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS*.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

---

##  License

This project is for academic purposes under CT-471 at NED University of Engineering & Technology.

---
