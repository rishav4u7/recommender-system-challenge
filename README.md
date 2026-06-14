# Recommendation System Challenge

A hybrid, scalable recommendation system built under realistic operational constraints. It combines collaborative filtering (Matrix Factorization & Item-CF) with content-based signals (user demographics & item metadata) to predict user clicks on promotional ads.

---

## Setup & Execution

### Environment Requirements
- **Python**: Version 3.10 or higher.
- **Libraries**: `pandas`, `numpy`, `scipy`, `scikit-learn`, `streamlit`.

### Foolproof One-Click Execution (Recommended for Recruiters)
We have packaged automated launchers in the root directory:
- **Windows**: Double-click [run_dashboard.bat](file:///d:/Gen%20AI/challenge_package/run_dashboard.bat) (or run `run_dashboard.bat` in terminal).
- **macOS/Linux**: Run `./run_dashboard.sh` in terminal.

These scripts check/install all dependencies automatically and launch the interactive dashboard.

### Manual Execution

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run Recommendation Engine (Generates CSV and evaluates)
```bash
python recommend.py
```
This saves top-10 recommended items to `recommendations.csv` and prints metrics to the console.

#### 3. Run Streamlit Dashboard
```bash
streamlit run app.py
```
*(Note: If your local Python environment has environment/wrapper path issues with the default `streamlit` command, launch it safely using the Python module executor: `python -m streamlit run app.py`)*

---

## Model Design & Evaluation Strategy

### Recommendation Approach (Hybrid Stacking)
We implement a **two-stage hybrid recommendation model**:
1. **Stage 1 (Unsupervised Collaboratives & CTR Stats)**:
   - **SVD Matrix Factorization**: Performs Singular Value Decomposition on the user-item click matrix using optimized latent factors ($k=10$) to extract clean user-item latent similarities.
   - **Item-based Collaborative Filtering**: Computes cosine similarities between items using click overlap and predicts item scores based on user click histories.
   - **Laplace CTR Smoothing**: Smoothes all CTR features using a Laplace parameter $\beta=5$ to prevent noise in user/item affinities with very few impressions.
2. **Stage 2 (Supervised Stacking)**:
   - A **Logistic Regression** classifier is trained to predict the probability of a click (`INTERACTION = 1`).
   - **Features**: SVD score, Item-CF score, user overall click-through rate (CTR), user category CTR affinity, user price-bucket CTR affinity, item overall CTR, user age group, item category, and price bucket.
   - **Generalization (Cold Start)**: By utilizing content-based features (age, category, price) alongside collaborative signals, the model successfully generates high-quality recommendations for cold-start users (users with no historical clicks).
   - **Inference Optimization**: Inference is fully vectorized using NumPy broadcasting (matrix operations), enabling the calculation of all 9.5 million user-item predictions in under 20 seconds.

### Data Splitting Strategy
To ensure a realistic offline evaluation, we utilize a **temporal split** instead of a random split:
- **Test Set (last 10% chronologically)**: Sorted by timestamp, the last 30,000 interactions (interactions in May–June 2024) serve as the test set. This simulates future deployment.
- **Train/Val Set (first 90% chronologically)**: The first 270,000 interactions (Jan–May 2024) are used for training.
- **Stacking Split**: Within the train/val set, we perform a temporal split of **90/10** (90% Part 1 for stats/CF, 10% Part 2 for Logistic Regression training) to avoid target leakage while maximizing the size and representation of collaborative features.

### Evaluation Strategy & Metrics
To address selection bias under realistic constraints, we evaluate the models using **two distinct evaluation strategies**:

#### 1. Strategy A: Global Catalog Recommendation (Standard)
- **Concept**: Recommends 10 items out of the entire pool of 1,000 candidate items.
- **Recall@10** (Primary): The percentage of items a user actually clicked in the test set that are successfully captured in the model's top 10 recommendations.
- **NDCG@10**: Normalized Discounted Cumulative Gain penalizes the model for placing clicked items lower in the top-10 list.
- *Performance*: **Recall@10 = 0.0179**, **NDCG@10 = 0.0093** (outperforms Popularity baseline of 0.0125 and SVD baseline of 0.0153).

#### 2. Strategy B: Within-Impression Ad Ranking (Targeted)
- **Concept**: Ranks only the items that were actually shown to the user in the test set. This is a direct test of the model's ad CTR optimization.
- **ROC AUC**: Measures the probability that the model ranks a clicked ad higher than a non-clicked ad.
- *Performance*: **ROC AUC = 0.8465**, **Recall@10 = 0.6696**, **NDCG@10 = 0.6210** (outperforms Popularity ranking NDCG@10 of 0.5516).

---

## Dataset Insights & Challenges

### Key Insights
- **Dataset Size**: 10,000 users, 1,000 items, and 300,000 interactions.
- **Click-Through Rate**: Overall CTR is **36.18%** (108,554 clicks out of 300,000 impressions), which indicates a high-engagement environment.
- **Test Set Clicks**: Test set contains 9,497 unique users with an average of **3.16 impressions** and **1.15 clicks** per user. 33.02% of test users have 0 clicks in the test period.

### Challenges & Biases
1. **Selection Bias (Unobserved Interactions)**: We only observe clicks on items that were actually sent to users (impressions). If we recommend an item that is highly relevant but was not sent to the user in the test set, it is marked as a "miss" ($Recall = 0$) in offline evaluation. This penalizes the model unfairly for exploring outside historical impressions.
2. **Cold Start**: Users or items with very few historic interactions are hard to model via collaborative filtering. We solve this by fallback heuristics (using global CTRs) and leveraging age demographics and category content affinities.

### Theoretical Upper Bound Estimation
When evaluating the top-10 recommendations from the entire pool of 1,000 items:
- **Recall@10 Upper Bound: 66.98%**. Because 33.02% of users in the test set have exactly 0 clicks, no model can achieve a Recall > 0 for them.
- **NDCG@10 Upper Bound: 66.98%**. Since NDCG is 0 for 0-click users.
- **Precision@10 Upper Bound: 11.46%**. Since the average number of clicked items per user in the test set is 1.146, the maximum possible precision is $\frac{1.146}{10} = 0.1146$.
- *Actual Model Performance*:
  - **Global Catalog (Strategy A)**: Recall@10 = 1.79%, NDCG@10 = 0.0093.
  - **Within-Impression (Strategy B)**: ROC AUC = 84.65%, NDCG@10 = 62.10%. This close-to-optimal NDCG proves the model's excellent rank-ordering capability.

## AI Tool Usage
AI was utilized for:
- Writing data exploration scripts to analyze shape, statistics, and class balance.
- Iterative speed tuning: optimizing the feature extraction step from a slow pandas `iterrows()` loop to vectorized NumPy tensor dot products, which reduced prediction time for 9.5M pairs from ~8 minutes to under 20 seconds.
- Creating the Markdown documentation and implementation plan.
