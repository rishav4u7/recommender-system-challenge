import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import time

def main():
    t_start_all = time.time()
    
    # ----------------- 1. Load Data -----------------
    print("Loading datasets...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    
    # Parse timestamps and sort chronologically
    interactions['TIMESTAMP'] = pd.to_datetime(interactions['TIMESTAMP'])
    interactions = sorted_interactions = interactions.sort_values('TIMESTAMP').reset_index(drop=True)
    
    print(f"Loaded {len(users)} users, {len(items)} items, and {len(interactions)} interactions.")
    
    # ----------------- 2. Split Data -----------------
    # 90% train+val, 10% test
    n_total = len(interactions)
    n_train_val = int(0.9 * n_total)
    
    train_val_df = interactions.iloc[:n_train_val].copy()
    test_df = interactions.iloc[n_train_val:].copy()
    
    # Within train+val, split 90/10 for stacking
    n_train_p1 = int(0.9 * len(train_val_df))
    train_p1 = train_val_df.iloc[:n_train_p1].copy()
    train_p2 = train_val_df.iloc[n_train_p1:].copy()
    
    print(f"Train/Val set size: {len(train_val_df)} (Part 1: {len(train_p1)}, Part 2: {len(train_p2)})")
    print(f"Test set size: {len(test_df)}")
    
    # Map users and items to unique matrix indices
    unique_users = users['USER_ID'].unique()
    unique_items = items['ITEM_ID'].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    # Pre-build lookup dictionaries for speed
    user_features_dict = users.set_index('USER_ID').to_dict()['AGE_BUCKET']
    item_cat_dict = items.set_index('ITEM_ID').to_dict()['CATEGORY']
    item_price_dict = items.set_index('ITEM_ID').to_dict()['PRICE_BUCKET']
    
    # One-hot encode features for vectorized inference later
    age_cols = sorted(list(users['AGE_BUCKET'].unique()))
    cat_cols = sorted(list(items['CATEGORY'].unique()))
    price_cols = sorted(list(items['PRICE_BUCKET'].unique()))
    
    # Feature columns mapping
    # NEW: age_cat_ctr and age_price_ctr give cold-start users demographic-group
    # preference signals instead of a useless global average fallback.
    expected_cols = [
        'user_ctr', 'user_imps', 'item_ctr', 'item_imps',
        'user_cat_ctr', 'user_price_ctr',
        'age_cat_ctr', 'age_price_ctr',   # <-- NEW demographic content-based features
        'svd_score', 'cf_score'
    ] + [f"age_bucket_{c}" for c in age_cols] + [f"category_{c}" for c in cat_cols] + [f"price_bucket_{c}" for c in price_cols]
    
    col_to_idx = {col: idx for idx, col in enumerate(expected_cols)}
    
    # Precompute user age-bucket lookup (needed for demographic CTR fallback)
    user_age_dict = users.set_index('USER_ID')['AGE_BUCKET'].to_dict()
    
    # Precompute demographic/content matrices
    age_dummies = pd.get_dummies(users['AGE_BUCKET']).reindex(columns=age_cols, fill_value=0).astype(float).values
    cat_dummies = pd.get_dummies(items['CATEGORY']).reindex(columns=cat_cols, fill_value=0).astype(float).values
    price_dummies = pd.get_dummies(items['PRICE_BUCKET']).reindex(columns=price_cols, fill_value=0).astype(float).values
    
    # ----------------- 3. Helper to Compute Features -----------------
    def compute_stats(history_df):
        global_ctr = history_df['INTERACTION'].mean()
        
        # User CTR (with Laplace smoothing, beta=5)
        user_clicks = history_df[history_df['INTERACTION'] == 1].groupby('USER_ID').size()
        user_imps = history_df.groupby('USER_ID').size()
        user_ctr = ((user_clicks + 5 * global_ctr) / (user_imps + 5)).fillna(global_ctr).to_dict()
        user_imp_counts = user_imps.to_dict()
        
        # Item CTR (with Laplace smoothing, beta=5)
        item_clicks = history_df[history_df['INTERACTION'] == 1].groupby('ITEM_ID').size()
        item_imps = history_df.groupby('ITEM_ID').size()
        item_ctr = ((item_clicks + 5 * global_ctr) / (item_imps + 5)).fillna(global_ctr).to_dict()
        item_imp_counts = item_imps.to_dict()
        
        # User-Category affinity (with Laplace smoothing, beta=5)
        history_with_items = history_df.merge(items, on='ITEM_ID', how='left')
        history_with_demo  = history_with_items.merge(users, on='USER_ID', how='left')
        user_cat_clicks = history_with_items[history_with_items['INTERACTION'] == 1].groupby(['USER_ID', 'CATEGORY']).size()
        user_cat_imps = history_with_items.groupby(['USER_ID', 'CATEGORY']).size()
        user_cat_ctr = ((user_cat_clicks + 5 * global_ctr) / (user_cat_imps + 5)).fillna(global_ctr).to_dict()
        
        # User-Price affinity (with Laplace smoothing, beta=5)
        user_price_clicks = history_with_items[history_with_items['INTERACTION'] == 1].groupby(['USER_ID', 'PRICE_BUCKET']).size()
        user_price_imps = history_with_items.groupby(['USER_ID', 'PRICE_BUCKET']).size()
        user_price_ctr = ((user_price_clicks + 5 * global_ctr) / (user_price_imps + 5)).fillna(global_ctr).to_dict()
        
        # Age-bucket × Category CTR (with Laplace smoothing, beta=5)
        age_cat_clicks = history_with_demo[history_with_demo['INTERACTION'] == 1].groupby(['AGE_BUCKET', 'CATEGORY']).size()
        age_cat_imps   = history_with_demo.groupby(['AGE_BUCKET', 'CATEGORY']).size()
        age_cat_ctr    = ((age_cat_clicks + 5 * global_ctr) / (age_cat_imps + 5)).fillna(global_ctr).to_dict()
        
        # Age-bucket × Price CTR (with Laplace smoothing, beta=5)
        age_price_clicks = history_with_demo[history_with_demo['INTERACTION'] == 1].groupby(['AGE_BUCKET', 'PRICE_BUCKET']).size()
        age_price_imps   = history_with_demo.groupby(['AGE_BUCKET', 'PRICE_BUCKET']).size()
        age_price_ctr    = ((age_price_clicks + 5 * global_ctr) / (age_price_imps + 5)).fillna(global_ctr).to_dict()
        
        # User-Item interaction matrix for CF
        train_clicks = history_df[history_df['INTERACTION'] == 1]
        rows = train_clicks['USER_ID'].map(user_to_idx)
        cols = train_clicks['ITEM_ID'].map(item_to_idx)
        data = np.ones(len(train_clicks))
        click_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))
        
        # SVD Matrix Factorization (using optimized k=20 factors)
        try:
            u_mat, s_mat, vt_mat = svds(click_matrix.astype(float), k=20, random_state=10)
            svd_pred_matrix = u_mat @ np.diag(s_mat) @ vt_mat
        except Exception:
            svd_pred_matrix = np.zeros((len(unique_users), len(unique_items)))
            
        # Item Similarity and CF scores
        item_sim = cosine_similarity(click_matrix.T)
        cf_scores_matrix = click_matrix.dot(item_sim)
        
        return {
            'global_ctr': global_ctr,
            'user_ctr': user_ctr,
            'user_imp_counts': user_imp_counts,
            'item_ctr': item_ctr,
            'item_imp_counts': item_imp_counts,
            'user_cat_ctr': user_cat_ctr,
            'user_price_ctr': user_price_ctr,
            'age_cat_ctr': age_cat_ctr,
            'age_price_ctr': age_price_ctr,
            'click_matrix': click_matrix,
            'svd_pred_matrix': svd_pred_matrix,
            'cf_scores_matrix': cf_scores_matrix
        }

    def extract_features_vectorized(df, stats):
        global_ctr       = stats['global_ctr']
        user_ctr         = stats['user_ctr']
        user_imp_counts  = stats['user_imp_counts']
        item_ctr         = stats['item_ctr']
        item_imp_counts  = stats['item_imp_counts']
        user_cat_ctr     = stats['user_cat_ctr']
        user_price_ctr   = stats['user_price_ctr']
        age_cat_ctr      = stats['age_cat_ctr']
        age_price_ctr    = stats['age_price_ctr']
        svd_pred_matrix  = stats['svd_pred_matrix']
        cf_scores_matrix = stats['cf_scores_matrix']
        
        u_idx = df['USER_ID'].map(user_to_idx).values
        i_idx = df['ITEM_ID'].map(item_to_idx).values
        
        n_samples = len(df)
        X = np.zeros((n_samples, len(expected_cols)))
        
        # Basic CTR stats
        X[:, col_to_idx['user_ctr']]  = df['USER_ID'].map(user_ctr).fillna(global_ctr).values
        X[:, col_to_idx['user_imps']] = df['USER_ID'].map(user_imp_counts).fillna(0).values
        X[:, col_to_idx['item_ctr']]  = df['ITEM_ID'].map(item_ctr).fillna(global_ctr).values
        X[:, col_to_idx['item_imps']] = df['ITEM_ID'].map(item_imp_counts).fillna(0).values
        
        # Personal category affinity (falls back to demographic group CTR for cold users)
        categories = df['ITEM_ID'].map(item_cat_dict).values
        user_ages  = df['USER_ID'].map(user_age_dict).values
        user_cat_ctr_arr = np.array([
            user_cat_ctr.get((u, c),
                age_cat_ctr.get((a, c), global_ctr))   # cold-start: use age-group fallback
            for u, c, a in zip(df['USER_ID'], categories, user_ages)
        ])
        X[:, col_to_idx['user_cat_ctr']] = user_cat_ctr_arr
        
        # Personal price affinity (falls back to demographic group CTR for cold users)
        price_buckets = df['ITEM_ID'].map(item_price_dict).values
        user_price_ctr_arr = np.array([
            user_price_ctr.get((u, p),
                age_price_ctr.get((a, p), global_ctr))  # cold-start: use age-group fallback
            for u, p, a in zip(df['USER_ID'], price_buckets, user_ages)
        ])
        X[:, col_to_idx['user_price_ctr']] = user_price_ctr_arr
        
        # NEW: Demographic content-based features (age-bucket × category / price)
        # These are always populated — they are the primary signal for cold users
        # and a regularising secondary signal for warm users.
        X[:, col_to_idx['age_cat_ctr']] = np.array([
            age_cat_ctr.get((a, c), global_ctr)
            for a, c in zip(user_ages, categories)
        ])
        X[:, col_to_idx['age_price_ctr']] = np.array([
            age_price_ctr.get((a, p), global_ctr)
            for a, p in zip(user_ages, price_buckets)
        ])
        
        # CF Scores
        X[:, col_to_idx['svd_score']] = svd_pred_matrix[u_idx, i_idx]
        X[:, col_to_idx['cf_score']]  = cf_scores_matrix[u_idx, i_idx]
        
        # One-hot encoded demographics & product content
        user_indices_df = df['USER_ID'].map(lambda x: user_to_idx[x]).values
        item_indices_df = df['ITEM_ID'].map(lambda x: item_to_idx[x]).values
        
        X[:, col_to_idx['age_bucket_10-19']:col_to_idx['age_bucket_10-19']+5]   = age_dummies[user_indices_df]
        X[:, col_to_idx['category_Accessories']:col_to_idx['category_Accessories']+5] = cat_dummies[item_indices_df]
        X[:, col_to_idx['price_bucket_$0-$50']:col_to_idx['price_bucket_$0-$50']+4]   = price_dummies[item_indices_df]
        
        return X, df['INTERACTION'].values

    # ----------------- 4. Train Model -----------------
    print("Training Stage 1 (Stats and CF models on Part 1)...")
    stats_p1 = compute_stats(train_p1)
    
    print("Extracting features for Stage 2 (Part 2)...")
    t0 = time.time()
    X_train, y_train = extract_features_vectorized(train_p2, stats_p1)
    print(f"Features extracted in {time.time() - t0:.2f} seconds.")
    
    print("Training Stage 2 (Logistic Regression classifier on Part 2)...")
    clf = LogisticRegression(C=0.1, max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Classifier trained successfully.")
    
    # ----------------- 5. Recompute Stats on Full Train+Val -----------------
    print("Re-computing features on full Train/Val dataset to predict on Test set...")
    stats_full = compute_stats(train_val_df)
    
    # ----------------- 6. Make Predictions on Test Set -----------------
    test_users = test_df['USER_ID'].unique()
    print(f"Predicting recommendations for {len(test_users)} test users...")
    
    t0 = time.time()
    
    U = len(test_users)
    I = len(unique_items)
    
    # Pre-extract user properties
    global_ctr = stats_full['global_ctr']
    user_ctr_dict = stats_full['user_ctr']
    user_imp_counts = stats_full['user_imp_counts']
    
    user_ctrs = np.array([user_ctr_dict.get(uid, global_ctr) for uid in test_users])
    user_imps = np.array([user_imp_counts.get(uid, 0) for uid in test_users])
    user_indices = np.array([user_to_idx[uid] for uid in test_users])
    
    # Pre-extract item properties
    item_ctr_dict = stats_full['item_ctr']
    item_imp_counts = stats_full['item_imp_counts']
    
    item_ctrs = np.array([item_ctr_dict.get(iid, global_ctr) for iid in unique_items])
    item_imps = np.array([item_imp_counts.get(iid, 0) for iid in unique_items])
    item_categories = np.array([item_cat_dict[iid] for iid in unique_items])
    item_price_buckets = np.array([item_price_dict[iid] for iid in unique_items])
    item_indices = np.array([item_to_idx[iid] for iid in unique_items])
    
    # Build user-item affinity grids
    user_cat_ctr_dict   = stats_full['user_cat_ctr']
    user_price_ctr_dict = stats_full['user_price_ctr']
    age_cat_ctr_dict    = stats_full['age_cat_ctr']
    age_price_ctr_dict  = stats_full['age_price_ctr']
    
    user_cat_ctr_grid   = np.zeros((U, I))
    user_price_ctr_grid = np.zeros((U, I))
    age_cat_ctr_grid    = np.zeros((U, I))
    age_price_ctr_grid  = np.zeros((U, I))
    
    for u_idx_in_test, uid in enumerate(test_users):
        u_ctr = user_ctrs[u_idx_in_test]
        age   = user_age_dict.get(uid, None)
        for i_idx, iid in enumerate(unique_items):
            cat   = item_categories[i_idx]
            price = item_price_buckets[i_idx]
            # Personal affinity — falls back to age-group CTR for cold-start users
            age_cat_fb   = age_cat_ctr_dict.get((age, cat), global_ctr)
            age_price_fb = age_price_ctr_dict.get((age, price), global_ctr)
            user_cat_ctr_grid[u_idx_in_test, i_idx]   = user_cat_ctr_dict.get((uid, cat), age_cat_fb)
            user_price_ctr_grid[u_idx_in_test, i_idx] = user_price_ctr_dict.get((uid, price), age_price_fb)
            # Demographic CTR grids (always set — primary signal for cold, secondary for warm)
            age_cat_ctr_grid[u_idx_in_test, i_idx]    = age_cat_fb
            age_price_ctr_grid[u_idx_in_test, i_idx]  = age_price_fb
            
    # CF matrices lookup
    svd_score_grid = stats_full['svd_pred_matrix'][user_indices][:, item_indices]
    cf_score_grid  = stats_full['cf_scores_matrix'][user_indices][:, item_indices]
    
    # Logits calculation using model weights
    coefs     = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    logits = np.zeros((U, I)) + intercept
    logits += user_ctrs.reshape(U, 1)  * coefs[col_to_idx['user_ctr']]
    logits += user_imps.reshape(U, 1)  * coefs[col_to_idx['user_imps']]
    logits += item_ctrs.reshape(1, I)  * coefs[col_to_idx['item_ctr']]
    logits += item_imps.reshape(1, I)  * coefs[col_to_idx['item_imps']]
    logits += user_cat_ctr_grid        * coefs[col_to_idx['user_cat_ctr']]
    logits += user_price_ctr_grid      * coefs[col_to_idx['user_price_ctr']]
    logits += age_cat_ctr_grid         * coefs[col_to_idx['age_cat_ctr']]
    logits += age_price_ctr_grid       * coefs[col_to_idx['age_price_ctr']]
    logits += svd_score_grid           * coefs[col_to_idx['svd_score']]
    logits += cf_score_grid            * coefs[col_to_idx['cf_score']]
    
    # Age bucket coefficients mapping
    age_coefs = coefs[col_to_idx['age_bucket_10-19']:col_to_idx['age_bucket_10-19']+5]
    age_contribution = age_dummies[user_indices] @ age_coefs
    logits += age_contribution.reshape(U, 1)
    
    # Category coefficients mapping
    cat_coefs = coefs[col_to_idx['category_Accessories']:col_to_idx['category_Accessories']+5]
    cat_contribution = cat_dummies[item_indices] @ cat_coefs
    logits += cat_contribution.reshape(1, I)
    
    # Price coefficients mapping
    price_coefs = coefs[col_to_idx['price_bucket_$0-$50']:col_to_idx['price_bucket_$0-$50']+4]
    price_contribution = price_dummies[item_indices] @ price_coefs
    logits += price_contribution.reshape(1, I)
    
    # Sigmoid to convert to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Generate recommendations excluding train clicked items
    test_recs = {}
    click_matrix_full = stats_full['click_matrix']
    
    for u_idx_in_test, uid in enumerate(test_users):
        u_idx = user_indices[u_idx_in_test]
        user_clicks = click_matrix_full[u_idx].toarray().flatten()
        user_probs = probs[u_idx_in_test].copy()
        user_probs[user_clicks > 0] = -1e9  # Exclude train clicked items
        
        top10_idx = np.argsort(-user_probs)[:10]
        test_recs[uid] = [unique_items[idx] for idx in top10_idx]
        
    print(f"Recommendations generated in {time.time() - t0:.2f} seconds.")
    
    # ----------------- 7. Save to CSV -----------------
    output_path = os.path.join(data_dir, "recommendations.csv")
    print(f"Saving recommendations to {output_path}...")
    
    # Create list of rows to save
    rows_to_save = []
    for uid in test_users:
        recs_str = " ".join(test_recs[uid])
        rows_to_save.append({'USER_ID': uid, 'RECOMMENDED_ITEMS': recs_str})
        
    output_df = pd.DataFrame(rows_to_save)
    output_df.to_csv(output_path, index=False)
    print("CSV file saved successfully.")
    
    # ----------------- 8. Evaluation on Test Set -----------------
    print("Evaluating model performance on Test Set...")
    # Get ground truth positive items for each user in test_df
    positives = test_df[test_df['INTERACTION'] == 1].groupby('USER_ID')['ITEM_ID'].apply(list).to_dict()
    
    # ── Strategy A: Global Catalog Recommendation Evaluation ──
    recalls = []
    ndcgs = []
    
    for uid in test_users:
        user_positives = positives.get(uid, [])
        user_recs = test_recs[uid]
        
        if len(user_positives) == 0:
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue
            
        # Recall@10
        hits = len(set(user_recs) & set(user_positives))
        recall = hits / len(user_positives)
        recalls.append(recall)
        
        # NDCG@10
        dcg = 0.0
        for rank, item in enumerate(user_recs):
            if item in user_positives:
                dcg += 1.0 / np.log2(rank + 2)
                
        idcg = 0.0
        for rank in range(min(len(user_positives), len(user_recs))):
            idcg += 1.0 / np.log2(rank + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        
    mean_recall = np.mean(recalls)
    mean_ndcg = np.mean(ndcgs)
    
    # ── Strategy B: Within-Impression Ad Ranking Evaluation ──
    # Evaluate model's ability to rank items that were actually shown in the test period.
    # We score shown items using the predicted probability from the stacking classifier.
    # Extract predicted test probabilities for each chronological test interaction
    X_test_eval, y_test_eval = extract_features_vectorized(test_df, stats_full)
    test_probs_eval = clf.predict_proba(X_test_eval)[:, 1]
    
    test_df_wi = test_df.copy()
    test_df_wi['prob'] = test_probs_eval
    
    # Rank shown items for each user
    recs_ranked = {}
    for uid, group in test_df_wi.groupby('USER_ID'):
        sorted_group = group.sort_values('prob', ascending=False)
        recs_ranked[uid] = list(sorted_group['ITEM_ID'].values)[:10]
        
    recalls_wi = []
    ndcgs_wi = []
    for uid in test_users:
        user_positives = positives.get(uid, [])
        user_recs = recs_ranked.get(uid, [])
        if len(user_positives) == 0:
            recalls_wi.append(0.0)
            ndcgs_wi.append(0.0)
            continue
            
        hits = len(set(user_recs) & set(user_positives))
        recalls_wi.append(hits / len(user_positives))
        
        dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(user_recs) if item in user_positives)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(user_positives), len(user_recs))))
        ndcgs_wi.append(dcg / idcg if idcg > 0 else 0.0)
        
    mean_recall_wi = np.mean(recalls_wi)
    mean_ndcg_wi = np.mean(ndcgs_wi)
    mean_auc_wi = roc_auc_score(y_test_eval, test_probs_eval)
    
    print("\n================ Evaluation Results ================")
    print("Strategy A: Global Catalog Recommendation (All 1,000 Items)")
    print(f"  Mean Recall@10: {mean_recall:.4f}")
    print(f"  Mean NDCG@10:   {mean_ndcg:.4f}")
    print("\nStrategy B: Within-Impression Ad Ranking (Shown Items Only)")
    print(f"  ROC AUC:        {mean_auc_wi:.4f}")
    print(f"  Mean Recall@10: {mean_recall_wi:.4f}")
    print(f"  Mean NDCG@10:   {mean_ndcg_wi:.4f}")
    print("====================================================\n")
    
    # Run-time stats
    t_end_all = time.time()
    print(f"Total script run time: {t_end_all - t_start_all:.2f} seconds.")

if __name__ == '__main__':
    main()
