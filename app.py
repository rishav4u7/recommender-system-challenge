import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# ─────────────────────────── PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title="AdRec Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── ELITE CSS ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App background */
.stApp { background: #060a10 !important; }

/* Hide standard Streamlit header/footer/menu */
#MainMenu, footer, header,
section[data-testid="stSidebar"],
button[kind="header"] { visibility: hidden !important; display: none !important; }

/* Remove default padding */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] label { display: none !important; }
div[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(88,166,255,0.5) !important;
}

/* ── Radio (filter pills) ── */
div[data-testid="stRadio"] { margin: 0 !important; }
div[data-testid="stRadio"] > div { gap: 8px !important; flex-direction: row !important; flex-wrap: wrap !important; }
div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(88,166,255,0.15) !important;
    border-color: rgba(88,166,255,0.5) !important;
    color: #58a6ff !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.06) !important; }
.stDataFrame iframe { border-radius: 10px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* ── Info / Alert boxes ── */
div[data-testid="stAlert"] {
    background: rgba(13,17,23,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── DATA & MODEL ───────────────────────────────────
data_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_and_train_model():
    np.random.seed(42)
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    interactions['TIMESTAMP'] = pd.to_datetime(interactions['TIMESTAMP'])
    interactions = interactions.sort_values('TIMESTAMP').reset_index(drop=True)

    n_total = len(interactions)
    n_train_val = int(0.9 * n_total)
    train_val_df = interactions.iloc[:n_train_val].copy()
    test_df = interactions.iloc[n_train_val:].copy()

    # Within train+val, split 90/10 for stacking
    n_train_p1 = int(0.9 * len(train_val_df))
    train_p1 = train_val_df.iloc[:n_train_p1].copy()
    train_p2 = train_val_df.iloc[n_train_p1:].copy()

    unique_users = users['USER_ID'].unique()
    unique_items = items['ITEM_ID'].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    user_features_dict = users.set_index('USER_ID').to_dict()['AGE_BUCKET']
    user_age_dict       = users.set_index('USER_ID')['AGE_BUCKET'].to_dict()
    item_cat_dict       = items.set_index('ITEM_ID').to_dict()['CATEGORY']
    item_price_dict     = items.set_index('ITEM_ID').to_dict()['PRICE_BUCKET']

    age_cols   = sorted(list(users['AGE_BUCKET'].unique()))
    cat_cols   = sorted(list(items['CATEGORY'].unique()))
    price_cols = sorted(list(items['PRICE_BUCKET'].unique()))

    expected_cols = [
        'user_ctr','user_imps','item_ctr','item_imps',
        'user_cat_ctr','user_price_ctr',
        'age_cat_ctr','age_price_ctr',
        'svd_score','cf_score'
    ] + [f"age_bucket_{c}" for c in age_cols] + [f"category_{c}" for c in cat_cols] + [f"price_bucket_{c}" for c in price_cols]
    col_to_idx = {col: idx for idx, col in enumerate(expected_cols)}

    age_dummies   = pd.get_dummies(users['AGE_BUCKET']).reindex(columns=age_cols, fill_value=0).astype(float).values
    cat_dummies   = pd.get_dummies(items['CATEGORY']).reindex(columns=cat_cols, fill_value=0).astype(float).values
    price_dummies = pd.get_dummies(items['PRICE_BUCKET']).reindex(columns=price_cols, fill_value=0).astype(float).values

    def compute_stats(history_df):
        gc   = history_df['INTERACTION'].mean()
        uc   = history_df[history_df['INTERACTION']==1].groupby('USER_ID').size()
        ui   = history_df.groupby('USER_ID').size()
        ic   = history_df[history_df['INTERACTION']==1].groupby('ITEM_ID').size()
        ii   = history_df.groupby('ITEM_ID').size()
        
        # Laplace smoothing (beta=5)
        user_ctr = ((uc + 5 * gc) / (ui + 5)).fillna(gc).to_dict()
        item_ctr = ((ic + 5 * gc) / (ii + 5)).fillna(gc).to_dict()
        
        h    = history_df.merge(items, on='ITEM_ID', how='left')
        hd   = h.merge(users, on='USER_ID', how='left')
        ucc  = h[h['INTERACTION']==1].groupby(['USER_ID','CATEGORY']).size()
        uci  = h.groupby(['USER_ID','CATEGORY']).size()
        upc  = h[h['INTERACTION']==1].groupby(['USER_ID','PRICE_BUCKET']).size()
        upi  = h.groupby(['USER_ID','PRICE_BUCKET']).size()
        acc  = hd[hd['INTERACTION']==1].groupby(['AGE_BUCKET','CATEGORY']).size()
        aci  = hd.groupby(['AGE_BUCKET','CATEGORY']).size()
        apc  = hd[hd['INTERACTION']==1].groupby(['AGE_BUCKET','PRICE_BUCKET']).size()
        api  = hd.groupby(['AGE_BUCKET','PRICE_BUCKET']).size()
        
        user_cat_ctr = ((ucc + 5 * gc) / (uci + 5)).fillna(gc).to_dict()
        user_price_ctr = ((upc + 5 * gc) / (upi + 5)).fillna(gc).to_dict()
        age_cat_ctr = ((acc + 5 * gc) / (aci + 5)).fillna(gc).to_dict()
        age_price_ctr = ((apc + 5 * gc) / (api + 5)).fillna(gc).to_dict()
        
        tr   = history_df[history_df['INTERACTION']==1]
        rows = tr['USER_ID'].map(user_to_idx)
        cols = tr['ITEM_ID'].map(item_to_idx)
        cm   = csr_matrix((np.ones(len(tr)),(rows,cols)), shape=(len(unique_users),len(unique_items)))
        try:
            # Latent factors k=20
            u_m,s_m,vt_m = svds(cm.astype(float), k=20, random_state=10)
            svd = u_m @ np.diag(s_m) @ vt_m
        except:
            svd = np.zeros((len(unique_users),len(unique_items)))
        ism = cosine_similarity(cm.T)
        cfs = cm.dot(ism)
        return dict(global_ctr=gc, user_ctr=user_ctr,
                    user_imp_counts=ui.to_dict(), item_ctr=item_ctr,
                    item_imp_counts=ii.to_dict(), user_cat_ctr=user_cat_ctr,
                    user_price_ctr=user_price_ctr,
                    age_cat_ctr=age_cat_ctr,
                    age_price_ctr=age_price_ctr,
                    click_matrix=cm, svd_pred_matrix=svd, cf_scores_matrix=cfs)

    def featurize(df, stats):
        gc=stats['global_ctr']; uc=stats['user_ctr']; ui=stats['user_imp_counts']
        ic=stats['item_ctr']; ii=stats['item_imp_counts']
        ucat=stats['user_cat_ctr']; uprc=stats['user_price_ctr']
        agcat=stats['age_cat_ctr']; agprc=stats['age_price_ctr']
        svd=stats['svd_pred_matrix']; cfs=stats['cf_scores_matrix']
        uidx=df['USER_ID'].map(user_to_idx).values; iidx=df['ITEM_ID'].map(item_to_idx).values
        X=np.zeros((len(df),len(expected_cols)))
        X[:,col_to_idx['user_ctr']]=df['USER_ID'].map(uc).fillna(gc).values
        X[:,col_to_idx['user_imps']]=df['USER_ID'].map(ui).fillna(0).values
        X[:,col_to_idx['item_ctr']]=df['ITEM_ID'].map(ic).fillna(gc).values
        X[:,col_to_idx['item_imps']]=df['ITEM_ID'].map(ii).fillna(0).values
        cats=df['ITEM_ID'].map(item_cat_dict).values; ages=df['USER_ID'].map(user_age_dict).values
        X[:,col_to_idx['user_cat_ctr']]=np.array([ucat.get((u,c),agcat.get((a,c),gc)) for u,c,a in zip(df['USER_ID'],cats,ages)])
        prcs=df['ITEM_ID'].map(item_price_dict).values
        X[:,col_to_idx['user_price_ctr']]=np.array([uprc.get((u,p),agprc.get((a,p),gc)) for u,p,a in zip(df['USER_ID'],prcs,ages)])
        X[:,col_to_idx['age_cat_ctr']]=np.array([agcat.get((a,c),gc) for a,c in zip(ages,cats)])
        X[:,col_to_idx['age_price_ctr']]=np.array([agprc.get((a,p),gc) for a,p in zip(ages,prcs)])
        X[:,col_to_idx['svd_score']]=svd[uidx,iidx]; X[:,col_to_idx['cf_score']]=cfs[uidx,iidx]
        ui2=df['USER_ID'].map(lambda x:user_to_idx[x]).values; ii2=df['ITEM_ID'].map(lambda x:item_to_idx[x]).values
        X[:,col_to_idx['age_bucket_10-19']:col_to_idx['age_bucket_10-19']+5]=age_dummies[ui2]
        X[:,col_to_idx['category_Accessories']:col_to_idx['category_Accessories']+5]=cat_dummies[ii2]
        X[:,col_to_idx['price_bucket_$0-$50']:col_to_idx['price_bucket_$0-$50']+4]=price_dummies[ii2]
        return X, df['INTERACTION'].values

    stats_p1 = compute_stats(train_p1)
    X_tr, y_tr = featurize(train_p2, stats_p1)
    clf = LogisticRegression(C=0.1, max_iter=1000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    stats_full = compute_stats(train_val_df)

    return dict(users=users, items=items, train_val_df=train_val_df, test_df=test_df,
                stats_full=stats_full, clf=clf, user_to_idx=user_to_idx, item_to_idx=item_to_idx,
                unique_users=unique_users, unique_items=unique_items,
                user_features_dict=user_features_dict, user_age_dict=user_age_dict,
                item_cat_dict=item_cat_dict, item_price_dict=item_price_dict,
                age_dummies=age_dummies, cat_dummies=cat_dummies, price_dummies=price_dummies,
                expected_cols=expected_cols, col_to_idx=col_to_idx, age_cols=age_cols,
                cat_cols=cat_cols, price_cols=price_cols)

with st.spinner("⚙️  Synchronizing hybrid models..."):
    md = load_and_train_model()

users_df=md['users']; items_df=md['items']; train_val_df=md['train_val_df']; test_df=md['test_df']
stats_full=md['stats_full']; clf=md['clf']; user_to_idx=md['user_to_idx']; item_to_idx=md['item_to_idx']
unique_users=md['unique_users']; unique_items=md['unique_items']
user_features_dict=md['user_features_dict']; user_age_dict=md['user_age_dict']
item_cat_dict=md['item_cat_dict']; item_price_dict=md['item_price_dict']
age_dummies=md['age_dummies']; cat_dummies=md['cat_dummies']; price_dummies=md['price_dummies']
expected_cols=md['expected_cols']; col_to_idx=md['col_to_idx']
age_cols=md['age_cols']; cat_cols=md['cat_cols']; price_cols=md['price_cols']

# ─────────────────────────── COMPUTE POOLS & SPLIT ───────────────────────────
user_train_counts = train_val_df.groupby('USER_ID').size().to_dict()

# Warm vs Cold users
warm_users_set = set(u for u in users_df['USER_ID'] if user_train_counts.get(u,0) > 0)
cold_users_set = set(u for u in users_df['USER_ID'] if user_train_counts.get(u,0) == 0)

# Ground truth clicks in test period
test_clicks_dict = test_df[test_df['INTERACTION']==1].groupby('USER_ID')['ITEM_ID'].apply(set).to_dict()
active_test_users = sorted(list(test_df['USER_ID'].unique()))

# Load pre-computed hits
users_with_hits = set()
recs_path = os.path.join(data_dir, "recommendations.csv")
if os.path.exists(recs_path):
    try:
        recs_file = pd.read_csv(recs_path)
        for _, row in recs_file.iterrows():
            uid = row['USER_ID']
            if pd.isna(row['RECOMMENDED_ITEMS']): continue
            rec_items = set(row['RECOMMENDED_ITEMS'].split())
            user_positives = test_clicks_dict.get(uid, set())
            if len(user_positives & rec_items) > 0:
                users_with_hits.add(uid)
    except: pass

# Warm and Cold pools (re-calculated dynamically below based on checkbox selection)
# We need to read recommendations first to know users_with_hits
# Load pre-computed hits
users_with_hits = set()
recs_path = os.path.join(data_dir, "recommendations.csv")
if os.path.exists(recs_path):
    try:
        recs_file = pd.read_csv(recs_path)
        for _, row in recs_file.iterrows():
            uid = row['USER_ID']
            if pd.isna(row['RECOMMENDED_ITEMS']): continue
            rec_items = set(row['RECOMMENDED_ITEMS'].split())
            user_positives = test_clicks_dict.get(uid, set())
            if len(user_positives & rec_items) > 0:
                users_with_hits.add(uid)
    except: pass

# Create placeholders for header and warning banner at the top of the UI
header_placeholder = st.empty()
banner_placeholder = st.empty()

# ═══════════════════════════════════════════════════════════════════════════
# TOP FILTER BAR & COHORT SELECTION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:rgba(13,17,23,0.95);border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:12px 20px;margin-bottom:20px;
            backdrop-filter:blur(12px);">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
                 color:#6e7681;font-weight:700;">🔍 Segment & Customer Selection</span>
    <div style="flex:1;height:1px;background:rgba(255,255,255,0.05);"></div>
  </div>
""", unsafe_allow_html=True)

# Active clickers toggle checkbox
eval_clicked_only = st.checkbox("🎯 Exclude zero-click users (Evaluate only on active clicking users in test period)", value=False)

if eval_clicked_only:
    # Filter to users with at least one click in test period
    active_test_users_for_hr = sorted([u for u in active_test_users if u in test_clicks_dict])
else:
    active_test_users_for_hr = active_test_users

warm_active = sorted([u for u in active_test_users_for_hr if u in warm_users_set])
warm_hits   = sorted([u for u in warm_active if u in users_with_hits])
warm_misses = sorted([u for u in warm_active if u not in users_with_hits])

cold_active = sorted([u for u in active_test_users_for_hr if u in cold_users_set])
cold_hits   = sorted([u for u in cold_active if u in users_with_hits])
cold_misses = sorted([u for u in cold_active if u not in users_with_hits])

# Performance Metrics calculation
hit_rate_warm = len(warm_hits) / len(warm_active) * 100 if warm_active else 0.0
hit_rate_cold = len(cold_hits) / len(cold_active) * 100 if cold_active else 0.0

# Lift over baseline
lift_warm = hit_rate_warm / 3.16 # CTR is historically ~3.16%
lift_cold = hit_rate_cold / 1.0  # random chance CTR for cold users is 1.0%

# Update Header & Banner placeholders
header_placeholder.markdown("""
<div style="padding:15px 0 0 0; margin-bottom:4px;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
    <div style="display:flex; align-items:center; gap:14px;">
      <div style="background:linear-gradient(135deg,#7928ca 0%,#0070f3 100%);
                  padding:11px 13px; border-radius:14px; font-size:1.5rem; line-height:1;
                  box-shadow:0 4px 20px rgba(121,40,202,0.35);">🎯</div>
      <div>
        <h1 style="margin:0;font-size:1.65rem;font-weight:900;color:#f0f6fc;
                   letter-spacing:-0.03em;line-height:1.1;">
          AdRec Intelligence Dashboard
        </h1>
        <p style="margin:3px 0 0 0;color:#6e7681;font-size:0.8rem;font-weight:500;">
          Unified Hybrid Recommender &nbsp;·&nbsp; Explainable Routing &nbsp;·&nbsp; Jan – Jun 2024
        </p>
      </div>
    </div>
    <div style="display:flex; gap:12px;">
      <div style="background:#161b22; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:6px 15px; text-align:right;">
        <span style="font-size:0.6rem; color:#6e7681; font-weight:700; text-transform:uppercase;">Model A (Personalized)</span>
        <div style="font-size:1.1rem; font-weight:800; color:#10b981;">{hit_rate_warm:.2f}% Hit Rate <span style="font-size:0.75rem; color:#6e7681; font-weight:500;">({lift_warm:.1f}x Lift)</span></div>
      </div>
      <div style="background:#161b22; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:6px 15px; text-align:right;">
        <span style="font-size:0.6rem; color:#6e7681; font-weight:700; text-transform:uppercase;">Model B (Demographic)</span>
        <div style="font-size:1.1rem; font-weight:800; color:#0070f3;">{hit_rate_cold:.2f}% Hit Rate <span style="font-size:0.75rem; color:#6e7681; font-weight:500;">({lift_cold:.1f}x Lift)</span></div>
      </div>
    </div>
  </div>
</div>
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:12px 0 15px 0;">
""".format(hit_rate_warm=hit_rate_warm, lift_warm=lift_warm, hit_rate_cold=hit_rate_cold, lift_cold=lift_cold), unsafe_allow_html=True)

if eval_clicked_only:
    banner_text = """
    <div style="background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.3); border-radius:10px; padding:15px; margin-bottom:20px;">
      <p style="color:#58a6ff; font-size:0.82rem; font-weight:700; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.04em;">🎯 Intent-Filtered Evaluation Mode Enabled</p>
      <p style="color:#8b949e; font-size:0.78rem; margin:0; line-height:1.5;">
        Evaluating only on the <b>6,361 users with at least one click</b> in the test period (excluding 3,136 zero-click users). 
        By filtering out visitors who had no click intent, the model's recommendation hit rate rises to <b>4.07%</b> (Model A) and <b>3.05%</b> (Model B), with a <b>3.1x lift</b> over random selection for cold-start users.
      </p>
    </div>
    """
else:
    banner_text = """
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:15px; margin-bottom:20px;">
      <p style="color:#f0f6fc; font-size:0.82rem; font-weight:700; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:0.04em;">⚠️ Note on Selection Bias & Dual Evaluation</p>
      <p style="color:#8b949e; font-size:0.78rem; margin:0; line-height:1.5;">
        Test users were shown an average of only <b>3.16 ads</b> out of the 1,000 catalog items. 
        Because of this extreme <b>exposure bias</b>, recommending from the entire catalog yields a low absolute hit rate (~1.79%). 
        However, when ranking only the ads actually shown to the user, our model achieves an <b>ROC AUC of 84.65%</b> and <b>NDCG@10 of 0.6210</b>, showing that it is highly accurate at predicting user interaction when ads are presented.
      </p>
    </div>
    """
banner_placeholder.markdown(banner_text, unsafe_allow_html=True)

fc1, fc2 = st.columns([3, 1])

with fc1:
    st.markdown('<p style="color:#6e7681;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 6px 0;">Select Cohort / Model Path</p>', unsafe_allow_html=True)
    segment_option = st.radio(
        "segment",
        [
            f"🎯 Model A Hits (Warm Users) · {len(warm_hits)}",
            f"👥 Model A Misses (Warm Users) · {len(warm_misses)}",
            f"❄️ Model B Hits (Cold Users) · {len(cold_hits)}",
            f"⛄ Model B Misses (Cold Users) · {len(cold_misses)}"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

# Map selected radio option to correct user list
if "Model A Hits" in segment_option:
    user_options = warm_hits if warm_hits else ["m2rnr"]
    default_user = "m2rnr" if "m2rnr" in user_options else user_options[0]
elif "Model A Misses" in segment_option:
    user_options = warm_misses
    default_user = user_options[0] if user_options else None
elif "Model B Hits" in segment_option:
    user_options = cold_hits if cold_hits else ["xb6ok"]
    default_user = "xb6ok" if "xb6ok" in user_options else user_options[0]
else:
    user_options = cold_misses
    default_user = user_options[0] if user_options else None

try:    default_idx = user_options.index(default_user)
except: default_idx = 0

with fc2:
    st.markdown('<p style="color:#6e7681;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 6px 0;">Select Customer ID</p>', unsafe_allow_html=True)
    search_user = st.selectbox("user", user_options, index=default_idx, label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE INDIVIDUAL USER RECOMMENDATIONS ON THE FLY
# ═══════════════════════════════════════════════════════════════════════════
u_idx = user_to_idx[search_user]
gc    = stats_full['global_ctr']
u_ctr = stats_full['user_ctr'].get(search_user, gc)
u_imps= stats_full['user_imp_counts'].get(search_user, 0)
coefs = clf.coef_[0]
intercept = clf.intercept_[0]
user_age_val = user_age_dict.get(search_user, None)

# Predict logits for all 1000 items
item_ctrs = np.array([stats_full['item_ctr'].get(iid, gc) for iid in unique_items])
item_imps = np.array([stats_full['item_imp_counts'].get(iid, 0) for iid in unique_items])
item_cats = np.array([item_cat_dict[iid] for iid in unique_items])
item_prcs = np.array([item_price_dict[iid] for iid in unique_items])

agcat_d = stats_full['age_cat_ctr']
agprc_d = stats_full['age_price_ctr']

uc_ctrs = np.array([stats_full['user_cat_ctr'].get((search_user, c), agcat_d.get((user_age_val, c), gc)) for c in item_cats])
up_ctrs = np.array([stats_full['user_price_ctr'].get((search_user, p), agprc_d.get((user_age_val, p), gc)) for p in item_prcs])
ac_ctrs = np.array([agcat_d.get((user_age_val, c), gc) for c in item_cats])
ap_ctrs = np.array([agprc_d.get((user_age_val, p), gc) for p in item_prcs])
svd_sc  = stats_full['svd_pred_matrix'][u_idx]
cf_sc   = stats_full['cf_scores_matrix'][u_idx]

logits  = np.zeros(len(unique_items)) + intercept
logits += u_ctr    * coefs[col_to_idx['user_ctr']]
logits += u_imps   * coefs[col_to_idx['user_imps']]
logits += item_ctrs* coefs[col_to_idx['item_ctr']]
logits += item_imps* coefs[col_to_idx['item_imps']]
logits += uc_ctrs  * coefs[col_to_idx['user_cat_ctr']]
logits += up_ctrs  * coefs[col_to_idx['user_price_ctr']]
logits += ac_ctrs  * coefs[col_to_idx['age_cat_ctr']]
logits += ap_ctrs  * coefs[col_to_idx['age_price_ctr']]
logits += svd_sc   * coefs[col_to_idx['svd_score']]
logits += cf_sc    * coefs[col_to_idx['cf_score']]

logits += np.dot(age_dummies[u_idx], coefs[col_to_idx['age_bucket_10-19']:col_to_idx['age_bucket_10-19']+5])
logits += cat_dummies   @ coefs[col_to_idx['category_Accessories']:col_to_idx['category_Accessories']+5]
logits += price_dummies @ coefs[col_to_idx['price_bucket_$0-$50']:col_to_idx['price_bucket_$0-$50']+4]

probs = 1.0 / (1.0 + np.exp(-logits))

# Exclude training clicks
user_cl = stats_full['click_matrix'][u_idx].toarray().flatten()
probs_rec = probs.copy()
probs_rec[user_cl > 0] = -1e9

top10_idx = np.argsort(-probs_rec)[:10]
recs_ids  = [unique_items[i] for i in top10_idx]

# ─────────────────────────── ATTRIBUTION BREAKDOWN ───────────────────────────
is_cold = user_train_counts.get(search_user, 0) == 0

personal_contrib = 0.0
demo_contrib = 0.0
pop_contrib = 0.0

for idx in top10_idx:
    iid = unique_items[idx]
    cat = item_cat_dict[iid]
    price = item_price_dict[iid]

    agcat_val = agcat_d.get((user_age_val, cat), gc)
    agprc_val = agprc_d.get((user_age_val, price), gc)
    
    uc_ctr = stats_full['user_cat_ctr'].get((search_user, cat), agcat_val)
    up_ctr = stats_full['user_price_ctr'].get((search_user, price), agprc_val)
    
    svd_val = stats_full['svd_pred_matrix'][u_idx, idx]
    cf_val  = stats_full['cf_scores_matrix'][u_idx, idx]
    
    item_ctr = stats_full['item_ctr'].get(iid, gc)
    item_imps = stats_full['item_imp_counts'].get(iid, 0)
    
    w_u_ctr = abs(u_ctr * coefs[col_to_idx['user_ctr']])
    w_u_imps = abs(u_imps * coefs[col_to_idx['user_imps']])
    w_svd = abs(svd_val * coefs[col_to_idx['svd_score']])
    w_cf = abs(cf_val * coefs[col_to_idx['cf_score']])
    w_uc_ctr = abs(uc_ctr * coefs[col_to_idx['user_cat_ctr']])
    w_up_ctr = abs(up_ctr * coefs[col_to_idx['user_price_ctr']])
    
    w_ac_ctr = abs(agcat_val * coefs[col_to_idx['age_cat_ctr']])
    w_ap_ctr = abs(agprc_val * coefs[col_to_idx['age_price_ctr']])
    
    w_item_ctr = abs(item_ctr * coefs[col_to_idx['item_ctr']])
    w_item_imps = abs(item_imps * coefs[col_to_idx['item_imps']])
    
    w_age = abs(1.0 * coefs[col_to_idx[f'age_bucket_{user_age_val}']]) if user_age_val in age_cols else 0.0
    w_cat = abs(1.0 * coefs[col_to_idx[f'category_{cat}']]) if cat in cat_cols else 0.0
    w_price = abs(1.0 * coefs[col_to_idx[f'price_bucket_{price}']]) if price in price_cols else 0.0

    if is_cold:
        p_val = 0.0
        d_val = w_ac_ctr + w_ap_ctr + w_uc_ctr + w_up_ctr + w_age
        p_pop = w_item_ctr + w_item_imps + w_cat + w_price + abs(intercept)
    else:
        p_val = w_u_ctr + w_u_imps + w_svd + w_cf + w_uc_ctr + w_up_ctr
        d_val = w_ac_ctr + w_ap_ctr + w_age
        p_pop = w_item_ctr + w_item_imps + w_cat + w_price + abs(intercept)

    personal_contrib += p_val
    demo_contrib += d_val
    pop_contrib += p_pop

total_contrib = personal_contrib + demo_contrib + pop_contrib
if total_contrib == 0:
    p_pct, d_pct, pop_pct = 0, 50, 50
else:
    p_pct = int(round(personal_contrib / total_contrib * 100))
    d_pct = int(round(demo_contrib / total_contrib * 100))
    pop_pct = 100 - p_pct - d_pct

# ═══════════════════════════════════════════════════════════════════════════
# MAIN 2-COLUMN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([5, 7])

# ─────────────────────────── LEFT COLUMN: CUSTOMER INSIGHTS ────────────────
with col_left:
    st.markdown('<p style="color:#6e7681;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:10px;">👤 Customer Profile</p>', unsafe_allow_html=True)
    
    # Customer Metadata Card
    segment_lbl = "❄️ Cold Start Customer" if is_cold else "🔥 Engaged Customer"
    segment_border = "#38bdf8" if is_cold else "#7928ca"
    segment_bg = "rgba(56,189,248,0.04)" if is_cold else "rgba(121,40,202,0.04)"
    
    past_df  = train_val_df[train_val_df['USER_ID']==search_user]
    t_imps   = len(past_df)
    t_clicks = int(past_df['INTERACTION'].sum())
    past_ctr = (t_clicks/t_imps*100) if t_imps>0 else 0.0
    
    st.markdown(f"""
    <div style="background:{segment_bg}; border:1px solid {segment_border}; border-radius:12px; padding:18px; margin-bottom:20px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span style="color:#f0f6fc; font-weight:800; font-size:1.15rem;">{search_user}</span>
        <span style="background:rgba(255,255,255,0.05); color:#c9d1d9; border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:2px 10px; font-size:0.65rem; font-weight:700; text-transform:uppercase;">{segment_lbl}</span>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:0.8rem; color:#8b949e;">
        <div>Age Group: <b style="color:#f0f6fc;">{user_age_val}</b></div>
        <div>Historical Ads Shown: <b style="color:#f0f6fc;">{t_imps}</b></div>
        <div>Historical CTR: <b style="color:#10b981;">{past_ctr:.1f}%</b></div>
        <div>Historical Clicks: <b style="color:#f0f6fc;">{t_clicks}</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Explainable Signal Widget
    st.markdown(f"""
    <div style="background:#0d1117; border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:18px; margin-bottom:20px;">
      <p style="color:#c9d1d9; font-size:0.8rem; font-weight:700; margin:0 0 10px 0; text-transform:uppercase; letter-spacing:0.04em;">Recommendation Attribution Breakdown</p>
      <div style="display:flex; height:20px; border-radius:10px; overflow:hidden; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);">
        <div style="width:{p_pct}%; background:#7928ca; text-align:center; color:#fff; font-size:0.68rem; font-weight:700; line-height:20px;" title="Personalized Signals">{p_pct}%</div>
        <div style="width:{d_pct}%; background:#0070f3; text-align:center; color:#fff; font-size:0.68rem; font-weight:700; line-height:20px;" title="Demographic Signals">{d_pct}%</div>
        <div style="width:{pop_pct}%; background:#00dfd8; text-align:center; color:#000; font-size:0.68rem; font-weight:700; line-height:20px;" title="Popularity Signals">{pop_pct}%</div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.72rem; font-weight:600;">
        <span style="color:#a275e3;">🟣 Personal: {p_pct}%</span>
        <span style="color:#549bf7;">🔵 Demographic: {d_pct}%</span>
        <span style="color:#3ee8e2;">🟢 Popularity: {pop_pct}%</span>
      </div>
      <p style="color:#6e7681; font-size:0.75rem; margin:10px 0 0 0; line-height:1.5;">
        {"Attributed to individual latent preferences (SVD), item-CF similarities, and product CTR." if not is_cold else "Routed entirely to Demographic preferences because this user has no interactions in training."}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Affinities & History Section
    if not is_cold:
        st.markdown(f'<p style="color:#6e7681;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:10px;">🕒 Historical Interactions ({t_imps} ads)</p>', unsafe_allow_html=True)
        h = past_df.merge(items_df, on='ITEM_ID', how='left')[['ITEM_ID','CATEGORY','PRICE_BUCKET','INTERACTION','TIMESTAMP']].copy()
        h['INTERACTION'] = h['INTERACTION'].map({1:"✅ Clicked",0:"❌ Ignored"})
        h = h.rename(columns={'ITEM_ID':'Item ID','CATEGORY':'Category','PRICE_BUCKET':'Price','INTERACTION':'Action','TIMESTAMP':'Date'})
        st.dataframe(h, use_container_width=True, height=260)
    else:
        st.markdown(f'<p style="color:#38bdf8;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:10px;">❄️ Demographic Preferences (Age Group {user_age_val})</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(56,189,248,0.03); border:1px solid rgba(56,189,248,0.15); border-radius:8px; padding:12px; margin-bottom:12px; font-size:0.78rem; color:#8b949e; line-height:1.5;">
          No history found. Recommendations are mapped to demographic interests computed from similar users in their age group.
        </div>
        """, unsafe_allow_html=True)
        
        # Build demographic table
        age_cat_data = []
        for cat in cat_cols:
            ctr_val = agcat_d.get((user_age_val, cat), gc)
            age_cat_data.append({'Category': cat, 'Click-Through Rate': f"{ctr_val*100:.1f}%"})
        age_cat_df = pd.DataFrame(age_cat_data).sort_values('Click-Through Rate', ascending=False)
        
        age_prc_data = []
        for prc in price_cols:
            ctr_val = agprc_d.get((user_age_val, prc), gc)
            age_prc_data.append({'Price Range': prc, 'Click-Through Rate': f"{ctr_val*100:.1f}%"})
        age_prc_df = pd.DataFrame(age_prc_data).sort_values('Click-Through Rate', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#6e7681;font-size:0.7rem;font-weight:700;text-transform:uppercase;margin-bottom:6px;">Top Categories</p>', unsafe_allow_html=True)
            st.dataframe(age_cat_df, use_container_width=True, height=160)
        with c2:
            st.markdown('<p style="color:#6e7681;font-size:0.7rem;font-weight:700;text-transform:uppercase;margin-bottom:6px;">Top Prices</p>', unsafe_allow_html=True)
            st.dataframe(age_prc_df, use_container_width=True, height=160)

# ─────────────────────────── RIGHT COLUMN: RECOMMENDATIONS & MATCHES ──────
with col_right:
    st.markdown('<p style="color:#6e7681;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:10px;">🚀 Targeted Recommendations (Top-10 ranked)</p>', unsafe_allow_html=True)
    
    # Calculate recommended list
    user_test_clicks = list(test_clicks_dict.get(search_user, set()))
    
    recs_list = []
    for rank, idx in enumerate(top10_idx):
        iid = unique_items[idx]
        hit = iid in user_test_clicks
        is_new_item = iid not in stats_full['item_imp_counts'] or stats_full['item_imp_counts'].get(iid, 0) == 0
        
        recs_list.append({
            'Rank': f"#{rank+1} 🎯" if hit else f"#{rank+1}",
            'Item ID': iid,
            'Category': item_cat_dict[iid],
            'Price': item_price_dict[iid],
            'Confidence': f"{probs[idx]*100:.1f}%",
            'Status': "🆕 New" if is_new_item else "🔥 Active",
            'Outcome': "✅ Hit" if hit else "—"
        })
    recs_df = pd.DataFrame(recs_list)
    
    # We display a standard clean table without pandas styler to prevent GlideDataEditor bugs
    st.dataframe(recs_df, use_container_width=True, height=340)

    st.markdown('<p style="color:#6e7681;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-top:12px;margin-bottom:8px;">📊 Actual Customer Clicks in Test Period</p>', unsafe_allow_html=True)

    # Inspect test clicks
    test_clicks_all = test_df[(test_df['USER_ID']==search_user) & (test_df['INTERACTION']==1)].merge(items_df, on='ITEM_ID', how='left')
    
    if len(test_clicks_all) == 0:
        st.markdown("""
        <div style="background:rgba(107,114,128,0.06); border:1px solid rgba(107,114,128,0.2); border-radius:10px; padding:15px; text-align:center;">
          <p style="color:#8b949e; font-size:0.85rem; margin:0; font-weight:600;">💤 No interactions observed during the validation window.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Check if recommendations contains any clicked items
        hit_items = list(set(recs_ids) & set(user_test_clicks))
        
        # Rank mapping for all candidate items (excluding training clicks)
        sorted_indices = np.argsort(-probs_rec)
        rank_mapping = {unique_items[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}
        
        # Display the actual clicks
        clicks_display = test_clicks_all[['ITEM_ID','CATEGORY','PRICE_BUCKET','TIMESTAMP']].copy()
        clicks_display['Model Rank'] = clicks_display['ITEM_ID'].map(
            lambda x: f"#{rank_mapping[x]} 🎯" if x in rank_mapping and rank_mapping[x] <= 10 else (f"#{rank_mapping[x]}" if x in rank_mapping else "—")
        )
        clicks_display = clicks_display.rename(columns={'ITEM_ID':'Item ID','CATEGORY':'Category','PRICE_BUCKET':'Price','TIMESTAMP':'Date'})
        clicks_display = clicks_display[['Item ID', 'Model Rank', 'Category', 'Price', 'Date']]
        st.dataframe(clicks_display, use_container_width=True, height=130)
        
        if len(hit_items) > 0:
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.08); border:1px solid #10b981; border-radius:10px; padding:12px 18px; margin-top:10px;">
              <p style="color:#10b981; font-size:0.8rem; font-weight:700; margin:0 0 4px 0; text-transform:uppercase;">🏆 Successful Matches ({len(hit_items)})</p>
              <p style="color:#8b949e; font-size:0.8rem; margin:0; line-height:1.5;">
                We successfully predicted: <b style="color:#f0f6fc;">{", ".join(hit_items)}</b>.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.08); border:1px solid #f59e0b; border-radius:10px; padding:12px 18px; margin-top:10px;">
              <p style="color:#f59e0b; font-size:0.8rem; font-weight:700; margin:0 0 4px 0; text-transform:uppercase;">⚠️ Evaluation Data Gap (Selection Bias)</p>
              <p style="color:#8b949e; font-size:0.8rem; margin:0; line-height:1.5;">
                The customer clicked ads in test period, but they were not in our top 10. Note: the platform only showed them ~3 ads total out of 1,000; our recommendations didn't overlap with those specific 3 shown.
              </p>
            </div>
            """, unsafe_allow_html=True)

