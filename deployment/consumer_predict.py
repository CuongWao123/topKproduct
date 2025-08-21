import json
from kafka import KafkaConsumer
import joblib
import pandas as pd
from pathlib import Path
import numpy as np 
from utils import reduce_memory_usage ,  clean_dataset , filter_data , create_enhanced_features 

import warnings 

warnings.filterwarnings("ignore")

# ================== LOAD MODEL ==================
def get_latest_model_path(model_prefix=None, models_dir="D:/Work/rec-sys/models"):
    md = Path(models_dir)
    if not md.exists():
        raise FileNotFoundError("models directory not found. Train và save model trước.")
    candidates = list(md.glob("*.joblib"))
    if model_prefix:
        prefix = model_prefix.lower().replace(" ", "_")
        candidates = [p for p in candidates if p.name.lower().startswith(prefix)]
    if not candidates:
        raise FileNotFoundError("Không tìm thấy model .joblib")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

print("📦 Loading latest model...")
latest_model = get_latest_model_path(None)
loaded_pipeline = joblib.load(latest_model)
print(f"✅ Model loaded: {latest_model}")

# bạn cần import hoặc định nghĩa từ training code:
# reduce_memory_usage, clean_dataset, filter_data, create_enhanced_features
# tar_cols, available_num_cols, available_cat_cols

tar_cols = [
    'current_accounts_final_label',
    'payroll_accounts_final_label',
    'junior_accounts_final_label',
    'more_particular_accounts_final_label',
    'particular_accounts_final_label',
    'particular_plus_accounts_final_label',
    'home_account_final_label',
    'payroll_final_label',
    'e_account_final_label'
]






# ================== CONSUMER ==================
consumer = KafkaConsumer(
    "topic1",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="rec-consumer"
)

print("📥 Listening for messages...")

def top2_from_proba(y_proba_batch, tar_cols):
    n_rows = y_proba_batch[0].shape[0]
    out = []
    for i in range(n_rows):
        scores = []
        for j, product in enumerate(tar_cols):
            proba_ij = y_proba_batch[j][i]
            p1 = float(proba_ij[1]) if getattr(proba_ij, 'shape', None) and len(proba_ij) > 1 else float(proba_ij[0])
            scores.append((product, p1))
        scores.sort(key=lambda x: x[1], reverse=True)
        top2 = [p for p, _ in scores[:2]]
        if len(top2) < 2:
            top2 += [''] * (2 - len(top2))
        out.append((top2[0], top2[1]))
    return out


exclude_features = [
    'date', 'customer_id', 'registration_date', 'last_primary_date'
]

numeric_features = [
    'year', 'month', 'quarter', 'day_of_week', 'is_weekend', 'is_month_end', 'is_quarter_end',
    'days_since_registration', 'registration_year', 'registration_month', 'years_since_registration',
    'is_young_adult', 'is_middle_aged', 'is_senior', 'age_squared',
    'income_vs_median', 'is_high_income', 'is_low_income', 'log_income',
    'seniority_years', 'is_new_relationship', 'is_established_relationship',
    'is_primary_customer', 'is_new_customer', 'is_active', 'is_domestic', 'is_foreigner',
    'total_products', 'has_any_product', 'is_single_product', 'is_multi_product',
    'product_diversity_ratio', 'has_current_account', 'has_savings_account', 'has_premium_account',
    'age_income_interaction', 'income_per_age', 'seniority_income_interaction', 'income_growth_proxy',
    'young_high_income', 'senior_established', 'is_digital_channel',
    'is_vip_segment', 'is_university_segment', 'customer_stability_score',
    'is_stable_customer', 'customer_potential_score', 'is_high_potential'
]

categorical_features = [
    'tenure_category', 'age_group', 'income_quartile', 'seniority_category', 'channel_type'
]


for msg in consumer:
    batch_records = msg.value
    if isinstance(batch_records, dict):
        batch_records = [batch_records]
    elif not isinstance(batch_records, list):
        print(f"⚠️ Unsupported message type: {type(batch_records)}")
        continue
    df = pd.DataFrame(batch_records)

    df = reduce_memory_usage(df)
    df = clean_dataset(df)
    df = filter_data(df)
    df = create_enhanced_features(df)

    _needed_num = [col for col in numeric_features if col not in exclude_features]
    _needed_cat = [col for col in categorical_features if col not in exclude_features]

    df = df.drop(columns=[c for c in tar_cols if c in df.columns], errors="ignore")

    for col in _needed_num:
        if col not in df.columns:
            df[col] = 0
    for col in _needed_cat:
        if col not in df.columns:
            df[col] = "Unknown"

    X = df[_needed_num + _needed_cat]

    # Lấy customer_id
    customer_ids = df["customer_id"] if "customer_id" in df.columns else pd.Series(np.arange(len(X)))

    # ========== PREDICT ==========
    try:
        y_proba_batch = loaded_pipeline.predict_proba(X)
        top2_batch = top2_from_proba(y_proba_batch, tar_cols)
        for i, (rec1, rec2) in enumerate(top2_batch):
            cid = customer_ids.iloc[i]
            print(f"🔮 customer_id={cid} → rec1={rec1}, rec2={rec2}")
    except Exception as e:
        print(f"⚠️ Lỗi khi dự đoán: {e}")
