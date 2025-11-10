import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier

CATEGORICAL_COLS = ["category", "gender", "state", "job", "merchant", "city"]

#    python z5618951.py train.csv test.csv 
#    python e.py test.csv z5618951_regression.csv z5618951_classification.csv

def add_advanced_features(df_clean):
    """添加高级特征"""
    
    # 1. 更细粒度的时间特征
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['trans_hour'] / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['trans_hour'] / 24)
    df_clean['day_sin'] = np.sin(2 * np.pi * df_clean['trans_day'] / 31)
    df_clean['day_cos'] = np.cos(2 * np.pi * df_clean['trans_day'] / 31)
    
    # 2. 交互特征
    df_clean['weekend_hour'] = df_clean['is_weekend'] * df_clean['trans_hour']
    df_clean['night_weekend'] = df_clean['is_night'] * df_clean['is_weekend']
    
    # 3. 距离和位置交互
    df_clean['lat_long_interaction'] = df_clean['lat'] * df_clean['long']
    df_clean['distance_pop_ratio'] = df_clean['customer_merchant_distance_km'] / (df_clean['city_pop'] + 1)
    
    # 4. 分组统计（如果数据量足够）
    # merchant平均交易距离
    # category平均交易距离
    
    return df_clean

def clean_and_prepare_data(df):
    """特征工程"""
    df_clean = df.copy()

    # 1. Convert datetime columns
    df_clean['trans_date_trans_time'] = pd.to_datetime(df_clean['trans_date_trans_time'])
    df_clean['dob'] = pd.to_datetime(df_clean['dob'])

    # 2. Age feature
    df_clean['age'] = (df_clean['trans_date_trans_time'] - df_clean['dob']).dt.days / 365.25

    # 3. Time-based features
    df_clean['trans_hour'] = df_clean['trans_date_trans_time'].dt.hour
    df_clean['trans_dayofweek'] = df_clean['trans_date_trans_time'].dt.dayofweek
    df_clean['trans_month'] = df_clean['trans_date_trans_time'].dt.month
    df_clean['trans_day'] = df_clean['trans_date_trans_time'].dt.day
    df_clean['trans_quarter'] = df_clean['trans_date_trans_time'].dt.quarter
    df_clean['trans_year'] = df_clean['trans_date_trans_time'].dt.year
    df_clean['is_weekend'] = (df_clean['trans_dayofweek'] >= 5).astype(int)
    
    df_clean['is_night'] = ((df_clean['trans_hour'] >= 22) | 
                            (df_clean['trans_hour'] <= 6)).astype(int)
    df_clean['is_business_hours'] = ((df_clean['trans_hour'] >= 9) & 
                                      (df_clean['trans_hour'] <= 17)).astype(int)
    df_clean['is_rush_hour'] = (((df_clean['trans_hour'] >= 7) & (df_clean['trans_hour'] <= 9)) |
                                  ((df_clean['trans_hour'] >= 17) & (df_clean['trans_hour'] <= 19))).astype(int)
    
    # 4. Geo-distance feature
    def haversine_np(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df_clean['customer_merchant_distance_km'] = haversine_np(
        df_clean['lat'], df_clean['long'], df_clean['merch_lat'], df_clean['merch_long']
    )
  
    # Distance features
    df_clean['distance_log'] = np.log1p(df_clean['customer_merchant_distance_km'])
    df_clean['distance_squared'] = df_clean['customer_merchant_distance_km'] ** 2
    df_clean['is_local_transaction'] = (df_clean['customer_merchant_distance_km'] < 10).astype(int)
    df_clean['is_very_far'] = (df_clean['customer_merchant_distance_km'] > 200).astype(int)
    df_clean['is_medium_distance'] = ((df_clean['customer_merchant_distance_km'] >= 10) & 
                                       (df_clean['customer_merchant_distance_km'] <= 200)).astype(int)
    
    # Population features
    df_clean['city_pop_log'] = np.log1p(df_clean['city_pop'])
    df_clean['city_pop_sqrt'] = np.sqrt(df_clean['city_pop'])
    df_clean['is_big_city'] = (df_clean['city_pop'] > 100000).astype(int)
    df_clean['is_small_town'] = (df_clean['city_pop'] < 5000).astype(int)
    df_clean['is_medium_city'] = ((df_clean['city_pop'] >= 5000) & 
                                   (df_clean['city_pop'] <= 100000)).astype(int)
    
    # Amount features
    if 'amt' in df_clean.columns:
        df_clean['amt_log'] = np.log1p(df_clean['amt'])
        df_clean['amt_sqrt'] = np.sqrt(df_clean['amt'])
        df_clean['amt_squared'] = df_clean['amt'] ** 2
        df_clean['is_high_amt'] = (df_clean['amt'] > 500).astype(int)
        df_clean['is_very_high_amt'] = (df_clean['amt'] > 1000).astype(int)
        df_clean['is_low_amt'] = (df_clean['amt'] < 10).astype(int)
        df_clean['is_medium_amt'] = ((df_clean['amt'] >= 10) & 
                                      (df_clean['amt'] <= 500)).astype(int)
    
    # Interaction features
    if 'amt' in df_clean.columns:
        df_clean['distance_amt_interaction'] = df_clean['customer_merchant_distance_km'] * df_clean['amt']
        df_clean['distance_amt_ratio'] = df_clean['customer_merchant_distance_km'] / (df_clean['amt'] + 1)
        df_clean['age_amt_interaction'] = df_clean['age'] * df_clean['amt']
        df_clean['hour_amt_interaction'] = df_clean['trans_hour'] * df_clean['amt']
        df_clean['city_pop_amt_ratio'] = df_clean['city_pop'] / (df_clean['amt'] + 1)
    
    # Age features
    df_clean['age_squared'] = df_clean['age'] ** 2
    df_clean['is_young'] = (df_clean['age'] < 30).astype(int)
    df_clean['is_senior'] = (df_clean['age'] > 60).astype(int)
    df_clean['is_middle_age'] = ((df_clean['age'] >= 30) & 
                                  (df_clean['age'] <= 60)).astype(int)
    
    # Combined features
    df_clean['night_far_transaction'] = df_clean['is_night'] * df_clean['is_very_far']
    df_clean['weekend_high_amt'] = df_clean['is_weekend'] * df_clean.get('is_high_amt', 0)
    
    drop_cols = [
        'trans_date_trans_time', 'dob', 'unix_time',
        'cc_num', 'first', 'last', 'street',
    ]
    df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], inplace=True)

    cat_cols = ['category', 'gender', 'state', 'job', 'merchant', 'city']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    df_clean = add_advanced_features(df_clean)

    return df_clean


def encode_features(df, is_train=True, encoders=None):
    """特征编码"""
    df_encoded = df.copy()

    if is_train:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df_encoded.columns:
                le = LabelEncoder()
                values = df_encoded[col].astype(str)
                df_encoded[f"{col}_encoded"] = le.fit_transform(values)
                encoders[col] = le
        df_encoded.drop(columns=CATEGORICAL_COLS, errors="ignore", inplace=True)
        return df_encoded, encoders
    else:
        if encoders is None:
            raise ValueError("When is_train=False, encoders must be provided.")
        for col in CATEGORICAL_COLS:
            if col in df_encoded.columns and col in encoders:
                le = encoders[col]
                mapping = {str(cls): idx for idx, cls in enumerate(le.classes_)}
                df_encoded[f"{col}_encoded"] = (
                    df_encoded[col].astype(str).map(mapping).fillna(-1).astype(int)
                )
        df_encoded.drop(columns=CATEGORICAL_COLS, errors="ignore", inplace=True)
        return df_encoded, encoders


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 z5618951.py <train_csv> <test_csv>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Feature engineering
    train_clean = clean_and_prepare_data(train_df)
    test_clean = clean_and_prepare_data(test_df)

    # Encode features
    train_encoded, encoders = encode_features(train_clean, is_train=True)
    test_encoded, _ = encode_features(test_clean, is_train=False, encoders=encoders)

    # ============================================================
    # Regression Task
    # ============================================================
    reg_feature_cols = [col for col in train_encoded.columns 
                    if col not in ['trans_num', 'is_fraud', 'amt']
                    and 'amt' not in col.lower()] 
    
    X_train_reg = train_encoded[reg_feature_cols]
    y_train_reg = train_encoded["amt"]
    X_test_reg = test_encoded[reg_feature_cols]

    model_reg = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model_reg.fit(X_train_reg, y_train_reg)
    pred_reg_test = model_reg.predict(X_test_reg)
    pred_reg_test = np.maximum(pred_reg_test, 0)
    
    regression_output = pd.DataFrame({
        "trans_num": test_encoded["trans_num"],
        "amt": pred_reg_test
    })
    regression_output.to_csv("z5618951_regression.csv", index=False)

    # ============================================================
    # Classification Task
    # ============================================================
    clf_feature_cols = [
        col for col in train_encoded.columns if col not in ["trans_num", "is_fraud"]
    ]
    
    X_train_clf = train_encoded[clf_feature_cols]
    y_train_clf = train_encoded["is_fraud"]
    X_test_clf = test_encoded[clf_feature_cols]

    fraud_count = y_train_clf.sum()
    normal_count = len(y_train_clf) - fraud_count

    scale_pos_weight = (normal_count / fraud_count) * 1.2
    print("scale_pos_weigth: ",scale_pos_weight)
    model_clf = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='logloss'
    )
    best_threshold = 0.74

    model_clf.fit(X_train_clf, y_train_clf)
    pred_proba = model_clf.predict_proba(X_test_clf)[:, 1]
    pred_clf_test = (pred_proba >= best_threshold).astype(int)

    classification_output = pd.DataFrame({
        "trans_num": test_encoded["trans_num"],
        "is_fraud": pred_clf_test
    })
    classification_output.to_csv("z5618951_classification.csv", index=False)


if __name__ == "__main__":
    main()
