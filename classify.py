import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

CATEGORICAL_COLS = ["category", "gender", "state", "job", "merchant", "city"]


def clean_and_prepare_data(df):
    """数据清洗和特征工程"""
    df_clean = df.copy()

    # 1. 转换日期列
    df_clean["trans_date_trans_time"] = pd.to_datetime(df_clean["trans_date_trans_time"])
    df_clean["dob"] = pd.to_datetime(df_clean["dob"])

    # 2. 创建年龄特征
    df_clean["age"] = (df_clean["trans_date_trans_time"] - df_clean["dob"]).dt.days / 365.25

    # 3. 时间特征
    df_clean["trans_hour"] = df_clean["trans_date_trans_time"].dt.hour
    df_clean["trans_dayofweek"] = df_clean["trans_date_trans_time"].dt.dayofweek
    df_clean["trans_month"] = df_clean["trans_date_trans_time"].dt.month
    df_clean["trans_day"] = df_clean["trans_date_trans_time"].dt.day
    df_clean["trans_quarter"] = df_clean["trans_date_trans_time"].dt.quarter

    # 4. 地理距离特征
    def haversine_np(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df_clean["customer_merchant_distance_km"] = haversine_np(
        df_clean["lat"], df_clean["long"], df_clean["merch_lat"], df_clean["merch_long"]
    )

    df_clean["distance_log"] = np.log1p(df_clean["customer_merchant_distance_km"])
    df_clean["city_pop_log"] = np.log1p(df_clean["city_pop"])

    # 删除不需要的列
    drop_cols = ["trans_date_trans_time", "dob", "unix_time", "cc_num", "first", "last", "street"]
    df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], inplace=True)

    # 转换分类列
    for col in CATEGORICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category")

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


def handle_class_imbalance(X_train, y_train, random_state=42):
    """使用SMOTE处理类别不平衡"""
    print("\n处理类别不平衡 (SMOTE)...")
    print(f"原始训练集大小: {len(X_train):,}")
    print(f"原始欺诈样本比例: {y_train.mean()*100:.2f}%")
    
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"平衡后训练集大小: {len(X_balanced):,}")
    print(f"平衡后欺诈样本比例: {y_balanced.mean()*100:.2f}%")
    
    return X_balanced, y_balanced


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 z5618951.py <train_csv> <test_csv>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    print("=" * 70)
    print("Part III: Classification Task - Fraud Detection")
    print("=" * 70)

    print("\n加载数据...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Training set: {len(train_df):,} rows")
    print(f"Test set: {len(test_df):,} rows")

    print("\n清洗和特征工程...")
    train_clean = clean_and_prepare_data(train_df)
    test_clean = clean_and_prepare_data(test_df)

    print("编码特征...")
    train_encoded, encoders = encode_features(train_clean, is_train=True)
    test_encoded, _ = encode_features(test_clean, is_train=False, encoders=encoders)

    # ============================================================
    # 准备分类任务数据
    # ============================================================
    print("\n准备分类任务数据...")
    
    clf_feature_cols = [
        col for col in train_encoded.columns 
        if col not in ["trans_num", "is_fraud"]
    ]
    
    X_train_clf = train_encoded[clf_feature_cols]
    y_train_clf = train_encoded["is_fraud"]
    X_test_clf = test_encoded[clf_feature_cols]

    print(f"特征数量: {len(clf_feature_cols)}")
    print(f"训练集大小: {len(X_train_clf):,}")
    print(f"测试集大小: {len(X_test_clf):,}")
    print(f"原始欺诈样本比例: {y_train_clf.mean()*100:.2f}%")

    # ============================================================
    # 🔥 划分验证集
    # ============================================================
    print("\n" + "=" * 70)
    print("Splitting Validation Set")
    print("=" * 70)
    
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train_clf, y_train_clf, test_size=0.15, random_state=42, stratify=y_train_clf
    )
    
    print(f"Training set: {len(X_train_val):,}")
    print(f"Validation set: {len(X_val):,}")

    # ============================================================
    # 🔥 处理类别不平衡（只对训练集部分）
    # ============================================================
    X_train_val_balanced, y_train_val_balanced = handle_class_imbalance(
        X_train_val, y_train_val, random_state=42  # ✅ 只对训练集部分做SMOTE
    )

    # ============================================================
    # 训练模型（用于验证）
    # ============================================================
    print("\n" + "=" * 70)
    print("训练分类模型 (用于验证)")
    print("=" * 70)

    best_clf_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    print(f"使用参数: {best_clf_params}")
    
    model_clf = RandomForestClassifier(**best_clf_params)
    
    print("\n开始训练...")
    model_clf.fit(X_train_val_balanced, y_train_val_balanced)  # ✅ 用训练集部分
    print("✓ 模型训练完成")

    # ============================================================
    # 🔥 在验证集上评估
    # ============================================================
    print("\n" + "=" * 70)
    print("Validation Set Evaluation")
    print("=" * 70)
    
    pred_val = model_clf.predict(X_val)  # ✅ 在独立验证集上评估
    f1_macro = f1_score(y_val, pred_val, average='macro')
    f1_weighted = f1_score(y_val, pred_val, average='weighted')
    
    print(f"Validation F1 Score (Macro): {f1_macro:.4f}")
    print(f"Validation F1 Score (Weighted): {f1_weighted:.4f}")
    
    # 预估得分
    if f1_macro >= 0.97:
        score = 5.0
        print(f"✓ Estimated score: {score:.2f}/5.0 🎉")
    elif f1_macro >= 0.85:
        score = ((f1_macro - 0.85) / 0.12) * 5
        print(f"⚠ Estimated score: {score:.2f}/5.0")
    else:
        score = 0.0
        print(f"✗ Estimated score: {score:.2f}/5.0 (F1 too low)")

    # ============================================================
    # 🔥 在全量数据上重新训练最终模型
    # ============================================================
    print("\n" + "=" * 70)
    print("Retraining on Full Training Data")
    print("=" * 70)
    
    # 对全量数据做SMOTE
    X_train_clf_balanced, y_train_clf_balanced = handle_class_imbalance(
        X_train_clf, y_train_clf, random_state=42  # ✅ 用全量数据
    )
    
    # 训练最终模型
    model_clf_final = RandomForestClassifier(**best_clf_params)
    print("\nTraining final model...")
    model_clf_final.fit(X_train_clf_balanced, y_train_clf_balanced)
    print("✓ Final model training complete")

    # ============================================================
    # 生成测试集预测
    # ============================================================
    print("\n" + "=" * 70)
    print("生成测试集预测")
    print("=" * 70)
    
    pred_clf_test = model_clf_final.predict(X_test_clf)  # ✅ 用最终模型

    print(f"✓ 预测完成: {len(pred_clf_test):,} 个样本")
    print(f"  预测为欺诈: {pred_clf_test.sum():,} ({pred_clf_test.mean()*100:.2f}%)")
    print(f"  预测为正常: {(pred_clf_test==0).sum():,} ({(pred_clf_test==0).mean()*100:.2f}%)")

    # ============================================================
    # 生成输出文件
    # ============================================================
    print("\n" + "=" * 70)
    print("生成输出文件")
    print("=" * 70)

    classification_output = pd.DataFrame({
        "trans_num": test_encoded["trans_num"],
        "is_fraud": pred_clf_test
    })
    classification_output.to_csv("z5618951_classification.csv", index=False)
    print(f"✓ z5618951_classification.csv ({len(classification_output):,} 行)")

    print("\n分类输出示例:")
    print(classification_output.head(10))

    # ============================================================
    # 最终总结
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Validation F1 Score (Macro): {f1_macro:.4f}")
    print(f"Estimated Score: {score:.2f}/5.0")
    print("\n✓ Part III Classification Task Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
