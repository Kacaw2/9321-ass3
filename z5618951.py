import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
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
    # 新增：周末和夜间标记
    df_clean["is_weekend"] = df_clean["trans_dayofweek"].isin([5, 6]).astype(int)
    df_clean["is_night"] = ((df_clean["trans_hour"] >= 22) | (df_clean["trans_hour"] <= 6)).astype(int)
    
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

    df_clean["is_long_distance"] = (df_clean["customer_merchant_distance_km"] > 100).astype(int)
    
    # 3. 金额特征
    df_clean["amt_log"] = np.log1p(df_clean["amt"])
    # 金额与距离的交互
    df_clean["amt_per_km"] = df_clean["amt"] / (df_clean["customer_merchant_distance_km"] + 1)
    
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

    # 训练回归模型（用于验证）
    print("\n训练回归模型 (用于验证)...")
    reg_params = {
        'max_iter': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'l2_regularization': 0.1,
        'random_state': 42
    }
    
    model_reg = HistGradientBoostingRegressor(**reg_params)

    print(f"使用参数: {reg_params}")
    print(f"使用模型: {model_reg}")

    print("=" * 70)

    # 训练分类模型（用于验证）
    print("\n训练分类模型 (用于验证)...")
    clf_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model_clf = RandomForestClassifier(**clf_params)
    
    print(f"使用参数: {clf_params}")
    print(f"使用模型: {model_clf}")

    print("=" * 70)
    print("Machine Learning Pipeline - Part II & Part III")
    print("=" * 70)

    # ============================================================
    # 数据加载和预处理（对 train 和 test 做完全一致的处理）
    # ============================================================
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
    # Part II - 回归任务（预测交易金额 amt）
    # ============================================================
    print("\n" + "=" * 70)
    print("Part II: Regression Task - Amount Prediction")
    print("=" * 70)

    # 准备回归特征（不包含 amt，因为这是目标变量）
    reg_feature_cols = [
        col for col in train_encoded.columns 
        if col not in ["trans_num", "is_fraud", "amt"]
    ]
    
    X_train_reg = train_encoded[reg_feature_cols]
    y_train_reg = train_encoded["amt"]
    X_test_reg = test_encoded[reg_feature_cols]

    print(f"\n特征数量: {len(reg_feature_cols)}")
    print(f"训练集大小: {len(X_train_reg):,}")
    print(f"测试集大小: {len(X_test_reg):,}")
    print(f"交易金额统计:")
    print(f"  均值: ${y_train_reg.mean():.2f}")
    print(f"  中位数: ${y_train_reg.median():.2f}")
    print(f"  标准差: ${y_train_reg.std():.2f}")

    
    model_reg.fit(X_train_reg, y_train_reg)
    print("✓ 回归模型训练完成")

    # 生成测试集预测
    print("\n生成测试集预测...")
    pred_reg_test = model_reg.predict(X_test_reg)
    print(f"✓ 预测完成: {len(pred_reg_test):,} 个样本")
    print(f"预测金额统计:")
    print(f"  均值: ${pred_reg_test.mean():.2f}")
    print(f"  中位数: ${np.median(pred_reg_test):.2f}")

    # 生成回归输出文件
    regression_output = pd.DataFrame({
        "trans_num": test_encoded["trans_num"],
        "amt": pred_reg_test
    })

    regression_output.to_csv("z5618951_regression.csv", index=False)
    print(f"\n✓ z5618951_regression.csv ({len(regression_output):,} 行)")

    print("\n" + "=" * 70)
    print("测试集性能评估 (回归)")
    print("=" * 70)
    
    y_test_reg = test_encoded["amt"]
    rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg_test))
    
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Test 金额统计 (真实):")
    print(f"  均值: ${y_test_reg.mean():.2f}")
    print(f"  中位数: ${y_test_reg.median():.2f}")

    # ============================================================
    # Part III - 分类任务（检测欺诈 is_fraud）
    # ============================================================
    print("\n" + "=" * 70)
    print("Part III: Classification Task - Fraud Detection")
    print("=" * 70)

    # 准备分类特征（包含 amt）
    clf_feature_cols = [
        col for col in train_encoded.columns if col not in ["trans_num", "is_fraud"]
    ]
    
    X_train_clf = train_encoded[clf_feature_cols]
    y_train_clf = train_encoded["is_fraud"]
    X_test_clf = test_encoded[clf_feature_cols]

    print(f"\n特征数量: {len(clf_feature_cols)}")
    print(f"训练集大小: {len(X_train_clf):,}")
    print(f"测试集大小: {len(X_test_clf):,}")
    print(f"原始欺诈样本比例: {y_train_clf.mean()*100:.2f}%")

    # 处理类别不平衡
    X_train_clf_balanced, y_train_clf_balanced = handle_class_imbalance(
        X_train_clf, y_train_clf, random_state=42
    )

    print("\n训练分类模型...")

    model_clf.fit(X_train_clf_balanced, y_train_clf_balanced )
    print("✓ 分类模型训练完成")

    # 验证集评估
    print("\n生成测试集预测...")
    pred_clf_test = model_clf.predict(X_test_clf)

    print(f"✓ 预测完成: {len(pred_clf_test):,} 个样本")
    print(f"  预测为欺诈: {pred_clf_test.sum():,} ({pred_clf_test.mean()*100:.2f}%)")
    print(f"  预测为正常: {(pred_clf_test==0).sum():,} ({(pred_clf_test==0).mean()*100:.2f}%)")

    # 生成分类输出文件
    classification_output = pd.DataFrame({
        "trans_num": test_encoded["trans_num"],
        "is_fraud": pred_clf_test
    })
    classification_output.to_csv("z5618951_classification.csv", index=False)
    print(f"\n✓ z5618951_classification.csv ({len(classification_output):,} 行)")

    print("\n" + "=" * 70)
    print("测试集性能评估")
    print("=" * 70)

    y_test_clf = test_encoded["is_fraud"]
    f1_macro = f1_score(y_test_clf, pred_clf_test, average='macro')
    f1_weighted = f1_score(y_test_clf, pred_clf_test, average='weighted')
        
    print(f"Test F1 Score (Macro): {f1_macro:.4f}")
    print(f"Test F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Test 欺诈样本比例 (真实): {y_test_clf.mean()*100:.2f}%")
        
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
    # 最终总结
    # ============================================================
    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)

    print(f"Part II  - RMSE: {rmse:.2f}")

    print(f"Part III - F1 Score (Macro): {f1_macro:.4f}")

    print(f"Part III - Estimated Score: {score:.2f}/5.0")


    print("\n生成的输出文件:")
    print("  1. z5618951_regression.csv")
    print("  2. z5618951_classification.csv")
    print("=" * 70)
    print("\n✓ 所有任务完成!")


if __name__ == "__main__":
    main()
