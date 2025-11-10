#!/usr/bin/env python3
"""
评估脚本 - 用于测试模型预测结果的性能
Usage: python3 evaluate.py <test_csv> <regression_output> <classification_output>
Example: python3 evaluate.py test.csv z5618951_regression.csv z5618951_classification.csv
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, f1_score, precision_recall_fscore_support, confusion_matrix


def evaluate_regression(test_df, pred_df):
    """评估回归任务"""
    print("=" * 70)
    print("REGRESSION EVALUATION")
    print("=" * 70)
    
    # 合并数据
    merged = test_df[['trans_num', 'amt']].merge(
        pred_df[['trans_num', 'amt']], 
        on='trans_num', 
        suffixes=('_true', '_pred')
    )
    
    y_true = merged['amt_true']
    y_pred = merged['amt_pred']
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算其他指标
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n📊 Regression Metrics:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # 计算分数
    if rmse >= 180:
        score = 0.0
    elif rmse <= 140:
        score = 5.0
    else:
        score = (1 - (rmse - 140) / (180 - 140)) * 5
    
    print(f"\n🎯 Estimated Regression Score: {score:.2f}/5.0")
    
    if rmse <= 140:
        print("  ✓ Excellent! Full marks! 🎉")
    elif rmse <= 160:
        print("  ⚠ Good, but can be improved")
    elif rmse <= 180:
        print("  ⚠ Need improvement")
    else:
        print("  ✗ Below threshold")
    
    return rmse, score


def evaluate_classification(test_df, pred_df):
    """评估分类任务"""
    print("\n" + "=" * 70)
    print("CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    # 合并数据
    merged = test_df[['trans_num', 'is_fraud']].merge(
        pred_df[['trans_num', 'is_fraud']], 
        on='trans_num', 
        suffixes=('_true', '_pred')
    )
    
    y_true = merged['is_fraud_true']
    y_pred = merged['is_fraud_pred']
    
    # 计算各类指标
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n📊 Classification Metrics:")
    print(f"\n  Class 0 (Normal):")
    print(f"    Precision: {precision[0]:.4f}")
    print(f"    Recall:    {recall[0]:.4f}")
    print(f"    F1-Score:  {f1_per_class[0]:.4f}")
    
    print(f"\n  Class 1 (Fraud) ← 关键:")
    print(f"    Precision: {precision[1]:.4f}")
    print(f"    Recall:    {recall[1]:.4f}")
    print(f"    F1-Score:  {f1_per_class[1]:.4f}")
    
    print(f"\n  🎯 F1 Macro: {f1_macro:.4f}")
    
    # 混淆矩阵
    print(f"\n📋 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Normal  Fraud")
    print(f"  Actual Normal  {cm[0,0]:6d} {cm[0,1]:5d}")
    print(f"         Fraud   {cm[1,0]:6d} {cm[1,1]:5d}")
    
    # 详细统计
    false_negatives = cm[1, 0]
    false_positives = cm[0, 1]
    true_positives = cm[1, 1]
    
    print(f"\n📈 Prediction Statistics:")
    print(f"  Predicted as Fraud: {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    print(f"  Actual Frauds:      {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"  True Positives:     {true_positives:,}")
    print(f"  False Positives:    {false_positives:,}")
    print(f"  False Negatives:    {false_negatives:,} ← (Missed frauds)")
    
    # 计算分数
    if f1_macro <= 0.85:
        score = 0.0
    elif f1_macro >= 0.97:
        score = 5.0
    else:
        score = ((f1_macro - 0.85) / (0.97 - 0.85)) * 5
    
    print(f"\n🎯 Estimated Classification Score: {score:.2f}/5.0")
    
    if f1_macro >= 0.97:
        print("  ✓ Excellent! Full marks! 🎉")
    elif f1_macro >= 0.93:
        print("  ✓ Very good!")
    elif f1_macro >= 0.90:
        print("  ⚠ Good, but can be improved")
    elif f1_macro >= 0.85:
        print("  ⚠ Need improvement")
    else:
        print("  ✗ Below threshold")
    
    # 建议
    if false_negatives > 50:
        print(f"\n💡 Suggestion: Too many missed frauds ({false_negatives})")
        print("   → Consider lowering classification threshold")
        print("   → Increase scale_pos_weight/class_weight")
    
    return f1_macro, score


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 evaluate.py <test_csv> <regression_output> <classification_output>")
        print("Example: python3 evaluate.py test.csv z5618951_regression.csv z5618951_classification.csv")
        sys.exit(1)
    
    test_path = sys.argv[1]
    regression_path = sys.argv[2]
    classification_path = sys.argv[3]
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION SCRIPT")
    print("=" * 70)
    print(f"\nTest file: {test_path}")
    print(f"Regression predictions: {regression_path}")
    print(f"Classification predictions: {classification_path}")
    
    # 读取数据
    try:
        test_df = pd.read_csv(test_path)
        regression_df = pd.read_csv(regression_path)
        classification_df = pd.read_csv(classification_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # 检查必要的列
    if 'amt' not in test_df.columns:
        print("\n⚠ Warning: 'amt' column not found in test file")
        print("   Cannot evaluate regression task")
        regression_score = None
        rmse = None
    else:
        rmse, regression_score = evaluate_regression(test_df, regression_df)
    
    if 'is_fraud' not in test_df.columns:
        print("\n⚠ Warning: 'is_fraud' column not found in test file")
        print("   Cannot evaluate classification task")
        classification_score = None
        f1_macro = None
    else:
        f1_macro, classification_score = evaluate_classification(test_df, classification_df)
    
    # 总结
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if rmse is not None:
        print(f"\n📊 Regression:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  Score: {regression_score:.2f}/5.0")
    
    if f1_macro is not None:
        print(f"\n📊 Classification:")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  Score: {classification_score:.2f}/5.0")
    
    if regression_score is not None and classification_score is not None:
        total_score = regression_score + classification_score
        print(f"\n🎯 Total Estimated Score: {total_score:.2f}/10.0")
        print(f"   (Plus up to 5.0 marks for Part I - Analysis)")
        
        if total_score >= 9.0:
            print("\n   ✓ Excellent work! 🎉🎉🎉")
        elif total_score >= 7.0:
            print("\n   ✓ Good job! 🎉")
        elif total_score >= 5.0:
            print("\n   ⚠ Needs improvement")
        else:
            print("\n   ✗ Significant improvement needed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
