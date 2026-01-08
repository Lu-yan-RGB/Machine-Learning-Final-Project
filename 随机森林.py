import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import joblib
import os
from collections import Counter
import time
import matplotlib
from matplotlib import font_manager
import warnings

warnings.filterwarnings('ignore')


def setup_chinese_font():
    font_candidates = [
        'SimHei',
        'Microsoft YaHei',
        'STHeiti'
    ]

    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    selected_font = None

    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams.update({'font.size': 12})

    return selected_font


selected_font = setup_chinese_font()


class ObesityDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_mapping = None
        self.mean_values = {}
        self.mode_values = {}

    def fit(self, df, target_col='NObeyesdad'):
        df_processed = df.copy()

        numeric_cols = ['Age', 'Height', 'Weight', 'NCP', 'FCVC', 'CH2O', 'FAF', 'TUE']
        for col in numeric_cols:
            if col in df_processed.columns:
                self.mean_values[col] = df_processed[col].mean()

        # 分类特征-众数
        categorical_cols = ['CAEC', 'CALC', 'MTRANS']
        for col in categorical_cols:
            if col in df_processed.columns:
                self.mode_values[col] = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'

        self.target_mapping = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6
        }

        return self

    def transform(self, df, target_col='NObeyesdad', is_train=True):
        df_processed = df.copy()

        numeric_cols = ['Age', 'Height', 'Weight', 'NCP', 'FCVC', 'CH2O', 'FAF', 'TUE']
        for col in numeric_cols:
            if col in df_processed.columns:
                if is_train:
                    fill_value = self.mean_values.get(col, df_processed[col].mean())
                else:
                    fill_value = self.mean_values.get(col, 0)
                df_processed[col] = df_processed[col].fillna(fill_value)

        if target_col in df_processed.columns and self.target_mapping:
            df_processed[target_col] = df_processed[target_col].map(self.target_mapping)
            df_processed[target_col] = df_processed[target_col].fillna(-1).astype(int)

        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df_processed.columns:
                if is_train:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        known_labels = set(le.classes_)
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in known_labels else 0
                        )

        # 有序分类
        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        if 'CAEC' in df_processed.columns:
            df_processed['CAEC'] = df_processed['CAEC'].map(caec_mapping)
            df_processed['CAEC'] = df_processed['CAEC'].fillna(1)

        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
        if 'CALC' in df_processed.columns:
            df_processed['CALC'] = df_processed['CALC'].map(calc_mapping)
            df_processed['CALC'] = df_processed['CALC'].fillna(0)

        # MTRANS编码
        if 'MTRANS' in df_processed.columns:
            if is_train:
                le_mtrans = LabelEncoder()
                df_processed['MTRANS'] = le_mtrans.fit_transform(df_processed['MTRANS'])
                self.label_encoders['MTRANS'] = le_mtrans
            else:
                le_mtrans = self.label_encoders.get('MTRANS')
                if le_mtrans:
                    known_labels = set(le_mtrans.classes_)
                    df_processed['MTRANS'] = df_processed['MTRANS'].apply(
                        lambda x: le_mtrans.transform([x])[0] if x in known_labels else
                        le_mtrans.transform(['Public_Transportation'])[0]
                    )

        return df_processed

    def fit_transform(self, df, target_col='NObeyesdad'):
        self.fit(df, target_col)
        return self.transform(df, target_col, is_train=True)


def create_additional_features(df_processed, use_bmi=True, bmi_as_feature=False):
    df_features = df_processed.copy()

# 特征工程
    if use_bmi and 'Height' in df_features.columns and 'Weight' in df_features.columns:
        bmi_values = df_features['Weight'] / (df_features['Height'] ** 2)

        if bmi_as_feature:
            df_features['BMI'] = bmi_values
            print(f"  创建BMI特征 - 均值: {bmi_values.mean():.2f}")

        df_features['BMI_Category'] = pd.cut(
            bmi_values,
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=[0, 1, 2, 3, 4, 5]
        )
        df_features['BMI_Category'] = df_features['BMI_Category'].cat.codes


    if 'Age' in df_features.columns:
        df_features['Age_Group'] = pd.cut(
            df_features['Age'],
            bins=[0, 20, 30, 40, 50, 100],
            labels=[0, 1, 2, 3, 4]
        )

    # 饮食交互
    if 'FCVC' in df_features.columns and 'FAVC' in df_features.columns:
        df_features['Healthy_Eater'] = ((df_features['FCVC'] > 2) & (df_features['FAVC'] == 0)).astype(int)

    # 活动水平
    if 'FAF' in df_features.columns and 'TUE' in df_features.columns:
        faf_mean = df_features['FAF'].mean()
        faf_std = df_features['FAF'].std()
        tue_mean = df_features['TUE'].mean()
        tue_std = df_features['TUE'].std()

        if faf_std > 0 and tue_std > 0:
            faf_normalized = (df_features['FAF'] - faf_mean) / faf_std
            tue_normalized = (df_features['TUE'] - tue_mean) / tue_std
            df_features['Activity_Index'] = faf_normalized - tue_normalized

    # 饮食规律性评分
    if 'NCP' in df_features.columns and 'CAEC' in df_features.columns:
        df_features['Regular_Eater'] = ((df_features['NCP'] == 3) & (df_features['CAEC'] <= 1)).astype(int)

    print(f"特征工程完成，新增特征后的数据形状: {df_features.shape}")

    return df_features


def evaluate_model_with_diagnostics(y_true, y_pred, y_train, y_train_pred,
                                    model=None, X_train=None):
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"准确率差异: {abs(train_accuracy - test_accuracy):.4f}")

    # 判断
    if train_accuracy > test_accuracy + 0.05:
        print("过拟合")
        overfitting_risk = "高"
    elif train_accuracy > test_accuracy + 0.02:
        print("轻微过拟合")
        overfitting_risk = "中"
    else:
        print("过拟合风险低")
        overfitting_risk = "低"

    # 计算精确率、召回率、F1
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")

    # 交叉验证
    cv_scores = None
    if model is not None and X_train is not None and len(y_train) > 10:
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

            print(f"\n5折分层交叉验证:")
            print(f"平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
            print(f"各折分数: {[f'{score:.4f}' for score in cv_scores]}")
        except Exception as e:
            print(f"交叉验证跳过: {e}")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'overfitting_risk': overfitting_risk,
        'cv_mean': cv_scores.mean() if cv_scores is not None else None,
        'cv_std': cv_scores.std() if cv_scores is not None else None
    }


def plot_learning_curve(estimator, X, y, title="学习曲线"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证得分")

    plt.xlabel("训练样本数", fontsize=14)
    plt.ylabel("准确率", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./result/学习曲线.png', dpi=300, bbox_inches='tight')
    plt.show()

    return train_sizes, train_scores_mean, test_scores_mean


def main():
    data_path = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"
    df = pd.read_csv(data_path)
    print(f"  目标变量分布:")
    print(df['NObeyesdad'].value_counts())

    preprocessor = ObesityDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    print(f"  目标变量分布:")
    print(df_processed['NObeyesdad'].value_counts().sort_index())

    df_features = create_additional_features(df_processed, use_bmi=True, bmi_as_feature=False)
# 数据划分
    X = df_features.drop('NObeyesdad', axis=1)
    y = df_features['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {X_train.shape[0]} 个样本, {X_train.shape[1]} 个特征")
    print(f"  测试集: {X_test.shape[0]} 个样本, {X_test.shape[1]} 个特征")

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        bootstrap=True,
        oob_score=True,
        max_samples=0.8,
        ccp_alpha=0.01
    )

    rf_model.fit(X_train, y_train)
    print(f"  OOB分数: {rf_model.oob_score_:.4f}")

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    print(f"  训练集预测样本数: {len(y_train_pred)}")
    print(f"  测试集预测样本数: {len(y_test_pred)}")

    metrics = evaluate_model_with_diagnostics(
        y_test, y_test_pred,
        y_train, y_train_pred,
        rf_model, X_train
    )

    plot_learning_curve(rf_model, X_train, y_train, "随机森林学习曲线")

    class_names = ['偏瘦', '正常', '超重I级', '超重II级', '肥胖I级', '肥胖II级', '肥胖III级']
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                zero_division=0))

# 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 11})
    plt.title('混淆矩阵 - 肥胖等级分类', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.tight_layout()
    cm_path = './result/混淆矩阵.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
# 特征重要性
    feature_importance = pd.DataFrame({
        '特征': X_train.columns,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("特征重要性排名 (Top 10):")
    print(feature_importance.head(10).to_string(index=False))

    # 可视化
    plt.figure(figsize=(14, 10))
    top_features = feature_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))

    bars = plt.barh(range(len(top_features)), top_features['重要性'], color=colors)
    plt.yticks(range(len(top_features)), top_features['特征'], fontsize=12)
    plt.xlabel('特征重要性', fontsize=14)
    plt.title('随机森林特征重要性排名 (Top 15)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    for i, (bar, importance) in enumerate(zip(bars, top_features['重要性'])):
        plt.text(importance + 0.001, i, f'{importance:.4f}',
                 va='center', fontsize=10, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = './result/特征重要性.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.show()

    bmi_weight = feature_importance[feature_importance['特征'].str.contains('BMI')]['重要性'].sum()
    weight_weight = feature_importance[feature_importance['特征'].str.contains('Weight')]['重要性'].sum()
    print(f"BMI相关特征总权重: {bmi_weight:.4f}")
    print(f"体重相关特征总权重: {weight_weight:.4f}")

    if bmi_weight > 0.5:
        print("BMI特征权重过高")
    else:
        print("BMI特征权重合理")

    model_path = './result/随机森林肥胖分类模型.pkl'
    joblib.dump(rf_model, model_path)

    preprocessor_path = './result/数据预处理器.pkl'
    joblib.dump(preprocessor, preprocessor_path)

    feature_columns_path = './result/特征列.pkl'
    joblib.dump(list(X_train.columns), feature_columns_path)

    results_df = pd.DataFrame({
        '真实标签': y_test.values,
        '预测标签': y_test_pred,
        '是否正确': y_test.values == y_test_pred
    })
    results_path = './result/预测结果.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')

    fi_csv_path = './result/特征重要性.csv'
    feature_importance.to_csv(fi_csv_path, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
