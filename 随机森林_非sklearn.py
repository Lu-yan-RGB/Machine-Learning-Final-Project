import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib
import os
import time
import matplotlib
from matplotlib import font_manager
import warnings
import math
from collections import Counter
from copy import deepcopy

warnings.filterwarnings('ignore')


def setup_chinese_font():
    font_candidates = ['SimHei', 'Microsoft YaHei', 'STHeiti']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]

    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False

    return selected_font


selected_font = setup_chinese_font()


class LabelEncoderCustom:
    def __init__(self):
        self.classes_ = None
        self.mapping = {}
        self.reverse_mapping = {}

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.mapping = {val: i for i, val in enumerate(self.classes_)}
        self.reverse_mapping = {i: val for val, i in self.mapping.items()}
        return self

    def transform(self, y):
        return np.array([self.mapping[val] for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.reverse_mapping[val] for val in y])


class StandardScalerCustom:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # 避免除零错误
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ObesityDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScalerCustom()
        self.target_mapping = None
        self.mean_values = {}
        self.mode_values = {}

    def fit(self, df, target_col='NObeyesdad'):
        df_processed = df.copy()

        numeric_cols = ['Age', 'Height', 'Weight', 'NCP', 'FCVC', 'CH2O', 'FAF', 'TUE']
        for col in numeric_cols:
            if col in df_processed.columns:
                self.mean_values[col] = df_processed[col].mean()

        categorical_cols = ['CAEC', 'CALC', 'MTRANS']
        for col in categorical_cols:
            if col in df_processed.columns:
                modes = df_processed[col].mode()
                self.mode_values[col] = modes[0] if len(modes) > 0 else 'Unknown'

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
                fill_value = self.mean_values.get(col, 0)
                df_processed[col] = df_processed[col].fillna(fill_value)

        if target_col in df_processed.columns and self.target_mapping:
            df_processed[target_col] = df_processed[target_col].map(self.target_mapping)
            df_processed[target_col] = df_processed[target_col].fillna(-1).astype(int)

        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df_processed.columns:
                if is_train:
                    le = LabelEncoderCustom()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        known_labels = set(le.classes_)
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([x])[0] if x in known_labels else 0
                        )

        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        if 'CAEC' in df_processed.columns:
            df_processed['CAEC'] = df_processed['CAEC'].map(caec_mapping).fillna(1)

        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
        if 'CALC' in df_processed.columns:
            df_processed['CALC'] = df_processed['CALC'].map(calc_mapping).fillna(0)

        if 'MTRANS' in df_processed.columns:
            if is_train:
                le_mtrans = LabelEncoderCustom()
                df_processed['MTRANS'] = le_mtrans.fit_transform(df_processed['MTRANS'])
                self.label_encoders['MTRANS'] = le_mtrans
            else:
                le_mtrans = self.label_encoders.get('MTRANS')
                if le_mtrans:
                    known_labels = set(le_mtrans.classes_)
                    df_processed['MTRANS'] = df_processed['MTRANS'].apply(
                        lambda x: le_mtrans.transform([x])[0] if x in known_labels else 3
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

        bins = [0, 18.5, 25, 30, 35, 40, 100]
        labels = [0, 1, 2, 3, 4, 5]
        df_features['BMI_Category'] = pd.cut(bmi_values, bins=bins, labels=labels)
        df_features['BMI_Category'] = df_features['BMI_Category'].cat.codes
        print(f"  创建BMI分类特征")

    if 'Age' in df_features.columns:
        bins = [0, 20, 30, 40, 50, 100]
        labels = [0, 1, 2, 3, 4]
        df_features['Age_Group'] = pd.cut(df_features['Age'], bins=bins, labels=labels)

    if 'FCVC' in df_features.columns and 'FAVC' in df_features.columns:
        df_features['Healthy_Eater'] = ((df_features['FCVC'] > 2) & (df_features['FAVC'] == 0)).astype(int)

    if 'FAF' in df_features.columns and 'TUE' in df_features.columns:
        faf_mean = df_features['FAF'].mean()
        faf_std = df_features['FAF'].std()
        tue_mean = df_features['TUE'].mean()
        tue_std = df_features['TUE'].std()

        if faf_std > 0 and tue_std > 0:
            faf_normalized = (df_features['FAF'] - faf_mean) / faf_std
            tue_normalized = (df_features['TUE'] - tue_mean) / tue_std
            df_features['Activity_Index'] = faf_normalized - tue_normalized

    if 'NCP' in df_features.columns and 'CAEC' in df_features.columns:
        df_features['Regular_Eater'] = ((df_features['NCP'] == 3) & (df_features['CAEC'] <= 1)).astype(int)

    print(f"特征工程完成，新增特征后的数据形状: {df_features.shape}")

    return df_features


class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, impurity=None, samples=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.samples = samples


def calculate_gini(y):
    # 基尼不纯度
    if len(y) == 0:
        return 0

    classes = np.unique(y)
    gini = 1.0

    for cls in classes:
        p = np.sum(y == cls) / len(y)
        gini -= p ** 2

    return gini


def find_best_split(X, y, n_features, random_state=None):
    n_samples, n_feat = X.shape
    if n_samples <= 1:
        return None, None

    # 特征子集
    if random_state is not None:
        np.random.seed(random_state)
    feature_indices = np.random.choice(n_feat, size=n_features, replace=False)

    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    current_gini = calculate_gini(y)

    for feature_idx in feature_indices:
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        if len(unique_values) > 10:
            percentiles = np.linspace(0, 100, 11)[1:-1]
            candidate_thresholds = np.percentile(feature_values, percentiles)
        else:
            candidate_thresholds = unique_values[:-1]

        for threshold in candidate_thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            # 加权基尼不纯度
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            gini_left = calculate_gini(y[left_mask])
            gini_right = calculate_gini(y[right_mask])
            weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold

    if best_gini >= current_gini:
        return None, None

    return best_feature, best_threshold


def build_tree(X, y, max_depth, min_samples_split, n_features, current_depth=0, random_state=None):
    n_samples = X.shape[0]

    # 终止条件
    if (current_depth >= max_depth or
            n_samples < min_samples_split or
            len(np.unique(y)) == 1):
        class_counts = Counter(y)
        most_common_class = class_counts.most_common(1)[0][0]
        impurity = calculate_gini(y)

        return TreeNode(
            value=most_common_class,
            impurity=impurity,
            samples=n_samples
        )

    # 寻找最佳分裂
    best_feature, best_threshold = find_best_split(
        X, y, n_features, random_state
    )

    if best_feature is None:
        # 无法分裂，创建叶子节点
        class_counts = Counter(y)
        most_common_class = class_counts.most_common(1)[0][0]
        impurity = calculate_gini(y)

        return TreeNode(
            value=most_common_class,
            impurity=impurity,
            samples=n_samples
        )

    # 分裂数据集
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree = build_tree(
        X[left_mask], y[left_mask], max_depth, min_samples_split,
        n_features, current_depth + 1, random_state
    )

    right_subtree = build_tree(
        X[right_mask], y[right_mask], max_depth, min_samples_split,
        n_features, current_depth + 1, random_state
    )

    return TreeNode(
        feature_idx=best_feature,
        threshold=best_threshold,
        left=left_subtree,
        right=right_subtree,
        impurity=calculate_gini(y),
        samples=n_samples
    )


def predict_tree(node, X):
    if node.value is not None:
        return node.value

    if X[node.feature_idx] <= node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


def predict_tree_proba(node, X, n_classes):
    if node.value is not None:
        proba = np.zeros(n_classes)
        proba[node.value] = 1.0
        return proba

    if X[node.feature_idx] <= node.threshold:
        return predict_tree_proba(node.left, X, n_classes)
    else:
        return predict_tree_proba(node.right, X, n_classes)


class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=10, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]

        n_features = max(1, int(np.sqrt(self.n_features_)))

        self.tree_ = build_tree(
            X, y, self.max_depth, self.min_samples_split,
            n_features, random_state=self.random_state
        )

        return self

    def predict(self, X):
        if self.tree_ is None:
            raise ValueError("模型尚未训练")

        predictions = []
        for i in range(X.shape[0]):
            pred = predict_tree(self.tree_, X[i])
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        if self.tree_ is None:
            raise ValueError("模型尚未训练")

        probas = []
        for i in range(X.shape[0]):
            proba = predict_tree_proba(self.tree_, X[i], self.n_classes_)
            probas.append(proba)

        return np.array(probas)


def bootstrap_sample(X, y, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)

    return X[indices], y[indices]


class RandomForestClassifierCustom:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]

        # 确定每个树使用的特征数量
        if self.max_features == 'sqrt':
            n_features_per_tree = max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == 'log2':
            n_features_per_tree = max(1, int(np.log2(self.n_features_)))
        else:
            n_features_per_tree = self.n_features_

        for i in range(self.n_estimators):
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            else:
                tree_random_state = None

            # 采样
            X_boot, y_boot = bootstrap_sample(X, y, tree_random_state)

            tree = DecisionTreeClassifierCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=tree_random_state
            )

            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        if len(self.trees_) == 0:
            raise ValueError("模型尚未训练")

        # 收集所有树的预测结果
        tree_predictions = []
        for tree in self.trees_:
            pred = tree.predict(X)
            tree_predictions.append(pred)

        tree_predictions = np.array(tree_predictions)

        final_predictions = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)

    def predict_proba(self, X):
        if len(self.trees_) == 0:
            raise ValueError("模型尚未训练")

        # 概率预测
        tree_probas = []
        for tree in self.trees_:
            proba = tree.predict_proba(X)
            tree_probas.append(proba)

        # 平均概率
        tree_probas = np.array(tree_probas)  # n_trees × n_samples × n_classes
        avg_proba = np.mean(tree_probas, axis=0)

        return avg_proba

    def feature_importances_(self, X_train):
        importances = np.zeros(self.n_features_)

        # 特征重要性
        for tree in self.trees_:
            tree_importances = self._calculate_tree_importance(tree.tree_, X_train)
            importances += tree_importances

        # 归一化
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

        return importances

    def _calculate_tree_importance(self, node, X_train, depth=0):
        if node.value is not None:  # 叶子节点
            return np.zeros(self.n_features_)

        feature_idx = node.feature_idx
        n_total = node.samples

        left_mask = X_train[:, feature_idx] <= node.threshold
        n_left = np.sum(left_mask)
        n_right = n_total - n_left
# 信息增益
        gini_parent = node.impurity
        gini_left = node.left.impurity
        gini_right = node.right.impurity

        importance = (n_total / len(X_train)) * (
                gini_parent -
                (n_left / n_total * gini_left) -
                (n_right / n_total * gini_right)
        )

        importances = np.zeros(self.n_features_)
        importances[feature_idx] = importance

        # 递归计算子节点的重要性
        if node.left is not None:
            left_importances = self._calculate_tree_importance(node.left, X_train[left_mask], depth + 1)
            importances += left_importances

        if node.right is not None:
            right_importances = self._calculate_tree_importance(node.right, X_train[~left_mask], depth + 1)
            importances += right_importances

        return importances


def evaluate_model_with_diagnostics(y_true, y_pred, y_train, y_train_pred,
                                    model=None, X_train=None):
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"准确率差异: {abs(train_accuracy - test_accuracy):.4f}")

    # 过拟合判断
    if train_accuracy > test_accuracy + 0.05:
        print("过拟合")
        overfitting_risk = "高"
    elif train_accuracy > test_accuracy + 0.02:
        print("轻微过拟合")
        overfitting_risk = "中"
    else:
        print("过拟合风险低")
        overfitting_risk = "低"

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'overfitting_risk': overfitting_risk
    }


def main():
    data_path = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"
    df = pd.read_csv(data_path)
    print(f"目标变量分布:")
    print(df['NObeyesdad'].value_counts())

    preprocessor = ObesityDataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    print(f"目标变量分布:")
    print(df_processed['NObeyesdad'].value_counts().sort_index())

    df_features = create_additional_features(df_processed, use_bmi=True, bmi_as_feature=False)

    X = df_features.drop('NObeyesdad', axis=1).values
    y = df_features['NObeyesdad'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {X_train.shape[0]} 个样本, {X_train.shape[1]} 个特征")
    print(f"  测试集: {X_test.shape[0]} 个样本, {X_test.shape[1]} 个特征")

    rf_model = RandomForestClassifierCustom(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    print(f"树的数量: {rf_model.n_estimators}")

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    print(f"  训练集预测样本数: {len(y_train_pred)}")
    print(f"  测试集预测样本数: {len(y_test_pred)}")

    metrics = evaluate_model_with_diagnostics(
        y_test, y_test_pred,
        y_train, y_train_pred
    )

    class_names = ['偏瘦', '正常', '超重I级', '超重II级', '肥胖I级', '肥胖II级', '肥胖III级']
    print(classification_report(y_test, y_test_pred,
                                target_names=class_names,
                                zero_division=0))

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
    cm_path = './result/混淆矩阵11.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()

    feature_names = df_features.drop('NObeyesdad', axis=1).columns.tolist()
    importances = rf_model.feature_importances_(X_train)
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=False)

    print("特征重要性排名 (Top 10):")
    print(feature_importance.head(10).to_string(index=False))

    plt.figure(figsize=(14, 10))
    top_features = feature_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))

    bars = plt.barh(range(len(top_features)), top_features['重要性'], color=colors)
    plt.yticks(range(len(top_features)), top_features['特征'], fontsize=12)
    plt.xlabel('特征重要性', fontsize=14)
    plt.title('随机森林特征重要性排名 (Top 15)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['重要性'])):
        plt.text(importance + 0.001, i, f'{importance:.4f}',
                 va='center', fontsize=10, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = './result/特征重要性11.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 特征重要性分析
    print(f"\n特征重要性分析:")
    bmi_weight = feature_importance[feature_importance['特征'].str.contains('BMI')]['重要性'].sum()
    weight_weight = feature_importance[feature_importance['特征'].str.contains('Weight')]['重要性'].sum()
    print(f"BMI相关特征总权重: {bmi_weight:.4f}")
    print(f"体重相关特征总权重: {weight_weight:.4f}")

    if bmi_weight > 0.5:
        print("BMI特征权重过高")
    else:
        print("BMI特征权重合理")

    model_info = {
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth,
        'min_samples_split': rf_model.min_samples_split,
        'max_features': rf_model.max_features,
        'n_classes': rf_model.n_classes_,
        'n_features': rf_model.n_features_,
        'feature_importances': importances
    }

    model_path = './result/随机森林模型参数11.pkl'
    joblib.dump(model_info, model_path)

    preprocessor_path = './result/数据预处理器11.pkl'
    joblib.dump(preprocessor, preprocessor_path)

    results_df = pd.DataFrame({
        '真实标签': y_test,
        '预测标签': y_test_pred,
        '是否正确': y_test == y_test_pred
    })
    results_path = './result/预测结果11.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')

    fi_csv_path = './result/特征重要性11.csv'
    feature_importance.to_csv(fi_csv_path, index=False, encoding='utf-8-sig')

    if feature_importance is not None:
        print(f"\n最重要的3个特征:")
        for i, row in feature_importance.head(3).iterrows():
            print(f"{row['特征']}: {row['重要性']:.4f}")


if __name__ == "__main__":
    main()
