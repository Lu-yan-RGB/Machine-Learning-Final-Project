import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
from typing import List, Dict, Tuple, Optional, Any
import math

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> Tuple:
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.class_to_index.get(val, 0) for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


class TreeNode:
    def __init__(self, depth=0, max_depth=None, min_samples_split=2, min_gain=1e-7,
                 lambda_l2=0.1, gamma=0.0):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.lambda_l2 = lambda_l2
        self.gamma = gamma

        self.feature_idx = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.gain = 0.0

    def fit(self, X, y, gradients, hessians, parent_value=None):
        n_samples = X.shape[0]

        # 计算当前节点的值
        G = np.sum(gradients)
        H = np.sum(hessians) + self.lambda_l2
        self.value = -G / H

        # 保存父节点的值用于初始化
        if parent_value is None:
            parent_value = self.value

        # 检查停止条件
        if (self.max_depth is not None and self.depth >= self.max_depth) or \
                n_samples < self.min_samples_split:
            self.is_leaf = True
            return

        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # 限制分裂点数量
            if len(unique_values) > 100:
                percentiles = np.linspace(10, 90, 20)
                thresholds = np.percentile(feature_values, percentiles)
            else:
                thresholds = unique_values

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue

                # 计算增益
                G_left = np.sum(gradients[left_mask])
                H_left = np.sum(hessians[left_mask]) + self.lambda_l2
                G_right = np.sum(gradients[right_mask])
                H_right = np.sum(hessians[right_mask]) + self.lambda_l2

                gain = (G_left ** 2 / H_left + G_right ** 2 / H_right - G ** 2 / H) / 2

                # 应用gamma正则化
                gain = gain - self.gamma

                if gain > best_gain and gain > self.min_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        if best_gain < self.min_gain or best_feature is None:
            self.is_leaf = True
            return

        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.gain = best_gain

        # 创建子节点
        self.left = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_gain=self.min_gain,
            lambda_l2=self.lambda_l2,
            gamma=self.gamma
        )
        self.right = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_gain=self.min_gain,
            lambda_l2=self.lambda_l2,
            gamma=self.gamma
        )

        self.left.fit(X[best_left_mask], y[best_left_mask],
                      gradients[best_left_mask], hessians[best_left_mask],
                      self.value)
        self.right.fit(X[best_right_mask], y[best_right_mask],
                       gradients[best_right_mask], hessians[best_right_mask],
                       self.value)

    def predict_single(self, x):
        if self.is_leaf:
            return self.value

        if x[self.feature_idx] <= self.threshold:
            return self.left.predict_single(x)
        else:
            return self.right.predict_single(x)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


class RegressionTree:
    def __init__(self, max_depth=6, min_samples_split=10, min_gain=1e-7,
                 lambda_l2=0.1, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.lambda_l2 = lambda_l2
        self.gamma = gamma
        self.root = None
        self.feature_importances = None

    def fit(self, X, y, gradients, hessians):
        n_samples, n_features = X.shape

        self.root = TreeNode(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_gain=self.min_gain,
            lambda_l2=self.lambda_l2,
            gamma=self.gamma
        )

        self.root.fit(X, y, gradients, hessians)

        # 计算特征重要性
        self.feature_importances = np.zeros(n_features)
        self._compute_feature_importance(self.root)

    def _compute_feature_importance(self, node):
        if node.is_leaf or node.feature_idx is None:
            return

        self.feature_importances[node.feature_idx] += node.gain

        self._compute_feature_importance(node.left)
        self._compute_feature_importance(node.right)

    def predict(self, X):
        if self.root is None:
            raise ValueError("模型未训练")
        return self.root.predict(X)


class ImprovedLightGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_samples_split=10, subsample=0.8, colsample=0.8,
                 lambda_l2=0.1, gamma=0.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.colsample = colsample
        self.lambda_l2 = lambda_l2
        self.gamma = gamma
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.base_prediction = None
        self.best_iteration = 0
        self.best_score = float('inf')

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=20, verbose=True):
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # 初始化预测值为目标值的均值
        self.base_prediction = np.mean(y)
        predictions = np.full_like(y, self.base_prediction, dtype=float)

        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)

        best_val_score = float('inf')
        no_improve_count = 0

        for i in range(self.n_estimators):
            residuals = y - predictions

            if self.subsample < 1.0:
                sample_size = max(10, int(n_samples * self.subsample))
                sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            else:
                sample_idx = np.arange(n_samples)

            gradients = -residuals[sample_idx]  # 负梯度
            hessians = np.ones_like(gradients)  # 二阶导数为常数1

            if self.colsample < 1.0:
                n_sampled_features = max(1, int(n_features * self.colsample))
                feature_indices = np.random.choice(n_features, n_sampled_features, replace=False)
                X_sampled = X[sample_idx][:, feature_indices]
                X_full = X[:, feature_indices]
            else:
                feature_indices = np.arange(n_features)
                X_sampled = X[sample_idx]
                X_full = X

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                lambda_l2=self.lambda_l2,
                gamma=self.gamma
            )

            tree.fit(X_sampled, y[sample_idx], gradients, hessians)

            tree_pred = tree.predict(X_full)
            predictions += self.learning_rate * tree_pred

            if tree.feature_importances is not None:
                for j, idx in enumerate(feature_indices):
                    self.feature_importances_[idx] += tree.feature_importances[j]

            self.trees.append((tree, feature_indices))

            train_score = calculate_mse(y, predictions)

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_score = calculate_mse(y_val, val_pred)

                if val_score < best_val_score:
                    best_val_score = val_score
                    no_improve_count = 0
                    self.best_iteration = i
                    self.best_score = val_score
                else:
                    no_improve_count += 1

                if verbose and (i % 10 == 0 or i < 10):
                    print(f"第 {i + 1:3d} 轮 - 训练MSE: {train_score:.4f}, 验证MSE: {val_score:.4f}")

                if no_improve_count >= early_stopping_rounds:
                    if verbose:
                        print(f"提前停止于第 {i + 1} 轮，最佳轮次: {self.best_iteration + 1}")
                    self.trees = self.trees[:self.best_iteration + 1]
                    break
            else:
                if verbose and i % 10 == 0:
                    print(f"第 {i + 1:3d} 轮 - 训练MSE: {train_score:.4f}")

        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

        if verbose:
            print(f"训练完成，共 {len(self.trees)} 棵树")

    def predict(self, X):
        if self.base_prediction is None:
            raise ValueError("模型未训练")

        predictions = np.full(X.shape[0], self.base_prediction, dtype=float)

        for tree, feature_indices in self.trees:
            tree_pred = tree.predict(X[:, feature_indices])
            predictions += self.learning_rate * tree_pred

        return predictions

    def get_feature_importance(self, feature_names=None):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]

        return pd.DataFrame({
            '特征': feature_names,
            '重要性': self.feature_importances_
        }).sort_values('重要性', ascending=False)


def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)

    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 计算BMI（目标变量）
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # 添加合理的特征工程（不包含目标变量信息）
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    df['Weight_Age_Interaction'] = df['Weight'] * df['Age'] / 1000
    df['Height_Age_Interaction'] = df['Height'] * df['Age'] / 100

    print(f"特征工程后形状: {df.shape}")

    return df


def preprocess_for_lightgbm(df):
    data = df.copy()
    columns_to_drop = ['BMI', 'NObeyesdad']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns_to_drop, axis=1)
    target = data['BMI']

    processed_features = features.copy()

    # 二分类编码
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    binary_cols = [col for col in binary_cols if col in processed_features.columns]

    for col in binary_cols:
        le = LabelEncoder()
        processed_features[col] = le.fit_transform(processed_features[col])

    # CAEC有序编码
    if 'CAEC' in processed_features.columns:
        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        processed_features['CAEC'] = processed_features['CAEC'].map(
            lambda x: caec_mapping.get(x, 1) if isinstance(x, str) else 1
        )

    # CALC有序编码
    if 'CALC' in processed_features.columns:
        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 2}
        processed_features['CALC'] = processed_features['CALC'].map(
            lambda x: calc_mapping.get(x, 0) if isinstance(x, str) else 0
        )

    # MTRANS标签编码
    if 'MTRANS' in processed_features.columns:
        le_mtrans = LabelEncoder()
        processed_features['MTRANS'] = le_mtrans.fit_transform(processed_features['MTRANS'])

    likert_cols = ['FCVC', 'CH2O', 'FAF', 'TUE', 'NCP']
    likert_cols = [col for col in likert_cols if col in processed_features.columns]

    for col in likert_cols:
        processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')

    # Age确保为数值
    if 'Age' in processed_features.columns:
        processed_features['Age'] = pd.to_numeric(processed_features['Age'], errors='coerce')

    processed_features = processed_features.fillna(processed_features.median())

    for col in processed_features.columns:
        processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')
    processed_features = processed_features.fillna(0)

    return processed_features, target


def train_lightgbm_model(X, y, result_dir):
    # 转换数据为numpy数组
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, pd.Series) else y

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    # 创建并训练改进的模型
    print("\n开始训练改进的LightGBM模型...")

    model = ImprovedLightGBM(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=10,
        subsample=0.8,
        colsample=0.8,
        lambda_l2=0.1,
        gamma=0.0,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        early_stopping_rounds=20,
        verbose=True
    )

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # 计算评估指标
    results = {
        '训练集': {
            'MSE': calculate_mse(y_train, y_pred_train),
            'RMSE': calculate_rmse(y_train, y_pred_train),
            'MAE': calculate_mae(y_train, y_pred_train),
            'R2': calculate_r2(y_train, y_pred_train)
        },
        '验证集': {
            'MSE': calculate_mse(y_val, y_pred_val),
            'RMSE': calculate_rmse(y_val, y_pred_val),
            'MAE': calculate_mae(y_val, y_pred_val),
            'R2': calculate_r2(y_val, y_pred_val)
        },
        '测试集': {
            'MSE': calculate_mse(y_test, y_pred_test),
            'RMSE': calculate_rmse(y_test, y_pred_test),
            'MAE': calculate_mae(y_test, y_pred_test),
            'R2': calculate_r2(y_test, y_pred_test)
        }
    }

    # 特征重要性
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X_np.shape[1])]
    feature_importance = model.get_feature_importance(feature_names)

    model_path = os.path.join(result_dir, 'improved_lightgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # 保存特征重要性
    feature_importance.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')

    return model, results, feature_importance, (X_train, X_val, X_test, y_train, y_val, y_test, y_pred_test)


def save_plot(fig, filepath):
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"已保存: {filepath}")
    plt.close(fig)


def plot_bmi_distribution(df, result_dir):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(df['BMI'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(df['BMI'].mean(), color='red', linestyle='--', label=f'均值: {df["BMI"].mean():.2f}')
        ax1.axvline(df['BMI'].median(), color='green', linestyle='--', label=f'中位数: {df["BMI"].median():.2f}')
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('频数')
        ax1.set_title('BMI分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.boxplot(df['BMI'], vert=False)
        ax2.set_xlabel('BMI')
        ax2.set_title('BMI箱线图')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(fig, os.path.join(result_dir, 'BMI分布.png'))
    except Exception as e:
        print(f"绘制BMI分布失败: {e}")


def plot_feature_importance(feature_importance, result_dir):
    fig = plt.figure(figsize=(10, 6))

    top_n = min(15, len(feature_importance))
    top_features = feature_importance.head(top_n)

    # 创建水平条形图
    bars = plt.barh(range(top_n), top_features['重要性'])
    plt.yticks(range(top_n), top_features['特征'])
    plt.xlabel('特征重要性')
    plt.title(f'特征重要性 (Top {top_n})')
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, v in enumerate(top_features['重要性']):
        plt.text(v, i, f' {v:.4f}', va='center')

    plt.tight_layout()
    save_plot(fig, os.path.join(result_dir, '特征重要性.png'))


def plot_predictions_vs_actual(lgbm_data, result_dir):
    fig = plt.figure(figsize=(8, 6))

    *_, y_test, y_pred = lgbm_data

    # 绘制散点图
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, label='预测点')

    # 绘制完美预测线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')

    # 绘制对角线
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    plt.grid(True, alpha=0.3)

    # 计算并显示R²
    r2 = calculate_r2(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    save_plot(fig, os.path.join(result_dir, '预测对比.png'))

    # 打印预测值范围
    print(f"\n预测值范围: {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f"真实值范围: {y_test.min():.2f} - {y_test.max():.2f}")
    print(f"预测值均值: {y_pred.mean():.2f}, 真实值均值: {y_test.mean():.2f}")


def plot_residuals(lgbm_data, result_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    *_, y_test, y_pred = lgbm_data
    residuals = y_test - y_pred

    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('残差')
    ax1.set_title('残差图')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('残差')
    ax2.set_ylabel('频数')
    ax2.set_title('残差分布')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, os.path.join(result_dir, '残差分析.png'))

    print(f"\n残差统计:")
    print(f"  均值: {residuals.mean():.4f}")
    print(f"  标准差: {residuals.std():.4f}")
    print(f"  最小残差: {residuals.min():.4f}")
    print(f"  最大残差: {residuals.max():.4f}")


def plot_model_performance(results, result_dir):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    datasets = ['训练集', '验证集', '测试集']
    x = np.arange(len(datasets))

    # 1. MSE
    mses = [results[d]['MSE'] for d in datasets]
    axs[0, 0].bar(x, mses, alpha=0.8, color='skyblue', width=0.6)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(datasets)
    axs[0, 0].set_ylabel('MSE')
    axs[0, 0].set_title('MSE指标')
    axs[0, 0].grid(True, alpha=0.3)

    for i, v in enumerate(mses):
        axs[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # 2. R²
    r2s = [results[d]['R2'] for d in datasets]
    axs[0, 1].bar(x, r2s, alpha=0.8, color='lightgreen', width=0.6)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(datasets)
    axs[0, 1].set_ylabel('R²')
    axs[0, 1].set_title('R²指标')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim([min(0, min(r2s) - 0.1), 1.1])

    for i, v in enumerate(r2s):
        axs[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # 3. MAE
    maes = [results[d]['MAE'] for d in datasets]
    axs[1, 0].bar(x, maes, alpha=0.8, color='lightcoral', width=0.6)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(datasets)
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].set_title('MAE指标')
    axs[1, 0].grid(True, alpha=0.3)

    for i, v in enumerate(maes):
        axs[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # 4. RMSE
    rmses = [results[d]['RMSE'] for d in datasets]
    axs[1, 1].bar(x, rmses, alpha=0.8, color='gold', width=0.6)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(datasets)
    axs[1, 1].set_ylabel('RMSE')
    axs[1, 1].set_title('RMSE指标')
    axs[1, 1].grid(True, alpha=0.3)

    for i, v in enumerate(rmses):
        axs[1, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    save_plot(fig, os.path.join(result_dir, '模型性能.png'))


def plot_error_analysis(y_true, y_pred, result_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    errors = y_true - y_pred

    # 误差分布
    ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(errors.mean(), color='red', linestyle='--', label=f'均值: {errors.mean():.4f}')
    ax1.axvline(errors.mean() + errors.std(), color='orange', linestyle='--',
                label=f'±1σ: [{errors.mean() - errors.std():.4f}, {errors.mean() + errors.std():.4f}]')
    ax1.axvline(errors.mean() - errors.std(), color='orange', linestyle='--')
    ax1.set_xlabel('预测误差')
    ax1.set_ylabel('频数')
    ax1.set_title('误差分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 误差vs真实值
    ax2.scatter(y_true, errors, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('误差')
    ax2.set_title('误差vs真实值')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, os.path.join(result_dir, '误差分析.png'))


def main():
    data_path = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"
    result_dir = './result/regression11'
    os.makedirs(result_dir, exist_ok=True)

    df = load_and_prepare_data(data_path)
    plot_bmi_distribution(df, result_dir)

    X_lgb, y_lgb = preprocess_for_lightgbm(df)
    print(f"处理后的特征形状: {X_lgb.shape}")
    print(f"目标变量形状: {y_lgb.shape}")
    print(f"特征数量: {len(X_lgb.columns)}")
    print(f"特征名: {list(X_lgb.columns)}")

    # 训练改进的LightGBM模型
    print("\n4. 训练改进的LightGBM模型")
    lgbm_model, lgbm_results, lgbm_importance, lgbm_data = train_lightgbm_model(X_lgb, y_lgb, result_dir)

    results_df = pd.DataFrame({
        '数据集': ['训练集', '验证集', '测试集'],
        '样本数量': [lgbm_data[0].shape[0], lgbm_data[1].shape[0], lgbm_data[2].shape[0]],
        'MSE': [lgbm_results['训练集']['MSE'], lgbm_results['验证集']['MSE'], lgbm_results['测试集']['MSE']],
        'RMSE': [lgbm_results['训练集']['RMSE'], lgbm_results['验证集']['RMSE'], lgbm_results['测试集']['RMSE']],
        'MAE': [lgbm_results['训练集']['MAE'], lgbm_results['验证集']['MAE'], lgbm_results['测试集']['MAE']],
        'R2': [lgbm_results['训练集']['R2'], lgbm_results['验证集']['R2'], lgbm_results['测试集']['R2']]
    })

    csv_path = os.path.join(result_dir, '详细结果.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: {csv_path}")

    plot_feature_importance(lgbm_importance, result_dir)
    plot_predictions_vs_actual(lgbm_data, result_dir)
    plot_residuals(lgbm_data, result_dir)
    plot_model_performance(lgbm_results, result_dir)
    *_, y_test, y_pred = lgbm_data
    plot_error_analysis(y_test, y_pred, result_dir)

    for dataset in ['训练集', '验证集', '测试集']:
        print(f"\n{dataset}:")
        print(f"  MSE:  {lgbm_results[dataset]['MSE']:.6f}")
        print(f"  RMSE: {lgbm_results[dataset]['RMSE']:.6f}")
        print(f"  MAE:  {lgbm_results[dataset]['MAE']:.6f}")
        print(f"  R²:   {lgbm_results[dataset]['R2']:.6f}")

    # 特征重要性分析
    for i, row in lgbm_importance.head(10).iterrows():
        print(f"  {i + 1:2d}. {row['特征']:25s}: {row['重要性']:.4f}")


    # 预测效果
    *_, y_test, y_pred = lgbm_data
    print(f"预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"真实值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"预测值标准差: {y_pred.std():.4f}")
    print(f"真实值标准差: {y_test.std():.4f}")

    # 检查过拟合
    train_r2 = lgbm_results['训练集']['R2']
    val_r2 = lgbm_results['验证集']['R2']
    test_r2 = lgbm_results['测试集']['R2']

    if train_r2 > 0.9 and val_r2 > 0.8 and test_r2 > 0.8:
        print("没有明显过拟合")
    elif train_r2 - val_r2 > 0.2:
        print("存在过拟合")
    else:
        print("欠拟合")


if __name__ == "__main__":
    main()