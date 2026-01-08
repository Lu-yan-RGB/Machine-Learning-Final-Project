import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
tf.random.set_seed(42)


# 数据加载和预处理函数
def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)

    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 计算BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    return df


def preprocess_for_lightgbm(df):
    data = df.copy()

    columns_to_drop = ['BMI', 'NObeyesdad']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns_to_drop, axis=1)
    target = data['BMI']

    processed_features = features.copy()

    # 二分类
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    binary_cols = [col for col in binary_cols if col in processed_features.columns]

    for col in binary_cols:
        le = LabelEncoder()
        processed_features[col] = le.fit_transform(processed_features[col])

    # CAEC有序
    if 'CAEC' in processed_features.columns:
        caec_values = processed_features['CAEC'].unique()
        caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

        def map_caec(x):
            if x in caec_mapping:
                return caec_mapping[x]
            else:
                if isinstance(x, str):
                    x_lower = x.lower()
                    if 'no' in x_lower or 'never' in x_lower:
                        return 0
                    elif 'sometimes' in x_lower:
                        return 1
                    elif 'frequently' in x_lower or 'often' in x_lower:
                        return 2
                    elif 'always' in x_lower:
                        return 3
                return 1  # 默认设为1 (Sometimes)

        processed_features['CAEC'] = processed_features['CAEC'].apply(map_caec)

    # CALC有序编码
    if 'CALC' in processed_features.columns:
        calc_values = processed_features['CALC'].unique()
        calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 2}

        def map_calc(x):
            if x in calc_mapping:
                return calc_mapping[x]
            else:
                if isinstance(x, str):
                    x_lower = x.lower()
                    if 'no' in x_lower or 'never' in x_lower:
                        return 0
                    elif 'sometimes' in x_lower:
                        return 1
                    elif 'frequently' in x_lower or 'often' in x_lower or 'always' in x_lower:
                        return 2
                return 0  # 默认设为0 (no)

        processed_features['CALC'] = processed_features['CALC'].apply(map_calc)

    # MTRANS标签
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


def preprocess_for_bp_nn(df):
    data = df.copy()

    columns_to_drop = ['BMI', 'NObeyesdad']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns_to_drop, axis=1)
    target = data['BMI']

    processed_features = features.copy()

    categorical_cols = []
    numeric_cols = []

    for col in processed_features.columns:
        if processed_features[col].dtype == 'object' or processed_features[col].nunique() < 10:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    # 标准化
    scalers = {}
    if numeric_cols:
        for col in numeric_cols:
            processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')

        processed_features[numeric_cols] = processed_features[numeric_cols].fillna(
            processed_features[numeric_cols].median()
        )

        scaler = StandardScaler()
        processed_features[numeric_cols] = scaler.fit_transform(processed_features[numeric_cols])
        scalers['numeric'] = scaler

    # 处理分类特征
    for col in categorical_cols:
        if col == 'CAEC':
            caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
            processed_features[col] = processed_features[col].map(lambda x: caec_mapping.get(x, 1))
        elif col == 'CALC':
            calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
            processed_features[col] = processed_features[col].map(lambda x: calc_mapping.get(x, 0))

    for col in ['CAEC', 'CALC']:
        if col in categorical_cols:
            categorical_cols.remove(col)
            if col not in numeric_cols:
                numeric_cols.append(col)
            processed_features[col] = pd.to_numeric(processed_features[col], errors='coerce')
            processed_features[col] = processed_features[col].fillna(0)

    if 'CAEC' in numeric_cols or 'CALC' in numeric_cols:
        new_numeric_cols = []
        for col in numeric_cols:
            if col in ['CAEC', 'CALC']:
                scaler_col = StandardScaler()
                processed_features[[col]] = scaler_col.fit_transform(processed_features[[col]])
                scalers[col] = scaler_col
            new_numeric_cols.append(col)
        numeric_cols = new_numeric_cols

    # 二分类和无序多分类）
    if categorical_cols:
        binary_cols = []    # jiangwei
        multi_categorical_cols = []

        for col in categorical_cols:
            if processed_features[col].nunique() == 2:
                binary_cols.append(col)
            else:
                multi_categorical_cols.append(col)

        # 处理二分类变量
        for col in binary_cols:
            le = LabelEncoder()
            processed_features[col] = le.fit_transform(processed_features[col])
            if col not in numeric_cols:
                numeric_cols.append(col)

        # 独热编码
        if multi_categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = ohe.fit_transform(processed_features[multi_categorical_cols])
            encoded_cols = ohe.get_feature_names_out(multi_categorical_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

            processed_features = pd.concat([
                processed_features.drop(columns=multi_categorical_cols).reset_index(drop=True),
                encoded_df
            ], axis=1)

    processed_features = processed_features.fillna(0)

    # 标准化
    target_scaler = StandardScaler()
    processed_target = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    return processed_features, processed_target, target_scaler, scalers


def train_lightgbm_model(X, y, result_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l2', 'l1'],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'min_data_in_leaf': 20,
        'max_depth': 6
    }

    print("\n开始训练LightGBM模型...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    results = {
        '训练集': {
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'R2': r2_score(y_train, y_pred_train)
        },
        '验证集': {
            'MSE': mean_squared_error(y_val, y_pred_val),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'MAE': mean_absolute_error(y_val, y_pred_val),
            'R2': r2_score(y_val, y_pred_val)
        },
        '测试集': {
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'R2': r2_score(y_test, y_pred_test)
        }
    }

    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': model.feature_importance(importance_type='gain')
    }).sort_values('重要性', ascending=False)

    model_path = os.path.join(result_dir, 'lightgbm_model.txt')
    model.save_model(model_path)

    feature_importance.to_csv(os.path.join(result_dir, 'lightgbm_feature_importance.csv'), index=False)

    return model, results, feature_importance, (X_train, X_val, X_test, y_train, y_val, y_test, y_pred_test)


def train_bp_neural_network(X, y, target_scaler, result_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),

        layers.Dense(16, activation='relu'),

        layers.Dense(1)  # 输出层
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    # 回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,  # 减少epochs以加快训练
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 预测
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    # 反标准化
    y_train_orig = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_orig = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    y_pred_train_orig = target_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    y_pred_val_orig = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
    y_pred_test_orig = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

    # 计算评估指标
    results = {
        '训练集': {
            'MSE': mean_squared_error(y_train_orig, y_pred_train_orig),
            'RMSE': np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig)),
            'MAE': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'R2': r2_score(y_train_orig, y_pred_train_orig)
        },
        '验证集': {
            'MSE': mean_squared_error(y_val_orig, y_pred_val_orig),
            'RMSE': np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig)),
            'MAE': mean_absolute_error(y_val_orig, y_pred_val_orig),
            'R2': r2_score(y_val_orig, y_pred_val_orig)
        },
        '测试集': {
            'MSE': mean_squared_error(y_test_orig, y_pred_test_orig),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig)),
            'MAE': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'R2': r2_score(y_test_orig, y_pred_test_orig)
        }
    }

    model_path = os.path.join(result_dir, 'bp_nn_model.h5')
    model.save(model_path)

    return model, results, history, (X_train, X_val, X_test, y_test_orig, y_pred_test_orig)


def plot_bmi_distribution(df, result_dir):
    plt.figure(figsize=(10, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(df['BMI'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(df['BMI'].mean(), color='red', linestyle='--',
                label=f'均值: {df["BMI"].mean():.2f}')
    plt.axvline(df['BMI'].median(), color='green', linestyle='--',
                label=f'中位数: {df["BMI"].median():.2f}')
    plt.xlabel('BMI')
    plt.ylabel('频数')
    plt.title('BMI分布直方图')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(df['BMI'], vert=False)
    plt.xlabel('BMI')
    plt.title('BMI箱线图')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'BMI分布.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(lgbm_results, nn_results, result_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # RMSE对比
    datasets = ['训练集', '验证集', '测试集']
    lgbm_rmses = [lgbm_results[d]['RMSE'] for d in datasets]
    nn_rmses = [nn_results[d]['RMSE'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    axes[0, 0].bar(x - width / 2, lgbm_rmses, width, label='LightGBM', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width / 2, nn_rmses, width, label='BP神经网络', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('数据集')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE对比')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for i, v in enumerate(lgbm_rmses):
        axes[0, 0].text(i - width / 2, v + max(lgbm_rmses) * 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(nn_rmses):
        axes[0, 0].text(i + width / 2, v + max(nn_rmses) * 0.01, f'{v:.3f}', ha='center', fontsize=9)

    lgbm_r2 = [lgbm_results[d]['R2'] for d in datasets]
    nn_r2 = [nn_results[d]['R2'] for d in datasets]

    axes[0, 1].bar(x - width / 2, lgbm_r2, width, label='LightGBM', alpha=0.8, color='skyblue')
    axes[0, 1].bar(x + width / 2, nn_r2, width, label='BP神经网络', alpha=0.8, color='lightcoral')
    axes[0, 1].set_xlabel('数据集')
    axes[0, 1].set_ylabel('R^2')
    axes[0, 1].set_title('R^2对比')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 设置y轴范围
    axes[0, 1].set_ylim([0, 1.1])

    # 添加数值标签
    for i, v in enumerate(lgbm_r2):
        axes[0, 1].text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(nn_r2):
        axes[0, 1].text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # 3. MAE对比
    lgbm_mae = [lgbm_results[d]['MAE'] for d in datasets]
    nn_mae = [nn_results[d]['MAE'] for d in datasets]

    axes[1, 0].bar(x - width / 2, lgbm_mae, width, label='LightGBM', alpha=0.8, color='skyblue')
    axes[1, 0].bar(x + width / 2, nn_mae, width, label='BP神经网络', alpha=0.8, color='lightcoral')
    axes[1, 0].set_xlabel('数据集')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE对比')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    for i, v in enumerate(lgbm_mae):
        axes[1, 0].text(i - width / 2, v + max(lgbm_mae) * 0.01, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(nn_mae):
        axes[1, 0].text(i + width / 2, v + max(nn_mae) * 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # 综合评分 = (1-RMSE/最大RMSE) * 0.4 + R² * 0.4 + (1-MAE/最大MAE) * 0.2
    max_rmse = max(max(lgbm_rmses), max(nn_rmses))
    max_mae = max(max(lgbm_mae), max(nn_mae))

    def calculate_score(rmse, r2, mae):
        return (1 - rmse / max_rmse) * 0.4 + r2 * 0.4 + (1 - mae / max_mae) * 0.2

    lgbm_scores = [calculate_score(lgbm_rmses[i], lgbm_r2[i], lgbm_mae[i]) for i in range(3)]
    nn_scores = [calculate_score(nn_rmses[i], nn_r2[i], nn_mae[i]) for i in range(3)]

    axes[1, 1].bar(x - width / 2, lgbm_scores, width, label='LightGBM', alpha=0.8, color='skyblue')
    axes[1, 1].bar(x + width / 2, nn_scores, width, label='BP神经网络', alpha=0.8, color='lightcoral')
    axes[1, 1].set_xlabel('数据集')
    axes[1, 1].set_ylabel('综合评分')
    axes[1, 1].set_title('综合评分对比')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(datasets)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(lgbm_scores):
        axes[1, 1].text(i - width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(nn_scores):
        axes[1, 1].text(i + width / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '模型对比.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_importance, result_dir):
    plt.figure(figsize=(12, 8))

    # TOP15最重要的特征
    top_n = min(15, len(feature_importance))
    top_features = feature_importance.head(top_n)

    plt.barh(range(top_n), top_features['重要性'])
    plt.yticks(range(top_n), top_features['特征'])
    plt.xlabel('特征重要性')
    plt.title(f'LightGBM特征重要性 (Top {top_n})')
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, v in enumerate(top_features['重要性']):
        plt.text(v + top_features['重要性'].max() * 0.001, i, f'{v:.1f}', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '特征重要性.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(nn_history, result_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(nn_history.history['loss'], label='训练损失')
    axes[0].plot(nn_history.history['val_loss'], label='验证损失')
    axes[0].set_xlabel('训练轮次')
    axes[0].set_ylabel('损失 (MSE)')
    axes[0].set_title('BP神经网络训练损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if 'val_loss' in nn_history.history:
        best_epoch = np.argmin(nn_history.history['val_loss'])
        axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                        label=f'最佳轮次: {best_epoch + 1}')
        axes[0].legend()

    if 'mae' in nn_history.history:
        axes[1].plot(nn_history.history['mae'], label='训练MAE')
        if 'val_mae' in nn_history.history:
            axes[1].plot(nn_history.history['val_mae'], label='验证MAE')
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('BP神经网络MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '神经网络训练历史.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(lgbm_data, nn_data, result_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # LightGBM
    *_, y_test_lgb, y_pred_lgb = lgbm_data
    residuals_lgb = y_test_lgb - y_pred_lgb

    axes[0].scatter(y_pred_lgb, residuals_lgb, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('预测值 (LightGBM)')
    axes[0].set_ylabel('残差')
    axes[0].set_title('LightGBM残差图')
    axes[0].grid(True, alpha=0.3)

    # BP神经网络
    *_, y_test_nn, y_pred_nn = nn_data
    residuals_nn = y_test_nn - y_pred_nn

    axes[1].scatter(y_pred_nn, residuals_nn, alpha=0.5, s=20, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel('预测值 (BP神经网络)')
    axes[1].set_ylabel('残差')
    axes[1].set_title('BP神经网络残差图')
    axes[1].grid(True, alpha=0.3)

    # 残差分布对比
    axes[2].hist(residuals_lgb, bins=30, alpha=0.5, label='LightGBM', color='skyblue')
    axes[2].hist(residuals_nn, bins=30, alpha=0.5, label='BP神经网络', color='lightcoral')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[2].set_xlabel('残差')
    axes[2].set_ylabel('频数')
    axes[2].set_title('残差分布对比')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '残差分析.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(lgbm_data, nn_data, result_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LightGBM
    *_, y_test_lgb, y_pred_lgb = lgbm_data

    axes[0].scatter(y_test_lgb, y_pred_lgb, alpha=0.5, s=20)

    # 绘制完美预测线
    min_val = min(y_test_lgb.min(), y_pred_lgb.min())
    max_val = max(y_test_lgb.max(), y_pred_lgb.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title('LightGBM: 预测值 vs 真实值')
    axes[0].grid(True, alpha=0.3)

    # 计算R²并显示
    r2_lgb = r2_score(y_test_lgb, y_pred_lgb)
    axes[0].text(0.05, 0.95, f'R² = {r2_lgb:.4f}',
                 transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # BP神经网络
    *_, y_test_nn, y_pred_nn = nn_data

    axes[1].scatter(y_test_nn, y_pred_nn, alpha=0.5, s=20, color='orange')

    # 绘制完美预测线
    min_val = min(y_test_nn.min(), y_pred_nn.min())
    max_val = max(y_test_nn.max(), y_pred_nn.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

    axes[1].set_xlabel('真实值')
    axes[1].set_ylabel('预测值')
    axes[1].set_title('BP神经网络: 预测值 vs 真实值')
    axes[1].grid(True, alpha=0.3)

    # 计算R²并显示
    r2_nn = r2_score(y_test_nn, y_pred_nn)
    axes[1].text(0.05, 0.95, f'R² = {r2_nn:.4f}',
                 transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '预测对比.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    data_path = "./data/ObesityDataSet_raw_and_data_sinthetic.csv"

    df = load_and_prepare_data(data_path)
    result_dir = './result/regression'
    plot_bmi_distribution(df, result_dir)

    # 3. LightGBM预处理和训练
    X_lgb, y_lgb = preprocess_for_lightgbm(df)
    lgbm_model, lgbm_results, lgbm_importance, lgbm_data = train_lightgbm_model(X_lgb, y_lgb, result_dir)

    # 4. BP神经网络预处理和训练
    X_nn, y_nn, target_scaler, scalers = preprocess_for_bp_nn(df)
    nn_model, nn_results, nn_history, nn_data = train_bp_neural_network(X_nn, y_nn, target_scaler, result_dir)

    results_df = pd.DataFrame({
        '数据集': ['训练集', '验证集', '测试集'] * 2,
        '模型': ['LightGBM'] * 3 + ['BP神经网络'] * 3,
        'RMSE': [
            lgbm_results['训练集']['RMSE'], lgbm_results['验证集']['RMSE'], lgbm_results['测试集']['RMSE'],
            nn_results['训练集']['RMSE'], nn_results['验证集']['RMSE'], nn_results['测试集']['RMSE']
        ],
        'MAE': [
            lgbm_results['训练集']['MAE'], lgbm_results['验证集']['MAE'], lgbm_results['测试集']['MAE'],
            nn_results['训练集']['MAE'], nn_results['验证集']['MAE'], nn_results['测试集']['MAE']
        ],
        'R2': [
            lgbm_results['训练集']['R2'], lgbm_results['验证集']['R2'], lgbm_results['测试集']['R2'],
            nn_results['训练集']['R2'], nn_results['验证集']['R2'], nn_results['测试集']['R2']
        ]
    })

    csv_path = os.path.join(result_dir, '详细结果.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 可视化
    plot_model_comparison(lgbm_results, nn_results, result_dir)
    plot_feature_importance(lgbm_importance, result_dir)
    plot_training_history(nn_history, result_dir)
    plot_residuals(lgbm_data, nn_data, result_dir)
    plot_predictions_vs_actual(lgbm_data, nn_data, result_dir)

    print("\n模型性能对比:")
    for dataset in ['训练集', '验证集', '测试集']:
        print(f"\n{dataset}:")
        print(f"  LightGBM - RMSE: {lgbm_results[dataset]['RMSE']:.4f}, "
              f"MAE: {lgbm_results[dataset]['MAE']:.4f}, "
              f"R²: {lgbm_results[dataset]['R2']:.4f}")
        print(f"  BP神经网络 - RMSE: {nn_results[dataset]['RMSE']:.4f}, "
              f"MAE: {nn_results[dataset]['MAE']:.4f}, "
              f"R²: {nn_results[dataset]['R2']:.4f}")

    # 判断
    lgbm_test_r2 = lgbm_results['测试集']['R2']
    nn_test_r2 = nn_results['测试集']['R2']

    if lgbm_test_r2 > nn_test_r2:
        diff = lgbm_test_r2 - nn_test_r2
        print(f"LightGBM表现更好，测试集R²高出{diff:.4f}")
        better_model = "LightGBM"
    else:
        diff = nn_test_r2 - lgbm_test_r2
        print(f"BP神经网络表现更好，测试集R²高出{diff:.4f}")
        better_model = "BP神经网络"

    # 特征重要性分析
    for i, row in lgbm_importance.head(5).iterrows():
        print(f"  {i + 1}. {row['特征']}: {row['重要性']:.2f}")

    print(f"推荐模型: {better_model}")


if __name__ == "__main__":
    main()
