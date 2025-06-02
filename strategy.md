# CATE推定実験リポジトリ概要

## プロジェクト概要

このリポジトリは、**EconML**で生成される人工データに対して**CausalML**の**Causal Tree**を用いてCATE（Conditional Average Treatment Effect：条件付き平均処置効果）を推定し、その推定精度を評価することを目的としています。

## 主な機能

- EconMLライブラリを参考にした人工データの生成
- CausalMLのCausalTreeRegressorを用いたCATE推定
- 推定精度の評価（RMSE、バイアスなど）
- 決定木の可視化機能

## ディレクトリ構成

```
class_imbalanced_cate_targeting/
├── main.py                     # メイン実行スクリプト
├── requirements.txt            # 依存パッケージ一覧
├── strategy.md                 # このファイル（プロジェクト概要）
├── data/                       # データ生成関連
│   └── data_generator.py       # 人工データ生成モジュール
├── evaluation/                 # 評価関連
│   └── evaluation.py           # CATE推定精度評価モジュール
└── shared/                     # 共通モジュール
    ├── model.py                # CausalTreeモデルの学習・推定
    └── visualization_utils.py  # 可視化ユーティリティ
```

## ファイル説明

### `main.py`
- プロジェクトのメインエントリーポイント
- データ生成→モデル学習→CATE推定→評価の一連の流れを実行
- `run_experiment()`関数で実験の全工程を管理
- **複数の報酬タイプでの実験実行**: linear, logistic, treeの3つの報酬関数での実験を自動実行

### `data/data_generator.py`
- 人工データの生成を担当
- 5つの特徴量を持つデータを生成
- **複数の報酬関数タイプに対応**:
  - `linear`: 線形なCATE（`0.5 + X[:, 0] + 0.8 * X[:, 1]`）
  - `logistic`: ロジスティック関数によるCATE（非線形パターン）
  - `tree`: 決定木による複雑なCATE（特徴量の相互作用を含む）
- 処置群（T=1）と対照群（T=0）をランダムに割り当て
- **既存DataFrameからの報酬生成機能**: 既存の共変量・処置データに対して報酬を生成するシミュレータとしても利用可能

### `shared/model.py`
- **統一的なCATEエスティメータインターフェース**: 抽象基底クラス`CATEEstimator`で全モデルを統一
- **多様なメタラーナー実装**:
  - **CausalTreeWrapper**: CausalMLのCausalTreeRegressorラッパー（既存との互換性）
  - **SLearner**: 単一学習器アプローチ（XGBoost使用）
  - **TLearner**: 二学習器アプローチ（処置群・対照群で別々に学習）
  - **DRLearner**: 二重ロバスト学習器（T-Learnerと同等実装）
  - **RLearner**: Rロビンソン学習器（残差ベースアプローチ）
  - **CausalForestWrapper**: EconMLのCausal Forest DML（高度なdoubly robust手法）
- **ファクトリーパターン**: `create_model()`関数で簡単にモデル生成
- **後方互換性**: 既存の`train_causal_tree()`、`estimate_cate()`関数も維持

### `evaluation/evaluation.py`
- CATE推定精度の評価指標を計算
- MSE（平均二乗誤差）、RMSE（平方根平均二乗誤差）、バイアスを算出

### `shared/visualization_utils.py`
- CausalMLのCausalTreeをsklearnのexport_graphvizで可視化するためのユーティリティ
- モッククラスを用いてCausalMLとsklearnの互換性を提供

## 使用方法

### 1. 環境構築
```bash
pip install -r requirements.txt
```

### 2. 実験実行
```bash
python main.py
```

### 3. 多様なモデルでの実験
新しい統一インターフェースを使用した実験：
```python
from shared.model import create_model
from data.data_generator import generate_synthetic_data
import numpy as np

# データ生成
X, T, Y, true_cate, _ = generate_synthetic_data(n_samples=1000, reward_type="linear")

# 各種モデルでの実験
models = {
    "causal_tree": create_model("causal_tree"),
    "s_learner": create_model("s_learner"), 
    "t_learner": create_model("t_learner"),
    "dr_learner": create_model("dr_learner"),
    "r_learner": create_model("r_learner"),
    "causal_forest": create_model("causal_forest", n_estimators=100)
}

# 学習・予測
for name, model in models.items():
    if name == "causal_forest":
        model.fit(X, T.astype(float), Y)  # Causal Forestは処置をfloatで要求
    else:
        model.fit(X, T, Y)
    
    cate_pred = model.predict(X)
    rmse = np.sqrt(np.mean((true_cate - cate_pred)**2))
    print(f"{name}: RMSE = {rmse:.4f}")
```

### 4. 包括的な実験実行
```python
from main_complete import run_complete_experiment

# 全モデル・全報酬タイプでの比較実験
results = run_complete_experiment(n_samples=1000)
```

### 4. 既存データでの報酬生成
```python
from data.data_generator import generate_synthetic_data
import pandas as pd

# 既存のDataFrameを準備
df = pd.DataFrame({
    'X0': [0.1, -0.5, 1.2],
    'X1': [0.8, -0.2, 0.3], 
    'X2': [0.0, 1.1, -0.7],
    'X3': [0.5, 0.2, -0.1],
    'X4': [1.0, -0.8, 0.6],
    'T': [1, 0, 1]
})

# 報酬を生成
X, T, Y, true_cate, features = generate_synthetic_data(
    df=df, reward_type="logistic"
)
```

### 5. 実行結果
実験を実行すると以下の情報が出力されます：
- 生成されたデータのサンプル数と特徴量数
- 訓練・テストデータの分割情報
- CATE推定のRMSEとバイアス
- モデル学習と推定の進行状況

## 実験パラメータ

現在の設定（`main.py`の`run_experiment()`関数）：
- サンプル数: 5,000
- テストデータ割合: 30%
- 報酬タイプ: linear, logistic, tree（3つすべてを実行）
- CausalTreeのパラメータ:
  - `min_samples_leaf`: 20
  - `max_depth`: 4
  - `random_state`: 123

## 依存ライブラリ

- **econml**: EconMLライブラリ（Causal Forest DML実装、バージョン0.15.1）
- **causalml**: CausalTreeRegressorの実装
- **scikit-learn**: データ分割・評価指標・決定木による複雑なCATE生成
- **xgboost**: XGBoostライブラリ（メタラーナーの基底学習器）
- **pandas**: データ操作
- **numpy**: 数値計算（バージョン1.24.3、EconMLとの互換性確保）
- **graphviz**: 決定木の可視化
- **typing**: 型ヒント（Optional等）

## モデル性能比較

### Linear Reward Typeでの性能（RMSE）
1. **T-Learner / DR-Learner**: 0.3120（最良）
2. **S-Learner**: 0.3357
3. **Causal Forest**: 0.3686
4. **Causal Tree**: 0.7089
5. **R-Learner**: 4.1099（不安定）

### Logistic Reward Typeでの性能（RMSE）
1. **S-Learner**: 0.2220（最良）
2. **T-Learner / DR-Learner**: 0.2247
3. **Causal Forest**: 0.2271
4. **Causal Tree**: 0.3117
5. **R-Learner**: 4.8194（不安定）

### Tree Reward Typeでの性能（RMSE）
1. **T-Learner / DR-Learner**: 0.1447（最良）
2. **Causal Forest**: 0.1789
3. **S-Learner**: 0.2092
4. **Causal Tree**: 0.2438
5. **R-Learner**: 1.6522（不安定）

## 出力

実験実行後、以下のファイルが生成される場合があります：
- `causalml_causal_tree_example.pdf/.png`: 学習されたCausal Treeの可視化

## 拡張可能性

- より複雑な真のCATE関数の定義
- 異なるモデル（Meta-learners等）との比較
- クラス不均衡データでの性能評価
- より高度な評価指標（PEHE等）の追加