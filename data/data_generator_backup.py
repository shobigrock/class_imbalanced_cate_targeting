# data_generator.py
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from econml.data.dgps import ihdp_surface_B # EconMLのデータ生成プロセスを利用する例


class DataGenerator(ABC):
    """
    CATE推定用データ生成の抽象基底クラス
    """
    
    def __init__(self, random_seed: int = 123):
        """
        Args:
            random_seed (int): 乱数シード
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    @abstractmethod
    def generate_data(self, **kwargs) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        データを生成する抽象メソッド
        
        Returns:
            X_df (pd.DataFrame): 特徴量データ
            T (np.ndarray): 処置データ (0 or 1)
            Y (np.ndarray): 観測結果
            true_cate (np.ndarray): 真のCATE（利用可能な場合）
            feature_names (List[str]): 特徴量名のリスト
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> dict:
        """
        データセットの情報を返す抽象メソッド
        
        Returns:
            dict: データセット情報（サンプル数、特徴量数、等）
        """
        pass


class SyntheticDataGenerator(DataGenerator):
    """
    合成データ生成クラス
    """
    
    def __init__(self, random_seed: int = 123):
        super().__init__(random_seed)
    
    def generate_data(
        self, 
        n_samples: int = 1000, 
        reward_type: str = "linear", 
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        合成データを生成
        
        Args:
            n_samples (int): 生成するサンプル数（dfが指定されていない場合に使用）
            reward_type (str): 報酬関数の種類 {"linear", "logistic", "tree"}
            df (Optional[pd.DataFrame]): 既存の共変量+介入変数のDataFrame
        
        Returns:
            X_df (pd.DataFrame): 特徴量データ
            T (np.ndarray): 処置データ
            Y (np.ndarray): 観測結果
            true_cate (np.ndarray): 真のCATE
            feature_names (List[str]): 特徴量名のリスト
        """
        return generate_synthetic_data(n_samples, self.random_seed, reward_type, df)
    
    def get_data_info(self) -> dict:
        """合成データの情報を返す"""
        return {
            "dataset_type": "synthetic",
            "available_reward_types": ["linear", "logistic", "tree"],
            "default_features": 5,
            "treatment_binary": True
        }


class CriteoDataGenerator(DataGenerator):
    """
    Criteo広告データセット用データ生成クラス
    """
    
    def __init__(self, data_path: str = None, random_seed: int = 123):
        """
        Args:
            data_path (str): Criteoデータファイルのパス
            random_seed (int): 乱数シード
        """
        super().__init__(random_seed)
        self.data_path = data_path or os.path.join("data", "criteo-research-uplift-v2.1.csv.gz")
        self.scaler = StandardScaler()
      def generate_data(
        self, 
        n_samples: Optional[int] = None,
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:        """
        CriteoデータからCATE推定用データを生成
        
        Args:
            n_samples (Optional[int]): 使用するサンプル数（Noneの場合全データ）
            treatment_col (str): 処置変数として使用する列名
            outcome_col (str): 結果変数として使用する列名
            feature_cols (Optional[List[str]]): 特徴量として使用する列名リスト
        
        Returns:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 観測結果
            treatment (np.ndarray): 処置データ
        """
        
        # Criteoデータの読み込み
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Criteo data file not found at {self.data_path}. "
                f"Please download the Criteo dataset and place it in the data/ folder."
            )
        
        print(f"Loading Criteo data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # サンプル数の制限
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=self.random_seed).reset_index(drop=True)
        
        # 特徴量の選択
        if feature_cols is None:
            # Criteoデータの特徴量列（f0-f11）を自動選択
            feature_cols = [f'f{i}' for i in range(12)]
            print(f"Using Criteo feature columns: {feature_cols}")
        
        # 利用可能な特徴量のみを選択
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) != len(feature_cols):
            print(f"Warning: Some feature columns not found. Using: {available_features}")
        feature_cols = available_features
        
        # 前処理
        X_df = df[feature_cols].copy()
        X_df = X_df.fillna(X_df.median())  # 欠損値処理
        X_df[feature_cols] = self.scaler.fit_transform(X_df[feature_cols])  # 標準化
        
        # 処置変数の準備
        if treatment_col not in df.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")
        T = df[treatment_col].astype(int).values
        
        # 結果変数の準備
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")
        Y = df[outcome_col].values
        
        # 真のCATEを推定（Criteoデータでは真の値は不明なのでS-learnerで推定）
        true_cate = self._estimate_cate_surrogate(X_df.values, T, Y)
        
        feature_names = X_df.columns.tolist()
        
        return X_df, T, Y, true_cate, feature_names
    
    def _estimate_cate_surrogate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """CATE代理推定（S-learnerベース）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 特徴量に処置を追加
        X_with_T = np.column_stack([X, T])
        
        # RandomForestで学習
        model = RandomForestRegressor(n_estimators=50, random_state=self.random_seed)
        model.fit(X_with_T, Y)
        
        # T=1とT=0での予測差を計算
        X_T1 = np.column_stack([X, np.ones(len(X))])
        X_T0 = np.column_stack([X, np.zeros(len(X))])
        
        pred_T1 = model.predict(X_T1)
        pred_T0 = model.predict(X_T0)
        
        return pred_T1 - pred_T0
    
    def get_data_info(self) -> dict:
        """Criteoデータの情報を返す"""
        if os.path.exists(self.data_path):
            try:
                # ヘッダーのみ読み込んでデータ構造を確認
                df_sample = pd.read_csv(self.data_path, nrows=100)
                
                return {
                    "dataset_type": "criteo_uplift",
                    "data_path": self.data_path,
                    "file_exists": True,
                    "columns": df_sample.columns.tolist(),
                    "sample_shape": df_sample.shape,
                    "feature_columns": [f'f{i}' for i in range(12)],
                    "treatment_column": "treatment",
                    "outcome_columns": ["conversion", "visit"],
                    "treatment_distribution": df_sample['treatment'].value_counts().to_dict() if 'treatment' in df_sample.columns else "N/A",
                    "conversion_rate": {
                        "overall": df_sample['conversion'].mean() if 'conversion' in df_sample.columns else "N/A",
                        "by_treatment": df_sample.groupby('treatment')['conversion'].mean().to_dict() if 'treatment' in df_sample.columns and 'conversion' in df_sample.columns else "N/A"
                    }
                }
            except Exception as e:
                return {
                    "dataset_type": "criteo_uplift", 
                    "data_path": self.data_path,
                    "file_exists": True,
                    "error": str(e)
                }
        else:
            return {
                "dataset_type": "criteo_uplift",
                "data_path": self.data_path,
                "file_exists": False,
                "note": "Criteo dataset file not found. Expected: criteo-research-uplift-v2.1.csv.gz"
            }


def create_data_generator(generator_type: str = "synthetic", **kwargs) -> DataGenerator:
    """
    データ生成器のファクトリー関数
    
    Args:
        generator_type (str): 生成器の種類 {"synthetic", "criteo"}
        **kwargs: 各生成器固有のパラメータ
    
    Returns:
        DataGenerator: データ生成器インスタンス
    """
    if generator_type == "synthetic":
        return SyntheticDataGenerator(**kwargs)
    elif generator_type == "criteo":
        return CriteoDataGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


# 後方互換性のための関数（既存コードとの互換性維持）

def generate_synthetic_data(n_samples=1000, random_seed=123, reward_type="linear", df=None):
    """
    EconMLのihdp_surface_Bを模倣した人工データを生成する関数。
    報酬関数の種類を選択でき、既存のDataFrameに対して報酬を生成することも可能。
    
    Args:
        n_samples (int): 生成するサンプル数（dfが指定されていない場合に使用）
        random_seed (int): 乱数シード
        reward_type (str): 報酬関数の種類 {"linear", "logistic", "tree"}
        df (Optional[pd.DataFrame]): 既存の共変量+介入変数のDataFrame
                                   列名は "X0", "X1", ..., "T" が期待される
    
    Returns:
        X_df (pd.DataFrame): 特徴量データ
        T (np.ndarray): 処置データ
        Y (np.ndarray): 観測結果
        true_cate (np.ndarray): 真のCATE
        feature_names (list): 特徴量名のリスト
    """
    np.random.seed(random_seed)
    
    # 既存のDataFrameが提供された場合の処理
    if df is not None:
        # DataFrameから特徴量と処置を抽出
        feature_cols = [col for col in df.columns if col.startswith('X')]
        if 'T' not in df.columns:
            raise ValueError("DataFrame must contain 'T' column for treatment assignment")
        
        X_df = df[feature_cols].copy()
        T = df['T'].values
        X = X_df.values
        feature_names = feature_cols
        n_samples = len(df)
    else:
        # 新しいデータを生成
        # 特徴量 (X)
        X = np.random.normal(size=(n_samples, 5))  # 5つの特徴量
        
        # 処置 (T): ランダムに割り当てる (0 or 1)
        T = np.random.binomial(1, 0.5, size=n_samples)
        
        # 特徴量の列名を付ける
        feature_names = [f'X{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    
    # 報酬関数の種類に応じた真のCATEの計算
    if reward_type == "linear":
        # 線形なCATE（従来の実装）
        true_cate = 0.5 + X[:, 0] + 0.8 * X[:, 1]
        
    elif reward_type == "logistic":
        # ロジスティック関数によるCATE
        # 共変量の線形結合をロジスティック関数に通す
        linear_combination = 0.5 + X[:, 0] + 0.8 * X[:, 1] - 0.3 * X[:, 2]
        true_cate = 2.0 / (1 + np.exp(-linear_combination))  # スケールを調整
        
    elif reward_type == "tree":
        # 決定木による複雑なCATE
        # より複雑な非線形パターンを生成
        tree_regressor = DecisionTreeRegressor(max_depth=3, random_state=random_seed)
        
        # ダミーのターゲット変数を作成（真のCATEの代理として）
        # 複数の特徴量の相互作用を含む関数
        dummy_target = (
            0.8 * X[:, 0] * X[:, 1] + 
            0.5 * np.where(X[:, 2] > 0, X[:, 3], -X[:, 3]) +
            0.3 * X[:, 4] ** 2 +
            np.random.normal(0, 0.1, size=n_samples)
        )
        
        tree_regressor.fit(X, dummy_target)
        true_cate = tree_regressor.predict(X)
        
        # スケールを調整（0.1から3.0の範囲にクリップ）
        true_cate = np.clip(true_cate, 0.1, 3.0)
        
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}. Choose from ['linear', 'logistic', 'tree']")
    
    # 介入なしの場合の結果 (Y0)
    Y0 = 0.2 * X[:, 0] - 0.3 * X[:, 2] + np.random.normal(0, 0.1, size=n_samples)
    
    # 介入ありの場合の結果 (Y1)
    Y1 = Y0 + true_cate
    
    # 観測される結果 (Y)
    Y = Y0 * (1 - T) + Y1 * T

    return X_df, T, Y, true_cate, feature_names

# 例: ihdp_surface_B のようなDGPを利用する場合
# from econml.data.dgps import ihdp_surface_B
# def generate_ihdp_like_data(n_variants=1, n_samples=None):
#     """
#     EconMLのIHDPライクなデータ生成プロセスを使用する。
#     """
#     data_gen = ihdp_surface_B(n_variants=n_variants)
#     df = data_gen.create_systematic_variations(num_samples=n_samples) # DataFrameを返す
#     # Y, T, X, W, TE を適切に抽出・整形する
#     # この例では単純化のため、上記の自作関数を使います。
#     # 実際には、EconMLのDGPから返されるデータ構造に合わせて処理が必要です。
#     # Y = df['y'].values
#     # T = df['t'].values
#     # X = df[[col for col in df.columns if col.startswith('x')]].values
#     # true_cate = df['true_effect'].values
#     # feature_names = [col for col in df.columns if col.startswith('x')]
#     # return pd.DataFrame(X, columns=feature_names), T, Y, true_cate, feature_names
#     pass # 上記の自作関数を当面は使用

if __name__ == '__main__':
    # このファイル単体で実行した場合のテスト用コード
    print("=" * 60)
    print("旧バージョン（後方互換性）テスト")
    print("=" * 60)
    
    print("=== Test 1: Default linear reward type ===")
    X_data, T_data, Y_data, cate_true_data, features = generate_synthetic_data(n_samples=10)
    print("Generated X:\n", X_data.head())
    print("\nGenerated T:\n", T_data[:10])
    print("\nGenerated Y:\n", Y_data[:10])
    print("\nTrue CATE (linear):\n", cate_true_data[:10])
    print("\nFeature names:", features)
    
    print("\n=== Test 2: Logistic reward type ===")
    X_data_log, T_data_log, Y_data_log, cate_true_data_log, _ = generate_synthetic_data(
        n_samples=10, reward_type="logistic", random_seed=123
    )
    print("True CATE (logistic):\n", cate_true_data_log[:10])
    
    print("\n=== Test 3: Tree reward type ===")
    X_data_tree, T_data_tree, Y_data_tree, cate_true_data_tree, _ = generate_synthetic_data(
        n_samples=10, reward_type="tree", random_seed=123
    )
    print("True CATE (tree):\n", cate_true_data_tree[:10])
    
    print("\n=== Test 4: Using existing DataFrame ===")
    # 既存のDataFrameを作成
    test_df = pd.DataFrame({
        'X0': [0.1, -0.5, 1.2],
        'X1': [0.8, -0.2, 0.3],
        'X2': [0.0, 1.1, -0.7],
        'X3': [0.5, 0.2, -0.1],
        'X4': [1.0, -0.8, 0.6],
        'T': [1, 0, 1]
    })
    
    X_data_df, T_data_df, Y_data_df, cate_true_data_df, features_df = generate_synthetic_data(
        df=test_df, reward_type="linear", random_seed=123
    )
    print("Input DataFrame:\n", test_df)
    print("\nGenerated Y from DataFrame:\n", Y_data_df)
    print("True CATE from DataFrame:\n", cate_true_data_df)
    
    print("\n" + "=" * 60)
    print("新バージョン（抽象クラス）テスト")
    print("=" * 60)
    
    # 合成データ生成器のテスト
    print("\n=== Test 5: SyntheticDataGenerator ===")
    synthetic_gen = create_data_generator("synthetic", random_seed=123)
    print("Data generator info:", synthetic_gen.get_data_info())
    
    X_syn, T_syn, Y_syn, cate_syn, features_syn = synthetic_gen.generate_data(
        n_samples=10, reward_type="linear"
    )
    print("Generated synthetic data - X shape:", X_syn.shape)
    print("Generated synthetic data - CATE sample:", cate_syn[:5])
    
    # Criteoデータ生成器のテスト（ファイルが存在しない場合の動作確認）
    print("\n=== Test 6: CriteoDataGenerator ===")
    criteo_gen = create_data_generator("criteo", random_seed=123)
    print("Criteo generator info:", criteo_gen.get_data_info())
    
    try:
        X_criteo, T_criteo, Y_criteo, cate_criteo, features_criteo = criteo_gen.generate_data(n_samples=10)
        print("Generated Criteo data - X shape:", X_criteo.shape)
        print("Generated Criteo data - CATE sample:", cate_criteo[:5])
    except FileNotFoundError as e:
        print("Expected error (Criteo file not found):", str(e))
    
    print("\n=== Test 7: サンプルCriteoデータ作成 ===")
    # Criteo風のサンプルデータを作成してテスト
    sample_criteo_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randint(0, 10, 100),
        'category_1': np.random.choice(['A', 'B', 'C'], 100),
        'category_2': np.random.choice(['X', 'Y'], 100),
        'click': np.random.binomial(1, 0.3, 100),
        'conversion': np.random.binomial(1, 0.1, 100)
    })
    
    # サンプルデータをファイルに保存
    sample_path = os.path.join("data", "sample_criteo_data.csv")
    os.makedirs("data", exist_ok=True)
    sample_criteo_data.to_csv(sample_path, index=False)
    print(f"Sample Criteo data saved to {sample_path}")
    
    # サンプルデータでテスト
    criteo_gen_sample = create_data_generator("criteo", data_path=sample_path, random_seed=123)
    X_sample, T_sample, Y_sample, cate_sample, features_sample = criteo_gen_sample.generate_data(
        n_samples=20, 
        treatment_col="click", 
        outcome_col="conversion"
    )
    print("Sample Criteo data - X shape:", X_sample.shape)
    print("Sample Criteo data - features:", features_sample)
    print("Sample Criteo data - CATE sample:", cate_sample[:5])