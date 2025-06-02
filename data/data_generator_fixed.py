# data_generator.py
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor


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
    def generate_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        データを生成する抽象メソッド
        
        Returns:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 観測結果
            treatment (np.ndarray): 処置データ (0 or 1)
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
        reward_type: str = "linear"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        合成データを生成
        
        Args:
            n_samples (int): 生成するサンプル数
            reward_type (str): 報酬関数の種類 {"linear", "logistic", "tree"}
        
        Returns:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 観測結果
            treatment (np.ndarray): 処置データ
        """
        np.random.seed(self.random_seed)
        
        # 特徴量 (X)
        X = np.random.normal(size=(n_samples, 5))  # 5つの特徴量
        
        # 処置 (T): ランダムに割り当てる (0 or 1)
        treatment = np.random.binomial(1, 0.5, size=n_samples)
        
        # 報酬関数の種類に応じた真のCATEの計算
        if reward_type == "linear":
            # 線形なCATE
            true_cate = 0.5 + X[:, 0] + 0.8 * X[:, 1]
            
        elif reward_type == "logistic":
            # ロジスティック関数によるCATE
            linear_combination = 0.5 + X[:, 0] + 0.8 * X[:, 1] - 0.3 * X[:, 2]
            true_cate = 2.0 / (1 + np.exp(-linear_combination))
            
        elif reward_type == "tree":
            # 決定木による複雑なCATE
            tree_regressor = DecisionTreeRegressor(max_depth=3, random_state=self.random_seed)
            dummy_target = (
                0.8 * X[:, 0] * X[:, 1] + 
                0.5 * np.where(X[:, 2] > 0, X[:, 3], -X[:, 3]) +
                0.3 * X[:, 4] ** 2 +
                np.random.normal(0, 0.1, size=n_samples)
            )
            tree_regressor.fit(X, dummy_target)
            true_cate = tree_regressor.predict(X)
            true_cate = np.clip(true_cate, 0.1, 3.0)
        else:
            raise ValueError(f"Unsupported reward_type: {reward_type}")
        
        # 介入なしの場合の結果 (Y0)
        Y0 = 0.2 * X[:, 0] - 0.3 * X[:, 2] + np.random.normal(0, 0.1, size=n_samples)
        
        # 介入ありの場合の結果 (Y1)
        Y1 = Y0 + true_cate
        
        # 観測される結果 (Y) - 低い変換率を持つようにクリップ
        Y = Y0 * (1 - treatment) + Y1 * treatment
        
        # バイナリ結果に変換（低い変換率を作るため）
        # シグモイド関数を使って確率に変換し、その後バイナリにサンプリング
        probs = 1 / (1 + np.exp(-Y + 2))  # +2でより低い変換率を作る
        y = np.random.binomial(1, probs)
        
        return X, y, treatment
    
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
        if data_path is None:
            # Default path to Criteo dataset
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_path = os.path.join(base_path, "data", "criteo-research-uplift-v2.1.csv.gz")
        else:
            self.data_path = data_path
        self.scaler = StandardScaler()
    
    def generate_data(
        self, 
        n_samples: Optional[int] = None,
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
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
        X_scaled = self.scaler.fit_transform(X_df[feature_cols])  # 標準化
        
        # 処置変数の準備
        if treatment_col not in df.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")
        treatment = df[treatment_col].astype(int).values
        
        # 結果変数の準備
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")
        y = df[outcome_col].values
        
        return X_scaled, y, treatment
    
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
    後方互換性のための関数
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    X, y, treatment = generator.generate_data(n_samples=n_samples, reward_type=reward_type)
    
    # 旧形式のレスポンスに合わせる
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    true_cate = np.random.normal(0.5, 0.2, len(y))  # ダミーのCATE
    feature_names = X_df.columns.tolist()
    
    return X_df, treatment, y, true_cate, feature_names


if __name__ == '__main__':
    # テスト用コード
    print("=== Testing SyntheticDataGenerator ===")
    synthetic_gen = create_data_generator("synthetic", random_seed=123)
    print("Data generator info:", synthetic_gen.get_data_info())
    
    X_syn, y_syn, treatment_syn = synthetic_gen.generate_data(n_samples=10, reward_type="linear")
    print("Generated synthetic data - X shape:", X_syn.shape)
    print("Generated synthetic data - y sample:", y_syn[:5])
    print("Generated synthetic data - treatment sample:", treatment_syn[:5])
    
    print("\n=== Testing CriteoDataGenerator ===")
    criteo_gen = create_data_generator("criteo", random_seed=123)
    print("Criteo generator info:", criteo_gen.get_data_info())
