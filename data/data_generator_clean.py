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
    def generate_data(self, **kwargs):
        """
        データを生成する抽象メソッド
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        合成データを生成
        
        Args:
            n_samples (int): 生成するサンプル数
            reward_type (str): 報酬関数の種類 {"linear", "logistic", "tree"}
            df (Optional[pd.DataFrame]): 既存の共変量+介入変数のDataFrame
        
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
            true_cate = 0.5 + X[:, 0] + 0.8 * X[:, 1]
        elif reward_type == "logistic":
            linear_combination = 0.5 + X[:, 0] + 0.8 * X[:, 1] - 0.3 * X[:, 2]
            true_cate = 2.0 / (1 + np.exp(-linear_combination))
        else:
            raise ValueError(f"Unsupported reward_type: {reward_type}")
        
        # 介入なしの場合の結果 (Y0)
        Y0 = 0.2 * X[:, 0] - 0.3 * X[:, 2] + np.random.normal(0, 0.1, size=n_samples)
        
        # 介入ありの場合の結果 (Y1)
        Y1 = Y0 + true_cate
        
        # 観測される結果 (Y)
        y = Y0 * (1 - treatment) + Y1 * treatment

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
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        CriteoデータからCATE推定用データを生成
        
        Args:
            n_samples (Optional[int]): 使用するサンプル数（Noneの場合全データ）
            treatment_col (str): 処置変数として使用する列名
            outcome_col (str): 結果変数として使用する列名
            feature_cols (Optional[List[str]]): 特徴量として使用する列名リスト
        
        Returns:
            X_df (pd.DataFrame): 特徴量データ（DataFrame形式）
            treatment (np.ndarray): 処置データ
            y (np.ndarray): 観測結果
            true_cate (np.ndarray): 真のCATE（推定値）
            feature_names (List[str]): 特徴量名リスト
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
        
        # 処置変数の準備
        if treatment_col not in df.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in data")
        treatment = df[treatment_col].astype(int).values
        
        # 結果変数の準備
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")
        y = df[outcome_col].values
        
        # 真のCATEの近似（簡単な推定）
        # 注意：これは真のCATEではなく推定値です
        try:
            # 簡単なCATE推定のため、各治療群でランダムフォレストを学習
            control_idx = (treatment == 0)
            treatment_idx = (treatment == 1)
            
            if np.sum(control_idx) > 10 and np.sum(treatment_idx) > 10:
                # Control群でのモデル
                rf_control = RandomForestRegressor(n_estimators=10, random_state=self.random_seed)
                rf_control.fit(X_df[control_idx], y[control_idx])
                y0_pred = rf_control.predict(X_df)
                
                # Treatment群でのモデル
                rf_treatment = RandomForestRegressor(n_estimators=10, random_state=self.random_seed)
                rf_treatment.fit(X_df[treatment_idx], y[treatment_idx])
                y1_pred = rf_treatment.predict(X_df)
                
                # CATE = E[Y|T=1, X] - E[Y|T=0, X]
                true_cate = y1_pred - y0_pred
            else:
                # データが少ない場合は単純な差分
                true_cate = np.zeros(len(y))
                
        except Exception:
            # エラーの場合は単純にゼロで埋める
            true_cate = np.zeros(len(y))
        
        feature_names = feature_cols
        
        return X_df, treatment, y, true_cate, feature_names
    
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


# 後方互換性のための関数
def generate_synthetic_data(n_samples=1000, random_seed=123, reward_type="linear", df=None):
    """
    後方互換性のための合成データ生成関数
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    X, y, treatment = generator.generate_data(n_samples=n_samples, reward_type=reward_type, df=df)
    
    # 従来の戻り値形式に合わせる
    feature_names = [f'X{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 真のCATEを計算（簡略版）
    if reward_type == "linear":
        true_cate = 0.5 + X[:, 0] + 0.8 * X[:, 1]
    else:
        true_cate = np.zeros(len(X))
    
    return X_df, treatment, y, true_cate, feature_names


if __name__ == '__main__':
    # テスト用コード
    print("=== Data Generator Test ===")
    
    # 合成データのテスト
    print("\n1. Synthetic Data Generator Test:")
    synthetic_gen = create_data_generator("synthetic", random_seed=42)
    X, y, treatment = synthetic_gen.generate_data(n_samples=100)
    print(f"X shape: {X.shape}, y shape: {y.shape}, treatment shape: {treatment.shape}")
    print(f"Conversion rate: {y.mean():.4f}")
    
    # Criteoデータのテスト
    print("\n2. Criteo Data Generator Test:")
    criteo_gen = create_data_generator("criteo", random_seed=42)
    info = criteo_gen.get_data_info()
    print(f"Criteo info: {info}")
    
    try:
        X_criteo, y_criteo, treatment_criteo = criteo_gen.generate_data(n_samples=100)
        print(f"Criteo X shape: {X_criteo.shape}, y shape: {y_criteo.shape}, treatment shape: {treatment_criteo.shape}")
        print(f"Criteo conversion rate: {y_criteo.mean():.4f}")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    print("\nTest completed!")
