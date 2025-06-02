# criteo_data_generator.py
"""
Criteo uplift データセット用データ生成クラス
"""
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from data_generator import DataGenerator


class CriteoUpliftDataGenerator(DataGenerator):
    """
    Criteo Uplift Research データセット用データ生成クラス
    
    データセット構造:
    - 特徴量: f0~f11 (12個の数値特徴量)
    - 処置: treatment (0/1)
    - 結果: conversion (0/1), visit (0/1)
    - その他: exposure (0/1)
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
        feature_cols: Optional[List[str]] = None,
        simulate_treatment: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        CriteoデータからCATE推定用データを生成
        
        Args:
            n_samples (Optional[int]): 使用するサンプル数（Noneの場合全データ）
            treatment_col (str): 処置変数として使用する列名
            outcome_col (str): 結果変数として使用する列名
            feature_cols (Optional[List[str]]): 特徴量として使用する列名リスト
            simulate_treatment (bool): 処置を人工的にシミュレートするかどうか
        
        Returns:
            X_df (pd.DataFrame): 特徴量データ
            T (np.ndarray): 処置データ
            Y (np.ndarray): 観測結果
            true_cate (np.ndarray): 推定されたCATE（真の値ではない）
            feature_names (List[str]): 特徴量名のリスト
        """
        
        # Criteoデータの読み込み
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Criteo data file not found at {self.data_path}. "
                f"Please download the Criteo uplift dataset and place it in the data/ folder."
            )
        
        print(f"Loading Criteo uplift data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"Original dataset shape: {df.shape}")
        
        # サンプル数の制限
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=self.random_seed).reset_index(drop=True)
            print(f"Sampled {n_samples} rows from dataset")
        
        # 特徴量の選択
        if feature_cols is None:
            # Criteoデータの特徴量列（f0-f11）を自動選択
            feature_cols = [f'f{i}' for i in range(12)]  # f0からf11まで
            print(f"Using Criteo feature columns: {feature_cols}")
        
        # 利用可能な特徴量のみを選択
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) != len(feature_cols):
            print(f"Warning: Some feature columns not found. Using: {available_features}")
        feature_cols = available_features
        
        # 前処理
        X_df = self._preprocess_features(df, feature_cols)
        
        # 処置変数の準備
        if simulate_treatment:
            # 処置をランダムに割り当て（傾向スコアベース）
            propensity_score = self._calculate_propensity_score(X_df)
            T = np.random.binomial(1, propensity_score)
            print("Simulated treatment assignment based on propensity scores")
        else:
            if treatment_col not in df.columns:
                raise ValueError(f"Treatment column '{treatment_col}' not found in data")
            T = df[treatment_col].astype(int).values
            print(f"Using existing treatment column: {treatment_col}")
        
        # 結果変数の準備
        if outcome_col not in df.columns:
            # 結果変数がない場合はシミュレート
            Y = self._simulate_outcome(X_df.values, T)
            print("Simulated outcome variable")
        else:
            Y = df[outcome_col].values
            print(f"Using existing outcome column: {outcome_col}")
        
        # データ統計の表示
        print(f"\nData Statistics:")
        print(f"  Treatment distribution: {np.bincount(T)}")
        print(f"  Treatment rates: {np.bincount(T) / len(T)}")
        print(f"  Outcome rate (overall): {np.mean(Y):.4f}")
        print(f"  Outcome rate (control): {np.mean(Y[T == 0]):.4f}")
        print(f"  Outcome rate (treatment): {np.mean(Y[T == 1]):.4f}")
        
        # 真のCATEをシミュレート（Criteoデータでは真の値は不明）
        true_cate = self._estimate_cate_surrogate(X_df.values, T, Y)
        
        feature_names = X_df.columns.tolist()
        
        return X_df, T, Y, true_cate, feature_names
    
    def _preprocess_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """特徴量の前処理（Criteoデータ用に簡略化）"""
        X_df = df[feature_cols].copy()
        
        # 欠損値処理（Criteoデータには欠損値はないが、念のため）
        X_df = X_df.fillna(X_df.median())
        
        # 標準化
        X_df[feature_cols] = self.scaler.fit_transform(X_df[feature_cols])
        
        return X_df
    
    def _calculate_propensity_score(self, X: pd.DataFrame) -> np.ndarray:
        """傾向スコアの計算（ロジスティック回帰ベース）"""
        # 特徴量の線形結合でロジスティック関数を適用
        linear_combination = np.sum(X.values * np.random.normal(0, 0.1, X.shape[1]), axis=1)
        propensity_score = 1 / (1 + np.exp(-linear_combination))
        
        # 0.1-0.9の範囲にクリップ
        return np.clip(propensity_score, 0.1, 0.9)
    
    def _simulate_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """結果変数のシミュレーション"""
        # ベースライン効果
        baseline = np.sum(X * np.random.normal(0, 0.2, X.shape[1]), axis=1)
        
        # 処置効果
        treatment_effect = T * (0.5 + 0.3 * X[:, 0] if X.shape[1] > 0 else 0.5)
        
        # ノイズ
        noise = np.random.normal(0, 0.1, len(X))
        
        # 確率に変換（シグモイド関数）
        logits = baseline + treatment_effect + noise
        probs = 1 / (1 + np.exp(-logits))
        
        return np.random.binomial(1, probs)
    
    def _estimate_cate_surrogate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """CATE代理推定（S-learnerベース）"""
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
                df_sample = pd.read_csv(self.data_path, nrows=1000)
                
                return {
                    "dataset_type": "criteo_uplift",
                    "data_path": self.data_path,
                    "file_exists": True,
                    "columns": df_sample.columns.tolist(),
                    "sample_shape": df_sample.shape,
                    "feature_columns": [f'f{i}' for i in range(12)],
                    "treatment_column": "treatment",
                    "outcome_columns": ["conversion", "visit"],
                    "treatment_distribution": df_sample['treatment'].value_counts().to_dict(),
                    "conversion_rate": {
                        "overall": df_sample['conversion'].mean(),
                        "by_treatment": df_sample.groupby('treatment')['conversion'].mean().to_dict()
                    },
                    "visit_rate": {
                        "overall": df_sample['visit'].mean(),
                        "by_treatment": df_sample.groupby('treatment')['visit'].mean().to_dict()
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
                "note": "Criteo uplift dataset file not found. Expected: criteo-research-uplift-v2.1.csv.gz"
            }


if __name__ == "__main__":
    # テスト実行
    print("=" * 60)
    print("Criteo Uplift Data Generator Test")
    print("=" * 60)
    
    # データ生成器の作成
    criteo_gen = CriteoUpliftDataGenerator(random_seed=123)
    
    # データ情報の表示
    print("\n=== Data Info ===")
    info = criteo_gen.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 小さなサンプルでテスト
    try:
        print("\n=== Small Sample Test ===")
        X, T, Y, cate, features = criteo_gen.generate_data(
            n_samples=1000, 
            outcome_col="conversion"
        )
        print(f"\nGenerated data shapes:")
        print(f"  X: {X.shape}")
        print(f"  T: {T.shape}")
        print(f"  Y: {Y.shape}")
        print(f"  CATE: {cate.shape}")
        print(f"  Features: {features}")
        print(f"\nCATE statistics:")
        print(f"  Mean: {np.mean(cate):.4f}")
        print(f"  Std: {np.std(cate):.4f}")
        print(f"  Range: [{np.min(cate):.4f}, {np.max(cate):.4f}]")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
