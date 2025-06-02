# imbalance_handler.py
"""
不均衡データに対するアンダーサンプリング手法の実装
論文: "Uplift Modeling with High Class Imbalance" (Nyberg et al., 2021)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


class UpliftUndersampler:
    """
    Uplift Modeling用アンダーサンプリングクラス
    
    論文の手法に基づき、処置群・対照群それぞれで独立にアンダーサンプリングを実行し、
    少数派クラス（正例）の割合をk倍に増加させる。
    """
    
    def __init__(self, k: float = 2.0, random_state: int = 42):
        """
        Args:
            k (float): 少数派クラス割合の増加倍率（k > 1.0）
            random_state (int): 乱数シード
        """
        if k <= 1.0:
            raise ValueError("k must be greater than 1.0")
        
        self.k = k
        self.random_state = random_state
        self.original_pos_rates_ = {}  # 各群の元の正例率を保存
        self.scaling_factor_ = k  # 予測値補正用
        
    def fit_resample(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        アンダーサンプリングを実行
        
        Args:
            X (pd.DataFrame): 特徴量
            T (np.ndarray): 処置変数 (0 or 1)
            Y (np.ndarray): 目的変数 (0 or 1)
            
        Returns:
            X_resampled, T_resampled, Y_resampled: アンダーサンプリング後のデータ
        """
        np.random.seed(self.random_state)
        
        # 処置群・対照群に分割
        treatment_mask = T == 1
        control_mask = T == 0
        
        X_resampled_list = []
        T_resampled_list = []
        Y_resampled_list = []
        
        for group_name, mask in [("treatment", treatment_mask), ("control", control_mask)]:
            if not np.any(mask):
                continue
                
            X_group = X[mask].copy()
            T_group = T[mask]
            Y_group = Y[mask]
            
            # 各群の正例率を計算
            N_pos = np.sum(Y_group == 1)
            N_neg = np.sum(Y_group == 0)
            N_total = len(Y_group)
            
            if N_pos == 0 or N_neg == 0:
                print(f"Warning: {group_name} group has no positive or negative samples")
                X_resampled_list.append(X_group)
                T_resampled_list.append(T_group)
                Y_resampled_list.append(Y_group)
                continue
            
            p_y_t = N_pos / N_total  # 元の正例率
            self.original_pos_rates_[group_name] = p_y_t
            
            print(f"{group_name.capitalize()} group - Original positive rate: {p_y_t:.4f}")
            
            # アンダーサンプリング後の目標正例率
            target_pos_rate = self.k * p_y_t
            
            if target_pos_rate >= 1.0:
                print(f"Warning: Target positive rate ({target_pos_rate:.4f}) >= 1.0. Skipping undersampling for {group_name} group.")
                X_resampled_list.append(X_group)
                T_resampled_list.append(T_group)
                Y_resampled_list.append(Y_group)
                continue
            
            # 論文の式に基づく負例のサンプリング比率
            # tilde_N_neg = ((1/k - p(y|t)) / (1 - p(y|t))) * N_neg
            sampling_ratio = ((1/self.k - p_y_t) / (1 - p_y_t))
            
            if sampling_ratio <= 0 or sampling_ratio > 1:
                print(f"Warning: Invalid sampling ratio ({sampling_ratio:.4f}) for {group_name} group. Skipping undersampling.")
                X_resampled_list.append(X_group)
                T_resampled_list.append(T_group)
                Y_resampled_list.append(Y_group)
                continue
            
            n_neg_sampled = int(N_neg * sampling_ratio)
            
            print(f"{group_name.capitalize()} group - Sampling {n_neg_sampled}/{N_neg} negative samples (ratio: {sampling_ratio:.4f})")
            
            # 負例をアンダーサンプリング
            neg_indices = np.where(Y_group == 0)[0]
            pos_indices = np.where(Y_group == 1)[0]
            
            sampled_neg_indices = np.random.choice(neg_indices, size=n_neg_sampled, replace=False)
            
            # 全正例 + サンプリングされた負例
            final_indices = np.concatenate([pos_indices, sampled_neg_indices])
            np.random.shuffle(final_indices)
            
            X_group_resampled = X_group.iloc[final_indices].copy()
            T_group_resampled = T_group[final_indices]
            Y_group_resampled = Y_group[final_indices]
            
            # アンダーサンプリング後の正例率を確認
            actual_pos_rate = np.mean(Y_group_resampled == 1)
            print(f"{group_name.capitalize()} group - Actual positive rate after undersampling: {actual_pos_rate:.4f}")
            
            X_resampled_list.append(X_group_resampled)
            T_resampled_list.append(T_group_resampled)
            Y_resampled_list.append(Y_group_resampled)
          # 結果をまとめる
        if X_resampled_list:
            X_resampled = pd.concat(X_resampled_list, ignore_index=True)
            T_resampled = np.concatenate(T_resampled_list)
            Y_resampled = np.concatenate(Y_resampled_list)
        else:
            # 空の場合の処理
            X_resampled = pd.DataFrame()
            T_resampled = np.array([])
            Y_resampled = np.array([])
        
        # インデックスをシャッフル
        shuffle_indices = np.arange(len(X_resampled))
        np.random.shuffle(shuffle_indices)
        
        X_resampled = X_resampled.iloc[shuffle_indices].reset_index(drop=True)
        T_resampled = T_resampled[shuffle_indices]
        Y_resampled = Y_resampled[shuffle_indices]
        
        print(f"\nTotal samples after undersampling: {len(X_resampled)} (original: {len(X)})")
        print(f"Overall positive rate after undersampling: {np.mean(Y_resampled == 1):.4f}")
        
        return X_resampled, T_resampled, Y_resampled
    
    def correct_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        アンダーサンプリング後のモデル予測値を補正
        
        Args:
            predictions (np.ndarray): モデルの予測値（CATE推定値）
            
        Returns:
            np.ndarray: 補正された予測値
        """
        return predictions / self.scaling_factor_


class IsotonicCalibrator:
    """
    Isotonic Regression によるキャリブレーション
    
    アンダーサンプリング後のモデル出力を実際のuplift確率に近づける
    """
    
    def __init__(self):
        self.isotonic_reg_treatment = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_reg_control = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray, model_predictions: np.ndarray):
        """
        キャリブレーション用のIsotonic Regressionを学習
        
        Args:
            X (pd.DataFrame): 特徴量
            T (np.ndarray): 処置変数
            Y (np.ndarray): 目的変数
            model_predictions (np.ndarray): モデルの予測値
        """
        treatment_mask = T == 1
        control_mask = T == 0
        
        if np.any(treatment_mask):
            self.isotonic_reg_treatment.fit(model_predictions[treatment_mask], Y[treatment_mask])
        
        if np.any(control_mask):
            self.isotonic_reg_control.fit(model_predictions[control_mask], Y[control_mask])
        
        self.is_fitted = True
    
    def transform(self, predictions: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        予測値をキャリブレーション
        
        Args:
            predictions (np.ndarray): 元の予測値
            T (np.ndarray): 処置変数
            
        Returns:
            np.ndarray: キャリブレーション後の予測値
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        calibrated_predictions = np.zeros_like(predictions)
        treatment_mask = T == 1
        control_mask = T == 0
        
        if np.any(treatment_mask):
            calibrated_predictions[treatment_mask] = self.isotonic_reg_treatment.transform(predictions[treatment_mask])
        
        if np.any(control_mask):
            calibrated_predictions[control_mask] = self.isotonic_reg_control.transform(predictions[control_mask])
        
        return calibrated_predictions


class ImbalancedUpliftPipeline:
    """
    不均衡データ対応のUplift Modeling パイプライン
    
    アンダーサンプリング + 予測値補正 + キャリブレーションを統合
    """
    
    def __init__(self, k: float = 2.0, use_calibration: bool = True, random_state: int = 42):
        """
        Args:
            k (float): アンダーサンプリングの倍率
            use_calibration (bool): キャリブレーションを使用するか
            random_state (int): 乱数シード
        """
        self.k = k
        self.use_calibration = use_calibration
        self.random_state = random_state
        
        self.undersampler = UpliftUndersampler(k=k, random_state=random_state)
        self.calibrator = IsotonicCalibrator() if use_calibration else None
        self.base_model = None
        
    def fit_transform_data(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        データにアンダーサンプリングを適用
        
        Args:
            X, T, Y: 元のデータ
            
        Returns:
            アンダーサンプリング後のデータ
        """
        return self.undersampler.fit_resample(X, T, Y)
    
    def fit_model_with_calibration(self, model, X_train: pd.DataFrame, T_train: np.ndarray, Y_train: np.ndarray,
                                 X_val: Optional[pd.DataFrame] = None, T_val: Optional[np.ndarray] = None, 
                                 Y_val: Optional[np.ndarray] = None):
        """
        モデルを学習し、キャリブレーションも行う
        
        Args:
            model: CATE推定モデル
            X_train, T_train, Y_train: 学習データ
            X_val, T_val, Y_val: キャリブレーション用検証データ（Noneの場合は学習データを分割）
        """
        self.base_model = model
        
        # キャリブレーション用データの準備
        if X_val is None or T_val is None or Y_val is None:
            # 学習データを分割してキャリブレーション用データを作成
            X_train_split, X_val, T_train_split, T_val, Y_train_split, Y_val = train_test_split(
                X_train, T_train, Y_train, test_size=0.2, random_state=self.random_state, stratify=T_train
            )
            X_train, T_train, Y_train = X_train_split, T_train_split, Y_train_split
        
        # モデル学習
        self.base_model.fit(X_train, T_train, Y_train)
        
        # キャリブレーション
        if self.use_calibration and self.calibrator is not None:
            val_predictions = self.base_model.predict(X_val)
            # アンダーサンプリング補正を適用
            val_predictions_corrected = self.undersampler.correct_predictions(val_predictions)
            
            # キャリブレーション学習（処置群・対照群の予測値を個別に使用）
            self.calibrator.fit(X_val, T_val, Y_val, val_predictions_corrected)
    
    def predict(self, X: pd.DataFrame, T: np.ndarray) -> np.ndarray:
        """
        予測（補正・キャリブレーション込み）
        
        Args:
            X: 特徴量
            T: 処置変数
            
        Returns:
            補正・キャリブレーション済みの予測値
        """
        if self.base_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # 基本予測
        predictions = self.base_model.predict(X)
        
        # アンダーサンプリング補正
        predictions_corrected = self.undersampler.correct_predictions(predictions)
        
        # キャリブレーション
        if self.use_calibration and self.calibrator is not None:
            predictions_final = self.calibrator.transform(predictions_corrected, T)
        else:
            predictions_final = predictions_corrected
            
        return predictions_final


# 使用例とテスト関数
def test_undersampling():
    """アンダーサンプリング手法のテスト"""
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    # 不均衡データを作成（正例率約5%）
    X = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    T = np.random.binomial(1, 0.5, n_samples)
    Y = np.random.binomial(1, 0.05, n_samples)  # 5%の正例率
    
    print("=== Original Data ===")
    print(f"Total samples: {len(X)}")
    print(f"Treatment group samples: {np.sum(T == 1)}")
    print(f"Control group samples: {np.sum(T == 0)}")
    print(f"Overall positive rate: {np.mean(Y):.4f}")
    print(f"Treatment group positive rate: {np.mean(Y[T == 1]):.4f}")
    print(f"Control group positive rate: {np.mean(Y[T == 0]):.4f}")
    
    # アンダーサンプリング実行
    undersampler = UpliftUndersampler(k=3.0, random_state=42)
    X_resampled, T_resampled, Y_resampled = undersampler.fit_resample(X, T, Y)
    
    print("\n=== After Undersampling ===")
    print(f"Total samples: {len(X_resampled)}")
    print(f"Treatment group samples: {np.sum(T_resampled == 1)}")
    print(f"Control group samples: {np.sum(T_resampled == 0)}")
    print(f"Overall positive rate: {np.mean(Y_resampled):.4f}")
    print(f"Treatment group positive rate: {np.mean(Y_resampled[T_resampled == 1]):.4f}")
    print(f"Control group positive rate: {np.mean(Y_resampled[T_resampled == 0]):.4f}")


if __name__ == "__main__":
    test_undersampling()
