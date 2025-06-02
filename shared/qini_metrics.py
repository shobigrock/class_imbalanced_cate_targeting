"""
QINI係数とアップリフト評価指標の計算

人工データでのCATE推定における追加評価指標を提供します。
QINIは反実仮想のあるデータでのみ計算可能です。
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional
# Removed numba import to avoid compilation issues


class QINICalculator:
    """QINI係数計算クラス"""
    
    def __init__(self):
        """初期化"""
        self.last_qini_points = None
        self.last_optimal_points = None
    
    def calculate_qini_coefficient(
        self, 
        y_true: np.ndarray, 
        y_pred_cate: np.ndarray, 
        treatment: np.ndarray,
        y_counterfactual: Optional[np.ndarray] = None
    ) -> float:
        """
        QINI係数を計算
        
        Args:
            y_true: 観測結果 (0 or 1)
            y_pred_cate: 予測されたCATE値
            treatment: 処置フラグ (0 or 1) 
            y_counterfactual: 反実仮想の結果（人工データのみ）
            
        Returns:
            QINI係数 (0-1の範囲、1に近いほど良い)
        """
        if y_counterfactual is None:
            warnings.warn("QINI係数は反実仮想データがないと正確に計算できません")
            # QINI係数は計算できないのでNoneを返す
            return None
        else:
            # 真のCATEを使用してQINI計算
            true_cate = self._calculate_true_cate(y_true, y_counterfactual, treatment)
            qini_coeff = self._calculate_qini_from_true_cate(
                y_true, true_cate, treatment
            )
        
        return qini_coeff
    
    def _calculate_true_cate(
        self, 
        y_observed: np.ndarray, 
        y_counterfactual: np.ndarray, 
        treatment: np.ndarray
    ) -> np.ndarray:
        """
        反実仮想から真のCATEを計算
        
        Args:
            y_observed: 観測された結果
            y_counterfactual: 反実仮想の結果
            treatment: 処置フラグ
            
        Returns:
            真のCATE値
        """
        true_cate = np.zeros_like(y_observed, dtype=float)
        
        # 処置群: Y(1) - Y(0) = y_observed - y_counterfactual
        treatment_mask = treatment == 1
        true_cate[treatment_mask] = (
            y_observed[treatment_mask] - y_counterfactual[treatment_mask]
        )
        
        # 対照群: Y(1) - Y(0) = y_counterfactual - y_observed  
        control_mask = treatment == 0
        true_cate[control_mask] = (
            y_counterfactual[control_mask] - y_observed[control_mask]
        )
        
        return true_cate
    
    def _calculate_qini_from_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred_cate: np.ndarray, 
        treatment: np.ndarray
    ) -> float:
        """
        予測CATEからQINI係数を計算（近似）
        """
        return self._qini_coefficient_numba(
            y_true.astype(bool), 
            y_pred_cate.astype(float), 
            treatment.astype(bool)
        )    
    def _calculate_qini_from_true_cate(
        self, 
        y_true: np.ndarray, 
        true_cate: np.ndarray, 
        treatment: np.ndarray
    ) -> float:
        """
        真のCATEからQINI係数を計算
        """
        return self._qini_coefficient_numba(
            y_true.astype(bool), 
            true_cate.astype(float), 
            treatment.astype(bool)
        )
    
    @staticmethod
    def _qini_coefficient_numba(
        data_class: np.ndarray, 
        data_score: np.ndarray, 
        data_group: np.ndarray
    ) -> float:
        """
        QINI係数のnumba最適化実装
        
        Args:
            data_class: 結果 (True/False)
            data_score: スコア（CATE予測値）
            data_group: 処置群フラグ (True=処置群, False=対照群)
        """
        # データを降順でソート
        data_idx = np.argsort(data_score)[::-1]
        data_class = data_class[data_idx]
        data_score = data_score[data_idx]
        data_group = data_group[data_idx]
        
        # QINI曲線のポイントを計算
        qini_points = QINICalculator._qini_points_numba(data_class, data_score, data_group)
        numerator = np.sum(qini_points)
        
        # 最適な順序でのQINI計算（分母）
        n_treatment = np.sum(data_group)
        n_control = np.sum(~data_group)
        
        # 最適順序: 処置群の成功 -> 処置群の失敗 -> 対照群の失敗 -> 対照群の成功
        optimal_class = np.concatenate([
            np.ones(np.sum(data_class[data_group]), dtype=np.bool_),  # 処置群成功
            np.zeros(np.sum(~data_class[data_group]), dtype=np.bool_), # 処置群失敗
            np.zeros(np.sum(~data_class[~data_group]), dtype=np.bool_), # 対照群失敗
            np.ones(np.sum(data_class[~data_group]), dtype=np.bool_)   # 対照群成功
        ])
        
        optimal_group = np.concatenate([
            np.ones(n_treatment, dtype=np.bool_),
            np.zeros(n_control, dtype=np.bool_)
        ])
        
        # 降順スコア（最適順序を維持）
        optimal_score = np.arange(len(data_group))[::-1].astype(np.float64)
        
        optimal_qini_points = QINICalculator._qini_points_numba(
            optimal_class, optimal_score, optimal_group
        )
        denominator = np.sum(optimal_qini_points)
        
        # QINI係数計算
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    @staticmethod
    def _qini_points_numba(
        data_class: np.ndarray, 
        data_score: np.ndarray, 
        data_group: np.ndarray
    ) -> np.ndarray:
        """
        QINI曲線のポイントを計算（numba最適化）
        
        Args:
            data_class: 結果配列
            data_score: スコア配列（降順ソート済み）
            data_group: 処置群フラグ配列
        """
        qini_points = []
        
        # 正規化因子 (N_t / N_c)
        n_treatment = np.sum(data_group)
        n_control = np.sum(~data_group)
        
        if n_control == 0:
            n_factor = 1.0
        else:
            n_factor = n_treatment / n_control
        
        treatment_goals = 0
        control_goals = 0
        score_previous = np.finfo(np.float32).min
        
        tmp_n_samples = 1
        tmp_treatment_goals = 0
        tmp_control_goals = 0
        
        for item_class, item_score, item_group in zip(data_class, data_score, data_group):
            if score_previous != item_score:
                # 新しいスコア値の場合、蓄積したサンプルを処理
                for i in range(1, tmp_n_samples + 1):
                    tmp_qini_point = (
                        (treatment_goals + i * tmp_treatment_goals / tmp_n_samples) -
                        (control_goals + i * tmp_control_goals / tmp_n_samples) * n_factor
                    )
                    qini_points.append(tmp_qini_point)
                
                # カウンターを更新
                treatment_goals += tmp_treatment_goals
                control_goals += tmp_control_goals
                
                # カウンターをリセット
                tmp_n_samples = 0
                tmp_treatment_goals = 0
                tmp_control_goals = 0
                score_previous = item_score
            
            # アイテムをカウンターに追加
            tmp_n_samples += 1
            tmp_treatment_goals += int(item_group) * item_class
            tmp_control_goals += int(~item_group) * item_class
        
        # 残りのサンプルを処理
        for i in range(1, tmp_n_samples + 1):
            tmp_qini_point = (
                (treatment_goals + i * tmp_treatment_goals / tmp_n_samples) -
                (control_goals + i * tmp_control_goals / tmp_n_samples) * n_factor
            )
            qini_points.append(tmp_qini_point)
        
        return np.array(qini_points)


class UpliftEvaluator:
    """アップリフト評価指標の統合クラス"""
    
    def __init__(self):
        """初期化"""
        self.qini_calculator = QINICalculator()
    
    def evaluate_cate_with_qini(
        self,
        y_true: np.ndarray,
        y_pred_cate: np.ndarray,
        treatment: np.ndarray,
        true_cate: Optional[np.ndarray] = None,
        y_counterfactual: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        CATE推定をQINI係数を含む複数指標で評価
        
        Args:
            y_true: 観測結果
            y_pred_cate: 予測CATE
            treatment: 処置フラグ
            true_cate: 真のCATE（利用可能な場合）
            y_counterfactual: 反実仮想結果（人工データ）
            
        Returns:
            評価指標の辞書
        """
        evaluation_results = {}
        
        # 基本的なCATE評価指標（MSE、RMSE等）
        if true_cate is not None:
            evaluation_results.update(self._calculate_basic_cate_metrics(true_cate, y_pred_cate))
        
        # QINI係数の計算
        try:
            qini_coeff = self.qini_calculator.calculate_qini_coefficient(
                y_true, y_pred_cate, treatment, y_counterfactual
            )
            evaluation_results['qini_coefficient'] = qini_coeff
        except Exception as e:
            warnings.warn(f"QINI係数の計算に失敗しました: {e}")
            evaluation_results['qini_coefficient'] = np.nan
        
        # アップリフト関連指標
        if y_counterfactual is not None:
            uplift_metrics = self._calculate_uplift_metrics(
                y_true, y_pred_cate, treatment, y_counterfactual
            )
            evaluation_results.update(uplift_metrics)
        
        return evaluation_results
    
    def _calculate_basic_cate_metrics(
        self, 
        true_cate: np.ndarray, 
        pred_cate: np.ndarray
    ) -> Dict[str, float]:
        """基本的なCATE評価指標を計算"""
        mse = np.mean((true_cate - pred_cate) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_cate - pred_cate))
        bias = np.mean(pred_cate - true_cate)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'bias': bias
        }
    
    def _calculate_uplift_metrics(
        self,
        y_observed: np.ndarray,
        y_pred_cate: np.ndarray,
        treatment: np.ndarray,
        y_counterfactual: np.ndarray
    ) -> Dict[str, float]:
        """アップリフト関連指標を計算"""
        # 真のCATEを計算
        true_cate = self.qini_calculator._calculate_true_cate(
            y_observed, y_counterfactual, treatment
        )
        
        # 処置群・対照群の転換率
        treatment_mask = treatment == 1
        control_mask = treatment == 0
        
        treatment_conversion_rate = np.mean(y_observed[treatment_mask]) if np.any(treatment_mask) else 0.0
        control_conversion_rate = np.mean(y_observed[control_mask]) if np.any(control_mask) else 0.0
        
        # 真のアップリフト効果
        true_uplift = treatment_conversion_rate - control_conversion_rate
        
        # 真のCATEの平均
        true_cate_mean = np.mean(true_cate)
        
        return {
            'treatment_conversion_rate': treatment_conversion_rate,
            'control_conversion_rate': control_conversion_rate,
            'true_uplift': true_uplift,
            'true_cate_mean': true_cate_mean,
            'cate_correlation': np.corrcoef(true_cate, y_pred_cate)[0, 1] if len(set(true_cate)) > 1 else 0.0
        }


def calculate_qini_coefficient(
    y_true: np.ndarray,
    y_pred_cate: np.ndarray,
    treatment: np.ndarray,
    y_counterfactual: Optional[np.ndarray] = None
) -> float:
    """
    QINI係数計算の便利関数
    
    Args:
        y_true: 観測結果
        y_pred_cate: 予測CATE
        treatment: 処置フラグ
        y_counterfactual: 反実仮想結果（オプション）
        
    Returns:
        QINI係数
    """
    calculator = QINICalculator()
    return calculator.calculate_qini_coefficient(y_true, y_pred_cate, treatment, y_counterfactual)


def evaluate_cate_with_uplift_metrics(
    y_true: np.ndarray,
    y_pred_cate: np.ndarray,
    treatment: np.ndarray,
    true_cate: Optional[np.ndarray] = None,
    y_counterfactual: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    CATE推定の包括的評価関数
    
    Args:
        y_true: 観測結果
        y_pred_cate: 予測CATE
        treatment: 処置フラグ
        true_cate: 真のCATE（利用可能な場合）
        y_counterfactual: 反実仮想結果（人工データ）
        
    Returns:
        評価指標の辞書
    """
    evaluator = UpliftEvaluator()
    return evaluator.evaluate_cate_with_qini(
        y_true, y_pred_cate, treatment, true_cate, y_counterfactual
    )


# テスト用の関数
def test_qini_calculation():
    """QINI計算のテスト"""
    np.random.seed(42)
    
    # テストデータ生成
    n_samples = 1000
    treatment = np.random.binomial(1, 0.5, n_samples)
      # 人工的なCATEデータ
    true_cate = np.random.normal(0.05, 0.1, n_samples)  # より小さな効果サイズ
    
    # 観測結果（処置効果あり）
    y_control = np.random.binomial(1, 0.3, n_samples)  # ベース転換率
    # 確率を0-1の範囲に制限
    treatment_prob = np.clip(0.3 + np.maximum(0, true_cate), 0.0, 1.0)
    y_treatment = np.random.binomial(1, treatment_prob, n_samples)
    
    y_observed = np.where(treatment == 1, y_treatment, y_control)
    y_counterfactual = np.where(treatment == 1, y_control, y_treatment)
    
    # 予測CATE（ノイズ付き）
    y_pred_cate = true_cate + np.random.normal(0, 0.1, n_samples)
    
    # QINI係数計算
    qini_coeff = calculate_qini_coefficient(y_observed, y_pred_cate, treatment, y_counterfactual)
    
    # 包括的評価
    evaluation_results = evaluate_cate_with_uplift_metrics(
        y_observed, y_pred_cate, treatment, true_cate, y_counterfactual
    )
    
    print("QINI係数テスト結果:")
    print(f"QINI係数: {qini_coeff:.4f}")
    print("\n評価指標:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")
    
    return qini_coeff, evaluation_results


if __name__ == "__main__":
    test_qini_calculation()
