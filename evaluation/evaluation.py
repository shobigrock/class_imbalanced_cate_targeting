"""
CATE推定の評価機能

真のCATEと推定CATEの比較、およびQINI係数による評価を提供します。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# QINI計算機能をインポート
try:
    from ..shared.qini_metrics import QINICalculator, UpliftEvaluator
except ImportError:
    # 相対インポートが失敗した場合の代替
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from shared.qini_metrics import QINICalculator, UpliftEvaluator


class CATEEvaluator:
    """CATE推定の総合評価クラス"""
    
    def __init__(self, enable_qini: bool = False):
        """
        Args:
            enable_qini: QINI係数の計算を有効にするか
        """
        self.enable_qini = enable_qini
        self.qini_calculator = QINICalculator() if enable_qini else None
        self.uplift_evaluator = UpliftEvaluator() if enable_qini else None
    
    def evaluate_cate_estimation(
        self, 
        true_cate: np.ndarray, 
        pred_cate: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        treatment: Optional[np.ndarray] = None,
        y_counterfactual: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        真のCATEと推定されたCATEを比較し、評価指標を計算する
        
        Args:
            true_cate: 真のCATE値
            pred_cate: 予測されたCATE値  
            y_true: 観測結果（QINI計算用）
            treatment: 処置フラグ（QINI計算用）
            y_counterfactual: 反実仮想の結果（QINI計算用）
            verbose: 詳細な出力を行うか
            
        Returns:
            評価指標の辞書
        """
        if len(true_cate) != len(pred_cate):
            raise ValueError("Length of true_cate and pred_cate must be the same.")

        # 基本的な評価指標
        mse = mean_squared_error(true_cate, pred_cate)
        mae = mean_absolute_error(true_cate, pred_cate)
        rmse = np.sqrt(mse)
        bias = np.mean(pred_cate - true_cate)
        
        # R²スコア（決定係数）
        try:
            r2 = r2_score(true_cate, pred_cate)
        except ValueError:
            # 分散が0の場合など
            r2 = float('nan')
        
        # 結果辞書
        results = {
            "mse": mse,
            "mae": mae, 
            "rmse": rmse,
            "bias": bias,
            "r2_score": r2
        }
        
        # QINI係数の計算（有効で必要なデータがある場合）
        if (self.enable_qini and self.qini_calculator is not None and 
            y_true is not None and treatment is not None):
            
            try:
                qini_coeff = self.qini_calculator.calculate_qini_coefficient(
                    y_true, pred_cate, treatment, y_counterfactual
                )
                results["qini_coefficient"] = qini_coeff
                  # 追加のアップリフト評価指標
                if self.uplift_evaluator is not None:
                    uplift_metrics = self.uplift_evaluator.evaluate_cate_with_qini(
                        y_true, pred_cate, treatment, true_cate, y_counterfactual
                    )
                    results.update(uplift_metrics)
                    
            except Exception as e:
                warnings.warn(f"QINI計算でエラーが発生しました: {e}")
                results["qini_coefficient"] = float('nan')

        # 詳細出力
        if verbose:
            self._print_evaluation_results(results)

        return results
    
    def _print_evaluation_results(self, results: Dict[str, Any]):
        """評価結果を詳細に出力"""
        print("📊 CATE Estimation Evaluation Results:")
        print(f"  Mean Squared Error (MSE): {results['mse']:.6f}")
        print(f"  Mean Absolute Error (MAE): {results['mae']:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {results['rmse']:.6f}")
        print(f"  Bias: {results['bias']:.6f}")
        print(f"  R² Score: {results['r2_score']:.6f}")
        
        if "qini_coefficient" in results:
            print(f"  QINI Coefficient: {results['qini_coefficient']:.6f}")
        
        if "auuc" in results:
            print(f"  AUUC (Area Under Uplift Curve): {results['auuc']:.6f}")


def evaluate_cate_estimation(true_cate, pred_cate):
    """
    後方互換性のための関数
    
    真のCATEと推定されたCATEを比較し、評価指標を計算する。
    """
    evaluator = CATEEvaluator(enable_qini=False)
    return evaluator.evaluate_cate_estimation(true_cate, pred_cate, verbose=True)

if __name__ == '__main__':
    # このファイル単体で実行した場合のテスト用コード
    print("🧪 Testing CATE Evaluation Module")
    
    # テストデータ
    true_effects = np.array([1.0, 1.5, 0.5, 2.0, 1.2])
    pred_effects = np.array([0.9, 1.3, 0.7, 2.2, 1.1])
    
    print("\n1. Basic CATE Evaluation:")
    results = evaluate_cate_estimation(true_effects, pred_effects)
    print("Evaluation results:", results)
    
    print("\n2. Advanced CATE Evaluation with QINI:")
    # 人工的なQINI用データ
    y_true = np.array([1, 0, 1, 1, 0])
    treatment = np.array([1, 0, 1, 0, 1])
    y_counterfactual = np.array([0, 1, 0, 0, 1])  # 反実仮想
    
    evaluator = CATEEvaluator(enable_qini=True)
    advanced_results = evaluator.evaluate_cate_estimation(
        true_effects, pred_effects, y_true, treatment, y_counterfactual
    )
    print("Advanced evaluation results:", advanced_results)