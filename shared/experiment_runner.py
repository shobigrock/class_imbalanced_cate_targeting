"""
実験実行ロジックを管理するモジュール
"""
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# 自作モジュールのインポート
from data.data_generator import generate_synthetic_data
from shared.model import train_causal_tree, estimate_cate, create_model
from shared.imbalance_handler import UpliftUndersampler
from shared.experiment_config import ExperimentConfig
from evaluation.evaluation import evaluate_cate_estimation, CATEEvaluator


class ExperimentRunner:
    """実験実行を管理するクラス"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: 実験設定
        """
        self.config = config
        self.results = {}
        self.data_cache = {}
    
    def run_single_experiment(self) -> Dict[str, Any]:
        """単一の実験を実行"""
        start_time = time.time()
        
        print(f"Starting CATE estimation experiment...")
        print(f"Configuration:\n{self.config.summary()}")
        
        # 1. データ生成
        X_df, T, Y, true_cate, train_data, test_data = self._prepare_data()
        
        # 2. アンダーサンプリング（オプション）
        X_train_df, T_train, Y_train = self._apply_undersampling(
            train_data["X"], train_data["T"], train_data["Y"]
        )
        
        # 3. モデル学習
        model = self._train_model(X_train_df, T_train, Y_train)
        
        # 4. CATE推定
        pred_cate = self._predict_cate(model, test_data["X"])
        
        # 5. 評価
        evaluation_results = self._evaluate_predictions(test_data["true_cate"], pred_cate, test_data)
        
        # 実行時間計算
        runtime = time.time() - start_time
        
        # 結果をまとめる
        results = {
            "config": self.config.to_dict(),
            "data_stats": {
                "total_samples": len(X_df),
                "train_samples": len(X_train_df),
                "test_samples": len(test_data["X"]),
                "original_positive_rate": np.mean(Y),
                "train_positive_rate": np.mean(Y_train)
            },
            "evaluation": evaluation_results,
            "runtime": runtime,
            "predictions": {
                "predicted_cate": pred_cate,
                "true_cate": test_data["true_cate"]
            }
        }
        
        self.results = results
        return results
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
        """データの準備"""
        print("\nStep 1: Generating synthetic data...")
        
        # データ生成
        X_data, T_data, Y_data, true_cate_data, feature_names_list = generate_synthetic_data(
            n_samples=self.config.n_samples,
            random_seed=self.config.random_seed,
            reward_type=self.config.reward_type
        )
        
        # DataFrame変換
        X_df = self._convert_to_dataframe(X_data, feature_names_list)
        
        print(f"Data generated: {X_df.shape[0]} samples, {X_df.shape[1]} features")
        print(f"Original positive rate: {np.mean(Y_data):.4f}")
        
        # データ分割
        split_result = train_test_split(
            X_df, T_data, Y_data, true_cate_data,
            test_size=self.config.test_size,
            random_state=self.config.random_seed
        )
        
        X_train_df, X_test_df, T_train, T_test, Y_train, Y_test, true_cate_train, true_cate_test = split_result
        
        train_data = {
            "X": X_train_df,
            "T": T_train,
            "Y": Y_train,
            "true_cate": true_cate_train
        }
        
        test_data = {
            "X": X_test_df,
            "T": T_test,
            "Y": Y_test,
            "true_cate": true_cate_test
        }
        
        print(f"Data split: {len(X_train_df)} train, {len(X_test_df)} test")
        
        return X_df, T_data, Y_data, true_cate_data, train_data, test_data
    
    def _convert_to_dataframe(self, X_data, feature_names_list) -> pd.DataFrame:
        """データをDataFrameに変換"""
        if isinstance(X_data, pd.DataFrame):
            return X_data
        
        if not feature_names_list:
            feature_names_list = [f'feature_{i}' for i in range(X_data.shape[1])]
        
        return pd.DataFrame(X_data, columns=feature_names_list)
    
    def _apply_undersampling(self, X_train_df: pd.DataFrame, T_train: np.ndarray, Y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """アンダーサンプリングの適用"""
        if not self.config.use_undersampling:
            print("\nStep 2: Skipping undersampling...")
            return X_train_df, T_train, Y_train
        
        print(f"\nStep 2: Applying undersampling with k={self.config.k_factor}...")
        
        undersampler = UpliftUndersampler(
            k=self.config.k_factor,
            random_state=self.config.random_seed
        )
        
        X_resampled, T_resampled, Y_resampled = undersampler.fit_resample(X_train_df, T_train, Y_train)
        
        print(f"After undersampling: {len(X_resampled)} samples")
        print(f"Positive rate change: {np.mean(Y_train):.4f} → {np.mean(Y_resampled):.4f}")
          # undersamplerを保存（予測補正用）
        self.undersampler = undersampler
        
        return X_resampled, T_resampled, Y_resampled
    
    def _train_model(self, X_train_df: pd.DataFrame, T_train: np.ndarray, Y_train: np.ndarray):
        """モデル学習"""
        print(f"\nStep 3: Training {self.config.model_type} model...")
        
        if self.config.model_type == "causal_tree":
            # 従来のCausal Tree実装
            model = train_causal_tree(
                X_train_df.values,
                T_train,
                Y_train,
                **self.config.model_params
            )
        else:
            # 新しいモデル実装
            model = create_model(self.config.model_type, **self.config.model_params)
            
            # モデルタイプに応じた学習
            if self.config.model_type == "causal_forest":
                model.fit(X_train_df, T_train.astype(float), Y_train)
            else:
                model.fit(X_train_df, T_train, Y_train)
        
        print("Model training complete.")
        return model
    
    def _predict_cate(self, model, X_test_df: pd.DataFrame) -> np.ndarray:
        """CATE推定"""
        print(f"\nStep 4: Estimating CATE on test data...")
        
        if self.config.model_type == "causal_tree":
            pred_cate = estimate_cate(model, X_test_df.values)
        else:
            pred_cate = model.predict(X_test_df)
        
        # アンダーサンプリング補正
        if self.config.use_undersampling and hasattr(self, 'undersampler'):
            pred_cate = self.undersampler.correct_predictions(pred_cate)
            print("Applied undersampling correction to predictions.")
        
        print(f"CATE estimation for {len(pred_cate)} test samples complete.")
        
        return pred_cate
    
    def _evaluate_predictions(self, true_cate: np.ndarray, pred_cate: np.ndarray, 
                            test_data: Dict[str, Any] = None) -> Dict[str, float]:
        """予測結果の評価"""
        print("\nStep 5: Evaluating CATE estimation...")
        
        # QINI評価が有効かチェック
        enable_qini = getattr(self.config, 'enable_qini', False) and test_data is not None
        
        if enable_qini:
            # 拡張評価（QINI含む）
            evaluator = CATEEvaluator(enable_qini=True)
            
            # 反実仮想データがあるかチェック（人工データの場合）
            y_counterfactual = None
            if hasattr(self.config, 'data_source') and self.config.data_source == "synthetic":
                # 人工データの場合、反実仮想を計算
                y_counterfactual = self._generate_counterfactual_outcomes(test_data)
            
            evaluation_results = evaluator.evaluate_cate_estimation(
                true_cate, 
                pred_cate,
                y_true=test_data.get("Y") if test_data else None,
                treatment=test_data.get("T") if test_data else None,
                y_counterfactual=y_counterfactual,
                verbose=True
            )
        else:
            # 基本評価のみ
            evaluation_results = evaluate_cate_estimation(true_cate, pred_cate)
        
        print("Evaluation complete.")
        return evaluation_results
    
    def _generate_counterfactual_outcomes(self, test_data: Dict[str, Any]) -> np.ndarray:
        """反実仮想の結果を生成（人工データ用）"""
        try:
            # 処置を反転させて反実仮想を計算
            X_test = test_data["X"]
            T_test = test_data["T"]
            Y_test = test_data["Y"]
            true_cate = test_data["true_cate"]
            
            # 反実仮想 = 観測結果 - (処置 * 真のCATE - (1-処置) * 真のCATE)
            # = 観測結果 - 真のCATE * (2*処置 - 1)
            y_counterfactual = Y_test - true_cate * (2 * T_test - 1)
            
            return y_counterfactual
            
        except Exception as e:
            print(f"Warning: Could not generate counterfactual outcomes: {e}")
            return None


class ComparisonExperiment:
    """比較実験を管理するクラス"""
    
    def __init__(self, base_config: ExperimentConfig):
        """
        Args:
            base_config: ベース設定
        """
        self.base_config = base_config
        self.results = {}
    
    def run_undersampling_comparison(self) -> Dict[str, Any]:
        """アンダーサンプリングありなしの比較実験"""
        print("=" * 60)
        print("UNDERSAMPLING COMPARISON EXPERIMENT")
        print("=" * 60)
        
        # ベースライン実験（アンダーサンプリングなし）
        baseline_config = ExperimentConfig(**self.base_config.to_dict())
        baseline_config.use_undersampling = False
        
        print("\n" + "=" * 40)
        print("BASELINE (No Undersampling)")
        print("=" * 40)
        
        baseline_runner = ExperimentRunner(baseline_config)
        baseline_results = baseline_runner.run_single_experiment()
        
        # アンダーサンプリング実験
        undersampling_config = ExperimentConfig(**self.base_config.to_dict())
        undersampling_config.use_undersampling = True
        
        print("\n" + "=" * 40)
        print("UNDERSAMPLING EXPERIMENT")
        print("=" * 40)
        
        undersampling_runner = ExperimentRunner(undersampling_config)
        undersampling_results = undersampling_runner.run_single_experiment()
        
        # 結果をまとめる
        comparison_results = {
            "baseline": baseline_results,
            "undersampling": undersampling_results,
            "comparison": self._calculate_improvements(baseline_results, undersampling_results)
        }
        
        self.results = comparison_results
        return comparison_results
    
    def _calculate_improvements(self, baseline: Dict, undersampling: Dict) -> Dict[str, float]:
        """改善率の計算"""
        baseline_eval = baseline["evaluation"]
        undersampling_eval = undersampling["evaluation"]
        
        rmse_improvement = (baseline_eval['rmse'] - undersampling_eval['rmse']) / baseline_eval['rmse'] * 100
        bias_improvement = (abs(baseline_eval['bias']) - abs(undersampling_eval['bias'])) / abs(baseline_eval['bias']) * 100
        
        return {
            "rmse_improvement_percent": rmse_improvement,
            "bias_improvement_percent": bias_improvement,
            "baseline_rmse": baseline_eval['rmse'],
            "undersampling_rmse": undersampling_eval['rmse'],
            "baseline_bias": baseline_eval['bias'],
            "undersampling_bias": undersampling_eval['bias']
        }
