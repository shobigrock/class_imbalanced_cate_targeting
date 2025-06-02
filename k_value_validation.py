#!/usr/bin/env python3
"""
k値の検証実験 - k=3での性能評価
論文: "Uplift Modeling with High Class Imbalance" (Nyberg et al., 2021)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import generate_synthetic_data
from shared.imbalance_handler import UpliftUndersampler
from shared.model import train_causal_tree, estimate_cate
from evaluation.evaluation import evaluate_cate_estimation

def create_imbalanced_dataset(n_samples=3000, positive_rate=0.05, random_seed=42):
    """
    不均衡データセットを作成
    
    Args:
        n_samples (int): サンプル数
        positive_rate (float): 正例率
        random_seed (int): 乱数シード
    
    Returns:
        X, T, Y, true_cate: 特徴量、処置変数、目的変数、真のCATE
    """
    np.random.seed(random_seed)
    
    # 特徴量生成（5次元）
    X = np.random.randn(n_samples, 5)
    
    # 処置変数（バランス取れた割り当て）
    T = np.random.binomial(1, 0.5, n_samples)
    
    # 真のCATE効果（特徴量に依存）
    true_cate = 0.02 + 0.01 * X[:, 0] + 0.005 * X[:, 1]  # 小さな効果
    
    # ベースライン確率（低い変換率）
    base_prob = positive_rate
    
    # 実際の変換確率（処置効果を含む）
    prob_control = np.clip(base_prob + 0.01 * X[:, 0], 0, 1)
    prob_treatment = np.clip(prob_control + true_cate, 0, 1)
    
    # 目的変数生成
    Y = np.zeros(n_samples)
    control_mask = T == 0
    treatment_mask = T == 1
    
    Y[control_mask] = np.random.binomial(1, prob_control[control_mask])
    Y[treatment_mask] = np.random.binomial(1, prob_treatment[treatment_mask])
    
    # DataFrameに変換
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    return X_df, T, Y, true_cate

def run_k_value_experiment(k_values=[1.5, 2.0, 3.0, 4.0, 5.0], n_samples=3000, positive_rate=0.05):
    """
    異なるk値での性能比較実験
    
    Args:
        k_values (list): 検証するk値のリスト
        n_samples (int): サンプル数
        positive_rate (float): 正例率
    """
    print("=" * 70)
    print("K値検証実験 - アンダーサンプリング効果の比較")
    print("=" * 70)
    print(f"実験設定:")
    print(f"- サンプル数: {n_samples}")
    print(f"- 目標正例率: {positive_rate:.3f}")
    print(f"- 検証するk値: {k_values}")
    print()
    
    # データセット作成
    print("Step 1: 不均衡データセット作成中...")
    X, T, Y, true_cate = create_imbalanced_dataset(
        n_samples=n_samples, 
        positive_rate=positive_rate, 
        random_seed=42
    )
    
    print(f"データセット情報:")
    print(f"- 総サンプル数: {len(X)}")
    print(f"- 実際の正例率: {np.mean(Y):.4f}")
    print(f"- 処置群正例率: {np.mean(Y[T == 1]):.4f}")
    print(f"- 対照群正例率: {np.mean(Y[T == 0]):.4f}")
    print()
    
    # データ分割
    X_train, X_test, T_train, T_test, Y_train, Y_test, true_cate_train, true_cate_test = train_test_split(
        X, T, Y, true_cate, test_size=0.3, random_state=42, stratify=T
    )
    
    print(f"データ分割:")
    print(f"- 訓練データ: {len(X_train)} サンプル")
    print(f"- テストデータ: {len(X_test)} サンプル")
    print()
    
    results = []
    
    # ベースライン（アンダーサンプリングなし）
    print("Step 2: ベースライン実験（アンダーサンプリングなし）")
    print("-" * 50)
    
    # ベースラインモデル学習
    baseline_model = train_causal_tree(
        X_train.values, T_train, Y_train,
        min_samples_leaf=20, max_depth=4, random_state=42
    )
    
    # ベースライン予測
    baseline_pred = estimate_cate(baseline_model, X_test.values)
    baseline_eval = evaluate_cate_estimation(true_cate_test, baseline_pred)
    
    results.append({
        'k_value': 'Baseline',
        'samples_after_undersampling': len(X_train),
        'positive_rate_after': np.mean(Y_train),
        'rmse': baseline_eval['rmse'],
        'bias': baseline_eval['bias'],
        'mae': baseline_eval['mae']
    })
    
    print(f"ベースライン結果:")
    print(f"- RMSE: {baseline_eval['rmse']:.4f}")
    print(f"- Bias: {baseline_eval['bias']:.4f}")
    print(f"- MAE: {baseline_eval['mae']:.4f}")
    print()
    
    # 各k値での実験
    for k in k_values:
        print(f"Step 3: k={k} でのアンダーサンプリング実験")
        print("-" * 50)
        
        try:
            # アンダーサンプリング実行
            undersampler = UpliftUndersampler(k=k, random_state=42)
            X_train_resampled, T_train_resampled, Y_train_resampled = undersampler.fit_resample(
                X_train, T_train, Y_train
            )
            
            print(f"アンダーサンプリング結果:")
            print(f"- 元サンプル数: {len(X_train)} → 新サンプル数: {len(X_train_resampled)}")
            print(f"- 元正例率: {np.mean(Y_train):.4f} → 新正例率: {np.mean(Y_train_resampled):.4f}")
            
            # モデル学習
            model = train_causal_tree(
                X_train_resampled.values, T_train_resampled, Y_train_resampled,
                min_samples_leaf=20, max_depth=4, random_state=42
            )
            
            # 予測と補正
            pred_cate = estimate_cate(model, X_test.values)
            pred_cate_corrected = undersampler.correct_predictions(pred_cate)
            
            # 評価
            eval_results = evaluate_cate_estimation(true_cate_test, pred_cate_corrected)
            
            results.append({
                'k_value': k,
                'samples_after_undersampling': len(X_train_resampled),
                'positive_rate_after': np.mean(Y_train_resampled),
                'rmse': eval_results['rmse'],
                'bias': eval_results['bias'],
                'mae': eval_results['mae']
            })
            
            print(f"k={k} 結果:")
            print(f"- RMSE: {eval_results['rmse']:.4f}")
            print(f"- Bias: {eval_results['bias']:.4f}")
            print(f"- MAE: {eval_results['mae']:.4f}")
            
            # ベースラインとの比較
            rmse_improvement = (baseline_eval['rmse'] - eval_results['rmse']) / baseline_eval['rmse'] * 100
            bias_improvement = (abs(baseline_eval['bias']) - abs(eval_results['bias'])) / abs(baseline_eval['bias']) * 100
            
            print(f"ベースラインからの改善:")
            print(f"- RMSE改善: {rmse_improvement:+.2f}%")
            print(f"- Bias改善: {bias_improvement:+.2f}%")
            print()
            
        except Exception as e:
            print(f"k={k} でエラーが発生: {e}")
            print()
            continue
    
    # 結果まとめ
    print("=" * 70)
    print("実験結果まとめ")
    print("=" * 70)
    
    # 結果テーブル
    print(f"{'K値':<10} {'サンプル数':<10} {'正例率':<10} {'RMSE':<10} {'Bias':<10} {'MAE':<10}")
    print("-" * 70)
    
    for result in results:
        k_str = str(result['k_value'])
        samples = result['samples_after_undersampling']
        pos_rate = result['positive_rate_after']
        rmse = result['rmse']
        bias = result['bias']
        mae = result['mae']
        
        print(f"{k_str:<10} {samples:<10} {pos_rate:<10.4f} {rmse:<10.4f} {bias:<10.4f} {mae:<10.4f}")
    
    # 最適k値の推奨
    print()
    print("=" * 70)
    print("推奨事項")
    print("=" * 70)
    
    # ベースライン除外してRMSEが最小のk値を見つける
    k_results = [r for r in results if r['k_value'] != 'Baseline']
    if k_results:
        best_rmse_result = min(k_results, key=lambda x: x['rmse'])
        best_bias_result = min(k_results, key=lambda x: abs(x['bias']))
        
        print(f"最小RMSE: k={best_rmse_result['k_value']} (RMSE: {best_rmse_result['rmse']:.4f})")
        print(f"最小Bias: k={best_bias_result['k_value']} (Bias: {best_bias_result['bias']:.4f})")
        
        # k=3の結果をハイライト
        k3_result = next((r for r in k_results if r['k_value'] == 3.0), None)
        if k3_result:
            print()
            print(f"📊 k=3の性能:")
            print(f"- RMSE: {k3_result['rmse']:.4f}")
            print(f"- Bias: {k3_result['bias']:.4f}")
            print(f"- データ削減率: {(1 - k3_result['samples_after_undersampling'] / len(X_train)) * 100:.1f}%")
            
            baseline_rmse = next(r['rmse'] for r in results if r['k_value'] == 'Baseline')
            improvement = (baseline_rmse - k3_result['rmse']) / baseline_rmse * 100
            print(f"- ベースラインからのRMSE改善: {improvement:+.2f}%")
    
    print("\n✅ K値検証実験完了!")
    return results

def quick_k3_test():
    """k=3での簡単なテスト"""
    print("=" * 50)
    print("K=3 クイックテスト")
    print("=" * 50)
    
    # 小さなデータセットで迅速テスト
    X, T, Y, true_cate = create_imbalanced_dataset(n_samples=1000, positive_rate=0.03, random_seed=42)
    
    print(f"テストデータ:")
    print(f"- サンプル数: {len(X)}")
    print(f"- 正例率: {np.mean(Y):.4f}")
    
    # k=3でアンダーサンプリング
    undersampler = UpliftUndersampler(k=3.0, random_state=42)
    X_resampled, T_resampled, Y_resampled = undersampler.fit_resample(X, T, Y)
    
    print(f"\nアンダーサンプリング後:")
    print(f"- サンプル数: {len(X_resampled)}")
    print(f"- 正例率: {np.mean(Y_resampled):.4f}")
    print(f"- データ保持率: {len(X_resampled) / len(X) * 100:.1f}%")
    
    print("\n✅ K=3テスト完了!")

if __name__ == "__main__":
    print("アンダーサンプリング K値検証プログラム")
    print("=" * 70)
    
    # まずクイックテスト
    quick_k3_test()
    print("\n")
    
    # 詳細なk値比較実験
    try:
        results = run_k_value_experiment(
            k_values=[2.0, 3.0, 4.0, 5.0],
            n_samples=2000,  # より小さなサンプルサイズで高速化
            positive_rate=0.04
        )
        
        print("\n🎉 全ての実験が正常に完了しました!")
        
    except Exception as e:
        print(f"\n❌ 実験中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
