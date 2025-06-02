# evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_cate_estimation(true_cate, pred_cate):
    """
    真のCATEと推定されたCATEを比較し、評価指標を計算する。
    """
    if len(true_cate) != len(pred_cate):
        raise ValueError("Length of true_cate and pred_cate must be the same.")

    mse = mean_squared_error(true_cate, pred_cate)
    rmse = np.sqrt(mse)
    bias = np.mean(pred_cate - true_cate) # 推定のバイアス

    print(f"  Mean Squared Error (MSE) of CATE: {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE) of CATE: {rmse:.4f}")
    print(f"  Bias of CATE estimation: {bias:.4f}")

    # その他の評価指標（例：PEHE - Precision in Estimating Heterogeneous Effect）
    # PEHE = sqrt(MSE(true_cate - pred_cate)) と同じ
    # policy_value など、より高度な評価も考えられる

    return {"mse": mse, "rmse": rmse, "bias": bias}

if __name__ == '__main__':
    # このファイル単体で実行した場合のテスト用コード
    true_effects = np.array([1.0, 1.5, 0.5, 2.0, 1.2])
    pred_effects = np.array([0.9, 1.3, 0.7, 2.2, 1.1])
    print("Evaluating CATE estimation (example):")
    results = evaluate_cate_estimation(true_effects, pred_effects)
    print("Evaluation results:", results)