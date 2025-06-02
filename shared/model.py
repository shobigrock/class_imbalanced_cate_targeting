# model.py
# CATE推定のための統一的なインターフェースとメタラーナーの実装
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

# CausalMLとEconMLからのインポート
from causalml.inference.tree import CausalTreeRegressor
from econml.dml import CausalForestDML


class CATEEstimator(ABC):
    """
    CATE推定器の抽象基底クラス
    main.pyの既存インターフェースとの互換性を保つ
    """
    
    def __init__(self, random_state: int = 123):
        self.random_state = random_state
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'CATEEstimator':
        """
        モデルを学習する
        
        Args:
            X: 特徴量 (n_samples, n_features)
            T: 処置変数 (n_samples,) - 0/1のバイナリ
            Y: 結果変数 (n_samples,)
        
        Returns:
            学習済みモデル
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        CATEを予測する
        
        Args:
            X: 特徴量 (n_samples, n_features)
        
        Returns:
            CATE予測値 (n_samples,)
        """
        pass


class CausalTreeWrapper(CATEEstimator):
    """
    CausalMLのCausalTreeRegressorのラッパークラス
    既存のtrain_causal_tree/estimate_cate関数との互換性を保つ
    """
    
    def __init__(self, min_samples_leaf: int = 10, max_depth: int = 5, 
                 random_state: int = 123):
        super().__init__(random_state)
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.model = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'CausalTreeWrapper':
        self.model = CausalTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X=X, treatment=T, y=Y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class SLearner(CATEEstimator):
    """
    S-Learner (Single Learner)
    処置変数を特徴量として含めた単一のモデルでCATE推定
    """
    
    def __init__(self, base_learner=None, random_state: int = 123):
        super().__init__(random_state)
        if base_learner is None:
            self.base_learner = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        else:
            self.base_learner = base_learner
        self.model = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'SLearner':
        # 特徴量に処置変数を追加
        X_augmented = np.column_stack([X, T])
        self.model = clone(self.base_learner)
        self.model.fit(X_augmented, Y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # T=1とT=0の場合の予測値を計算
        X_treated = np.column_stack([X, np.ones(X.shape[0])])
        X_control = np.column_stack([X, np.zeros(X.shape[0])])
        
        y1_pred = self.model.predict(X_treated)
        y0_pred = self.model.predict(X_control)
        
        return y1_pred - y0_pred


class TLearner(CATEEstimator):
    """
    T-Learner (Two Learner)
    処置群と対照群で別々のモデルを学習してCATE推定
    """
    
    def __init__(self, base_learner=None, random_state: int = 123):
        super().__init__(random_state)
        if base_learner is None:
            base_learner = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        
        self.model_treated = clone(base_learner)
        self.model_control = clone(base_learner)
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'TLearner':
        # 処置群と対照群に分割
        treated_idx = T == 1
        control_idx = T == 0
        
        # 各群で別々にモデルを学習
        if np.sum(treated_idx) > 0:
            self.model_treated.fit(X[treated_idx], Y[treated_idx])
        if np.sum(control_idx) > 0:
            self.model_control.fit(X[control_idx], Y[control_idx])
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 各モデルで予測
        y1_pred = self.model_treated.predict(X)
        y0_pred = self.model_control.predict(X)
        
        return y1_pred - y0_pred


class DRLearner(CATEEstimator):
    """
    DR-Learner (Doubly Robust Learner)
    傾向スコアと結果回帰の両方を使用したメタラーナー
    """
    
    def __init__(self, outcome_learner=None, propensity_learner=None, 
                 random_state: int = 123):
        super().__init__(random_state)
        
        if outcome_learner is None:
            self.outcome_learner = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        else:
            self.outcome_learner = outcome_learner
            
        if propensity_learner is None:
            self.propensity_learner = xgb.XGBClassifier(random_state=random_state, n_estimators=100)
        else:
            self.propensity_learner = propensity_learner
        
        self.outcome_model_treated = None
        self.outcome_model_control = None
        self.propensity_model = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DRLearner':
        # 傾向スコアモデルの学習
        self.propensity_model = clone(self.propensity_learner)
        self.propensity_model.fit(X, T)
        
        # 結果回帰モデルの学習（処置群・対照群別々）
        treated_idx = T == 1
        control_idx = T == 0
        
        self.outcome_model_treated = clone(self.outcome_learner)
        self.outcome_model_control = clone(self.outcome_learner)
        
        if np.sum(treated_idx) > 0:
            self.outcome_model_treated.fit(X[treated_idx], Y[treated_idx])
        if np.sum(control_idx) > 0:
            self.outcome_model_control.fit(X[control_idx], Y[control_idx])
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 傾向スコアの予測
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # 結果回帰の予測
        mu1_pred = self.outcome_model_treated.predict(X)
        mu0_pred = self.outcome_model_control.predict(X)
        
        # DR推定値の計算（簡略化版）
        cate_pred = mu1_pred - mu0_pred
        
        return cate_pred


class RLearner(CATEEstimator):
    """
    R-Learner (Robinson Learner)
    残差に基づくメタラーナー
    """
    
    def __init__(self, outcome_learner=None, propensity_learner=None, 
                 final_learner=None, random_state: int = 123):
        super().__init__(random_state)
        
        if outcome_learner is None:
            self.outcome_learner = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        else:
            self.outcome_learner = outcome_learner
            
        if propensity_learner is None:
            self.propensity_learner = xgb.XGBClassifier(random_state=random_state, n_estimators=100)
        else:
            self.propensity_learner = propensity_learner
            
        if final_learner is None:
            self.final_learner = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        else:
            self.final_learner = final_learner
        
        self.outcome_model = None
        self.propensity_model = None
        self.tau_model = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'RLearner':
        # 結果変数の予測モデル
        self.outcome_model = clone(self.outcome_learner)
        self.outcome_model.fit(X, Y)
        
        # 傾向スコアモデル
        self.propensity_model = clone(self.propensity_learner)
        self.propensity_model.fit(X, T)
        
        # 残差の計算
        y_pred = self.outcome_model.predict(X)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # クリッピングして数値安定性を確保
        propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
        
        # R-learnerの残差
        y_residual = Y - y_pred
        t_residual = T - propensity_scores
        
        # 重み付き残差（簡略化版）
        weights = 1.0 / (propensity_scores * (1 - propensity_scores))
        weights = np.clip(weights, 0, 100)  # 極端な重みをクリップ
        
        # 最終モデルの学習
        self.tau_model = clone(self.final_learner)
        
        # ゼロ除算を避けるため、t_residualが0に近い場合は除外
        valid_idx = np.abs(t_residual) > 1e-6
        if np.sum(valid_idx) > 10:  # 最低限のサンプル数を確保
            pseudo_outcome = (y_residual[valid_idx] / t_residual[valid_idx]) * weights[valid_idx]
            self.tau_model.fit(X[valid_idx], pseudo_outcome)
        else:
            # フォールバック: T-learnerと同様の方法
            treated_idx = T == 1
            control_idx = T == 0
            if np.sum(treated_idx) > 0 and np.sum(control_idx) > 0:
                y1_mean = np.mean(Y[treated_idx])
                y0_mean = np.mean(Y[control_idx])
                # 定数CATEとして学習
                pseudo_outcome = np.full(X.shape[0], y1_mean - y0_mean)
                self.tau_model.fit(X, pseudo_outcome)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.tau_model.predict(X)


class CausalForestWrapper(CATEEstimator):
    """
    EconMLのCausal Forest DMLのラッパークラス
    Doubly Robust learningとCausal Forestを組み合わせた高度なCATEエスティメータ
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_leaf: int = 5, random_state: int = 123):
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        # CausalForestDMLを初期化
        # バイナリ処置の場合は分類器、連続処置の場合は回帰器を使用
        self.model = CausalForestDML(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            model_y=RandomForestRegressor(n_estimators=50, random_state=random_state),
            model_t=RandomForestRegressor(n_estimators=50, random_state=random_state)  # 回帰器に変更
        )
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'CausalForestWrapper':
        """
        Causal Forest DMLモデルを学習
        """
        # NumPy配列をそのまま使用
        self.model.fit(Y, T, X=X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        CATEを予測
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # effectメソッドでCATEを取得
        cate = self.model.effect(X)
        
        # 結果が2次元の場合は1次元に変換
        if cate.ndim > 1:
            cate = cate.flatten()
        
        return cate


# 既存関数との互換性を保つラッパー関数
def train_causal_tree(X, T, Y, feature_names=None, min_samples_leaf=10, max_depth=5, random_state=123):
    """
    CausalMLのCausalTreeRegressorモデルを学習し、学習済みモデルを返す。
    既存のmain.pyとの互換性を保つ
    """
    model = CausalTreeWrapper(
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=random_state
    )
    return model.fit(X, T, Y)


def estimate_cate(model, X_test):
    """
    学習済みモデルを用いて、新しいデータポイントに対するCATEを推定する。
    既存のmain.pyとの互換性を保つ
    """
    if hasattr(model, 'predict'):
        return model.predict(X_test)
    else:
        # CausalMLのCausalTreeRegressorの場合
        return model.model.predict(X_test)


def create_model(model_type: str, **kwargs) -> CATEEstimator:
    """
    モデルタイプに応じてCATEEstimatorを作成する便利関数
    
    Args:
        model_type: "causal_tree", "s_learner", "t_learner", "dr_learner", "r_learner", "causal_forest"
        **kwargs: モデル固有のパラメータ
    
    Returns:
        CATEEstimatorのインスタンス
    """
    random_state = kwargs.get('random_state', 123)
    
    if model_type == "causal_tree":
        return CausalTreeWrapper(
            min_samples_leaf=kwargs.get('min_samples_leaf', 10),
            max_depth=kwargs.get('max_depth', 5),
            random_state=random_state
        )
    elif model_type == "s_learner":
        return SLearner(random_state=random_state)
    elif model_type == "t_learner":
        return TLearner(random_state=random_state)
    elif model_type == "dr_learner":
        return DRLearner(random_state=random_state)
    elif model_type == "r_learner":
        return RLearner(random_state=random_state)
    elif model_type == "causal_forest":
        return CausalForestWrapper(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_leaf=kwargs.get('min_samples_leaf', 5),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # このファイル単体で実行した場合のテスト用コード
    # data_generator.py は causalml に適した形式でデータを生成するように
    # 修正されているか、このテストコード内でデータを調整する必要があります。

    # --- ダミーデータの準備 (causalml.dataset.synthetic_data を使う方が適切) ---
    # ここでは、data_generator.py が適切に0/1のtreatmentと、
    # X, Y, feature_names を返すことを期待します。
    # もし data_generator.py が econml 前提のままだと、調整が必要です。
    # 例として、ここで簡単なデータを生成します。
    from causalml.dataset import synthetic_data as causalml_synth_data
    import graphviz # 可視化テストのため

    print("Generating synthetic data for CausalML CausalTreeRegressor test...")
    # y, X, treatment_str, tau, b, e
    y_s, X_s_np, treatment_s_str, _, _, _ = causalml_synth_data(
        mode='continuous', n_samples=120, n_features=5, random_seed=42
    )
    # 特徴量名を生成
    f_names_s = [f'feature_{i}' for i in range(X_s_np.shape[1])]
    X_s_df = pd.DataFrame(X_s_np, columns=f_names_s)

    # treatment を 0/1 に変換
    # 'treatment1' が処置群、それ以外(例: 'control')が対照群と仮定
    T_s_binary = np.where(treatment_s_str == 'treatment1', 1, 0)

    # 簡単な分割 (通常は train_test_split を使用)
    X_train, T_train, Y_train = X_s_df.iloc[:100], T_s_binary[:100], y_s[:100]
    X_test = X_s_df.iloc[100:]
    # --- ダミーデータ準備ここまで ---


    print("\nTraining CausalML CausalTreeRegressor...")
    # feature_names は train_causal_tree 関数には不要になりましたが、
    # 可視化のために保持しておきます。
    ct_model = train_causal_tree(X_train, T_train, Y_train, feature_names=f_names_s)
    print("CausalML CausalTreeRegressor trained.")

    print("\nEstimating CATE on test data using CausalML model...")
    cate_estimates = estimate_cate(ct_model, X_test)
    print("Estimated CATE for test data (first 5):")
    if len(cate_estimates) > 0:
        print(cate_estimates[:5])
    else:
        print("No CATE estimates generated (test data might be empty).")


    # CausalTreeRegressorの可視化 (graphvizが必要)
    print("\nVisualizing CausalML CausalTreeRegressor...")
    try:
        # CausalMLのCausalTreeRegressorはexport_graphvizメソッドを持っています
        dot_data = ct_model.export_graphviz(
            out_file=None,
            feature_names=f_names_s, # ここで特徴量名を渡します
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=True,
            proportion=True,
            precision=2
        )
        graph = graphviz.Source(dot_data)
        graph.render("causalml_causal_tree_example", view=False, cleanup=True) # "causalml_causal_tree_example.pdf" として保存
        print("CausalML Causal Tree structure saved to causalml_causal_tree_example.pdf and .png")
    except ImportError:
        print("graphviz is not installed. Skipping tree visualization.")
    except Exception as e:
        print(f"Could not visualize CausalML tree: {e}")
        print("Make sure graphviz executables are in your system's PATH if you are on Windows.")