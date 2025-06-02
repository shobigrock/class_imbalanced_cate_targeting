# visualization_utils.py (または main.py に直接定義)

import graphviz
from sklearn.tree import export_graphviz
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np # np.array のために追加

class _MockSklearnTreeEstimatorV3(BaseEstimator, RegressorMixin):
    """
    CausalMLのTreeオブジェクトをsklearn.tree.export_graphvizに渡せるようにするための
    さらに改良版モック（模擬）Estimatorクラス。
    fitメソッドを追加し、より多くの属性を設定。
    """
    def __init__(self, causalml_fitted_model, n_input_features):
        # CausalMLの学習済みモデルインスタンスとtreeオブジェクトを保持
        self._causalml_model = causalml_fitted_model
        if not hasattr(causalml_fitted_model, 'tree_') or causalml_fitted_model.tree_ is None:
            raise ValueError("渡されたCausalMLモデルに 'tree_' 属性がないか、Noneです。")
        self.tree_ = causalml_fitted_model.tree_
        
        # scikit-learnの推定器が持つべき基本的な属性
        self.n_features_in_ = int(n_input_features) # int型であることを保証
        self.n_outputs_ = 1     # CATEの場合は通常1
        self._estimator_type = "regressor"

        # CausalMLモデルのパラメータを可能な範囲でコピー (get_paramsで使われる)
        self.max_depth = getattr(self._causalml_model, 'max_depth', None)
        # min_samples_split は BaseEstimator の repr で表示されていたので、それを元に設定
        self.min_samples_split = getattr(self._causalml_model, 'min_samples_split', 2)
        self.min_samples_leaf = getattr(self._causalml_model, 'min_samples_leaf', 1)
        
        # criterion は export_graphviz で使われる可能性がある
        # CausalMLの実際のcriterionと異なるかもしれないが、sklearnが認識する文字列を設定
        self.criterion = 'squared_error' # DecisionTreeRegressorの一般的なデフォルト
        
        self.random_state = getattr(self._causalml_model, 'random_state', None)

        # scikit-learnの推定器がfit後に持つことが多い属性
        self.max_features_ = self.n_features_in_ # 全特徴量を使ったと仮定
        
        # n_classes_ と classes_ (Regressorの場合の典型的な設定)
        if self.n_outputs_ == 1:
            self.n_classes_ = 1 # 単一出力回帰の場合
        else:
            # マルチ出力回帰の場合の n_classes_ の扱い (通常は出力ごとに1)
            self.n_classes_ = np.ones(self.n_outputs_, dtype=int)

        # classes_ は通常、分類器で使われるが、Regressorでも属性として存在することがある
        # (中身はNoneやダミー値)
        # export_graphviz が class_names=True の場合などに参照する可能性がある
        if self.n_outputs_ == 1:
            self.classes_ = np.array([None]) # 単一出力の場合
        else:
            self.classes_ = np.array([None] * self.n_outputs_)


        # 学習済みであることを示すためのダミー属性 (check_is_fittedが参照する)
        # 実際には、tree_ や n_features_in_ のような _ で終わる属性の存在で判断される
        self._is_fitted = True # scikit-learn 0.24以降では _is_fitted() メソッドが推奨されることもある

    # BaseEstimatorに必要なメソッド
    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state,
            # 他にもCausalTreeRegressorが持つパラメータがあれば追加
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # scikit-learnの推定器としての体裁を整えるためのダミーfitメソッド
    def fit(self, X, y=None):
        """
        このモック推定器は既に「学習済み」のtree_オブジェクトを持つため、
        このfitメソッドは実際には何もしません。
        scikit-learnの check_is_fitted が通るようにするために存在します。
        """
        # n_features_in_ が X から設定されることを模倣
        if X is not None:
             self.n_features_in_ = X.shape[1]
             self.max_features_ = X.shape[1] # fit時に確定
        # _is_fitted() メソッドまたは _is_fitted 属性の存在がチェックされることがある
        return self

    # RegressorMixinに必要なメソッド (形式上)
    def predict(self, X):
        raise NotImplementedError("MockEstimator.predict should not be called.")

    def _more_tags(self):
        # scikit-learnの互換性テストなどで使われることがある
        return {'X_types': ['2darray'], 'allow_nan': False, 'requires_y': True} # requires_y=True for fit


def visualize_causalml_tree_0_15_4(causalml_model, X_input_features_df, feature_names, filename_prefix="causalml_tree_v0154"):
    """
    CausalML 0.15.4 の CausalTreeRegressor の木構造をGraphvizで可視化する（V3）。
    """
    if not hasattr(causalml_model, 'tree_') or causalml_model.tree_ is None:
        print("エラー: モデルに 'tree_' 属性が存在しないか、学習されていません。可視化をスキップします。")
        return

    print(f"モデルのtree_属性の型: {type(causalml_model.tree_)}")

    try:
        n_features = X_input_features_df.shape[1]
        # 改良版V3モックEstimatorを作成
        mock_estimator = _MockSklearnTreeEstimatorV3(causalml_model, n_features)
        # ダミーfitを呼び出して、内部的にn_features_in_などが設定されるようにする
        # (実際にはコンストラクタで設定済みだが、fitを持つことが重要)
        mock_estimator.fit(X_input_features_df, None) # yは使わないのでNone
        
        dot_data = export_graphviz(
            mock_estimator, # モックEstimatorを渡す
            out_file=None,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=True,
            proportion=True,
            precision=2
        )
    except Exception as e:
        print(f"DOTデータの生成中にエラーが発生しました: {e}")
        print("CausalML 0.15.4 の tree_ オブジェクトの内部構造、またはモックEstimatorの定義が、scikit-learnの期待と異なる可能性があります。")
        return

    try:
        graph = graphviz.Source(dot_data, format="png")
        graph.render(filename_prefix, view=False, cleanup=True)
        print(f"CausalMLツリー構造が {filename_prefix}.png (および.pdf) として保存されました。")
    except graphviz.backend.execute.ExecutableNotFound:
        print("エラー: Graphvizの実行ファイルが見つかりません。Graphvizをインストールし、PATHに追加してください。")
        print("Windowsの場合、Graphvizをインストール後、OSを再起動するとPATHが認識されることがあります。")
    except Exception as e:
        print(f"Graphvizでのレンダリング中にエラーが発生しました: {e}")