"""
実験設定の管理モジュール
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ExperimentConfig:
    """実験設定を管理するデータクラス"""
    
    # データ生成設定
    n_samples: int = 2000
    test_size: float = 0.3
    random_seed: int = 42
    reward_type: str = "linear"  # "linear", "logistic", "tree"
    
    # アンダーサンプリング設定
    use_undersampling: bool = False
    k_factor: float = 2.0
    
    # モデル設定
    model_type: str = "causal_tree"  # "causal_tree", "s_learner", "t_learner", etc.
    
    # モデル固有パラメータ
    model_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.model_params is None:
            self.model_params = self._get_default_model_params()
    
    def _get_default_model_params(self) -> Dict[str, Any]:
        """モデルタイプに応じたデフォルトパラメータを返す"""
        if self.model_type == "causal_tree":
            return {
                "min_samples_leaf": 20,
                "max_depth": 4,
                "random_state": self.random_seed
            }
        else:
            return {"random_state": self.random_seed}
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で返す"""
        return {
            "n_samples": self.n_samples,
            "test_size": self.test_size,
            "random_seed": self.random_seed,
            "reward_type": self.reward_type,
            "use_undersampling": self.use_undersampling,
            "k_factor": self.k_factor,
            "model_type": self.model_type,
            "model_params": self.model_params
        }
    
    def summary(self) -> str:
        """設定の要約を文字列で返す"""
        summary_lines = [
            f"Samples: {self.n_samples}",
            f"Test size: {self.test_size}",
            f"Random seed: {self.random_seed}",
            f"Reward type: {self.reward_type}",
            f"Model type: {self.model_type}",
            f"Use undersampling: {self.use_undersampling}"
        ]
        
        if self.use_undersampling:
            summary_lines.append(f"K-factor: {self.k_factor}")
        
        return "\n".join(summary_lines)


# 事前定義された実験設定
PRESET_CONFIGS = {
    "basic_linear": ExperimentConfig(
        n_samples=3000,
        reward_type="linear",
        model_type="causal_tree"
    ),
    
    "basic_logistic": ExperimentConfig(
        n_samples=3000,
        reward_type="logistic",
        model_type="causal_tree"
    ),
    
    "basic_tree": ExperimentConfig(
        n_samples=3000,
        reward_type="tree",
        model_type="causal_tree"
    ),
    
    "undersampling_comparison": ExperimentConfig(
        n_samples=5000,
        reward_type="logistic",
        model_type="causal_tree",
        use_undersampling=True,
        k_factor=3.0
    ),
    
    "large_scale": ExperimentConfig(
        n_samples=10000,
        reward_type="logistic",
        model_type="causal_forest",
        use_undersampling=True,
        k_factor=2.5
    )
}


def get_config(preset_name: str) -> ExperimentConfig:
    """事前定義された設定を取得"""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESET_CONFIGS[preset_name]


def create_custom_config(**kwargs) -> ExperimentConfig:
    """カスタム設定を作成"""
    return ExperimentConfig(**kwargs)
