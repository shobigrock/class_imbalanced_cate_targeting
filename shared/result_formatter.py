"""
実験結果の表示とフォーマットを管理するモジュール
"""
from typing import Dict, Any, List
import numpy as np


class ResultFormatter:
    """実験結果のフォーマット表示を管理するクラス"""
    
    @staticmethod
    def print_experiment_summary(results: Dict[str, Any]) -> None:
        """単一実験の結果サマリーを表示"""
        config = results["config"]
        data_stats = results["data_stats"]
        evaluation = results["evaluation"]
        
        print("\n--- Experiment Summary ---")
        print(f"Number of samples: {config['n_samples']}")
        print(f"Reward type: {config['reward_type']}")
        print(f"Model type: {config['model_type']}")
        print(f"Use undersampling: {config['use_undersampling']}")
        
        if config['use_undersampling']:
            print(f"K-factor: {config['k_factor']}")
        
        print(f"Test set size: {data_stats['test_samples']}")
        print(f"Original positive rate: {data_stats['original_positive_rate']:.4f}")
        print(f"Training positive rate: {data_stats['train_positive_rate']:.4f}")
        print(f"CATE Estimation RMSE: {evaluation['rmse']:.4f}")
        print(f"CATE Estimation Bias: {evaluation['bias']:.4f}")
        
        if 'r2_score' in evaluation:
            print(f"R² Score: {evaluation['r2_score']:.4f}")
    
    @staticmethod
    def print_comparison_results(comparison_results: Dict[str, Any]) -> None:
        """比較実験の結果を表示"""
        comparison = comparison_results["comparison"]
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        # テーブルヘッダー
        print(f"{'Metric':<20} {'Baseline':<15} {'Undersampling':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # RMSE
        print(f"{'RMSE':<20} {comparison['baseline_rmse']:<15.4f} "
              f"{comparison['undersampling_rmse']:<15.4f} "
              f"{comparison['rmse_improvement_percent']:+.2f}%")
        
        # Bias
        print(f"{'Bias (abs)':<20} {abs(comparison['baseline_bias']):<15.4f} "
              f"{abs(comparison['undersampling_bias']):<15.4f} "
              f"{comparison['bias_improvement_percent']:+.2f}%")
    
    @staticmethod
    def print_detailed_evaluation(evaluation: Dict[str, float]) -> None:
        """詳細な評価結果を表示"""
        print("\n--- Detailed Evaluation ---")
        for metric, value in evaluation.items():
            print(f"{metric.upper()}: {value:.4f}")
    
    @staticmethod
    def print_data_statistics(data_stats: Dict[str, Any]) -> None:
        """データ統計を表示"""
        print("\n--- Data Statistics ---")
        print(f"Total samples: {data_stats['total_samples']}")
        print(f"Training samples: {data_stats['train_samples']}")
        print(f"Test samples: {data_stats['test_samples']}")
        print(f"Original positive rate: {data_stats['original_positive_rate']:.4f}")
        print(f"Training positive rate: {data_stats['train_positive_rate']:.4f}")
    
    @staticmethod
    def print_configuration(config: Dict[str, Any]) -> None:
        """設定内容を表示"""
        print("\n--- Configuration ---")
        for key, value in config.items():
            if key != 'model_params':
                print(f"{key}: {value}")
        
        if 'model_params' in config and config['model_params']:
            print("Model parameters:")
            for param, value in config['model_params'].items():
                print(f"  {param}: {value}")
    
    @staticmethod
    def create_results_summary(results_list: List[Dict[str, Any]]) -> str:
        """複数実験結果のサマリーを作成"""
        if not results_list:
            return "No results to summarize."
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("EXPERIMENT RESULTS SUMMARY")
        summary_lines.append("=" * 80)
        
        # ヘッダー
        header = f"{'Experiment':<25} {'Reward':<10} {'Model':<15} {'Undersampling':<12} {'RMSE':<8} {'Bias':<8}"
        summary_lines.append(header)
        summary_lines.append("-" * 80)
        
        # 各実験の結果
        for i, result in enumerate(results_list):
            config = result["config"]
            evaluation = result["evaluation"]
            
            exp_name = f"Experiment {i+1}"
            reward_type = config['reward_type'][:8]  # 短縮
            model_type = config['model_type'][:13]  # 短縮
            undersampling = "Yes" if config['use_undersampling'] else "No"
            rmse = f"{evaluation['rmse']:.4f}"
            bias = f"{evaluation['bias']:.4f}"
            
            line = f"{exp_name:<25} {reward_type:<10} {model_type:<15} {undersampling:<12} {rmse:<8} {bias:<8}"
            summary_lines.append(line)
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)
    
    @staticmethod
    def print_experiment_header(title: str) -> None:
        """実験開始時のヘッダーを表示"""
        border = "=" * len(title)
        print(f"\n{border}")
        print(title)
        print(border)
    
    @staticmethod
    def print_step_header(step_num: int, step_name: str) -> None:
        """ステップヘッダーを表示"""
        print(f"\nStep {step_num}: {step_name}")
        print("-" * (10 + len(step_name)))
    
    @staticmethod
    def print_completion_message() -> None:
        """実験完了メッセージを表示"""
        print("\n" + "=" * 60)
        print("=== ALL EXPERIMENTS COMPLETED ===")
        print("To test different models:")
        print("  python test/test_models.py")
        print("For comprehensive analysis:")
        print("  python test/main_complete.py")
        print("=" * 60)


class ProgressReporter:
    """実験進行状況の報告を管理するクラス"""
    
    def __init__(self, total_experiments: int):
        """
        Args:
            total_experiments: 総実験数
        """
        self.total_experiments = total_experiments
        self.completed_experiments = 0
    
    def start_experiment(self, experiment_name: str) -> None:
        """実験開始を報告"""
        self.completed_experiments += 1
        progress = (self.completed_experiments / self.total_experiments) * 100
        
        print(f"\n[{self.completed_experiments}/{self.total_experiments}] ({progress:.1f}%) {experiment_name}")
    
    def complete_experiment(self, experiment_name: str) -> None:
        """実験完了を報告"""
        print(f"✓ Completed: {experiment_name}")
    
    def final_summary(self) -> None:
        """最終サマリーを表示"""
        print(f"\n🎉 All {self.total_experiments} experiments completed successfully!")


def format_metric_table(metrics_data: Dict[str, Dict[str, float]], 
                       experiment_names: List[str]) -> str:
    """メトリクステーブルをフォーマット"""
    if not metrics_data or not experiment_names:
        return "No data to format."
    
    # メトリクス名を取得
    metric_names = list(next(iter(metrics_data.values())).keys())
    
    # ヘッダー作成
    header = f"{'Experiment':<20}"
    for metric in metric_names:
        header += f" {metric.upper():<12}"
    
    lines = [header, "-" * len(header)]
    
    # データ行作成
    for exp_name in experiment_names:
        if exp_name in metrics_data:
            line = f"{exp_name:<20}"
            for metric in metric_names:
                value = metrics_data[exp_name].get(metric, 0.0)
                line += f" {value:<12.4f}"
            lines.append(line)
    
    return "\n".join(lines)
