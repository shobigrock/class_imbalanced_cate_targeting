"""
CATE推定実験のメインエントリーポイント

このファイルは実験の実行制御のみを行い、
具体的な実装は shared/ モジュールに委譲します。
"""

from shared.experiment_config import get_config, create_custom_config, PRESET_CONFIGS
from shared.experiment_runner import ExperimentRunner, ComparisonExperiment
from shared.result_formatter import ResultFormatter, ProgressReporter


def run_basic_experiments():
    """基本的な実験セットを実行"""
    ResultFormatter.print_experiment_header("BASIC EXPERIMENTS")
    
    # 基本実験の設定
    basic_experiments = ["basic_linear", "basic_logistic", "basic_tree"]
    
    progress = ProgressReporter(len(basic_experiments))
    results = []
    
    for preset_name in basic_experiments:
        progress.start_experiment(f"{preset_name.replace('_', ' ').title()} Experiment")
        
        # 設定取得と実験実行
        config = get_config(preset_name)
        runner = ExperimentRunner(config)
        result = runner.run_single_experiment()
        
        # 結果表示
        ResultFormatter.print_experiment_summary(result)
        results.append(result)
        
        progress.complete_experiment(preset_name)
    
    return results


def run_undersampling_comparison():
    """アンダーサンプリング比較実験を実行"""
    ResultFormatter.print_experiment_header("UNDERSAMPLING COMPARISON EXPERIMENT")
    
    # 比較実験用の設定
    config = get_config("undersampling_comparison")
    
    # 比較実験実行
    comparison = ComparisonExperiment(config)
    comparison_results = comparison.run_undersampling_comparison()
    
    # 結果表示
    ResultFormatter.print_comparison_results(comparison_results)
    
    return comparison_results


def run_custom_experiment():
    """カスタム実験の例"""
    ResultFormatter.print_experiment_header("CUSTOM EXPERIMENT EXAMPLE")
    
    # カスタム設定作成
    custom_config = create_custom_config(
        n_samples=4000,
        reward_type="logistic",
        model_type="s_learner",
        use_undersampling=True,
        k_factor=2.5
    )
    
    # 実験実行
    runner = ExperimentRunner(custom_config)
    result = runner.run_single_experiment()
    
    # 結果表示
    ResultFormatter.print_experiment_summary(result)
    ResultFormatter.print_detailed_evaluation(result["evaluation"])
    
    return result


def main():
    """メイン実行関数"""
    print("🚀 Starting CATE Estimation Experiments")
    print("📊 Imbalanced Data Undersampling Analysis")
    
    # 利用可能な設定を表示
    print(f"\n📋 Available preset configurations: {', '.join(PRESET_CONFIGS.keys())}")
    
    try:
        # 1. 基本実験
        basic_results = run_basic_experiments()
        
        # 2. アンダーサンプリング比較実験
        comparison_result = run_undersampling_comparison()
        
        # 3. カスタム実験例
        custom_result = run_custom_experiment()
        
        # 最終サマリー
        print("\n" + "=" * 80)
        print("📈 EXPERIMENT SUMMARY")
        summary = ResultFormatter.create_results_summary(basic_results + [custom_result])
        print(summary)
        
        # 完了メッセージ
        ResultFormatter.print_completion_message()
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        raise


if __name__ == '__main__':
    main()