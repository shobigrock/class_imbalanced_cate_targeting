"""
CATE推定実験のメインエントリーポイント

このファイルは実験の実行制御のみを行い、
具体的な実装は shared/ モジュールに委譲します。
"""

import sys
from typing import List, Dict, Any, Optional
from shared.cli_parser import CLIParser, CLIConfig
from shared.experiment_config import (
    get_config, create_custom_config, create_config_from_cli, 
    PRESET_CONFIGS, ExperimentConfig
)
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


def run_cli_experiments(cli_config: CLIConfig) -> List[Dict[str, Any]]:
    """CLIコンフィグに基づいて実験を実行"""
    print(f"🚀 Running CLI Experiments")
    print(f"📊 Data source: {cli_config.data_source}")
    print(f"🔧 CATE methods: {', '.join(cli_config.cate_methods)}")
    print(f"📈 K values: {cli_config.k_values}")
    print(f"🎯 QINI enabled: {cli_config.enable_qini}")
    
    # CLIコンフィグから実験設定を作成
    configs = create_config_from_cli(cli_config)
    total_experiments = len(configs)
    
    progress = ProgressReporter(total_experiments)
    results = []
    
    for i, config in enumerate(configs):
        exp_name = f"{config.model_type}_k{config.k_factor if config.use_undersampling else 'none'}"
        progress.start_experiment(f"Experiment {i+1}/{total_experiments}: {exp_name}")
        
        # 実験実行
        runner = ExperimentRunner(config)
        result = runner.run_single_experiment()
        
        # 詳細ログ（verbose時のみ）
        if cli_config.verbose:
            ResultFormatter.print_experiment_summary(result)
        
        results.append(result)
        progress.complete_experiment(exp_name)
    
    return results


def save_results_to_file(results: List[Dict[str, Any]], output_file: str):
    """結果をCSVファイルに保存"""
    import pandas as pd
    
    # 結果データを平坦化
    flattened_data = []
    for result in results:
        row = {
            'model_type': result['config']['model_type'],
            'use_undersampling': result['config']['use_undersampling'],
            'k_factor': result['config'].get('k_factor', 0),
            'data_source': result['config'].get('data_source', 'synthetic'),
            'n_samples': result['config']['n_samples'],
            'test_size': result['config']['test_size'],
            'rmse': result['evaluation']['rmse'],
            'bias': result['evaluation']['bias'],
            'r2_score': result['evaluation'].get('r2_score', 0.0),
            'runtime': result['runtime']
        }
        
        # QINI係数がある場合は追加
        if 'qini_coefficient' in result['evaluation']:
            row['qini_coefficient'] = result['evaluation']['qini_coefficient']
        
        flattened_data.append(row)
    
    # DataFrame作成とCSV保存
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")


def main():
    """メイン実行関数"""
    print("🚀 CATE Estimation Experiments with Class Imbalance")
    print("📊 Command-line Interface for Flexible Experimentation")
    
    # CLI引数解析
    parser = CLIParser()
    cli_config = parser.parse_args()
    
    if cli_config is None:
        # ヘルプが表示されたか、引数エラーがあった場合
        return
    
    try:
        # CLI指定の実験を実行
        results = run_cli_experiments(cli_config)
        
        # 結果サマリーを表示
        print("\n" + "=" * 80)
        print("📈 EXPERIMENT RESULTS SUMMARY")
        summary = ResultFormatter.create_results_summary(results)
        print(summary)
        
        # CSVファイルに保存（指定された場合）
        if cli_config.output_file:
            save_results_to_file(results, cli_config.output_file)
        
        # 完了メッセージ
        ResultFormatter.print_completion_message()
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        if cli_config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_preset_experiments():
    """プリセット実験を実行（後方互換性のため）"""
    print("🚀 Running Preset Experiments")
    print("📋 Use 'python main.py --help' for custom CLI options")
    
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
    # コマンドライン引数があるかチェック
    if len(sys.argv) > 1:
        # CLI引数が提供された場合はCLI実験を実行
        main()
    else:
        # 引数がない場合はプリセット実験を実行（後方互換性）
        print("📋 No CLI arguments provided. Running preset experiments...")
        print("💡 Use 'python main.py --help' for CLI options")
        run_preset_experiments()