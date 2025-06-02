"""
コマンドライン引数の解析機能

CATE推定実験のためのコマンドライン引数を定義・検証します。
"""

import argparse
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CLIConfig:
    """コマンドライン引数の設定を管理するデータクラス"""
    
    # データ設定
    data_source: str = "synthetic"  # "synthetic" or "criteo"
    n_samples: int = 2000
    random_seed: int = 42
    
    # CATE手法設定
    cate_methods: List[str] = None  # ["causal_tree", "s_learner", "t_learner"]
    
    # アンダーサンプリング設定
    k_values: List[int] = None  # [0, 1, 2, ..., 15]
    
    # 実験設定
    reward_type: str = "linear"  # "linear", "logistic", "tree"
    test_size: float = 0.3
    
    # 評価設定
    enable_qini: bool = True  # 人工データでのQINI計算
    
    # 出力設定
    verbose: bool = False
    output_file: Optional[str] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.cate_methods is None:
            self.cate_methods = ["causal_tree"]
        if self.k_values is None:
            self.k_values = [0, 2, 5]  # デフォルトでいくつかのk値をテスト


class CLIParser:
    """コマンドライン引数の解析クラス"""
    
    AVAILABLE_DATA_SOURCES = ["synthetic", "criteo"]
    AVAILABLE_CATE_METHODS = [
        "causal_tree", "s_learner", "t_learner", "dr_learner", 
        "r_learner", "causal_forest"
    ]
    AVAILABLE_REWARD_TYPES = ["linear", "logistic", "tree"]
    K_VALUE_RANGE = (0, 15)
    
    def __init__(self):
        """初期化"""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """argparseパーサーを作成"""
        parser = argparse.ArgumentParser(
            description="Class Imbalanced CATE Targeting Experiment",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        
        # データ設定
        data_group = parser.add_argument_group('データ設定')
        data_group.add_argument(
            '--data-source', 
            choices=self.AVAILABLE_DATA_SOURCES,
            default="synthetic",
            help="使用データソース (default: %(default)s)"
        )
        data_group.add_argument(
            '--n-samples', 
            type=int, 
            default=2000,
            help="サンプル数 (default: %(default)s)"
        )
        data_group.add_argument(
            '--reward-type',
            choices=self.AVAILABLE_REWARD_TYPES,
            default="linear",
            help="報酬関数タイプ (人工データのみ) (default: %(default)s)"
        )
        data_group.add_argument(
            '--seed', 
            type=int, 
            default=42,
            help="乱数シード (default: %(default)s)"
        )
        
        # CATE手法設定
        method_group = parser.add_argument_group('CATE手法設定')
        method_group.add_argument(
            '--cate-methods',
            nargs='+',
            choices=self.AVAILABLE_CATE_METHODS,
            default=["causal_tree"],
            help="使用するCATE手法のリスト (default: %(default)s)"
        )
        
        # アンダーサンプリング設定
        sampling_group = parser.add_argument_group('アンダーサンプリング設定')
        sampling_group.add_argument(
            '--k-values',
            nargs='+',
            type=int,
            metavar='K',
            default=[0, 2, 5],
            help=f"アンダーサンプリングk値のリスト ({self.K_VALUE_RANGE[0]}-{self.K_VALUE_RANGE[1]}) (default: %(default)s)"
        )
        
        # 実験設定
        experiment_group = parser.add_argument_group('実験設定')
        experiment_group.add_argument(
            '--test-size',
            type=float,
            default=0.3,
            help="テストデータの割合 (default: %(default)s)"
        )
        experiment_group.add_argument(
            '--disable-qini',
            action='store_true',
            help="QINI係数計算を無効化"
        )
        
        # 出力設定
        output_group = parser.add_argument_group('出力設定')
        output_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help="詳細出力モード"
        )
        output_group.add_argument(
            '--output',
            type=str,
            help="結果出力ファイル (CSV形式)"
        )
        
        # バージョン情報
        parser.add_argument(
            '--version',
            action='version',
            version='CATE Targeting Experiment v1.0.0'
        )
        
        return parser
    
    def _get_epilog(self) -> str:
        """ヘルプの最後に表示する使用例"""
        return """
使用例:
  # 基本実行（人工データ、causal_tree、k=0,2,5）
  python main.py
  
  # 複数手法での比較実験
  python main.py --cate-methods causal_tree s_learner t_learner
  
  # k値を詳細にスキャン
  python main.py --k-values 0 1 2 3 4 5 10 15
  
  # Criteoデータでの実験
  python main.py --data-source criteo --k-values 0 3 5
  
  # 大規模実験（詳細出力、結果保存）
  python main.py --n-samples 10000 --verbose --output results.csv
  
  # ロジスティック報酬関数での実験
  python main.py --reward-type logistic --cate-methods s_learner t_learner
"""
    
    def parse_args(self, args: Optional[List[str]] = None) -> CLIConfig:
        """引数を解析してCLIConfigを返す"""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        # 引数の検証
        self._validate_args(parsed_args)
        
        # CLIConfigオブジェクトの作成
        config = CLIConfig(
            data_source=parsed_args.data_source,
            n_samples=parsed_args.n_samples,
            random_seed=parsed_args.seed,
            cate_methods=parsed_args.cate_methods,
            k_values=parsed_args.k_values,
            reward_type=parsed_args.reward_type,
            test_size=parsed_args.test_size,
            enable_qini=not parsed_args.disable_qini,
            verbose=parsed_args.verbose,
            output_file=parsed_args.output
        )
        
        return config
    
    def _validate_args(self, args) -> None:
        """引数の妥当性を検証"""
        errors = []
        
        # サンプル数の検証
        if args.n_samples <= 0:
            errors.append("サンプル数は正の値である必要があります")
        
        # k値の検証
        for k in args.k_values:
            if not (self.K_VALUE_RANGE[0] <= k <= self.K_VALUE_RANGE[1]):
                errors.append(f"k値は{self.K_VALUE_RANGE[0]}-{self.K_VALUE_RANGE[1]}の範囲である必要があります: {k}")
        
        # テストサイズの検証
        if not (0.0 < args.test_size < 1.0):
            errors.append("テストサイズは0.0-1.0の範囲である必要があります")
        
        # Criteoデータでの報酬タイプ検証
        if args.data_source == "criteo" and args.reward_type != "linear":
            print("警告: Criteoデータでは報酬タイプ設定は無視されます")
        
        # QINI係数とデータソースの組み合わせ検証
        if not args.disable_qini and args.data_source != "synthetic":
            print("警告: QINI係数は人工データでのみ計算されます")
        
        # エラーがあれば例外発生
        if errors:
            print("引数エラー:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
    
    def print_help(self):
        """ヘルプを表示"""
        self.parser.print_help()


def create_cli_parser() -> CLIParser:
    """CLIParserインスタンスを作成"""
    return CLIParser()


def parse_command_line_args(args: Optional[List[str]] = None) -> CLIConfig:
    """コマンドライン引数を解析する便利関数"""
    parser = create_cli_parser()
    return parser.parse_args(args)


# テスト用の関数
def test_cli_parser():
    """CLIパーサーのテスト"""
    parser = create_cli_parser()
    
    # テストケース
    test_cases = [
        # 基本ケース
        [],
        
        # 複数手法
        ["--cate-methods", "causal_tree", "s_learner", "t_learner"],
        
        # k値スキャン
        ["--k-values", "0", "1", "2", "3", "4", "5"],
        
        # Criteoデータ
        ["--data-source", "criteo", "--k-values", "0", "3", "5"],
        
        # 詳細設定
        ["--n-samples", "5000", "--reward-type", "logistic", "--verbose", "--output", "test.csv"]
    ]
    
    print("CLIパーサーテスト:")
    for i, test_args in enumerate(test_cases):
        try:
            config = parser.parse_args(test_args)
            print(f"テスト {i+1}: ✓ 成功")
            print(f"  引数: {test_args}")
            print(f"  設定: data_source={config.data_source}, methods={config.cate_methods}, k_values={config.k_values}")
        except SystemExit:
            print(f"テスト {i+1}: ✗ 失敗")
        print()


if __name__ == "__main__":
    test_cli_parser()
