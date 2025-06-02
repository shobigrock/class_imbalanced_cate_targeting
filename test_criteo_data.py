# test_criteo_data.py
"""
Criteoデータセットの読み込みと処理をテストするスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import create_data_generator

def test_criteo_data_info():
    """Criteoデータの情報を確認"""
    print("=" * 60)
    print("Criteo Data Generator Information Test")
    print("=" * 60)
    
    # Criteoデータジェネレータを作成
    criteo_gen = create_data_generator("criteo", random_state=42)
    
    # データ情報を取得
    data_info = criteo_gen.get_data_info()
    
    print("Data Info:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    return data_info

def test_criteo_data_loading():
    """Criteoデータの実際の読み込みをテスト"""
    print("\n" + "=" * 60)
    print("Criteo Data Loading Test")
    print("=" * 60)
    
    try:
        # Criteoデータジェネレータを作成
        criteo_gen = create_data_generator("criteo", random_state=42)
        
        # 小さなサンプルでデータを読み込み
        print("Loading small sample of Criteo data...")
        X_df, T, Y, true_cate, feature_names = criteo_gen.generate_data(n_samples=1000)
        
        print(f"Successfully loaded Criteo data!")
        print(f"  Features shape: {X_df.shape}")
        print(f"  Treatment shape: {T.shape}")
        print(f"  Outcome shape: {Y.shape}")
        print(f"  True CATE shape: {true_cate.shape}")
        print(f"  Feature names: {feature_names}")
        
        # データの基本統計
        print(f"\nData Statistics:")
        print(f"  Treatment distribution: {dict(zip(*np.unique(T, return_counts=True)))}")
        print(f"  Outcome mean: {Y.mean():.4f}")
        print(f"  True CATE mean: {true_cate.mean():.4f}")
        print(f"  True CATE std: {true_cate.std():.4f}")
        
        # 不均衡データかチェック
        conversion_rate = Y.mean()
        print(f"\nImbalance Analysis:")
        print(f"  Overall conversion rate: {conversion_rate:.4f}")
        print(f"  Treatment group conversion: {Y[T==1].mean():.4f}")
        print(f"  Control group conversion: {Y[T==0].mean():.4f}")
        
        if conversion_rate < 0.1:
            print("  -> This is an imbalanced dataset (conversion rate < 10%)")
        else:
            print("  -> This dataset is relatively balanced")
        
        return X_df, T, Y, true_cate, feature_names
        
    except Exception as e:
        print(f"Error loading Criteo data: {e}")
        return None

if __name__ == "__main__":
    import numpy as np
    
    # データ情報のテスト
    data_info = test_criteo_data_info()
    
    # ファイルが存在する場合のみデータ読み込みテスト
    if data_info.get("file_exists", False):
        test_criteo_data_loading()
    else:
        print("\nCriteo data file not found - skipping data loading test")
        print("Expected file: data/criteo-research-uplift-v2.1.csv.gz")
