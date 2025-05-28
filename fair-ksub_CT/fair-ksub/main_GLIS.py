import argparse
import pandas as pd
from greedy_lazy import greedy

def main():

    parser = argparse.ArgumentParser(
        description="Chạy Lazy Greedy nhiều lần"
    )
    parser.add_argument("--input", type=str, default="graph250.csv", help="CSV source,target,weight*")
    parser.add_argument("--k", type=int, default=5, help="Số seed cần chọn")
    parser.add_argument("--mc", type=int, default=1000, help="MC simulations per estimate")
    parser.add_argument("--workers", type=int, default=None, help="Số process for parallel")
    parser.add_argument("--repeat", type=int, default=10, help="Số lần chạy")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    args = parser.parse_args()

    print(f"📥 Đang đọc đồ thị từ: {args.input}")
    print(f"🔍 Tổng số node cần chọn: {args.k}")

    x, spread = greedy(args.input, args.k)

    print("\n✅ Seed sets chọn được theo từng chủ đề:")
    for i, seed in enumerate(x):
        print(f"  Chủ đề {i+1}: {seed}")
    print(f"\n✅ Ảnh hưởng trung bình cuối cùng: {spread:.2f}")

if __name__ == "__main__":
    main()
