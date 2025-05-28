import argparse
from greedy import greedy

def main():
    parser = argparse.ArgumentParser(description="Chạy thuật toán Greedy Influence Maximization nhiều chủ đề")
    parser.add_argument("--input", type=str, default="graph250.csv", help="Đường dẫn tới file CSV đồ thị")
    parser.add_argument("--k", type=int, default=5, help="Tổng số node cần chọn")

    args = parser.parse_args()

    print(f"📥 Đang đọc đồ thị từ: {args.input}")
    print(f"🔍 Tổng số node cần chọn: {args.k}")

    x, spread = greedy(args.input, args.k,1000,)

    print("\n✅ Seed sets chọn được theo từng chủ đề:")
    for i, seed in enumerate(x):
        print(f"  Chủ đề {i+1}: {seed}")
    print(f"\n✅ Ảnh hưởng trung bình cuối cùng: {spread:.2f}")

if __name__ == "__main__":
    main()
