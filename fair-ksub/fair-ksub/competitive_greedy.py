# main.py
import pandas as pd
import argparse
import time
from greedy import greedy
from lazy_greedy import greedy_celf
from lazy_mp import greedy_lazy_mp
def main():
    parser = argparse.ArgumentParser(description="Chạy thuật toán Greedy Influence Maximization nhiều chủ đề")
    parser.add_argument("--input", type=str, default="graph.csv", help="Đường dẫn tới file CSV đồ thị")
    parser.add_argument("--k", type=int, default=5, help="Tổng số node cần chọn")
    parser.add_argument("--mc", type=int, default=1000, help="MC simulations per estimate")
    parser.add_argument("--workers", type=int, default=None, help="Số process for parallel")
    args = parser.parse_args()

    print(f"📥 Đang đọc đồ thị từ: {args.input}")
    print(f"🔍 Tổng số node cần chọn: {args.k}\n")

    # 1. Thử greedy thường
    start1 = time.time()
    x1, spread1 = greedy(args.input, args.k)
    t1 = time.time() - start1

    print("🕒 Kết quả greedy thường:")
    for i, seed in enumerate(x1):
        print(f"  Chủ đề {i+1}: {seed}")
    print(f"  ➤ Ảnh hưởng: {spread1:.2f}")
    print(f"  ⏱ Thời gian chạy: {t1:.3f} giây\n")

    # 2. Thử lazy greedy (CELF)
    start2 = time.time()
    x2, spread2 = greedy_celf(args.input, args.k)
    t2 = time.time() - start2

    print("🕒 Kết quả lazy greedy lazy (CELF):")
    for i, seed in enumerate(x2):
        print(f"  Chủ đề {i+1}: {seed}")
    print(f"  ➤ Ảnh hưởng: {spread2:.2f}")
    print(f"  ⏱ Thời gian chạy: {t2:.3f} giây\n")
    df = pd.read_csv(args.input)
    # 2. Thử lazy greedy (CELF)
    start3 = time.time()

    x3, spread3 = greedy_lazy_mp(df, args.k, mc=args.mc, n_workers=args.workers)
    t3 = time.time() - start2

    print("🕒 Kết quả lazy greedy mp (CELF):")
    for i, seed in enumerate(x3):
        print(f"  Chủ đề {i + 1}: {seed}")
    print(f"  ➤ Ảnh hưởng: {spread2:.2f}")
    print(f"  ⏱ Thời gian chạy: {t2:.3f} giây\n")

    # 3. Tóm tắt so sánh
    speedup = t1 / t2 if t2 > 0 else float('inf')
    print("🔍 So sánh tốc độ:")
    print(f"  - Greedy thường: {t1:.3f}s")
    print(f"  - Lazy greedy:  {t2:.3f}s")
    print(f"  ⇒ Speed-up:     {speedup:.2f}×")

if __name__ == "__main__":
    main()
