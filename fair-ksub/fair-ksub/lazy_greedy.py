import pandas as pd
from diffuse import IC
import heapq
from tqdm import tqdm
import time
# from ic_parallel_noisy import IC
def greedy_celf(filename, k):
    """
    Cải tiến greedy bằng Lazy Forward (CELF) cho Multi-Topic IC.
    filename: đường dẫn file CSV edge list với các cột source, target, weight1...weightK
    k: tổng số seed cần chọn (chia cho các topic linh hoạt)
    Trả về: danh sách seeds theo từng topic và giá trị influence cuối cùng
    """
    # 1. Đọc graph và xác định topics
    G = pd.read_csv(filename)
    weight_cols = [c for c in G.columns if c.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(G['target'])

    # 2. Các cấu trúc lưu kết quả
    seeds = [[] for _ in range(K)]
    current_spread = 0.0

    # 3. Khởi tạo heap CELF: [(-gain, node, topic, last_iter)]
    heap = []
    for node in tqdm(nodes, desc="Khởi tạo CELF"):
        for topic in range(K):
            # thử chỉ seed node đơn lẻ trên topic
            tmp_seeds = [[] for _ in range(K)]
            tmp_seeds[topic] = [node]
            spread = IC(G, tmp_seeds)
            gain = spread - current_spread
            # lưu negative gain để heapq thành max-heap
            heapq.heappush(heap, (-gain, node, topic, 0))

    # 4. Vòng lặp chọn k seeds
    for i in range(1, k + 1):
        while True:
            # lấy phần tử có gain cao nhất
            neg_gain, node, topic, last_iter = heapq.heappop(heap)
            gain = -neg_gain

            if last_iter < i:
                # gain đã lỗi thời, tính lại
                tmp_seeds = [s.copy() for s in seeds]
                tmp_seeds[topic].append(node)
                new_spread = IC(G, tmp_seeds)
                new_gain = new_spread - current_spread
                # đẩy lại với last_iter = i
                heapq.heappush(heap, (-new_gain, node, topic, i))
            else:
                # gain còn đúng, chọn node này
                seeds[topic].append(node)
                current_spread += gain
                # rồi thoát vòng while để chuyển sang i+1
                break

    return seeds, current_spread


import pandas as pd
import time
from lazy_mp import greedy_lazy_mp  # Đảm bảo import đúng nếu tách file, nếu không thì chỉ cần dùng trực tiếp hàm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Chạy Lazy Greedy nhiều lần"
    )
    parser.add_argument("--input", type=str, default="graph.csv", help="CSV source,target,weight*")
    parser.add_argument("--k", type=int, default=5, help="Số seed cần chọn")
    parser.add_argument("--mc", type=int, default=1000, help="MC simulations per estimate")
    parser.add_argument("--workers", type=int, default=None, help="Số process for parallel")
    parser.add_argument("--repeat", type=int, default=10, help="Số lần chạy")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    for i in range(1, args.repeat + 1):
        start = time.time()
        seeds, spread = greedy_lazy_mp(df, args.k, mc=args.mc, n_workers=args.workers)
        elapsed = time.time() - start

        print(f"\n🟦 Lần chạy {i}:")
        print("🕒 Kết quả lazy greedy mp (CELF + MP):")
        for idx, s in enumerate(seeds, 1):
            print(f"  Chủ đề {idx}: {s}")
        print(f"  ➤ Spread (union) = {spread:.2f}")
        print(f"  ⏱ Time = {elapsed:.3f}s")
