import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- パラメータ設定 ---
# ==========================================
L = 1.0       # 棒の長さ (m)
nx = 50       # 空間の分割数
dx = L / nx   # 空間ステップ (m)
alpha = 0.01  # 熱拡散率 (m^2/s)

T = 5.0       # シミュレーションの総時間 (s)
nt = 1000     # 時間の分割数
dt = T / nt   # 時間ステップ (s)

# 安定性条件の確認 (フーリエ数 F <= 0.5)
F = alpha * dt / dx**2
print(f"フーリエ数 (F): {F:.3f}")
if F > 0.5:
    print("警告: 安定性条件を満たしていません。dtを小さくするか、dxを大きくしてください。")

# ==========================================
# --- 初期条件と境界温度設定 ---
# ==========================================
# 初期温度分布の設定
u = np.zeros(nx)                        # 全体を0℃に設定
u[int(nx/2)-5:int(nx/2)+5] = 100.0      # 中央部分を100℃に加熱
u_new = np.copy(u)

# ==========================================
# --- 記録の設定 ---
# ==========================================
# 複数の時刻における温度の空間分布を記録します
record_times = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] # グラフに表示したい時刻 (秒)
history_dist = {}
history_dist[0.0] = np.copy(u) # 初期状態を記録

# ==========================================
# --- メインループ（時間発展） ---
# ==========================================
for n in range(1, nt + 1):
    # 内側（両端を除く領域）の熱伝導を計算（対流なし）
    for i in range(1, nx - 1):
        u_new[i] = u[i] + F * (u[i+1] - 2*u[i] + u[i-1])
    
    # 境界が無限に続いていると仮定し、境界付近の温度勾配が一定（2階微分が0）であるとみなす線形外挿
    u_new[0] = 2.0 * u_new[1] - u_new[2]
    u_new[-1] = 2.0 * u_new[-2] - u_new[-3]
    
    u = np.copy(u_new)
    
    # 現在の時刻を計算
    current_time = n * dt
    
    # 設定した記録時刻に達したら、その時点の温度分布全体を保存
    for t_rec in record_times:
        if t_rec > 0.0 and abs(current_time - t_rec) < dt / 2:
            history_dist[t_rec] = np.copy(u)

# ==========================================
# --- 結果の可視化 ---
# ==========================================
# 横軸を「熱源（中央）からの距離」とするため、-L/2 から L/2 の座標配列を作成
x_coords = np.linspace(-L/2, L/2, nx)

plt.figure(figsize=(10, 6))

# 保存した各時刻の温度分布をプロット
for t_rec in record_times:
    if t_rec in history_dist:
        plt.plot(x_coords, history_dist[t_rec], label=f"Time = {t_rec:.1f} s")

plt.title("1D Heat Conduction - Temperature Distribution")
plt.xlabel("Distance from Heat Source (m)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()