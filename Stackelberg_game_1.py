import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 使用 Agg 后端进行图像保存
# 设置随机种子以保证结果可重复

np.random.seed(520)

# 系统参数
A = np.array([[0, 1], [0, -1]])
B1 = np.array([[0.3], [0.8]])
B2 = np.array([[0.8], [0.2]])
C = np.array([[1, 0], [0, -1]])
D1 = np.array([[0.3], [0.8]])
D2 = np.array([[-0.8], [0.2]])

# 性能指标参数
Q1 = np.array([[0.1, 0], [0, 0.1]])
Q2 = np.array([[0.5, 0], [0, 0.5]])
R1, R2 = 0.5, 0.5
theta1, theta2 = 0.1, 0.1
rho1, rho2 = 3, 3
lambda1, lambda2 = 10, 10

# 初始化参数
W1 = np.array([1, 1, 1, 1])
W2 = np.array([1, 1, 1, 1])
alpha = 0
gamma = 0
Sigma_pi = lambda1 / (2 * R1)
Sigma_gamma = lambda2 / (2 * R2)
epsilon = 1e-3
M =50  # 数据点数量

# 生成探测噪声的频率参数
w1_freqs = np.random.uniform(-100, 100, size=100)
w2_freqs = np.random.uniform(-100, 100, size=100)

# 生成初始数据点
dt = 0.01
X = []
x = np.array([3.0, 3.0])  # 初始状态

for _ in range(M):
    t = _ * dt
    # 计算探测噪声
    e1 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w1_freqs])
    e2 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w2_freqs])

    # 生成控制输入
    u1 = np.random.normal(alpha + e1, np.sqrt(Sigma_pi))
    u2 = np.random.normal(gamma + e2, np.sqrt(Sigma_gamma))

    # 计算状态增量
    deterministic = (A @ x.reshape(-1, 1) + B1 * u1 + B2 * u2) * dt
    noise_C = C @ x.reshape(-1, 1) * np.sqrt(dt) * np.random.normal()
    noise_D1 = D1 * u1 * np.sqrt(dt) * np.random.normal()
    noise_D2 = D2 * u2 * np.sqrt(dt) * np.random.normal()

    dx = deterministic.flatten() + noise_C.flatten() + noise_D1.flatten() + noise_D2.flatten()
    x += dx
    X.append(x.copy())

X = np.array(X)
print("系统数据点为:", X)

# 辅助函数定义
def phi(x):
    return np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2, 1])


def compute_grad_phi(x):
    return np.array([
        [2 * x[0], 0],
        [x[1], x[0]],
        [0, 2 * x[1]],
        [0, 0]
    ])


def compute_hessian_phi():
    return [
        np.array([[2, 0], [0, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 0], [0, 2]]),
        np.array([[0, 0], [0, 0]])
    ]


hessians = compute_hessian_phi()


def compute_dVdx(W, x):
    grad_phi = compute_grad_phi(x)
    return W.T @ grad_phi


def compute_d2Vdx2(W):
    return sum(W[i] * hessians[i] for i in range(4))

W1_history = []
W2_history = []
# 主循环
converged = False
s = 0
max_iterations = 100

while not converged and s < max_iterations:
    # ================== 领导者更新 ==================
    # Step 2a: 计算LAM1和LAM2
    # 使用追随者当前的值函数参数W2
    d2V2_dx2 = compute_d2Vdx2(W2)
    matrix_term = 2 * R2 + D2.T @ d2V2_dx2 @ D2

    # 计算LAM1_s
    LAM1_s = -(2 * theta2 * R2 + D2.T @ d2V2_dx2 @ D1)/matrix_term

    # 计算LAM2_s（使用第6个状态x[6]）
    x_m = X[6].T
    dV2_dx_m = compute_dVdx(W2, x_m)
    LAM2_s = -(D2.T @ d2V2_dx2 @ C @ x_m + B2.T @ dV2_dx_m)/matrix_term

    # Step 2b: 值函数更新
    xi1 = np.array([phi(x) for x in X]).T
    H1_list = []

    for x in X:
        # 计算各项
        x = x.reshape(-1, 1)
        term1 = x.T @ Q1 @ x
        term2 = R1 * (alpha + theta1 * gamma) ** 2 + R1 * Sigma_pi + (theta1 ** 2) * R1 * Sigma_gamma
        term3 = -lambda1 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_pi))

        dV1_dx = compute_dVdx(W1, x.flatten())
        term4 = dV1_dx @ (A @ x + B1 * alpha + B2 * gamma)

        d2V1_dx2 = compute_d2Vdx2(W1)
        term5 = 0.5 * x.T @ C.T @ d2V1_dx2 @ C @ x
        term6 = 0.5 * D1.T @ d2V1_dx2 @ D1 * (alpha ** 2 + Sigma_pi)
        term7 = 0.5 * D2.T @ d2V1_dx2 @ D2 * (gamma ** 2 + Sigma_gamma)
        term8 = x.T @ C.T @ d2V1_dx2 @ D1 * alpha
        term9 = x.T @ C.T @ d2V1_dx2 @ D2 * gamma
        term10 = alpha * D1.T @ d2V1_dx2 @ D2 * gamma

        H1 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10).item()
        H1_list.append(H1)

    chi1 = np.array(H1_list) / rho1
    print("chi1为:", chi1)
    W1_new = np.linalg.inv(xi1 @ xi1.T) @ xi1 @ chi1  #不可逆时候用伪逆np.linalg.pinv(xi1 @ xi1.T)

    # Step 2c: 策略更新
    d2V1_new_dx2 = compute_d2Vdx2(W1_new)
    Sigma_pi_new = lambda1 / (2 * R1 + D1.T @ d2V1_new_dx2 @ D1).item()

    # 计算新alpha
    denominator1 = (2 * R1 + D1.T @ d2V1_new_dx2 @ D1+ D1.T @ d2V1_new_dx2 @ D2* LAM1_s+ LAM1_s * D2.T @ d2V1_new_dx2 @ D2 * LAM1_s+ 2 * theta1 ** 2 * LAM1_s * R1 * LAM1_s+ 4 * theta1 * R1* LAM1_s+ D2.T @ d2V1_new_dx2 @ D1* LAM1_s)
    numerator = (B1.T @ compute_dVdx(W1_new, x_m) +
                 LAM1_s * B2.T @ compute_dVdx(W1_new, x_m) +
                 D1.T @ d2V1_new_dx2 @ C @ x_m.reshape(-1, 1) +
                 LAM1_s * D2.T @ d2V1_new_dx2 @ C @ x_m.reshape(-1, 1) +
                 (D2.T @ d2V1_new_dx2 @ D2 * LAM1_s + 2 * theta1 * R1 +
                  2 * theta1 ** 2 * R1 * LAM1_s + D1.T @ d2V1_new_dx2 @ D2) * LAM2_s)

    alpha_new = -numerator / denominator1

    # ================== 追随者更新 ==================
    # Step 3a: 值函数更新
    xi2 = np.array([phi(x) for x in X]).T
    H2_list = []

    for x in X:
        x = x.reshape(-1, 1)
        term1 = x.T @ Q2 @ x
        term2 = R2 * (gamma + theta2 * alpha_new) ** 2 + R2 * Sigma_gamma + (theta2 ** 2) * R2 * Sigma_pi
        term3 = -lambda2 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_gamma))

        dV2_dx = compute_dVdx(W2, x.flatten())
        term4 = dV2_dx @ (A @ x + B1 * alpha_new + B2 * gamma)

        d2V2_dx2 = compute_d2Vdx2(W2)
        term5 = 0.5 * x.T @ C.T @ d2V2_dx2 @ C @ x
        term6 = 0.5 * D1.T @ d2V2_dx2 @ D1 * (alpha_new ** 2 + Sigma_pi)
        term7 = 0.5 * D2.T @ d2V2_dx2 @ D2 * (gamma ** 2 + Sigma_gamma)
        term8 = x.T @ C.T @ d2V2_dx2 @ D1 * alpha_new
        term9 = x.T @ C.T @ d2V2_dx2 @ D2 * gamma
        term10 = alpha_new * D1.T @ d2V2_dx2 @ D2 * gamma

        H2 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10).item()
        H2_list.append(H2)

    chi2 = np.array(H2_list) / rho2
    W2_new = np.linalg.inv(xi2 @ xi2.T) @ xi2 @ chi2  #不可逆时候用伪逆np.linalg.pinv(xi2 @ xi2.T)

    # Step 3b: 策略更新
    d2V2_new_dx2 = compute_d2Vdx2(W2_new)
    Sigma_gamma_new = lambda2 / (2 * R2 + D2.T @ d2V2_new_dx2 @ D2).item()

    denominator4 = -(2 * R2 + D2.T @ d2V2_new_dx2 @ D2)
    denominator5 = (2 * theta2 * R2 + D2.T @ d2V2_new_dx2 @ D1)
    denominator6 = D2.T @ d2V2_new_dx2 @ C @ x_m + B2.T @ compute_dVdx(W2_new, x_m)
    gamma_new = denominator5 * alpha_new/denominator4 + denominator6 / denominator4

    # 检查收敛
    V1_diff = np.linalg.norm(W1_new - W1)
    V2_diff = np.linalg.norm(W2_new - W2)
    print("V1_diff为:", V1_diff)
    print("V2_diff为:", V2_diff)

    if V1_diff < epsilon and V2_diff < epsilon:
        converged = True
    else:
        W1 = W1_new
        W2 = W2_new
        alpha = alpha_new.item()
        gamma = gamma_new.item()
        Sigma_pi = Sigma_pi_new
        Sigma_gamma = Sigma_gamma_new
        s += 1

 # 输出结果
    print("收敛于迭代次数:", s)
    print("最优策略:")
    print("领导者策略: N({:.4f}, {:.4f})".format(alpha, Sigma_pi))
    print("追随者策略: N({:.4f}, {:.4f})".format(gamma, Sigma_gamma))
    print("值函数参数:")
    print("W1:", W1)
    print("W2:", W2)
    W1_history.append(W1.copy())
    W2_history.append(W2.copy())

# ========================= 绘制权重图 ================================
# 将历史记录转换为数组
W1_history = np.array(W1_history)
W2_history = np.array(W2_history)

# 创建图表
plt.figure(figsize=(14, 6))

# 绘制W1各分量
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.plot(W1_history[:, i], label=f'W1_{i}', color='orange',linestyle='--')
    plt.title(f'W1 Component {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

# 绘制W2各分量
for i in range(4):
    plt.subplot(2, 4, i+5)
    plt.plot(W2_history[:, i], label=f'W2_{i}',color='blue',linestyle='-')
    plt.title(f'W2 Component {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# ========================= 绘制最终策略的正态分布 ================================
# 生成x轴数据
x = np.linspace(-50, 50, 1000)

# 计算概率密度函数
pdf_leader = stats.norm.pdf(x, alpha, Sigma_pi)
pdf_follower = stats.norm.pdf(x, gamma, Sigma_gamma)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_leader, label=f'Leader: N({alpha:.2f}, {Sigma_pi:.2f})', color='red', linewidth=2, linestyle='--')
plt.plot(x, pdf_follower, label=f'Follower: N({gamma:.2f}, {Sigma_gamma:.2f})', color='blue', linewidth=2, linestyle='-')


plt.title('Optimal Strategies', fontsize=14)
plt.xlabel('Control Input', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(title=f'λ1=λ2={lambda1},Θ1=Θ2={theta1}', loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ========================= 绘制价值函数变化图像 ================================
# 假设使用第6个数据点，计算每次迭代的V1和V2
V1_values = [W.dot(phi(X[6])) for W in W1_history]
V2_values = [W.dot(phi(X[6])) for W in W2_history]

# 创建迭代次数列表
iterations = range(len(V1_values))
# 绘制价值函数变化曲线
plt.figure(figsize=(8, 6))
plt.plot(iterations, V1_values, 'r--', linewidth=2, label='Leader V1')
plt.plot(iterations, V2_values, 'b-', linewidth=2, label='Follower V2')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Value Function', fontsize=12)
plt.title('Convergence of Value Functions', fontsize=14)
plt.legend(title=f'λ1=λ2={lambda1}',loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# ================================= hcd====================================