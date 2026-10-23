"""
Self-Attention (Scaled Dot-Product Attention)
来源: Vaswani et al., "Attention Is All You Need" (2017), 公式 (1):
    Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V

本文件用 numpy 把每一步显式写出来, 方便对照数学公式逐行理解。
直接运行: python self_attention.py
"""

import numpy as np


def softmax(x, axis=-1):
    """对指定维度做 softmax, 带数值稳定处理 (减去 max 防止 exp 溢出)。"""
    x_max = np.max(x, axis=axis, keepdims=True)   # 每行最大值
    e = np.exp(x - x_max)                          # 平移后再取指数, 结果不变但更稳定
    return e / np.sum(e, axis=axis, keepdims=True) # 归一化成概率分布 (每行和为 1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    核心注意力函数。
    Q: (n, d_k)  查询   query
    K: (m, d_k)  键     key
    V: (m, d_v)  值     value
    返回: (n, d_v) 的输出, 以及注意力权重 (n, m)
    """
    d_k = Q.shape[-1]                       # 每个 query/key 向量的维度
    scores = Q @ K.T                        # 步骤1: 相似度打分 QKᵀ, 形状 (n, m)
    scores = scores / np.sqrt(d_k)          # 步骤2: 缩放, 除以 sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # 步骤3(可选): 被屏蔽的位置设成 -∞
    weights = softmax(scores, axis=-1)      # 步骤4: softmax 得到注意力权重 (每行和=1)
    output = weights @ V                    # 步骤5: 用权重对 V 加权求和
    return output, weights


class SelfAttention:
    """
    自注意力层: 输入序列自己生成 Q, K, V。
    用三个可学习的权重矩阵 W_q, W_k, W_v 把输入 X 投影成 Q, K, V。
    """

    def __init__(self, d_model, d_k, d_v, seed=0):
        rng = np.random.default_rng(seed)
        # 权重初始化 (真实训练中这些是 nn.Linear 的参数, 由反向传播更新)
        self.W_q = rng.normal(0, 0.1, size=(d_model, d_k))  # (d_model, d_k)
        self.W_k = rng.normal(0, 0.1, size=(d_model, d_k))  # (d_model, d_k)
        self.W_v = rng.normal(0, 0.1, size=(d_model, d_v))  # (d_model, d_v)

    def forward(self, X, mask=None):
        """
        X: (seq_len, d_model)  输入序列, 每行是一个 token 的向量
        """
        Q = X @ self.W_q   # 投影出 query, 形状 (seq_len, d_k)
        K = X @ self.W_k   # 投影出 key,   形状 (seq_len, d_k)
        V = X @ self.W_v   # 投影出 value, 形状 (seq_len, d_v)
        return scaled_dot_product_attention(Q, K, V, mask)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # 造一个长度为 4 的序列, 每个 token 用 8 维向量表示
    seq_len, d_model, d_k, d_v = 4, 8, 8, 8
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(seq_len, d_model))

    attn = SelfAttention(d_model, d_k, d_v)

    print("=== 1. 普通自注意力 (每个 token 能看到所有 token) ===")
    out, w = attn.forward(X)
    print("注意力权重 (每行和应为 1):")
    print(w)
    print("每行求和:", w.sum(axis=1))
    print("输出形状:", out.shape)

    print("\n=== 2. 因果掩码 (第 i 个 token 只能看到 <= i 的 token) ===")
    # 下三角为 True 表示允许关注
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    out_m, w_m = attn.forward(X, mask=causal_mask)
    print("掩码矩阵:\n", causal_mask)
    print("掩码后的注意力权重 (上三角被清零):")
    print(w_m)
