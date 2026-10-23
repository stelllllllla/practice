import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)   # 作用在 query 序列上
        self.w_k = nn.Linear(d_model, d_model)   # 作用在 key/value 序列上
        self.w_v = nn.Linear(d_model, d_model)   # 作用在 key/value 序列上
        self.w_o = nn.Linear(d_model, d_model)   # 输出投影

    def forward(self, x_q, x_kv, mask=None):
        B, Lq, _ = x_q.shape
        Lkv = x_kv.shape[1]

        # 1) 线性投影
        Q = self.w_q(x_q)    # (B, Lq,  d_model)
        K = self.w_k(x_kv)   # (B, Lkv, d_model)
        V = self.w_v(x_kv)   # (B, Lkv, d_model)

        # 2) 拆成多头
        Q = Q.view(B, Lq,  self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, Lkv, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, Lkv, self.num_heads, self.d_head).transpose(1, 2)

        # 3) 打分 + 缩放
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4) 可选 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5) softmax 归一化
        attn = F.softmax(scores, dim=-1)

        # 6) 加权求和
        out = attn @ V

        # 7) 合并多头
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)

        # 8) 输出投影
        out = self.w_o(out)
        return out, attn


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Lq, Lkv, d_model, num_heads = 2, 4, 6, 16, 4

    x_q  = torch.randn(B, Lq,  d_model)   # 比如解码器序列
    x_kv = torch.randn(B, Lkv, d_model)   # 比如编码器输出

    cross_attn = MultiHeadCrossAttention(d_model, num_heads)
    out, attn = cross_attn(x_q, x_kv)

    print("输入 query 形状 :", x_q.shape)
    print("输入 kv 形状    :", x_kv.shape)
    print("输出形状        :", out.shape)       # (B, Lq, d_model) = (2, 4, 16)
    print("注意力权重形状  :", attn.shape)       # (B, heads, Lq, Lkv) = (2, 4, 4, 6)
    print("每行权重和(应≈1):", attn.sum(dim=-1)[0, 0])
