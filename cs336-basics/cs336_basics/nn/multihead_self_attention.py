import torch
from torch import nn
import einops

from .rope import RoPE
from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiheadSelfAttention(nn.Module):
    r"""
    Causal Multi-Head Self-Attention 因果多头自注意力，实现 Vaswani 等人原始 Transformer 多头注意力结构
    数学定义：
    $$
    \begin{align*}
    \text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,\dots,\text{head}_h) \\
    \text{head}_i &= \text{Attention}(Q_i,K_i,V_i)
    \end{align*}
    $$
    其中 $Q_i,K_i,V_i$ 分别为 $Q,K,V$ 在特征维度上划分出的第 $i$ 个头，单头执行缩放点积注意力运算；
    完整自注意力前向运算：
    $$
    \text{MultiHeadSelfAttention}(x) = W_O \cdot \text{MultiHead}\big(W_Q x,\ W_K x,\ W_V x\big)
    $$
    可学习参数：
    $W_Q \in \mathbb{R}^{h d_k \times d_\text{model}},\ W_K \in \mathbb{R}^{h d_k \times d_\text{model}},\ W_V \in \mathbb{R}^{h d_v \times d_\text{model}},\ W_O \in \mathbb{R}^{d_\text{model} \times h d_v}$
    内置因果掩码约束：token $i$ 仅能访问位置 $j \le i$，无法看到未来时序位置；
    可选启用 RoPE 旋转位置编码，在分头后对 Q、K 施加位置旋转变换。

    Args:
        d_model: 模型整体嵌入维度 $d_\text{model}$
        num_heads: 多头注意力头数量 $h$
        device: 张量运算设备
        dtype: 张量精度类型
        theta: RoPE 基础角度超参；传入 None 则不启用 RoPE
        max_seq_len: RoPE 预计算最大序列长度
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
        theta: float | None = None,
        max_seq_len: int | None = -1,
    ):
        r"""
        初始化因果多头自注意力模块，包含QKV投影、RoPE位置编码、多头注意力计算组件

        Args:
            d_model: 模型总嵌入维度 $d_\text{model}$，要求可被头数 num_heads 整除
            num_heads: 注意力头总数量 $h$
            device: 所有参数与张量运算所在设备（CPU / CUDA）
            dtype: 权重与前向张量数据精度（float32 / bfloat16 等）
            theta: RoPE 旋转位置编码基础角度超参数；传入 `None` 代表不启用 RoPE
            max_seq_len: RoPE 预计算三角函数的最大支持序列长度
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 判断是否应用RoPE
        self.use_rope = False
        if theta is not None:
            self.use_rope = True

        self.rope = None
        if self.use_rope:
            # 共享RoPE
            self.rope = RoPE(theta, self.d_head, max_seq_len, device, dtype)

        # 注意力
        self.attn = ScaledDotProductAttention()

        # ================初始化投影矩阵参数================
        self.Proj_q = Linear(d_model, d_model, device, dtype)

        """
        NOTE: As a stretch goal, try combining the key, query, and value projections 
        into a single weight matrix so you only need a single matrix multiply.

        所以这里其实是生成了num_heads组K, V
        """
        self.Proj_k = Linear(d_model, d_model, device, dtype)
        self.Proj_v = Linear(d_model, d_model, device, dtype)

        # 最后还有个线性层
        self.W_o = Linear(d_model, d_model, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        前向传播执行因果多头自注意力计算

        Args:
            x: 输入特征张量，形状 $(\dots,\ \text{seq\_len},\ d_\text{model})$
            token_positions: RoPE 所需位置下标张量，形状匹配序列维度；不启用RoPE仍需占位传参

        Returns:
            torch.Tensor: 多头注意力输出特征，形状与输入 x 完全一致 $(\dots,\ \text{seq\_len},\ d_\text{model})$
        """
        q = self.Proj_q(x)
        k = self.Proj_k(x)
        v = self.Proj_v(x)

        # 分头
        q = einops.rearrange(
            q,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
            d=self.d_head,
        )
        k = einops.rearrange(
            k,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
            d=self.d_head,
        )
        v = einops.rearrange(
            v,
            "... seq (head d) -> ... head seq d",
            head=self.num_heads,
            d=self.d_head,
        )

        """
        NOTE: in multi-head attention, attention is being applied independently for each head. This means that precisely the same RoPE rotation should be applied to the ** query ** and ** key ** vectors for each head.

        NOTE: 这一步必须在分头后做，不然d这一维的下标会对不上
        """
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 因果掩码
        # "Your implementation should prevent the model from attending to future tokens in the sequence."
        # "we’ll use causal attention masking, which allows token i to attend to all positions j ≤ i in the sequence."
        seq_len = x.size()[-2]
        # triu只保留上三角，下部分全弄成0或者False，取决于dtype
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,  # 向上偏移主对角线1
        )
        # 实际上要取下三角
        causal_mask = ~causal_mask

        # 并行计算Attention
        attn = self.attn(q, k, v, causal_mask)

        # 合并
        res = einops.rearrange(attn, "... head seq d -> ... seq (head d)")

        # 最后过个线性层
        res = self.W_o(res)
        return res
