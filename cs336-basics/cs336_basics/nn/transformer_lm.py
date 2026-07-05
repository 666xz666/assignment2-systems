import torch
from torch import nn

from .embedding import Embedding
from .transformer import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear
from .softmax import Softmax


class TransformerLM(nn.Module):
    r"""
    Decoder-only 结构 Transformer 自回归语言模型（Pre-Norm 架构，搭配 RoPE 旋转位置编码）
    整体前向流程：
    $$
    \begin{align*}
    &\text{token\_emb} = \text{Embedding}(tokens) \\
    &h_0 = \text{token\_emb} \\
    &h_l = \text{TransformerBlock}_l(h_{l-1},\ \text{pos\_indices}),\quad l=1,2,\dots,\text{num\_layers} \\
    &h_{\text{final}} = \text{RMSNorm}(h_{\text{num\_layers}}) \\
    &\text{logits} = h_{\text{final}} \cdot \text{W}_{\text{lm\_head}}^\top
    \end{align*}
    $$
    模块组成：
    1. Token 嵌入层：将离散 token id 映射为 $d_\text{model}$ 维度向量
    2. 堆叠多层因果 Pre-Norm Transformer Block（内置多头自注意力 + SwiGLU FFN + RoPE）
    3. 顶层最终归一化
    4. 语言模型头：将特征投影至词表维度，输出每个位置词汇分布 logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        theta: float | None = None,
    ) -> None:
        """
        Args:
            vocab_size: int
                词表总大小，决定 token 嵌入矩阵与最终输出头维度
            context_length: int
                模型支持最大上下文长度，用于 RoPE 预计算三角函数缓冲区
            d_model: int
                全局嵌入/特征维度，贯穿整个模型
            num_heads: int
                单个 Transformer Block 内多头注意力头数量
            d_ff: int
                前馈网络 SwiGLU 内部隐藏层维度
            num_layers: int
                Transformer Block 堆叠层数
            device: torch.device
                全部参数与张量运算设备
            dtype: torch.dtype
                权重与前向传播张量精度类型
            theta: float | None = None
                RoPE 旋转位置编码基础角度超参；传入 None 则关闭 RoPE
        """
        super().__init__()

        # 模型预设的最长上下文
        self.context_length = context_length

        # Token Embedding 层
        self.emb = Embedding(vocab_size, d_model, device, dtype)

        # 堆叠多层 Transformer Block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                    theta=theta,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        # 模型顶层最终归一化
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # LM 输出头：映射到词表维度
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        r"""
        Transformer 语言模型前向推理，输出逐位置词汇 logits

        Args:
            token_ids: torch.Tensor
                输入 token 序号张量，形状 $(\dots,\ \text{seq\_len})$，取值范围 $[0,\ vocab\_size-1]$

        Returns:
            torch.Tensor
                输出词汇 logits，形状 $(\dots,\ \text{seq\_len},\ \text{vocab\_size})$，
                对应每个位置下所有词表 token 的原始得分
        """
        # Embedding
        x = self.emb(token_ids)

        # 生成token_positions
        seq_len = token_ids.size(-1)
        # 形状就是(seq_len)后续利用广播机制展开计算
        token_positions = torch.arange(
            0, seq_len, device=token_ids.device, dtype=torch.long
        )

        # Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Norm
        x = self.final_norm(x)

        # 输出Next Token Logits
        # 交叉熵损失函数自带softmax，这里不用先处理
        x = self.lm_head(x)
        return x
