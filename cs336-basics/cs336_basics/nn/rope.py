import torch
import torch.nn as nn
import einops


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        NOTE: 有缓存，可以考虑全局就一个RoPE，所有层共享
        构建RoPE模块，预计算并注册cos、sin旋转位置编码缓冲区
        RoPE 角度计算公式 LaTeX：
        $$
        \omega_k = \Theta^{-\frac{2k}{d_k}},\quad
        \theta_{i,k} = i\cdot \omega_k
        $$
        其中：
        $i$ 为 token 绝对位置，$k=0,1,\dots,\frac{d_k}{2}-1$ 为维度分组下标；
        每组二维旋转矩阵：
        $$
        R_k^i=
        \begin{pmatrix}
        \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
        \sin(\theta_{i,k}) & \cos(\theta_{i,k})
        \end{pmatrix}
        $$
        :param theta: float, RoPE公式中的基底超参数$\Theta$
        :param d_k: int, Query/Key向量的单头维度，需为偶数（RoPE按两两维度成对旋转）
        :param max_seq_len: int, 模型支持的最大输入序列长度，用于预计算位置编码
        :param device: torch.device | None = None, 缓冲区张量的存储设备
        :param dtype: torch.dtype | None = None, 缓存区张量的数据类型
        """
        super().__init__()
        # d_k必须被2整除
        assert d_k % 2 == 0, "RoPE requires even dimension d_k"

        # 实现预计算频率、各位置cos/sin值并注册buffer的逻辑
        K = torch.arange(d_k // 2, device=device, dtype=dtype)
        # 注意这里k是从1开始的，但下标是从0开始，公式化简了：
        f_K = 1.0 / theta ** (2 * K / d_k)
        f_K = einops.rearrange(f_K, "k -> 1 k")  # shape: (1, d_k // 2)
        I = torch.arange(max_seq_len, device=device, dtype=dtype)
        I = einops.rearrange(I, "i -> i 1")  # shape: (max_seq_len, 1)
        theta_table = I * f_K  # shape: (max_seq_len, d_k // 2) 广播机制
        cos_table = torch.cos(theta_table)
        sin_table = torch.sin(theta_table)

        # 保存到缓存中
        # persistent=False表示不存到model.state_dict()，也就是不保存到权重文件中
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用旋转位置编码RoPE
        :param x: torch.Tensor, 待旋转的向量，形状为(..., seq_len, d_k)，支持任意数量的前置batch维度
        :param token_positions: torch.Tensor, 每个token对应的序列位置，形状为(..., seq_len)，前置维度需和x广播兼容
        :return: torch.Tensor, 经过RoPE旋转后的张量，形状与输入x完全一致

        NOTE: 注意这里为什么不在方法里生成token_position，因为每一层TransformerBlock都要用，所以在TransformerLM中生成一次然后共享
        """
        # shape: (..., seq_len, d_k // 2)
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        # 把最后一维 d_k 拆成 (pair_num, 2) 两两分组
        # shape: (..., seq_len, d_k // 2, 2)
        x_pairs = einops.rearrange(x, "... (half_d_k two)-> ... half_d_k two", two=2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]

        # RoPE 旋转核心计算
        # shape: (..., seq_len, d_k // 2)
        x_rot0 = x0 * cos - x1 * sin
        x_rot1 = x0 * sin + x1 * cos

        # 拼接变形
        x_rot = torch.stack([x_rot0, x_rot1], dim=0)
        x_out = einops.rearrange(x_rot, "two ... half_d_k -> ... (half_d_k two)")

        return x_out
