import torch
import torch.nn as nn
from torch import autocast, GradScaler
import time
from contextlib import nullcontext


class ToyModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def print_dtype_autocast():

    # 1. 初始化模型，参数默认FP32，放GPU
    device = "cuda"
    model = ToyModel(in_features=5, out_features=2).to(device)
    print("=== 初始模型参数权重 dtype ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    # 构造输入与标签
    x = torch.randn(32, 5, device=device)
    y = torch.randn(32, 2, device=device)

    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # 开启autocast FP16混合精度
    with autocast(device_type="cuda", dtype=torch.float16):
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        print("[Loss 张量 dtype]:", loss.dtype)

    # 反向传播
    optimizer.zero_grad()
    scaler.scale(loss).backward()

    print("\n=== 反向后参数梯度 grad dtype ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}.grad: {param.grad.dtype}")

    scaler.step(optimizer)
    scaler.update()


"""
python cs336_systems/benchmarking_mixed_precision.py
=== 初始模型参数权重 dtype ===
fc1.weight: torch.float32
ln.weight: torch.float32
ln.bias: torch.float32
fc2.weight: torch.float32
[fc1 输出 dtype]: torch.float16
[LayerNorm 输出 dtype]: torch.float32
[模型最终logits dtype]: torch.float16
[Loss 张量 dtype]: torch.float32

=== 反向后参数梯度 grad dtype ===
fc1.weight.grad: torch.float32
ln.weight.grad: torch.float32
ln.bias.grad: torch.float32
fc2.weight.grad: torch.float32

NOTE: 参数和梯度始终用单精度，
各层输出的矩阵，除了norm和loss计算保留单精度，其余全部用半精度
autocast要奏效，必须使用pytorch原生自带算子
"""


def benchmark(model, input_tensor, use_bf16: bool, warmup=5, repeat=20):
    """
    use_bf16=True: BF16混合精度 autocast
    use_bf16=False: 全FP32 nullcontext空上下文
    返回平均前向耗时、平均反向耗时
    """
    # 选择上下文管理器
    if use_bf16:
        ctx = autocast(device_type="cuda", dtype=torch.bfloat16)
        scaler = GradScaler(enabled=False)  # BF16一般禁用scaler
    else:
        ctx = nullcontext()
        scaler = None

    # 预热GPU，消除初始化开销
    for _ in range(warmup):
        with ctx:
            out = model(input_tensor)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    for _ in range(repeat):
        model.zero_grad()
        torch.cuda.synchronize()  # 1. 阻塞：GPU清空历史任务，本轮计时绝对起点
        t_start = time.perf_counter()
        # ========== 【前向计时区间：t_start ~ t0】 ==========
        with ctx:
            out = model(input_tensor)  # 仅模型前传GPU计算
        torch.cuda.synchronize()  # 等待前传100%跑完，再往下走
        t0 = time.perf_counter()
        # ========== 前传计时彻底结束 ==========
        loss = out.sum()  # sum放到前向计时之外，不会污染fwd耗时
        torch.cuda.synchronize()  # 确保loss计算完毕，GPU队列清空
        # ========== 【反向计时区间：t0之后到下一次同步结束】 ==========
        loss.backward()  # 仅反向传播梯度计算
        torch.cuda.synchronize()
        # 先算差值，循环末尾再append
        fwd_dur = t0 - t_start
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        bwd_dur = t_end - t0
        # 所有计时结束再追加，完全不侵入GPU计时窗口
        fwd_times.append(fwd_dur)
        bwd_times.append(bwd_dur)

    avg_fwd = sum(fwd_times) / repeat
    avg_bwd = sum(bwd_times) / repeat
    return avg_fwd, avg_bwd


if __name__ == "__main__":
    default_hyper_params = {
        "vocab_size": 10_000,
        "batch_size": 4,
        "context_length": 512,
    }
    vocab_size = default_hyper_params["vocab_size"]
    B = default_hyper_params["batch_size"]
    L = default_hyper_params["context_length"]

    model_sizes = [
        {
            "size": "small",
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        {
            "size": "medium",
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        {
            "size": "large",
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        {
            "size": "xl",
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
        {
            "size": "10B",
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36,
        },
    ]

    for cfg in model_sizes:
        name = cfg["size"]
        d_model = cfg["d_model"]
        print(f"\n===== Model Size: {name} | d_model = {d_model} =====")

        # 实例化你的原版ToyModel，参数名完全不动
        model = ToyModel(
            in_features=d_model, hidden_dim=d_model, out_features=vocab_size
        ).to("cuda")

        # 构造输入
        input_tensor = torch.randn(B, L, d_model, device="cuda")

        # 分别测速 FP32 / BF16
        fwd_fp32, bwd_fp32 = benchmark(model, input_tensor, use_bf16=False)
        fwd_bf16, bwd_bf16 = benchmark(model, input_tensor, use_bf16=True)

        print(f"FP32  前向: {fwd_fp32:.4f}s | 反向: {bwd_fp32:.4f}s")
        print(f"BF16  前向: {fwd_bf16:.4f}s | 反向: {bwd_bf16:.2f}s")
        print(f"前向加速比: {fwd_fp32 / fwd_bf16:.2f}x")
        print(f"反向加速比: {bwd_fp32 / bwd_bf16:.2f}x")
