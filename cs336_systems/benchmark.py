# 覆盖添加注解
from cs336_systems.annotated import ScaledDotProductAttention
import cs336_basics.nn.scaled_dot_product_attention as sdpa

sdpa.ScaledDotProductAttention = ScaledDotProductAttention
# 强制重载多头注意力模块，刷新内部缓存的类引用
import cs336_basics.nn.multihead_self_attention
import importlib

importlib.reload(cs336_basics.nn.multihead_self_attention)

# 库依赖
import torch
import timeit
from typing import Callable, List, Dict
import math
import pandas as pd
import os
import torch.cuda.nvtx as nvtx

# 作业一实现
from cs336_basics.nn import TransformerLM
from cs336_basics.utils import try_gpu, cross_entropy
from cs336_basics.optim import AdamW


class Timer:
    def __enter__(self):
        self.cost = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize()  # 在计时器内部做同步
        self.cost = timeit.default_timer() - self.cost


def mean(x: list[float]) -> float:
    """Returns the average value of a list of float number"""
    return sum(x) / len(x)


def std_sample(x: list[float]) -> float:
    """Returns the standard deviation of a list of float number"""
    mean_val = mean(x)
    var = sum((xi - mean_val) ** 2 for xi in x) / (len(x) - 1)
    return math.sqrt(var)


def init_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float | None = 10_000.0,
) -> TransformerLM:
    """
    Given hyperparameters (e.g., number of layers), initialize a model.
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
        device=try_gpu(),
        dtype=torch.float32,
    )
    return model


def generate_random_data(
    batch_size: int, context_len: int, d_model: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = try_gpu()
    x = torch.randint(
        0, vocab_size, size=(batch_size, context_len), dtype=torch.long, device=dv
    )
    y = torch.randint(
        0, vocab_size, size=(batch_size, context_len), dtype=torch.long, device=dv
    )
    return x, y


def benchmark_full(
    warm_up_steps: int,
    num_steps: int,
    forward_step: Callable,
    backward_step: Callable,
    train_step: Callable,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # 预热整体NVTX标记，nsys可过滤掉预热
    with nvtx.range(f"Warmup_{warm_up_steps}_Steps"):
        for _ in range(warm_up_steps):
            out = forward_step()
            backward_step(out)
            train_step()
    torch.cuda.synchronize()

    f_times = []
    b_times = []
    opt_times = []

    # 测量总区间
    with nvtx.range(f"Measurement_{num_steps}_Iter"):
        for _ in range(num_steps):
            # 单独标记前向
            with nvtx.range("ForwardPass"):
                with Timer() as t_f:
                    out = forward_step()
                f_times.append(t_f.cost)

            # 单独标记反向
            with nvtx.range("BackwardPass"):
                with Timer() as t_b:
                    backward_step(out)
                b_times.append(t_b.cost)

            # 完整训练步：前向+反向+优化器（用于题目d）
            with nvtx.range("FullTrainStep_ForwardBackward_Optimizer"):
                with Timer() as t_opt:
                    o = forward_step()
                    backward_step(o)
                    train_step()
                opt_times.append(t_opt.cost)

    f_mean, f_std = mean(f_times), std_sample(f_times)
    b_mean, b_std = mean(b_times), std_sample(b_times)
    opt_mean, opt_std = mean(opt_times), std_sample(opt_times)
    return (f_mean, f_std), (b_mean, b_std), (opt_mean, opt_std)


default_hyper_params = {"vocab_size": 10_000, "batch_size": 4, "context_length": 512}
model_sizes = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    # {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # {"size": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    # {"size": "10B", "d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
]


def single_model_bench(cfg, warm_up_steps):
    model = init_model(
        vocab_size=default_hyper_params["vocab_size"],
        context_length=default_hyper_params["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10_000,
    )
    x, y = generate_random_data(
        batch_size=default_hyper_params["batch_size"],
        context_len=default_hyper_params["context_length"],
        d_model=cfg["d_model"],
        vocab_size=default_hyper_params["vocab_size"],
    )
    opt = AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    def forward_step() -> torch.Tensor:
        return model(x)

    def backward_step(out: torch.Tensor):
        model.zero_grad()
        loss = cross_entropy(out, y)
        loss.backward()

    def train_step():
        opt.step()

    f_res, b_res, opt_res = benchmark_full(
        warm_up_steps=warm_up_steps,
        num_steps=10,
        forward_step=forward_step,
        backward_step=backward_step,
        train_step=train_step,
    )
    del model, x, y, opt
    torch.cuda.empty_cache()
    return f_res, b_res, opt_res


def run_benchmark():
    res_list: List[Dict] = []
    # warmup_configs = [0, 2, 5] # 测试不warm up、warm up 1-2步
    warmup_configs = [5]

    for warmup in warmup_configs:
        print(f"===== Running warm_up_steps = {warmup} =====")
        for cfg in model_sizes:
            f_res, b_res, opt_res = single_model_bench(cfg, warm_up_steps=warmup)
            row = {
                "warmup_steps": warmup,
                "Size": cfg["size"],
                "Tf_mean": f_res[0],
                "Tf_std": f_res[1],
                "Tb_mean": b_res[0],
                "Tb_std": b_res[1],
                "Topt_mean": opt_res[0],
                "Topt_std": opt_res[1],
            }
            res_list.append(row)

    df = pd.DataFrame(res_list)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "bench_warmup_compare.csv")
    df.to_csv(out_path, index=False)
    print("\n==== All benchmark results ====")
    print(df)
    return df


OUT_DIR = "./output/benchmark"
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    run_benchmark()


"""
uv run nsys profile \
--trace=cuda,cudnn,cublas,osrt,nvtx \
--pytorch=functions-trace,autograd-shapes-nvtx \
-o ./output/nsys/benchmark.nsys-rep \
-- python cs336_systems/benchmark.py
"""
