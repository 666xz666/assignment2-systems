import torch
import timeit
from typing import Callable, List, Dict
import math
import pandas as pd
import os

from cs336_basics.nn import TransformerLM
from cs336_basics.utils import try_gpu, cross_entropy


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
    """
    Generate a random batch of data.
    """
    dv = try_gpu()
    x = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, context_len),
        dtype=torch.long,
        device=dv,
    )
    y = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, context_len),
        dtype=torch.long,
        device=dv,
    )
    return x, y


def benchmark(
    warm_up_steps: int,
    num_steps: int,
    forward_step: Callable,
    backward_step: Callable,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Run w warm-up steps (before you start measuring time), then time the execution of n  steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument).
    """
    # warm up
    for _ in range(warm_up_steps):
        # 消除算子编译时间开销
        out = forward_step()
        backward_step(out)
    torch.cuda.synchronize()  # 等待所有cuda异步线程完成

    # 正式计数
    f_times = []
    b_times = []
    for _ in range(num_steps):
        # 单独计时前向
        with Timer() as t_f:
            out = forward_step()
        f_times.append(t_f.cost)

        # 复用前向输出，单独计时反向
        with Timer() as t_b:
            backward_step(out)
        b_times.append(t_b.cost)

    f_mean, f_std = mean(f_times), std_sample(f_times)
    b_mean, b_std = mean(b_times), std_sample(b_times)
    return (f_mean, f_std), (b_mean, b_std)


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
    """单模型执行测速，返回(前向均值std, 反向均值std)"""
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

    def forward_step() -> torch.Tensor:
        return model(x)

    def backward_step(out: torch.Tensor):
        model.zero_grad()
        loss = cross_entropy(out, y)
        loss.backward()

    f_res, b_res = benchmark(
        warm_up_steps=warm_up_steps,
        num_steps=10,
        forward_step=forward_step,
        backward_step=backward_step,
    )
    # 释放显存，避免下一个模型OOM
    del model, x, y
    torch.cuda.empty_cache()
    return f_res, b_res


def run_benchmark():
    """
    同时测试 0 / 2 / 5 轮预热，保存全部结果用于对比
    Use w warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass?
    """
    res_list: List[Dict] = []
    # 三组预热配置
    warmup_configs = [0, 2, 5]

    for warmup in warmup_configs:
        print(f"===== Running warm_up_steps = {warmup} =====")
        for cfg in model_sizes:
            f_res, b_res = single_model_bench(cfg, warm_up_steps=warmup)
            row = {
                "warmup_steps": warmup,
                "Size": cfg["size"],
                "Tf_mean": f_res[0],
                "Tf_std": f_res[1],
                "Tb_mean": b_res[0],
                "Tb_std": b_res[1],
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
    # run benchmark
    run_benchmark()
