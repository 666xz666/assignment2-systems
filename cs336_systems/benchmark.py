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
import argparse

# 导入autocast
from torch import autocast

# 作业一实现
from cs336_basics.nn import TransformerLM
from cs336_basics.utils import try_gpu, cross_entropy
from cs336_basics.optim import AdamW


class Timer:
    def __enter__(self):
        self.cost = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize()
        self.cost = timeit.default_timer() - self.cost


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def std_sample(x: list[float]) -> float:
    mean_val = mean(x)
    var = sum((xi - mean_val) ** 2 for xi in x) / (len(x) - 1)
    return math.sqrt(var)


def build_config_tag(
    selected_sizes: List[str],
    context_length: int,
    batch_size: int,
    vocab_size: int,
    warmup_steps: List[int],
    dtype: str,
    run_mode: str,
    enable_mem_profile: bool,
) -> str:
    """生成带全量配置的标识字符串，用于目录/文件名，一眼识别运行参数"""
    parts = []
    # 模型规模：单模型显式标注，多模型标数量
    if len(selected_sizes) == 1:
        parts.append(f"model_{selected_sizes[0]}")
    else:
        parts.append(f"models{len(selected_sizes)}")
    # 核心超参
    parts.append(f"ctx{context_length}")
    parts.append(f"bs{batch_size}")
    # 词表大小简写：10000 → 10k
    vocab_str = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    parts.append(f"vocab{vocab_str}")
    # 预热步数：多个用短横线连接
    warm_str = "-".join(str(w) for w in warmup_steps)
    parts.append(f"warm{warm_str}")
    # 计算精度（fp32 / bf16）
    parts.append(dtype)
    # 运行模式
    parts.append(run_mode)
    # 是否开启内存剖面
    parts.append("mem1" if enable_mem_profile else "mem0")
    return "_".join(parts)


def init_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float | None = 10_000.0,
) -> TransformerLM:
    # 固定模型权重永远初始化 FP32，autocast 只控制中间激活，不修改参数存储类型
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
    batch_size: int, context_len: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = try_gpu()
    x = torch.randint(
        0, vocab_size, size=(batch_size, context_len), dtype=torch.long, device=dv
    )
    y = torch.randint(
        0, vocab_size, size=(batch_size, context_len), dtype=torch.long, device=dv
    )
    return x, y


def dump_memory_snapshot(save_path: str):
    """封装 PyTorch CUDA 内存快照导出逻辑"""
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(save_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Memory snapshot saved to {save_path}")


def benchmark_full(
    warm_up_steps: int,
    num_steps: int,
    forward_step: Callable,
    backward_step: Callable,
    train_step: Callable,
    run_mode: str,
    enable_mem_profile: bool,
    snapshot_save_path: str = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # 预热阶段
    with nvtx.range(f"Warmup_{warm_up_steps}_Steps"):
        for _ in range(warm_up_steps):
            out = forward_step()
            backward_step(out)
            train_step()
    torch.cuda.synchronize()

    f_times = []
    b_times = []
    opt_times = []

    # 测量阶段
    with nvtx.range(f"Measurement_{num_steps}_Iter"):
        for step_idx in range(num_steps):
            capture_mem = enable_mem_profile and step_idx == 0

            # 前向传播
            with nvtx.range("ForwardPass"):
                with Timer() as t_f:
                    out = forward_step()
                    if capture_mem and run_mode == "infer":
                        dump_memory_snapshot(snapshot_save_path)
                f_times.append(t_f.cost)

            # 反向传播
            with nvtx.range("BackwardPass"):
                with Timer() as t_b:
                    backward_step(out)
                b_times.append(t_b.cost)

            # 完整训练步
            with nvtx.range("FullTrainStep_ForwardBackward_Optimizer"):
                with Timer() as t_opt:
                    o = forward_step()
                    backward_step(o)
                    train_step()
                    if capture_mem and run_mode == "train":
                        dump_memory_snapshot(snapshot_save_path)
                opt_times.append(t_opt.cost)

    f_mean, f_std = mean(f_times), std_sample(f_times)
    b_mean, b_std = mean(b_times), std_sample(b_times)
    opt_mean, opt_std = mean(opt_times), std_sample(opt_times)
    return (f_mean, f_std), (b_mean, b_std), (opt_mean, opt_std)


# 全部模型配置（基准配置，不修改；运行时按命令行筛选）
ALL_MODEL_SIZES = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"size": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    # {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "10B", "d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
]


def single_model_bench(
    cfg: Dict,
    vocab_size: int,
    batch_size: int,
    context_length: int,
    warm_up_steps: int,
    run_mode: str,
    enable_mem_profile: bool,
    snapshot_root_dir: str,
    amp_enable: bool,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    model = init_model(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10_000,
    )
    x, y = generate_random_data(
        batch_size=batch_size,
        context_len=context_length,
        vocab_size=vocab_size,
    )
    opt = AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    # 推理模式关闭梯度
    if run_mode == "infer":
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    def forward_step() -> torch.Tensor:
        # 开启BF16 autocast 上下文
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enable):
            return model(x)

    def backward_step(out: torch.Tensor):
        if run_mode == "infer":
            return
        model.zero_grad(set_to_none=True)
        loss = cross_entropy(out, y)
        loss.backward()

    def train_step():
        if run_mode == "infer":
            return
        opt.step()

    # 构造快照文件名：全局配置已在目录体现，这里仅标模型+模式
    snapshot_path = None
    if enable_mem_profile:
        os.makedirs(snapshot_root_dir, exist_ok=True)
        snapshot_name = f"{cfg['size']}_{run_mode}.pickle"
        snapshot_path = os.path.join(snapshot_root_dir, snapshot_name)

    f_res, b_res, opt_res = benchmark_full(
        warm_up_steps=warm_up_steps,
        num_steps=10,
        forward_step=forward_step,
        backward_step=backward_step,
        train_step=train_step,
        run_mode=run_mode,
        enable_mem_profile=enable_mem_profile,
        snapshot_save_path=snapshot_path,
    )

    del model, x, y, opt
    torch.cuda.empty_cache()
    return f_res, b_res, opt_res


def run_benchmark(
    selected_sizes: List[str],
    vocab_size: int,
    batch_size: int,
    context_length: int,
    warmup_steps_list: List[int],
    run_mode: str,
    enable_mem_profile: bool,
    out_dir: str,
    amp_dtype_arg: str,
):
    # 筛选要跑的模型规模
    model_sizes = [cfg for cfg in ALL_MODEL_SIZES if cfg["size"] in selected_sizes]
    if not model_sizes:
        raise ValueError(
            f"No valid model sizes selected. Available: {[c['size'] for c in ALL_MODEL_SIZES]}"
        )

    # 解析AMP配置
    if amp_dtype_arg == "bf16":
        amp_enable = True
        dtype_tag = "bf16"
    else:
        amp_enable = False
        dtype_tag = "fp32"

    # 生成全量配置标签，创建专属输出目录
    config_tag = build_config_tag(
        selected_sizes=selected_sizes,
        context_length=context_length,
        batch_size=batch_size,
        vocab_size=vocab_size,
        warmup_steps=warmup_steps_list,
        dtype=dtype_tag,
        run_mode=run_mode,
        enable_mem_profile=enable_mem_profile,
    )
    run_out_dir = os.path.join(out_dir, config_tag)
    os.makedirs(run_out_dir, exist_ok=True)
    mem_snap_dir = os.path.join(run_out_dir, "memory_snapshots")

    res_list: List[Dict] = []

    for warmup in warmup_steps_list:
        print(
            f"===== Warmup steps = {warmup} | Mode = {run_mode} | AMP={dtype_tag} | Mem profile = {enable_mem_profile} ====="
        )
        for cfg in model_sizes:
            print(f"  Running model: {cfg['size']}")
            f_res, b_res, opt_res = single_model_bench(
                cfg=cfg,
                vocab_size=vocab_size,
                batch_size=batch_size,
                context_length=context_length,
                warm_up_steps=warmup,
                run_mode=run_mode,
                enable_mem_profile=enable_mem_profile,
                snapshot_root_dir=mem_snap_dir,
                amp_enable=amp_enable,
            )
            row = {
                "warmup_steps": warmup,
                "size": cfg["size"],
                "vocab_size": vocab_size,
                "batch_size": batch_size,
                "context_length": context_length,
                "run_mode": run_mode,
                "amp_precision": dtype_tag,
                "tf_mean": f_res[0],
                "tf_std": f_res[1],
                "tb_mean": b_res[0],
                "tb_std": b_res[1],
                "topt_mean": opt_res[0],
                "topt_std": opt_res[1],
            }
            res_list.append(row)

    df = pd.DataFrame(res_list)
    # CSV 直接放在配置目录下，文件名简洁
    csv_path = os.path.join(run_out_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n==== Benchmark Results ====")
    print(df.to_string(index=False))
    print(f"\nAll outputs saved to: {run_out_dir}")
    print(f"  - CSV results: {csv_path}")
    if enable_mem_profile:
        print(f"  - Memory snapshots: {mem_snap_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="CS336 Assignment2 Transformer Benchmark (兼容所有子题目 + BF16混合精度AMP)"
    )

    # 模型规模选择
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=[c["size"] for c in ALL_MODEL_SIZES],
        help="选择要跑的模型规模，可选 small/medium/large/xl/10B，默认全部",
    )

    # 超参数覆盖（对应作业默认配置）
    parser.add_argument(
        "--vocab-size", type=int, default=10000, help="词表大小，默认 10000"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小，默认 4")
    parser.add_argument(
        "--context-length", type=int, default=512, help="上下文长度，默认 512"
    )

    # 预热步数配置（兼容 2.1.3 不同 warmup 测试）
    parser.add_argument(
        "--warmup-steps",
        nargs="+",
        type=int,
        default=[5],
        help="预热步数，可传多个值，例如 --warmup-steps 0 2 5，默认 [5]",
    )

    # 运行模式与内存 profiling（兼容 2.1.6）
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["infer", "train"],
        default="train",
        help="infer: 仅前向推理；train: 完整训练步(fw+bw+opt)，默认 train",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="开启 PyTorch CUDA 内存快照导出（2.1.6 题目用）",
    )

    # 新增混合精度参数
    parser.add_argument(
        "--amp",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="fp32 全精度 / bf16 开启autocast混合精度，默认 fp32",
    )

    # 输出目录
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./output/benchmark",
        help="结果输出根目录，默认 ./output/benchmark",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_benchmark(
        selected_sizes=args.model_sizes,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps_list=args.warmup_steps,
        run_mode=args.run_mode,
        enable_mem_profile=args.memory_profile,
        out_dir=args.out_dir,
        amp_dtype_arg=args.amp,
    )


if __name__ == "__main__":
    main()


"""
uv run python cs336_systems/benchmark.py \
--model-sizes xl \
--context-length 128 \
--run-mode infer \
--amp bf16 \
--memory-profile

# nsys  profiling 示例（输出路径自动携带amp标识）
uv run nsys profile \
--trace=cuda,cudnn,cublas,osrt,nvtx \
--pytorch=functions-trace,autograd-shapes-nvtx \
-o ./output/benchmark/xxx/nsys_trace \
-- python cs336_systems/benchmark.py \
--model-sizes xl \
--context-length 1024 \
--amp bf16
"""
