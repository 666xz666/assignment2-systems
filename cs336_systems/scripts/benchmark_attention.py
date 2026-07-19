"""Benchmark naive scaled dot-product attention at multiple sizes.

Example:
    python benchmark_attention.py --output results/attention.csv
    python benchmark_attention.py --compile

The default configuration follows the CS336 Assignment 2 prompt.  The
benchmark intentionally materializes the attention matrix, so that its
quadratic memory cost is visible.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch

from cs336_basics import ScaledDotProductAttention


DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]
MIB = 1024**2


AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def build_attention(
    *, use_compile: bool, compile_backend: str, compile_mode: str
) -> AttentionFn:
    """Create one attention module for one benchmark configuration."""

    attention: AttentionFn = ScaledDotProductAttention()
    if use_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0 or newer")
        attention = torch.compile(
            attention,
            backend=compile_backend,
            mode=compile_mode,
        )
    return attention


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list such as ``16,32,64``."""

    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all list values must be positive integers")
    return values


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(dtypes))
        raise argparse.ArgumentTypeError(
            f"unknown dtype {name!r}; choose from {valid}"
        ) from exc


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def cleanup_cuda(device: torch.device) -> None:
    """Release references and allocator cache after a configuration ends."""

    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            # An OOM can leave CUDA in an error state; the next process will
            # still contain the complete CSV record written by the caller.
            pass


def reset_compile_cache() -> None:
    """Prevent shape-specialized graphs from leaking across configurations."""

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
        torch._dynamo.reset()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


def theoretical_memory(
    batch_size: int, sequence_length: int, d_model: int, dtype: torch.dtype
) -> dict[str, int]:
    element_size = torch.empty((), dtype=dtype).element_size()
    qkv_bytes = 3 * batch_size * sequence_length * d_model * element_size
    score_bytes = batch_size * sequence_length * sequence_length * element_size
    return {
        "theoretical_qkv_bytes": qkv_bytes,
        "theoretical_one_attention_matrix_bytes": score_bytes,
        # Naive autograd commonly needs both scores and softmax output.
        "theoretical_two_attention_matrices_bytes": 2 * score_bytes,
    }


def benchmark_configuration(
    *,
    attention: AttentionFn,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Run one configuration and return measurements in a CSV-friendly dict."""

    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    q = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    after_inputs = memory_stats(device)

    # Warm-up includes both paths so CUDA kernels, allocator state, and
    # autograd behavior are initialized before measurements begin. For a
    # compiled module, the first warm-up also absorbs torch.compile overhead.
    for _ in range(warmup):
        warmup_output = attention(q, k, v)
        synchronize(device)
        warmup_output.sum().backward()
        synchronize(device)
        del warmup_output
        q.grad = None
        k.grad = None
        v.grad = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Keep only the final output graph.  Earlier graphs are released when
    # output is overwritten, preventing 100 forward graphs from accumulating.
    synchronize(device)
    forward_start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = attention(q, k, v)
        synchronize(device)
    forward_ms = (time.perf_counter() - forward_start) * 1000.0 / iterations

    synchronize(device)
    before_backward = memory_stats(device)
    del output

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # A backward pass consumes its graph, so build a fresh graph for each
    # iteration. Forward execution is synchronized but excluded from the
    # backward timer.
    backward_elapsed = 0.0
    for _ in range(iterations):
        q.grad = None
        k.grad = None
        v.grad = None
        backward_output = attention(q, k, v)
        synchronize(device)
        backward_start = time.perf_counter()
        backward_output.sum().backward()
        synchronize(device)
        backward_elapsed += time.perf_counter() - backward_start
        del backward_output
    backward_ms = backward_elapsed * 1000.0 / iterations
    after_backward = memory_stats(device)

    theoretical = theoretical_memory(batch_size, sequence_length, d_model, dtype)
    result = {
        "status": "ok",
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "memory_after_inputs_mb": after_inputs["allocated_bytes"] / MIB,
        "memory_before_backward_mb": before_backward["allocated_bytes"] / MIB,
        "attention_extra_before_backward_mb": (
            before_backward["allocated_bytes"] - after_inputs["allocated_bytes"]
        )
        / MIB,
        "forward_peak_allocated_mb": before_backward["peak_allocated_bytes"] / MIB,
        "backward_peak_allocated_mb": after_backward["peak_allocated_bytes"] / MIB,
        "peak_reserved_mb": after_backward["peak_reserved_bytes"] / MIB,
        "error": "",
    }
    result.update(
        {
            "theoretical_qkv_mb": theoretical["theoretical_qkv_bytes"] / MIB,
            "theoretical_one_attention_matrix_mb": (
                theoretical["theoretical_one_attention_matrix_bytes"] / MIB
            ),
            "theoretical_two_attention_matrices_mb": (
                theoretical["theoretical_two_attention_matrices_bytes"] / MIB
            ),
        }
    )
    return result


def dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def default_output_path(args: argparse.Namespace) -> Path:
    """Build a filename that identifies the important benchmark settings."""

    attention_label = "compiled" if args.use_compile else "eager"
    if args.use_compile:
        attention_label += f"-{args.compile_backend}-{args.compile_mode}"
    d_models = "-".join(str(value) for value in args.d_models)
    sequence_lengths = "-".join(str(value) for value in args.sequence_lengths)
    filename = (
        f"attention_benchmark_{attention_label}_{dtype_label(args.dtype)}"
        f"_b{args.batch_size}_w{args.warmup}_n{args.iterations}"
        f"_dm{d_models}_s{sequence_lengths}.csv"
    )
    return Path("output/attention") / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=parse_int_list, default=DEFAULT_D_MODELS)
    parser.add_argument(
        "--sequence-lengths", type=parse_int_list, default=DEFAULT_SEQUENCE_LENGTHS
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warm-up forward/backward steps per configuration",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="measured forward and backward passes",
    )
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument(
        "--device", default="cuda", help="for example cuda, cuda:0, or cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile",
        "--torch-compile",
        dest="use_compile",
        action="store_true",
        help="compile the attention module with torch.compile",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="backend passed to torch.compile",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
        help="mode passed to torch.compile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="explicit CSV path; otherwise a configuration-aware path is generated",
    )
    oom_group = parser.add_mutually_exclusive_group()
    oom_group.add_argument(
        "--stop-on-oom",
        action="store_true",
        help="stop after recording the first OOM (default: continue)",
    )
    # Keep this alias so existing commands remain valid after changing the
    # default to the full 20-configuration sweep.
    oom_group.add_argument(
        "--continue-on-oom",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def write_row(writer: csv.DictWriter, output_file: Any, row: dict[str, Any]) -> None:
    writer.writerow(row)
    output_file.flush()
    try:
        os.fsync(output_file.fileno())
    except OSError:
        pass


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError(
            "batch size and iterations must be positive; warmup cannot be negative"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if args.use_compile and not hasattr(torch, "compile"):
        raise RuntimeError("--compile requires PyTorch 2.0 or newer")

    torch.manual_seed(args.seed)
    output_path = args.output or default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "d_model",
        "sequence_length",
        "batch_size",
        "dtype",
        "attention_mode",
        "compile_backend",
        "compile_mode",
        "status",
        "forward_ms",
        "backward_ms",
        "memory_after_inputs_mb",
        "memory_before_backward_mb",
        "attention_extra_before_backward_mb",
        "forward_peak_allocated_mb",
        "backward_peak_allocated_mb",
        "peak_reserved_mb",
        "theoretical_qkv_mb",
        "theoretical_one_attention_matrix_mb",
        "theoretical_two_attention_matrices_mb",
        "error",
    ]

    attention_mode = "compiled" if args.use_compile else "eager"
    print(
        f"device={device}, dtype={args.dtype}, attention={attention_mode}, "
        f"output={output_path}"
    )
    print(
        f"configs={len(args.d_models) * len(args.sequence_lengths)}, iterations={args.iterations}"
    )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        output_file.flush()

        stop_after_oom = False
        config_index = 0
        for d_model in args.d_models:
            for sequence_length in args.sequence_lengths:
                config_index += 1
                base = {
                    "d_model": d_model,
                    "sequence_length": sequence_length,
                    "batch_size": args.batch_size,
                    "dtype": dtype_label(args.dtype),
                    "attention_mode": attention_mode,
                    "compile_backend": args.compile_backend if args.use_compile else "",
                    "compile_mode": args.compile_mode if args.use_compile else "",
                }
                theoretical = theoretical_memory(
                    args.batch_size,
                    sequence_length,
                    d_model,
                    args.dtype,
                )
                base.update(
                    {
                        "theoretical_qkv_mb": theoretical["theoretical_qkv_bytes"]
                        / MIB,
                        "theoretical_one_attention_matrix_mb": (
                            theoretical["theoretical_one_attention_matrix_bytes"] / MIB
                        ),
                        "theoretical_two_attention_matrices_mb": (
                            theoretical["theoretical_two_attention_matrices_bytes"]
                            / MIB
                        ),
                    }
                )
                print(
                    f"[{config_index}/{len(args.d_models) * len(args.sequence_lengths)}] "
                    f"d_model={d_model}, sequence_length={sequence_length} ...",
                    flush=True,
                )
                attention = None
                try:
                    if args.use_compile:
                        reset_compile_cache()
                    attention = build_attention(
                        use_compile=args.use_compile,
                        compile_backend=args.compile_backend,
                        compile_mode=args.compile_mode,
                    )
                    result = benchmark_configuration(
                        attention=attention,
                        batch_size=args.batch_size,
                        sequence_length=sequence_length,
                        d_model=d_model,
                        dtype=args.dtype,
                        device=device,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    row = {**base, **result}
                    write_row(writer, output_file, row)
                    print(
                        f"  OK: forward={result['forward_ms']:.3f} ms, "
                        f"backward={result['backward_ms']:.3f} ms, "
                        f"before_backward={result['memory_before_backward_mb']:.1f} MiB",
                        flush=True,
                    )
                except Exception as exc:
                    status = "OOM" if is_oom_error(exc) else "ERROR"
                    row = {
                        **base,
                        "status": status,
                        "forward_ms": "",
                        "backward_ms": "",
                        "memory_after_inputs_mb": "",
                        "memory_before_backward_mb": "",
                        "attention_extra_before_backward_mb": "",
                        "forward_peak_allocated_mb": "",
                        "backward_peak_allocated_mb": "",
                        "peak_reserved_mb": "",
                        "error": str(exc).replace("\n", " ")[:1000],
                    }
                    write_row(writer, output_file, row)
                    print(f"  {status}: {exc}", flush=True)
                    cleanup_cuda(device)
                    if status == "OOM" and args.stop_on_oom:
                        stop_after_oom = True
                        break
                    if status == "ERROR":
                        raise
                finally:
                    attention = None
                    cleanup_cuda(device)
            if stop_after_oom:
                break

    if stop_after_oom:
        print(f"Stopped after OOM. Partial results were saved to {output_path}")
    else:
        print(f"Finished. Results were saved to {output_path}")


if __name__ == "__main__":
    main()
