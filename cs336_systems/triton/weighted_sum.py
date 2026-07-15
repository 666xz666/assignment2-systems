"""Triton 加权求和算子及其逐元素正确性测试。"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def weight_sum_forward(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """计算 output[row] = sum(x[row, dim] * weight[dim])。

    Args：
        x_ptr：输入矩阵的起始指针，逻辑形状为 [NUM_ROWS, D]。
        weight_ptr：权重向量的起始指针，逻辑形状为 [D]。
        output_ptr：输出向量的起始指针，逻辑形状为 [NUM_ROWS]。
        x_stride_row：x 沿行维度移动一个元素的 stride。
        x_stride_dim：x 沿特征维度移动一个元素的 stride。
        weight_stride_dim：weight 沿特征维度的 stride。
        output_stride_row：output 沿行维度的 stride。
        NUM_ROWS：输入展平后的总行数。
        D：输入的特征维度大小。
        ROWS_TILE_SIZE：一个 program 处理的行数，编译期常量。
        D_TILE_SIZE：一个循环迭代处理的特征数，编译期常量。
    """

    # 一个 Triton program 负责一组连续的行。
    row_tile_idx = tl.program_id(0)

    # 指向 x 的二维 tile，初始位于当前行 tile 的第 0 列。
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # weight 是一维向量，每次读取 D_TILE_SIZE 个元素。
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 输出是一维向量，每个输入行对应一个输出元素。
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 使用 FP32 累加，减少输入为低精度时的累加误差。
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # 沿特征维度分块遍历。最后一个 D tile 可能不完整，因此需要边界检查。
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # weight[None, :] 的形状是 [1, D_TILE_SIZE]，会广播到每一行。
        output += tl.sum(row * weight[None, :], axis=1)

        # 两个输入指针沿特征维度向右移动一个 tile。
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # 最后一个行 tile 可能超出 NUM_ROWS，越界位置不会写入。
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weight_sum_backward(
    x_ptr,
    weight_ptr,
    grad_output_ptr,
    grad_x_ptr,
    partial_grad_weight_ptr,
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    """计算 x 和 weight 的反向梯度。

    Args：
        x_ptr：输入矩阵的起始指针，逻辑形状为 [NUM_ROWS, D]。
        weight_ptr：权重向量的起始指针，逻辑形状为 [D]。
        grad_output_ptr：输出梯度的起始指针，逻辑形状为 [NUM_ROWS]。
        grad_x_ptr：输入 x 的梯度写入地址，逻辑形状为 [NUM_ROWS, D]。
        partial_grad_weight_ptr：每个行 tile 的局部 weight 梯度地址。
        stride_xr、stride_xd：x 沿行维度和特征维度的 stride。
        stride_wd：weight 沿特征维度的 stride。
        stride_gr：grad_output 的行 stride。
        stride_gxr、stride_gxd：grad_x 沿行维度和特征维度的 stride。
        stride_gwb、stride_gwd：局部 weight 梯度沿 tile 和特征维度的 stride。
        NUM_ROWS：输入展平后的总行数。
        D：输入的特征维度大小。
        ROWS_TILE_SIZE：一个 program 处理的行数，编译期常量。
        D_TILE_SIZE：一个循环迭代处理的特征数，编译期常量。
    """

    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # 读取当前行 tile 对应的输出梯度。
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 读取当前 tile 的输入 x。
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 读取权重向量的当前特征 tile。
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 写入当前行 tile 对应的 x 梯度。
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 每个 program 将自己的 weight 梯度写入独立的临时行，避免原子加。
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        # y[row] = sum_d x[row, d] * weight[d]，所以 dy/dx = weight。
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # dy/dweight[d] = sum_row x[row, d] * grad_output[row]。
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance(
            (0, D_TILE_SIZE)
        )
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightSumFunc(torch.autograd.Function):
    """将 Triton forward/backward kernel 接入 PyTorch autograd。"""

    @staticmethod
    def forward(ctx, x, weight):
        """执行 Triton 前向，并保存反向所需的输入。

        Args：
            ctx：PyTorch autograd 用于保存张量和元数据的上下文对象。
            x：最后一维为特征维度的 CUDA 输入张量。
            weight：形状为 [D] 的 CUDA 权重向量。

        Returns：
            删除最后一维后的输出张量。
        """

        # 记录原始形状，kernel 内部统一处理成 [NUM_ROWS, D]。
        input_shape = x.shape
        D = input_shape[-1]
        output_shape = input_shape[:-1]
        x_2d = rearrange(x, "... d -> (...) d").contiguous()

        if len(weight.shape) != 1 or weight.shape[0] != D:
            raise ValueError("weight 必须是一维向量，且长度等于 x 的最后一维")
        if not x_2d.is_cuda or not weight.is_cuda:
            raise ValueError("x 和 weight 必须位于 CUDA 设备")

        ctx.save_for_backward(x_2d, weight)
        ctx.D_TILE_SIZE = max(1, triton.next_power_of_2(D) // 16)
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        n_rows = output.numel()
        n_row_tiles = triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE)

        weight_sum_forward[(n_row_tiles,)](
            x_2d,
            weight,
            output,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            output.reshape(-1).stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return output

    @staticmethod
    def backward(ctx, grad_out):
        """执行 Triton 反向并返回 x 与 weight 的梯度。

        Args：
            ctx：前向阶段保存的 autograd 上下文。
            grad_out：来自后续算子的输出梯度。

        Returns：
            与 forward 输入一一对应的 grad_x 和 grad_weight。
        """

        x_2d, weight = ctx.saved_tensors
        rows_tile_size = ctx.ROWS_TILE_SIZE
        d_tile_size = ctx.D_TILE_SIZE
        n_rows, D = x_2d.shape

        # grad_out 的形状与 output 相同，展平后才能匹配 kernel 的一维布局。
        grad_out_1d = grad_out.contiguous().reshape(-1)
        n_row_tiles = triton.cdiv(n_rows, rows_tile_size)

        partial_grad_weight = torch.empty(
            (n_row_tiles, D),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        grad_x_2d = torch.empty_like(x_2d)

        weight_sum_backward[(n_row_tiles,)](
            x_2d,
            weight,
            grad_out_1d,
            grad_x_2d,
            partial_grad_weight,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            grad_out_1d.stride(0),
            grad_x_2d.stride(0),
            grad_x_2d.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=rows_tile_size,
            D_TILE_SIZE=d_tile_size,
        )

        grad_weight = partial_grad_weight.sum(dim=0)
        grad_x = grad_x_2d.reshape(ctx.input_shape)
        return grad_x, grad_weight


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """打印误差并检查两个张量的逐元素数值是否一致。

    Args：
        name：当前检查项目的名称。
        actual：Triton 算子产生的结果。
        expected：PyTorch 参考实现产生的结果。
    """

    max_abs_error = (actual - expected).abs().max().item()
    print(f"{name}: 最大绝对误差 = {max_abs_error:.6e}")
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def run_value_test(device: torch.device) -> None:
    """用非整除的行数和特征数测试 forward 及两个反向梯度。

    Args：
        device：执行 Triton kernel 的 CUDA 设备。
    """

    torch.manual_seed(2024)
    test_shapes = [
        (2, 3, 5),
        (3, 17, 37),
    ]

    for case_index, shape in enumerate(test_shapes):
        print(f"\n===== 测试 {case_index + 1}: x.shape={shape} =====")
        x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(
            shape[-1], device=device, dtype=torch.float32, requires_grad=True
        )

        output = WeightSumFunc.apply(x, weight)
        expected_output = (x * weight).sum(dim=-1)
        check_close("forward output", output, expected_output)

        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        custom_grad_x = x.grad.detach().clone()
        custom_grad_weight = weight.grad.detach().clone()

        reference_x = x.detach().clone().requires_grad_(True)
        reference_weight = weight.detach().clone().requires_grad_(True)
        reference_output = (reference_x * reference_weight).sum(dim=-1)
        reference_output.backward(grad_output)

        check_close("grad_x", custom_grad_x, reference_x.grad)
        check_close("grad_weight", custom_grad_weight, reference_weight.grad)

        if case_index == 0:
            print("自定义 forward 输出：")
            print(output.detach().cpu())
            print("参考 forward 输出：")
            print(expected_output.detach().cpu())

    print("\n所有逐元素数值测试通过。")


def main() -> None:
    """解析命令行参数并启动逐元素数值测试。"""

    parser = argparse.ArgumentParser(description="Triton 加权求和算子数值测试")
    parser.add_argument("--device", default="cuda", help="测试设备，默认 cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("该 Triton 测试需要可用的 CUDA 设备")
    run_value_test(device)


if __name__ == "__main__":
    main()
