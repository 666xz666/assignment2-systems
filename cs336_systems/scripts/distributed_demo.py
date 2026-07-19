import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------- CPU 4进程 Gloo 分布式 ----------------------
def setup_cpu(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cpu_worker(rank, world_size):
    setup_cpu(rank, world_size)
    data = torch.randint(0, 10, (3,), device="cpu")
    print(f"【CPU进程{rank}】all_reduce前: {data}")
    dist.all_reduce(data, async_op=False)
    print(f"【CPU进程{rank}】all_reduce后: {data}")
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------- GPU 单进程 NCCL 分布式 ----------------------
def setup_gpu(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"  # 换端口避免和CPU进程组冲突
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def gpu_worker(rank, world_size):
    setup_gpu(rank, world_size)
    torch.cuda.set_device(rank)
    data = torch.randint(0, 10, (3,), device=f"cuda:{rank}")
    print(f"【GPU进程{rank}】单进程无需跨卡all_reduce，数据: {data}")
    # world_size=1时all_reduce不会修改张量，可注释或保留
    # dist.all_reduce(data, async_op=False)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # 启动4个CPU进程
    cpu_world = 4
    mp.spawn(fn=cpu_worker, args=(cpu_world,), nprocs=cpu_world, join=True)

    # 启动1个GPU进程
    gpu_world = 1
    mp.spawn(fn=gpu_worker, args=(gpu_world,), nprocs=gpu_world, join=True)
