import gc
import logging
import os

import psutil
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    vm = psutil.virtual_memory()
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host_total_GB": _byte_to_gb(vm.total),
        "host_available_GB": _byte_to_gb(vm.available),
        "host_used_GB": _byte_to_gb(vm.used),
        "host_free_GB": _byte_to_gb(vm.free),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def log_process_diagnostics(tag: str = ""):
    """Logs open file descriptors (sockets, /dev/shm, /dev/nvidia*), network connections, and CUDA device state."""
    pid = os.getpid()
    rank = dist.get_rank() if dist.is_initialized() else -1

    # 1. Open FDs & Sockets
    fd_dir = f"/proc/{pid}/fd"
    sockets = []
    shm_files = []
    nvidia_devs = set()
    if os.path.exists(fd_dir):
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
                if target.startswith("socket:"):
                    sockets.append(target)
                elif "/dev/shm" in target or "/dev/hugepages" in target:
                    shm_files.append(target)
                elif "/dev/nvidia" in target:
                    nvidia_devs.add(target)
            except OSError:
                continue

    # 2. CUDA Device State
    cuda_state = {}
    if torch.cuda.is_available():
        cur_dev = torch.cuda.current_device()
        dev_count = torch.cuda.device_count()
        cuda_state["current_device"] = cur_dev
        cuda_state["device_count"] = dev_count
        dev_allocs = {}
        p2p_peers = []
        for d in range(dev_count):
            alloc = torch.cuda.memory_allocated(d)
            res = torch.cuda.memory_reserved(d)
            if alloc > 0 or res > 0:
                dev_allocs[d] = {
                    "allocated_MB": round(alloc / (1024**2), 2),
                    "reserved_MB": round(res / (1024**2), 2),
                }
            if d != cur_dev:
                try:
                    if torch.cuda.can_device_access_peer(cur_dev, d):
                        p2p_peers.append(d)
                except Exception:
                    pass
        cuda_state["active_devices"] = dev_allocs
        cuda_state["p2p_nvlink_peers"] = p2p_peers

    logger.info(
        f"[Rank {rank}] [Diagnostics - {tag}] PID={pid} | "
        f"Sockets={len(sockets)} ({sockets[:5]}...) | "
        f"SHM={len(shm_files)} ({shm_files}) | "
        f"NVIDIA_FDs={sorted(list(nvidia_devs))} | "
        f"CUDA={cuda_state}"
    )

