"""
Utility functions for PyTorch training with support for distributed computing,
device management (CUDA/MPS/CPU), and training metrics logging.

This module provides utilities for:
- Cross-device compatibility (CUDA, MPS, CPU)
- Distributed training setup and management
- Training metrics collection and smoothing
- Memory usage tracking and logging
- Progress monitoring during training loops
"""
import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from .device_check import check_set_gpu


def _get_device_for_distributed():
    """
    Get the appropriate device for distributed operations.
    
    Prioritizes devices in the following order:
    1. CUDA (if available)
    2. MPS (Apple Silicon, if available)
    3. CPU (fallback)
    
    Returns:
        torch.device: The best available device for distributed operations.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def _get_memory_usage():
    """
    Get memory usage for the current device.
    
    Returns the maximum memory allocated on the current device.
    For CUDA devices, returns actual memory usage.
    For MPS and CPU, returns 0 as a placeholder since direct
    memory tracking is not available.
    
    Returns:
        int: Maximum memory allocated in bytes, or 0 if not trackable.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    elif torch.backends.mps.is_available():
        # MPS doesn't have a direct equivalent to max_memory_allocated
        # Return 0 as a placeholder
        return 0
    else:
        return 0


def _has_gpu_memory_tracking():
    """
    Check if current device supports memory tracking for logging.
    
    Currently only CUDA devices support detailed memory tracking.
    
    Returns:
        bool: True if memory tracking is supported, False otherwise.
    """
    return torch.cuda.is_available()


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    
    This class maintains both a sliding window of recent values and
    global statistics across all updates. Useful for monitoring
    training metrics like loss, accuracy, etc.
    
    Attributes:
        deque (collections.deque): Sliding window of recent values
        total (float): Sum of all values ever added
        count (int): Total number of values added
        fmt (str): Format string for string representation
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Initialize SmoothedValue tracker.
        
        Args:
            window_size (int, optional): Size of sliding window. Defaults to 20.
            fmt (str, optional): Format string for __str__ method. 
                                Defaults to "{median:.4f} ({global_avg:.4f})".
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Add a new value to the tracker.
        
        Args:
            value (float): The value to add
            n (int, optional): Number of samples this value represents. Defaults to 1.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize statistics across distributed processes.
        
        Warning: This only synchronizes total and count, not the deque!
        After synchronization, global_avg will be correct across all processes,
        but window-based statistics (median, avg) will remain process-local.
        """
        if not is_dist_avail_and_initialized():
            return
        device = _get_device_for_distributed()
        # Create tensor with count and total for all_reduce operation
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Get median of values in the current window."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Get average of values in the current window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Get global average across all values ever added."""
        return self.total / self.count

    @property
    def max(self):
        """Get maximum value in the current window."""
        return max(self.deque)

    @property
    def value(self):
        """Get the most recent value added."""
        return self.deque[-1]

    def __str__(self):
        """Return formatted string with key statistics."""
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Gather arbitrary picklable data from all processes in distributed training.
    
    This function works with any picklable Python object, not just tensors.
    In single-process mode, simply returns the data wrapped in a list.
    
    Args:
        data: Any picklable object to gather from all processes
        
    Returns:
        list: List containing data from each rank. Length equals world_size.
              Order corresponds to rank order (index 0 = rank 0, etc.)
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values across all distributed processes.
    
    All tensor values in the dictionary are reduced (summed) across processes.
    Optionally averages the results by world_size.
    
    Args:
        input_dict (dict): Dictionary with string keys and tensor values
        average (bool, optional): Whether to average results by world_size. 
                                 If False, returns sum. Defaults to True.
                                 
    Returns:
        dict: Dictionary with same keys, containing reduced values.
              In single-process mode, returns input_dict unchanged.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # Sort keys to ensure consistent order across all processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)  # Sum across all processes
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    """
    Logger for tracking and displaying training metrics.
    
    Maintains SmoothedValue objects for each metric and provides
    convenient logging with progress tracking, ETA estimation,
    and memory usage monitoring.
    
    Attributes:
        meters (defaultdict): Dictionary mapping metric names to SmoothedValue objects
        delimiter (str): String used to separate metrics in output
    """
    
    def __init__(self, delimiter="\t"):
        """
        Initialize MetricLogger.
        
        Args:
            delimiter (str, optional): Delimiter for joining metrics in output. 
                                     Defaults to tab character.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update multiple metrics at once.
        
        Args:
            **kwargs: Keyword arguments where keys are metric names and 
                     values are numeric values (int, float, or scalar tensor)
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        Allow direct access to meters as attributes.
        
        Args:
            attr (str): Name of the meter to access
            
        Returns:
            SmoothedValue: The requested meter
            
        Raises:
            AttributeError: If the attribute doesn't exist
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """Return formatted string of all current metrics."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize all meters across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Add a custom meter.
        
        Args:
            name (str): Name of the meter
            meter (SmoothedValue): Pre-configured SmoothedValue object
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log metrics at regular intervals during iteration.
        
        This is a generator that yields items from the iterable while
        periodically printing progress information including:
        - Current iteration and total
        - ETA (estimated time remaining)
        - All tracked metrics
        - Timing information (iteration time, data loading time)
        - Memory usage (if GPU memory tracking available)
        
        Args:
            iterable: Any iterable object (e.g., DataLoader)
            print_freq (int): Print progress every N iterations
            header (str, optional): Header text for log messages
            
        Yields:
            Any: Items from the input iterable
        """
        i = 1
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        # Use device-agnostic memory tracking
        has_gpu_memory = _has_gpu_memory_tracking()
        if has_gpu_memory:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0  # Convert bytes to megabytes
        
        for obj in iterable:
            # Track data loading time
            data_time.update(time.time() - end)
            yield obj
            # Track iteration time (including forward/backward pass)
            iter_time.update(time.time() - end)
            
            # Print progress at specified frequency or on last iteration
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if has_gpu_memory:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=_get_memory_usage() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        
        # Print total time summary
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    """
    Default collate function for DataLoader.
    
    Transposes a list of samples to create a batch where each element
    is a tuple of corresponding items from each sample.
    
    Args:
        batch (list): List of samples from dataset
        
    Returns:
        tuple: Tuple of batched elements
    """
    return tuple(zip(*batch))


def mkdir(path):
    """
    Create directory if it doesn't exist.
    
    Safely creates a directory, ignoring the error if it already exists.
    
    Args:
        path (str): Path to directory to create
        
    Raises:
        OSError: If directory creation fails for reasons other than already existing
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    Configure print function for distributed training.
    
    Modifies the built-in print function so that only the master process
    prints by default. Other processes remain silent unless 'force=True'
    is passed to print().
    
    Args:
        is_master (bool): Whether this process is the master (rank 0)
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    
    Returns:
        bool: True if distributed training is ready to use
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get total number of processes in distributed training.
    
    Returns:
        int: Number of processes (1 if not distributed)
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get rank (ID) of current process in distributed training.
    
    Returns:
        int: Process rank (0 if not distributed)
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if current process is the main process (rank 0).
    
    Returns:
        bool: True if this is rank 0 or not distributed
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save checkpoint only on master process.
    
    Wrapper around torch.save that only executes on rank 0 to prevent
    multiple processes from writing the same file simultaneously.
    
    Args:
        *args: Arguments passed to torch.save
        **kwargs: Keyword arguments passed to torch.save
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initialize distributed training mode.
    
    Sets up distributed training by detecting the environment (SLURM, torchrun, etc.)
    and configuring appropriate backend (NCCL for CUDA, Gloo for CPU/MPS).
    
    Modifies args object in-place with distributed training configuration:
    - args.distributed: Whether distributed training is enabled
    - args.rank: Process rank
    - args.world_size: Total number of processes  
    - args.gpu: Local GPU ID
    - args.dist_backend: Backend to use ('nccl' or 'gloo')
    
    Supports multiple launching methods:
    1. Environment variables (RANK, WORLD_SIZE, LOCAL_RANK) - torchrun
    2. SLURM environment (SLURM_PROCID) - SLURM clusters
    3. Single process fallback
    
    Args:
        args: Arguments object that will be modified with distributed config
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Standard torchrun/distributed launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # SLURM cluster environment
        args.rank = int(os.environ["SLURM_PROCID"])
        if torch.cuda.is_available():
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            # For MPS or CPU, set gpu to 0 as there's typically only one device
            args.gpu = 0
    else:
        # Single process mode
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    # Set device and backend based on hardware availability
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"  # Optimal for CUDA
    else:
        # MPS doesn't support distributed training with NCCL
        # Fall back to Gloo backend for CPU/MPS
        args.dist_backend = "gloo"
        print("Warning: Using Gloo backend for distributed training. Performance may be reduced.")
    
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()  # Synchronize all processes
    setup_for_distributed(args.rank == 0)  # Configure printing for distributed mode