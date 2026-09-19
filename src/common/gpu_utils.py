import torch


def setup_gpu_memory_limit(fraction=0.8):
    """Limit GPU memory to a fraction of total VRAM so other apps can share the GPU."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory limit setup")
        return

    try:
        torch.cuda.set_per_process_memory_fraction(fraction)

        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (
            1024**3
        )  # GB
        limited_memory = total_memory * fraction

        print("\nGPU Memory Configuration:")
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print(f"  Total VRAM: {total_memory:.2f} GB")
        print(f"  Limited to: {limited_memory:.2f} GB ({fraction * 100:.0f}%)")
        print(f"  Reserved for system: {total_memory - limited_memory:.2f} GB\n")

    except Exception as e:
        print(f"Warning: Could not set GPU memory limit: {e}")
