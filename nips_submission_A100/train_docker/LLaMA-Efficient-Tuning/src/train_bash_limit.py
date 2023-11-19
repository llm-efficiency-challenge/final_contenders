from llmtuner import run_exp
import torch
import os



def main():
    torch.cuda.set_per_process_memory_fraction(0.51, 0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024' 
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
