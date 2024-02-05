# Install llama_cpp with NVidia CUDA acceleration
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Install compiled wheel: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels
# python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117

import time
import statistics
from llama_cpp import Llama

# Initialize the Llama model for benchmarking
llm = Llama(
    model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Path to your model file
    n_ctx=2048,  # Maximum sequence length
    n_threads=8,  # Number of CPU threads
    n_gpu_layers=35  # Number of layers offloaded to GPU
)

def benchmark_llama(prompt: str, num_trials: int = 10):
    elapsed_times = []
    token_counts = []
    tokens_per_seconds = []

    for _ in range(num_trials):
        start_time = time.time()
        
        # Perform inference
        output = llm(
            f"[INST] {prompt} [/INST]",  # Format the prompt
            max_tokens=2000,  # Limit the number of tokens generated
            stop=["</s>"],  # Define stop tokens
            echo=True  # Echo the prompt in the output
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

        # Count the number of tokens in the output
        # Assuming the Llama model outputs a string, we need to approximate token count.
        # This approximation might not be accurate for all tokenizers.
        token_count = len(output.split())
        token_counts.append(token_count)

        tokens_per_second = token_count / elapsed_time
        tokens_per_seconds.append(tokens_per_second)

        print(f"Output: {output[:50]}...")  # Print a snippet of the output for reference
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Tokens per second: {tokens_per_second}")

    # Calculate statistics
    mean_elapsed_time = statistics.mean(elapsed_times)
    std_dev_elapsed_time = statistics.stdev(elapsed_times)

    mean_tokens_per_second = statistics.mean(tokens_per_seconds)
    std_dev_tokens_per_second = statistics.stdev(tokens_per_seconds)

    mean_token_count = statistics.mean(token_counts)
    std_dev_token_count = statistics.stdev(token_counts)

    print(f"\n--- Benchmark Results ---")
    print(f"Mean Elapsed Time: {mean_elapsed_time} seconds")
    print(f"Std Dev Elapsed Time: {std_dev_elapsed_time} seconds")
    print(f"Mean Tokens per Second: {mean_tokens_per_second}")
    print(f"Std Dev Tokens per Second: {std_dev_tokens_per_second}")
    print(f"Mean Token Count: {mean_token_count}")
    print(f"Std Dev Token Count: {std_dev_token_count}")

if __name__ == "__main__":
    prompt = "User: What is your favourite condiment? Assistant: Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen! User: Do you have mayonnaise recipes?"
    benchmark_llama(prompt, num_trials=10)
