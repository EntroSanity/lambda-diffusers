# https://gist.github.com/tsubasakong/aa9157cb79d6a0653201e9548dbc030b
# Lamma tokenizer playground

from transformers import AutoTokenizer
import transformers
import torch
import time
import statistics

# text = "The fall of Empire, gentlemen, is a massive thing, however, and not easily fought. It is dictated by a rising bureaucracy, a receding initiative, a freezing of caste, a damming of curiosity—a hundred other factors. It has been going on, as I have said, for centuries, and it is too majestic and massive a movement to stop."

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-hf


tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device="cuda:0",
)



def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response and the number of tokens per second.
    """
    start_time = time.time()
    
    encoded_input = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    print("encoded_input",encoded_input)

    sequences = llama_pipeline(
        encoded_input,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=150,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    response_text = sequences[0]['generated_text']
    num_tokens = len(tokenizer.encode(response_text))

    tokens_per_second = num_tokens / elapsed_time

    print("Chatbot:", response_text)
    print(f"Number of Tokens: {num_tokens}")
    print(f"Total time in secs: {elapsed_time}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    return num_tokens, elapsed_time, tokens_per_second

# prompt = [
#   {"role": "user", "content": "Hello, how are you?"},
#   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#   {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]

prompt = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

if __name__ == "__main__":
    # get_llama_response(prompt)
    num_trials = 10
    token_counts = []
    elapsed_times = []
    tokens_per_seconds = []
    for i in range(num_trials):
        token_count,elapsed_time,  tokens_per_second= get_llama_response(prompt)
        token_counts.append(token_count)
        elapsed_times.append(elapsed_time)
        tokens_per_seconds.append(tokens_per_second)


    mean_tokens = statistics.mean(token_counts)
    variance_tokens = statistics.variance(token_counts)
    std_dev_tokens = statistics.stdev(token_counts)

    mean_elapsed_time = statistics.mean(elapsed_times)
    variance_elapsed_time = statistics.variance(elapsed_times)
    std_dev_elapsed_time = statistics.stdev(elapsed_times)

    mean_tokens_per_second = statistics.mean(tokens_per_seconds)
    variance_tokens_per_second = statistics.variance(tokens_per_seconds)
    std_dev_tokens_per_second = statistics.stdev(tokens_per_seconds)


    print(f"Mean elapsed time in secs: {mean_elapsed_time}")
    print(f"Variance of elapsed time in secs: {variance_elapsed_time}")
    print(f"Standard deviation of elapsed time in secs: {std_dev_elapsed_time}")

    print(f"Mean tokens per second: {mean_tokens_per_second:.2f}")
    print(f"Variance of tokens per second: {variance_tokens_per_second:.2f}")
    print(f"Standard deviation of tokens per second: {std_dev_tokens_per_second:.2f}")

    
    print(f"Mean number of tokens: {mean_tokens}")
    print(f"Variance of number of tokens: {variance_tokens}")
    print(f"Standard deviation of number of tokens: {std_dev_tokens:.2f}")