from transformers import pipeline, AutoTokenizer
import torch
import time
import statistics

proxy = False
if proxy:
    import os
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
    os.environ['ALL_PROXY'] = "socks5://127.0.0.1:7890"

model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Specify the model to use

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the pipeline for text-generation with the specified model
mistral_pipeline = pipeline(
    "text-generation",  # Specify the task
    model=model_name,
    tokenizer=tokenizer,  # Use the specified tokenizer
    torch_dtype=torch.float16,  # Specify the data type for computation
    device="cuda:0",
)

# pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

def get_mistral_response(messages: list) -> None:
    """
    Generate a response from the Mistral model based on the conversation history.

    Parameters:
        messages (list): A list of message dictionaries with "role" and "content" keys.

    Returns:
        None: Prints the model's response.
    """
    # Encode the conversation using the chat template
    encoded_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("encoded_input", encoded_input)
    
    start_time = time.time()
    # Generate a response
    sequences = mistral_pipeline(
        encoded_input, 
        max_new_tokens=150,  # Limit the number of new tokens to generate
        do_sample=True,  # Enable sampling to generate diverse responses
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Extract the generated response
    response_text = sequences[0]['generated_text']
   
    num_tokens = len(tokenizer.encode(response_text))
    tokens_per_second = num_tokens / elapsed_time
    
    print("Chatbot:", response_text)
    print(f"Number of Tokens: {num_tokens}")
    print(f"Total time in secs: {elapsed_time}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    return num_tokens, elapsed_time, tokens_per_second


# Example conversation
prompt = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

if __name__ == "__main__":
    # get_mistral_response(messages)
    num_trials = 10
    token_counts = []
    elapsed_times = []
    tokens_per_seconds = []
    for i in range(num_trials):
        token_count,elapsed_time,  tokens_per_second= get_mistral_response(prompt)
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