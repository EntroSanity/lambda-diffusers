## 1 Follow the instruction on https://github.com/abetlen/llama-cpp-python 
Only changed is at https://github.com/abetlen/llama-cpp-python/blob/main/docker/cuda_simple/Dockerfile#L1, replaced by `ARG CUDA_IMAGE="12.3.1-devel-ubuntu22.04"` for Ubuntu.
## 2 build and run docker
Follow the example below to build the docker
```
cd ./cuda_simple
docker build -t cuda_simple .
```

make sure to have `nvidia-container-toolkit` installed before run the docker
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

download an model example from https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF 
```
huggingface-cli download TheBloke/zephyr-7B-beta-GGUF zephyr-7b-beta.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```
to run the docker
```
docker run -p 8000:8000 --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e MODEL=/var/model/zephyr-7b-beta.Q4_K_M.gguf  -v /home/administrator/llama_cpp_python/models/:/var/model -t cuda_simple
```
followed by the format as 
```
docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e MODEL=/var/model/<model-path> -v <model-root-path>:/var/model -t cuda_simple
```

## 3 call API examples

```
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'
```

```
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```






