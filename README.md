IPEX-LLM Fine tune App written in Golang to let you finetune model with huggingface datasets using Intel GPU and convert it to GGUF format

1. git clone https://github.com/shungyan/IPEX-LLM-Finetune-App.git into docker container
2. docker build -t ipex-llm-finetuneapp -f ./Dockerfile .
3. docker run -itd --net=host --device=/dev/dri --memory="32G" --name=ipex-llm-finetuneapp --shm-size="16g" ipex-llm-finetuneapp
4. docker exec -it ipex-llm-finetuneapp bash
5. cd /app
6. go run main.go
7. go to localhost:12345
   
