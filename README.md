IPEX-LLM Fine tune App written in Golang to let you finetune model with huggingface datasets using Intel GPU and convert it to GGUF format

1. Setup a docker container based on this link: https://github.com/intel-analytics/ipex-llm/tree/main/docker/llm/finetune/xpu
2. git clone https://github.com/shungyan/IPEX-LLM-Finetune-App.git into docker container
3. Replace LLM-Finetuning folder in docker container with LLM-Finetuning folder in this repo
4. apt install update && apt install golang
5. git clone https://github.com/ggerganov/llama.cpp
6. Create a write token and copy it to huggingface-cli login
7. cd IPEX-LLM-Finetune-APP && go run main.go
