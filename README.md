# IPEX-LLM Fine-tune App

This application allows you to fine-tune Hugging Face models with custom datasets using Intel GPUs via **IPEX-LLM**, and then convert them into **GGUF** format for optimized inference.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shungyan/IPEX-LLM-Finetune-App.git
cd IPEX-LLM-Finetune-App
```
###2. Build the Docker Image
```bash
docker build -t ipex-llm-finetuneapp -f ./Dockerfile .
```
###3. Run the Docker Container
```bash
docker run -itd \
  --net=host \
  --device=/dev/dri \
  --memory="32G" \
  --shm-size="16g" \
  --name=ipex-llm-finetuneapp \
  ipex-llm-finetuneapp
```
###4. Enter the Container
```bash
docker exec -it ipex-llm-finetuneapp bash
```
###5. Start the Application
```bash
cd /app
go run main.go
```
6. Access the Web UI

Open your browser and navigate to:
http://localhost:12345

🖥️ Requirements

Intel GPU with proper drivers installed

Docker with GPU support enabled

At least 32GB memory and 16GB shared memory recommended for fine-tuning

📦 Features

Fine-tune Hugging Face models with your datasets

Optimized for Intel GPUs with IPEX-LLM

Convert fine-tuned models into GGUF format for lightweight inference

Simple web interface to manage projects and sessions

   
