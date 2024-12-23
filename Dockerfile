FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04
ARG http_proxy
ARG https_proxy
ENV TZ=Asia/Shanghai
ARG PIP_NO_CACHE_DIR=false

# When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. 
ENV SYCL_CACHE_PERSISTENT=1

# retrive oneapi repo public key
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    # update dependencies
    apt-get update && \
    # install basic dependencies
    apt-get install -y --no-install-recommends curl wget git libunwind8-dev vim less && \
    # install Intel GPU driver
    apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero level-zero-dev --allow-downgrades && \
    # install python 3.11
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    apt-get install -y --no-install-recommends gnupg gpg-agent software-properties-common && \
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    # avoid axolotl lib conflict
    apt-get remove -y python3-blinker && apt autoremove -y && \
    # link to python 3.11
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # remove apt cache
    rm -rf /var/lib/apt/lists/* && \
    # upgrade pip
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && \
    # install XPU ipex-llm
    pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    # install transformers & peft dependencies
    pip install transformers==4.36.0 && \
    pip install peft==0.10.0 datasets && \
    pip install bitsandbytes scipy fire &&\
    pip install trl==0.9.6

#install intel GPU driver
RUN wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16695.4/intel-igc-core_1.0.16695.4_amd64.deb && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16695.4/intel-igc-opencl_1.0.16695.4_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-level-zero-gpu-dbgsym_1.3.29377.6_amd64.ddeb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-level-zero-gpu_1.3.29377.6_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-opencl-icd-dbgsym_24.17.29377.6_amd64.ddeb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-opencl-icd_24.17.29377.6_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/libigdgmm12_22.3.19_amd64.deb && \
    dpkg -i *.deb && \
    rm *.deb && \
    rm *.ddeb

#Setup golang app
RUN apt-get update && apt-get install -y golang
RUN mkdir -p /app
COPY ./main.go /app/main.go
COPY ./templates /app/templates
COPY ./go.mod /app/go.mod
COPY ./go.sum /app/go.sum

# prepare finetune code and scripts
COPY ./LLM-Finetuning /LLM-Finetuning
COPY ./convert_ollama /convert_ollama

# Prepare accelerate config
RUN mkdir -p /root/.cache/huggingface/accelerate 
RUN mv /LLM-Finetuning/axolotl/default_config.yaml /root/.cache/huggingface/accelerate

#Setup llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp

# Run the Go application
#CMD ["go", "run", "/app/main.go"]