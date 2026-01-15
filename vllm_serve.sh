vllm serve \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 16384 \
    --attention-backend FLASHINFER \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --max-cudagraph-capture-size 4 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.81 \
    --served-model-name Heineken_qwen-3-8B-8bit \
    --prefix-caching-hash-algo xxhash \
    --disable-cascade-attn \
    --block-size 32 \
    --disable-sliding-window \
    --quantization fp8 \
    --trust-remote-code

###
    # --chat-template configs/chat_template.jinja2 \
###