vllm serve \
    --model Heineken_qwen-3-8B_chatbot_merged \
    --chat-template configs/chat_template.jinja2  \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 2048 \
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
    --reasoning-parser qwen3 

###
    # --chat-template configs/chat_template.jinja2 \
###