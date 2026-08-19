# chatbot

An educational LLM which visualizes different stages of the LLM! Built with both privacy and education in mind.

## Implementation progress

**Phase 0** (Basic scaffolding) — *Complete*
- Create the project's environment (`pyproject.toml`, `pytest`, `venv`, etc.)
- Import my decoder-only GPT from `gradcore/`, and get the basics working
- Some small smoke tests

**Phase 1** (GPT-2-style model) — *Complete*
- Mold the model in the direction of GPT-2. GPT-2-style implementations added:
    - Abstracted configuration into `GPTConfig`
    - Vectorized causal self-attention in `CausalSelfAttention` (removed `SingleHeadAttention`, `MultiHeadAttention`)
    - Added `MLP` with tanh-approximation GELU (same as GPT-2) (removed `FeedForward`)
    - Weight tying between input and output layers (`lm_head` and `token_embedding_table`)
    - Weight initialization with `std=0.02`, scaled by `1/√(2*num_layers)` for residual-output projections 
    - Rewrite `train.py` completely with basic checkpointing and a `TrainConfig` abstraction
        - Improve logging and analytics to prepare for system analyzing
        - Gradient clipping to combat exploding gradients
        - Decaying learning rate
        - AdamW configuration (similar to Karpathy's nanogpt)

**Phase 2** (Rewrite Python tokenizer; port rewrite to C++/Rust) — *In-progress*
- Three different languages will be used to create versions of the same BPE tokenizer, and compare/benchmark results:
    1. Python 
    2. C++
    3. Rust
- Firstly, I'll rewrite a single Tokenizer specification used by all three implementations. All implementations will:
    - Consume the exact same learned merge table.
    - Produce identical token IDs and decoded bytes.
    - Report the same analytics: MB/s, token/s, peak memory, compression ratio.
- The Python implementation will be used as a baseline, and to offer basic readability of my algorithm.
- Both C++ and Rust implementations will "compete" in performance (relative to baseline), and utilize low-level optimizations like SIMD.

## Brainstorming

I plan to include as much of the stuff listed below in my project as I can. There's a lot I really want to learn, so I'm using this project as an opportunity to do just that.

Model / ML
- Privacy-focused deployment
- Graphic to visualize LLM computation (like a verbose ChatGPT lol)
- Re-use and substantially improve my from-scratch GPT
- Conversational/instructional fine-tuning
- Determine appropriate model size and training corpus
- Rent GPU for training
- Checkpointing and model versioning
- Custom Rust/C++ tokenizer implementation
- Custom CUDA kernel for one or more operations
- Benchmark custom implementations against PyTorch/reference versions

Inference
- GPU inference worker
- Token streaming
- Request queue
- Dynamic or continuous batching
- KV cache
- Generation controls
- Graceful cancellation / timeouts
- Measure tokens/sec, time-to-first-token, p50/p95 latency

Backend
- FastAPI
- REST API + SSE or WebSocket streaming
- PostgreSQL for users/conversations/model metadata
- Redis for caching / rate limiting
- Background job or message queue
- Retry / timeout / idempotency behavior
- Health/readiness endpoints

Frontend
- Next.js + TypeScript
- Streaming chat UI
- Conversation history

Infrastructure
- Docker
- AWS
- S3 for model checkpoints/artifacts
- EC2 GPU for inference if needed
- RDS PostgreSQL
- SQS for asynchronous jobs
- CI/CD with GitHub Actions
- Terraform
- Separate development / production configuration

Observability
- Structured logging
- OpenTelemetry
- Prometheus + Grafana OR AWS CloudWatch
- Request/inference metrics
- Error-rate and latency monitoring

Testing / reliability
- Unit tests
- Integration tests
- End-to-end tests
- Load testing
- Failure testing for queue/worker/model-service failures

## Note about AI usage

For this project, my reliance on AI tools is very constrained. AI tools will be used as bounded engineering assistants, rather than autonomous developers. 

All code and architectural decisions will be primarily my own. I'm using this to learn.

## References
- https://github.com/karpathy/build-nanogpt
- *Language Models are Unsupervised Multitask Learners* 
- *Attention Is All You Need*
