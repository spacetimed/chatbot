# Chatbot

An chatbot that visualizes different stages of an LLM. Built with privacy and education in mind! 

This project is an extension of my work in the [Gradcore](https://github.com/spacetimed/gradcore) repository, where I learned, designed, and implemented various language models, finally ending with a GPT-style transformer.

## Progress

**Phase 0: Preliminary** — *Complete*
- Created the project's environment (`pyproject.toml`, `pytest`, `venv`, etc).
- Imported my decoder-only GPT from `gradcore/`, and got the basics working.
- Some small smoke tests to ensure it trains.

**Phase 1: Upgrade to GPT-2-like Architecture** — *Complete*
- Molded the model in the direction of GPT-2. GPT-2-style implementations added:
    - Abstracted configuration into `GPTConfig`
    - Vectorized causal self-attention in `CausalSelfAttention` (removed `SingleHeadAttention`, `MultiHeadAttention`)
    - Added `MLP` with tanh-approximation GELU (same as GPT-2) (removed `FeedForward`)
    - Weight tying between input and output layers (`lm_head` and `token_embedding_table`)
    - Weight initialization with `std=0.02`, scaled by `1/√(2*num_layers)` for residual-output projections 
    - Rewrote `train.py` completely with basic checkpointing and a `TrainConfig` abstraction
        - Improved logging and analytics to prepare for system analyzing
        - Gradient clipping to combat exploding gradients
        - Decaying learning rate
        - AdamW configuration (similar to Karpathy's nanogpt)

**Phase 2: Tokenizer challenge, implemented in Python, C++, and Rust** — *In-progress*

- Preliminary: [Tokenizer specification that I wrote](docs/tokenizer.md)
    - The tokenizer will be created three times in each language: Python, C++, and Rust. 
    - It will be similar to the GPT-2 tokenizer, such as using the same Regex pattern.
    - The Python one serves as a naive tokenizer implementation, whereas the C++ and Rust serve as optimized performant rewrites.
    - Results will be benchmarked, and the most optimal implementation will be used in production.
    - The C++ and Rust variants will need to deterministically create the same output as the Python baseline (tested through their artifacts).
    - Different low-level optimization techniques can be used in both the C++ and Rust variants (e.g. SIMD), so long as the same output is created as Python.
    - *Why?* Curiosity, optimization is fun. And because a tokenizer is simple enough that it provides a nice exercise to toy with other languages.
- While working on the tokenizer, I upgraded the toy dataset from the Plato text used in Gradcore to streaming [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) with HuggingFace datasets. 
- I rewrote `tokenizer_driver.py` to serve as a harness for running different language tokenizer implementations.
- I rewrote `tokenizer.py` (from my Gradcore repo) to be a naive and readable implementation for the C++ and Rust variants to follow. 
- Moved dataset loading to `dataset_loader.py` so that both `train.py` and `tokenizer_driver.py` can stream HF datasets.

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

## References (todo, cite properly and hyperlink)
- [NanoGPT by Karpathy](https://github.com/karpathy/build-nanogpt)
- [GPT-2 Tokenizer by OpenAI](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- *Language Models are Unsupervised Multitask Learners* 
- *Attention Is All You Need*