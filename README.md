# chatbot

An educational LLM which visualizes different stages of the LLM! Built with both privacy and education in mind.

# Roadmap

**Phase 0**
- Create the project's environment (`pyproject.toml`, `pytest`, etc.)
- Import my decoder-only GPT from `gradcore/`, and get the basics working
- Some small smoke tests

**Phase 1**
- Mold the model in the direction of GPT-2. Implementations include:

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
- Custom C/C++ tokenizer implementation
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