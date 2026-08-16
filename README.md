# chatbot

---

My goal for this project is to build, train, serve, and deploy a small GPT-like language model. This document marks the beginning of the journey.

**Project components (initial brainstorming)**

Model / ML
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

**Note about AI usage**

For this project, I want to keep my reliance on AI tools constrained. AI tools will be used as bounded engineering assistants rather than autonomous developers. 

I will hold ownership of the system design, technical decisions, integration, testing, etc. Code will be primarily my own; the goal of this project is learning.
