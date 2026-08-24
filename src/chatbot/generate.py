# generate.py rewrite

# target usage:
#   python -m chatbot.generate \
#       --checkpoint checkpoints/best.pt \
#       --prompt "Machine learning is" \
#       --max-new-tokens 200 \
#       --temperature 0.8 \
#       --top-k 50 \
#       --seed 1337

# today's scope:
#  1. load input checkpoint
#  2. reconstruct GPTConfig/weighs/tokenizer
#  3. prompt functionality
#  4. temperature/top-k/token count/seed/device options
#  5. generate/decode a completion
#  6. confirm seed produces deterministic output
