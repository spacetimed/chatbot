#!/bin/sh

# basically just quickly generate a random prompt
python -m chatbot.generate --checkpoint checkpoints/best.pt --prompt "I want to" --tokens 200 --seed 1337 --temperature 0.7
