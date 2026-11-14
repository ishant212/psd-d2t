#!/usr/bin/env bash
set -e

# ==============================================================================
# Cross-Model Ablation Runner for Top 3 Models (Gemma-4B, LLaMA-3B, Qwen-3B)
# ==============================================================================

# Session name
SESSION_NAME="ablation_suite"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Attaching to it..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

echo "Creating new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Send execution commands inside the tmux session
tmux send-keys -t $SESSION_NAME "source ~/venv_pjt1/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "cd ~/PJT1" C-m
tmux send-keys -t $SESSION_NAME "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" C-m
tmux send-keys -t $SESSION_NAME "python3 -m src.psd.run_cross_model_ablation --models gemma_4b llama_3b qwen_3b --ablation_types all --steps 150" C-m

echo "======================================================================"
echo "Ablation run launched inside tmux session: $SESSION_NAME"
echo "To view live progress, run:"
echo "  tmux attach -t $SESSION_NAME"
echo "To detach safely without stopping the run:"
echo "  Press Ctrl + B, then D"
echo "======================================================================"

tmux attach-session -t $SESSION_NAME
