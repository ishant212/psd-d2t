#!/usr/bin/env bash
set -e

# ==============================================================================
# Execution Runner for Remaining 5 Models across L=1024 and L=2048
# Models: Phi-3.5-mini, Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Gemma-2-9B
# ==============================================================================

SESSION_NAME="remaining_5_models"

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
tmux send-keys -t $SESSION_NAME "python3 -m src.psd.run_cross_model_evaluation --models phi_3_5 qwen_7b mistral_7b llama_8b gemma_9b --modes baseline psd --steps 150 --context_lengths 1024 2048" C-m

echo "======================================================================"
echo "Job successfully launched inside tmux session: $SESSION_NAME"
echo "Models: phi_3_5, qwen_7b, mistral_7b, llama_8b, gemma_9b"
echo "Context lengths: 1024, 2048"
echo "Modes: baseline, psd (150 steps each)"
echo "----------------------------------------------------------------------"
echo "To monitor live progress:"
echo "  tmux attach -t $SESSION_NAME"
echo "To detach safely without stopping the run:"
echo "  Press Ctrl + B, then D"
echo "======================================================================"

tmux attach-session -t $SESSION_NAME
