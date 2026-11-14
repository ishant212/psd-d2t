#!/usr/bin/env bash
set -e

# ==============================================================================
# Complete Pipeline Runner: Remaining 10 Model Runs (L=2048) + Ablation Suite
# ==============================================================================

SESSION_NAME="pjt1_runs"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Attaching to it..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

echo "======================================================================"
echo "Creating new tmux session: $SESSION_NAME"
echo "======================================================================"
tmux new-session -d -s $SESSION_NAME

# Setup environment inside tmux
tmux send-keys -t $SESSION_NAME "source ~/venv_pjt1/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "cd ~/PJT1" C-m
tmux send-keys -t $SESSION_NAME "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" C-m

# ------------------------------------------------------------------------------
# STEP 1: Complete the remaining 10 long-context model runs at L=2048
# Models: Phi-3.5-mini, Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Gemma-2-9B
# Modes: Baseline and PSD (150 steps each)
# ------------------------------------------------------------------------------
echo "Queueing Step 1: Remaining 5 models at L=2048..."
tmux send-keys -t $SESSION_NAME "echo '>>> STARTING STEP 1: REMAINING 5 MODELS AT L=2048'" C-m
tmux send-keys -t $SESSION_NAME "python3 -m src.psd.run_cross_model_evaluation --models phi_3_5 qwen_7b mistral_7b llama_8b gemma_9b --modes baseline psd --steps 150 --context_lengths 2048" C-m

# ------------------------------------------------------------------------------
# STEP 2: Automatically chain into the Cross-Model Ablation Suite
# Evaluates: Lambda loss sensitivity, Layer placement, and Component decomposition
# Models: Gemma-4B, Llama-3B, Qwen-3B
# ------------------------------------------------------------------------------
echo "Queueing Step 2: Cross-Model Ablation Suite..."
tmux send-keys -t $SESSION_NAME "echo '>>> STARTING STEP 2: CROSS-MODEL ABLATION SUITE'" C-m
tmux send-keys -t $SESSION_NAME "python3 -m src.psd.run_cross_model_ablation --models gemma_4b llama_3b qwen_3b --ablation_types all --steps 150" C-m

tmux send-keys -t $SESSION_NAME "echo '======================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'ALL MODEL RUNS AND ABLATIONS COMPLETED SUCCESSFULLY!'" C-m
tmux send-keys -t $SESSION_NAME "echo '======================================================'" C-m

echo "======================================================================"
echo "Pipeline successfully started inside tmux session: $SESSION_NAME"
echo "----------------------------------------------------------------------"
echo "To view live progress:   tmux attach -t $SESSION_NAME"
echo "To safely detach:        Press Ctrl + B, then D"
echo "======================================================================"

tmux attach-session -t $SESSION_NAME
