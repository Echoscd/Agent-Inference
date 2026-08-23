#!/bin/bash
# Environment for ToolOrchestra HLE eval (smoke + scaled-down repro)
# All experts mapped to gpt-5-mini via OpenRouter.

# --- Paths ---
export AGENT_ROOT="/home/jerryzhou/chendong/agent-inference"
export REPO_PATH="${AGENT_ROOT}/ThunderAgent/examples/inference/ToolOrchestra"
export THUNDERAGENT_ROOT="${AGENT_ROOT}/ThunderAgent"
export CKPT_DIR="${AGENT_ROOT}/Nemotron-Orchestrator-8B"
export USER_PATH="${HOME}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export INDEX_DIR=""  # unused; we mock retrieval

# Skip ThunderAgent code requiring Tavily, since we mock retrieval
export TAVILY_KEY="${TAVILY_KEY:-unused}"
export TOGETHER_API_KEY="${TOGETHER_API_KEY:-unused}"

# --- OpenRouter routing (used by openai SDK as base_url) ---
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# --- OPENAI_API_KEY: paste your sk-or-v1-... key on the next line ---
export OPENAI_API_KEY="${OPENAI_API_KEY:?set your sk-or-v1-... key in the environment}"   # set externally OR replace this line

# --- venv for python ---
export VENV_BIN="${AGENT_ROOT}/.venv/bin"
export PATH="${VENV_BIN}:${PATH}"

# --- HLE eval knobs ---
export HLE_EXAMPLE_PATH="${AGENT_ROOT}/hle_sampled_200.jsonl"
export HLE_TOKENIZER_NAME="${CKPT_DIR}"
export HLE_JUDGE_MODEL="deepseek/deepseek-chat"  # OpenAI blocked in HK; use DeepSeek via OpenRouter

# --- Proxy: OpenRouter needs to be reachable; clash exits US in your setup ---
# Uncomment if direct connection to OpenRouter is slow/blocked
# export HTTPS_PROXY="http://127.0.0.1:7890"
# export HTTP_PROXY="http://127.0.0.1:7890"

# --- Sanity print ---
echo "[setup_envs] CKPT_DIR=${CKPT_DIR}"
echo "[setup_envs] REPO_PATH=${REPO_PATH}"
echo "[setup_envs] OPENAI_API_KEY=${OPENAI_API_KEY:0:14}... (len=${#OPENAI_API_KEY})"
echo "[setup_envs] OPENAI_BASE_URL=${OPENAI_BASE_URL}"
