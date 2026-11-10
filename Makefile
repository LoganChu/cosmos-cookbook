PY := .venv/bin/python
HF := uvx hf@latest
FRAMEWORK := third_party/cosmos-framework

.PHONY: help setup check framework download-edge download-nano download-nano-fp8 fp8-deps \
        reason generate forward inverse policy watch serve-reasoner serve-generator

help:
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

setup: ## Create .venv and install the inference stack (TORCH_BACKEND=cu130|cu128)
	scripts/setup.sh

check: ## Show GPU/RAM headroom, which models fit right now, and Hugging Face access
	$(PY) scripts/check_env.py

framework: ## Clone NVIDIA's cosmos-framework (sample episodes, FP8 export, policy server)
	test -d $(FRAMEWORK) || git clone --depth 1 https://github.com/NVIDIA/cosmos-framework.git $(FRAMEWORK)

download-edge: ## Cosmos3-Edge + Edge DROID policy (about 9 GB each)
	$(HF) download nvidia/Cosmos3-Edge
	$(HF) download nvidia/Cosmos3-Edge-Policy-DROID

download-nano: ## Cosmos3-Nano BF16 (35 GB)
	$(HF) download nvidia/Cosmos3-Nano

download-nano-fp8: ## Cosmos3-Nano FP8 weights (21 GB, the `fp8` branch)
	$(HF) download nvidia/Cosmos3-Nano --revision fp8

fp8-deps: ## Install NVIDIA ModelOpt, needed to load FP8 checkpoints
	uv pip install --python $(PY) "nvidia-modelopt==0.44.0" requests

reason: ## Edge reasoner on the bundled robot scene
	$(PY) scripts/reason.py

generate: ## Edge image-to-video, 5 s at 480p
	$(PY) scripts/generate.py

forward: ## Edge forward dynamics: actions -> video
	$(PY) scripts/dynamics.py forward

inverse: ## Edge inverse dynamics: video -> actions
	$(PY) scripts/dynamics.py inverse

policy: framework ## Edge DROID policy: cameras + instruction -> action chunks
	$(PY) scripts/policy.py

watch: ## Reasoner as a live monitor over a video stream
	$(PY) scripts/watch.py

# OpenAI-compatible servers, as given on NVIDIA's model cards. MODEL=nvidia/Cosmos3-Nano to switch.
MODEL ?= nvidia/Cosmos3-Edge
DOCKER_RUN := docker run --rm --gpus all --ipc host -p 8000:8000 \
	-v $(HOME)/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN

serve-reasoner: ## vLLM chat endpoint (text out) on :8000
	$(DOCKER_RUN) vllm/vllm-openai:cosmos3 $(MODEL) --host 0.0.0.0 --port 8000 \
		--max-model-len 32768 --allowed-local-media-path / \
		--media-io-kwargs '{"video": {"num_frames": 64}}'

serve-generator: ## vLLM-Omni video/action endpoint on :8000
	$(DOCKER_RUN) vllm/vllm-omni:cosmos3 vllm serve $(MODEL) --omni --host 0.0.0.0 --port 8000 \
		--init-timeout 1800 --enable-layerwise-offload
