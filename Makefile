PY := .venv/bin/python
HF := uvx hf@latest
FRAMEWORK := third_party/cosmos-framework

.PHONY: help setup check framework download-edge download-nano download-nano-fp8 fp8-deps \
        reason generate forward inverse policy watch serve-reasoner serve-generator \
        wam-describe wam-train wam-eval wam-imagine wam-bench wam-adapt gr00t-setup gr00t-n17 \
        sim-setup sim-server sim sim-batch sim-grid

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

# Miniature GR00T 2-style world-action model; see docs/gr00t2.md. NAME=<checkpoint> to switch.
NAME ?= base

wam-describe: ## Mini world-action model: token layout and parameter counts
	$(PY) scripts/wam.py describe

wam-train: ## Train it on the toy tabletop (about 13 min)
	$(PY) scripts/wam.py train --name $(NAME)

wam-eval: ## Closed-loop success per robot and colour
	$(PY) scripts/wam.py eval --name $(NAME)

wam-imagine: ## PNG of the future it imagined next to what happened
	$(PY) scripts/wam.py imagine --name $(NAME)

wam-bench: ## Latency and success across denoising schedules, KV cache, bf16, torch.compile
	$(PY) scripts/wam.py bench --name $(NAME) --compile

wam-adapt: ## LoRA few-shot on the held-out task (SHOTS=1 or 5)
	$(PY) scripts/wam.py adapt --name $(NAME) --shots $(or $(SHOTS),5)

# GR00T N1.7, the open predecessor of GR00T 2. It pins its own Python and Transformers,
# so it gets its own environment inside the checkout (15 GB, about 25 min to build).
GR00T := third_party/Isaac-GR00T

gr00t-setup: ## Clone Isaac-GR00T and build its environment (needs git-lfs)
	test -d $(GR00T) || git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git $(GR00T)
	cd $(GR00T) && uv sync --python 3.12

gr00t-n17: ## GR00T N1.7 on a recorded DROID episode: shapes, parameter counts, latency (needs ~12 GB free RAM)
	$(GR00T)/.venv/bin/python scripts/gr00t_n17.py --describe

# The Edge DROID policy driving a simulated robot. RoboLab runs the scene on Isaac Sim 6.0.1 +
# Isaac Lab 3.0 beta (RoboLab's default Isaac Sim 5.0 crashes on this machine's 595 driver);
# NVIDIA's reference policy server runs the model. Each has its own environment, and they run in
# two terminals: `make sim-server`, then `make sim`. See docs/sim.md.
ROBOLAB := third_party/RoboLab
SIM_PY := $(ROBOLAB)/.venv-60/bin/python
SERVER_PY := $(FRAMEWORK)/.venv/bin/python
TASK ?= BananaInBowlTask
CFG ?= 1 3 6
# Each job in its own memory-capped scope, so running out of RAM kills that job and not the
# desktop session. NOCAP=1 runs without it.
capped = $(if $(NOCAP),,systemd-run --user --scope -q -p MemorySwapMax=0 -p MemoryMax=$(1))

sim-setup: framework ## Clone RoboLab, build the simulator and policy-server environments (about 40 GB)
	test -d $(ROBOLAB) || git clone --depth 1 https://github.com/NVlabs/RoboLab.git $(ROBOLAB)
	cd $(ROBOLAB) && UV_HTTP_TIMEOUT=600 UV_PROJECT_ENVIRONMENT=.venv-60 uv sync --python 3.12 --extra isaac60
	# RoboLab's websockets override only covers Python < 3.12, so its Isaac Sim 6 env lacks the
	# package its policy clients import. --no-config keeps that override out of this install.
	uv pip install --no-config --python $(SIM_PY) "websockets>=14.0"
	cd $(FRAMEWORK) && uvx uv@0.11.29 sync --group cu130-torch213 --group policy-server
	# The server module imports these from the (much larger) training extra.
	cd $(FRAMEWORK) && uvx uv@0.11.29 pip install --python .venv/bin/python \
		webdataset pandas wandb "multi-storage-client==0.44.0"

sim-server: ## Edge DROID policy server for the simulator on 127.0.0.1:8000 (CFG settable per plan)
	COSMOS_VRAM_CAP_GIB=14 $(call capped,12G) $(SERVER_PY) scripts/sim_policy_server.py

# OMNI_KIT_ACCEPT_EULA=Y accepts the NVIDIA Omniverse license that Isaac Sim asks for on first run.
sim: ## Simulated DROID robot you instruct by typing (TASK=BananaInBowlTask; needs sim-server)
	OMNI_KIT_ACCEPT_EULA=Y $(call capped,14G) $(SIM_PY) scripts/sim_droid.py --task $(TASK) --viz kit

sim-batch: ## Headless: every PROMPTS x CFG x SEEDS run from a fresh scene, videos + success table
	OMNI_KIT_ACCEPT_EULA=Y $(call capped,14G) $(SIM_PY) scripts/sim_droid.py --task $(TASK) \
		--prompt $(or $(PROMPTS),"Pick up the banana and place it in the bowl") --cfg $(CFG) \
		--seed $(or $(SEEDS),0)

sim-grid: ## One labeled comparison video of a batch (RUN=outputs/sim/<folder>, default newest; VIEWS=1)
	python3 scripts/sim_grid.py $(RUN) $(if $(VIEWS),--views)

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
