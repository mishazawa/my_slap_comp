export HEADROOM_TELEMETRY := off

INPUT_FILE ?= ./test_data/0001.exr
INPUT_DIR ?= ./test_data/sequence
NOISE_DIR ?= ./test_data/noise
PAPER_DIR ?= ./test_data/paper
EXR_FILES := $(wildcard $(INPUT_DIR)/*.exr)

.PHONY: run aider seq

run:
	uv run main.py $(INPUT_FILE) -p $(PAPER_DIR) -n $(NOISE_DIR)

aider:
	headroom wrap aider --watch-files --config .aider.conf.yml

seq:
	@echo "Processing EXR files in '$(INPUT_DIR)'..."
	@for file in $(EXR_FILES); do \
		echo "Running: uv run main.py $$file"; \
		uv run main.py "$$file" -p $(PAPER_DIR) -n $(NOISE_DIR) || exit 1; \
	done