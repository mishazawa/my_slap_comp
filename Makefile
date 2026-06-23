export HEADROOM_TELEMETRY := off

INPUT_DIR ?= ./test_data/sequence
EXR_FILES := $(wildcard $(INPUT_DIR)/*.exr)

.PHONY: run aider seq

run:
	uv run main.py

aider:
	headroom wrap aider

seq:
	@echo "Processing EXR files in '$(INPUT_DIR)'..."
	@for file in $(EXR_FILES); do \
		echo "Running: uv run main.py $$file"; \
		uv run main.py "$$file" || exit 1; \
	done