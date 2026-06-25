export HEADROOM_TELEMETRY := off

INPUT_FILE ?= ./test_data/0001.exr
INPUT_DIR ?= ./test_data/sequence
export NOISE_DIR ?= ./test_data/noise
export PAPER_DIR ?= ./test_data/paper
EXR_FILES := $(wildcard $(INPUT_DIR)/*.exr)

.PHONY: run aider seq

run:
	uv run main.py

aider:
	headroom wrap aider --watch-files --config .aider.conf.yml

test:
	uv run pytest -s tests/

test-run:
	uv run pytest -s tests/test_image_processing.py