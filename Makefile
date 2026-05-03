PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip

.PHONY: setup prepare train evaluate clean

setup:
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

prepare:
	$(VENV)/bin/python scripts/prepare_dataset.py --yolo-export data/yolo_export/

train:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	$(VENV)/bin/python scripts/train.py

evaluate:
	$(VENV)/bin/python scripts/evaluate.py \
		--weights $(shell find runs -name best.pt | tail -1)

clean:
	rm -rf data/dataset runs __pycache__ .venv *.egg-info
