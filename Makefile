.PHONY: install data train-neural eval pipeline reproduce notebooks clean help

help:
	@echo "Targets:"
	@echo "  install        - pip install -r requirements.txt"
	@echo "  data           - download + canonicalize Goodreads CSVs"
	@echo "  train-neural   - train Two-Tower model (feature engineering from train split only)"
	@echo "  eval           - run evaluation, write reports/metrics_final.json"
	@echo "  pipeline       - full end-to-end run (skips completed stages)"
	@echo "  notebooks      - regenerate and execute EDA/comparison notebooks"
	@echo "  reproduce      - pipeline + notebooks"
	@echo "  clean          - remove generated artifacts"

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py
	python -m src.data.load_real --archive archive --output data/raw --config configs/config.yaml

train-neural:
	python scripts/train_two_tower.py --config configs/config.yaml

eval:
	python -m src.evaluate_baseline --config configs/config.yaml --output reports/metrics_final.json

pipeline:
	python -m src.pipeline --config configs/config.yaml

notebooks:
	python scripts/build_notebooks.py

reproduce: pipeline notebooks

clean:
	rm -rf data/raw/*.csv data/processed/*.pt reports/*.json
