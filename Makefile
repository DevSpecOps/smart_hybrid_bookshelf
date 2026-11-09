.PHONY: install data features train-neural eval pipeline reproduce clean notebooks help

help:
	@echo "Targets:"
	@echo "  install        - pip install -r requirements.txt"
	@echo "  data           - download Goodreads CSVs into archive/"
	@echo "  features       - build data/raw/ and data/processed/ from archive/"
	@echo "  train-neural   - train Two-Tower model (GPU auto-detect)"
	@echo "  eval           - run evaluation, write reports/metrics_final.json"
	@echo "  pipeline       - full end-to-end run (skips completed stages)"
	@echo "  reproduce      - alias for pipeline"
	@echo "  clean          - remove generated artifacts"

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

features:
	python -m src.data.load_real --archive archive --output data/raw --config configs/config.yaml
	python -m src.data.build_features --raw data/raw --output data/processed

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
	rm -rf data/raw/*.csv data/processed/*.csv data/processed/*.pt reports/*.json
