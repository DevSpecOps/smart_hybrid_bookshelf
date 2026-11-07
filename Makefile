.PHONY: install data features eval pipeline reproduce clean

install:
	pip install -r requirements.txt

data:
	python -m src.data.make_dataset --config configs/config.yaml --output data/raw

features:
	python -m src.data.build_features --raw data/raw --output data/processed

eval:
	python -m src.evaluate_baseline --config configs/config.yaml --output reports/metrics_baseline.json

pipeline:
	python -m src.pipeline --config configs/config.yaml

reproduce: data features eval

clean:
	rm -rf data/processed reports/*.json
