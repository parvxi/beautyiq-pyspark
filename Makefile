export JAVA_HOME ?= /opt/homebrew/opt/openjdk@17
PY := .venv/bin/python
HF := https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw

setup:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

download:
	mkdir -p data/raw
	curl -L -C - -o data/raw/Beauty_and_Personal_Care.jsonl $(HF)/review_categories/Beauty_and_Personal_Care.jsonl
	curl -L -C - -o data/raw/meta_Beauty_and_Personal_Care.jsonl $(HF)/meta_categories/meta_Beauty_and_Personal_Care.jsonl

bronze:
	$(PY) scripts/01_bronze_ingest.py

run:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 beauty_pipeline.ipynb
	cd notebooks && ../.venv/bin/jupyter nbconvert --to html beauty_pipeline.ipynb

history:
	@echo "Spark History Server: http://localhost:18080  (Ctrl+C to stop)"
	$(JAVA_HOME)/bin/java -cp ".venv/lib/python3.9/site-packages/pyspark/jars/*" \
		-Dspark.history.fs.logDirectory=file://$(HOME)/.spark-events \
		org.apache.spark.deploy.history.HistoryServer

.PHONY: setup download bronze run history
