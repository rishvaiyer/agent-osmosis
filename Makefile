.PHONY: install demo test dashboard all clean

install:      ## install engine + viz deps (CPU, no downloads)
	pip install -e ".[viz]"

demo:         ## run all reproducible experiments (warm-vs-cold, compounding, invariance, baselines)
	python examples/01_warm_vs_cold.py
	python examples/02_compounding.py
	python examples/03_influence_invariance.py
	python examples/05_baselines.py

transformer:  ## run the real (from-scratch) transformer validation — needs .[llm]
	python examples/04_real_transformer.py

test:         ## run the test suite
	pytest -q

dashboard:    ## launch the Streamlit control panel
	streamlit run dashboard/app.py

all: install test demo

clean:
	rm -f examples/out_*.png
	rm -rf __pycache__ */__pycache__ .pytest_cache *.egg-info
