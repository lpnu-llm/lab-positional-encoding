PYTHON ?= python3

.PHONY: all notebook figures check

all: figures notebook

notebook: positional_encoding.ipynb

positional_encoding.ipynb: positional_encoding.py tools/percent_to_ipynb.py
	$(PYTHON) tools/percent_to_ipynb.py $< $@

figures:
	$(PYTHON) tools/plot_attention.py

check:
	$(PYTHON) -m py_compile positional_encoding.py tools/percent_to_ipynb.py tools/plot_attention.py
