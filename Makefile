PYTHON ?= python3

.PHONY: all notebook check

all: notebook

notebook: positional_encoding.ipynb

positional_encoding.ipynb: positional_encoding.py tools/percent_to_ipynb.py
	$(PYTHON) tools/percent_to_ipynb.py $< $@

check:
	$(PYTHON) -m py_compile positional_encoding.py tools/percent_to_ipynb.py
