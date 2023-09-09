
PYTHON?=python

CHECK_PY_FILES := \
	analysis_utils.py \
	coq_export_utils.py \
	training_utils.py \
	#

.PHONY: check
check:
	$(PYTHON) $(CHECK_PY_FILES)

BROKEN_PACKAGES_GREP_PATTERN:='^mkl'

.PHONY: save-env-quick
save-env-quick:
	conda env export > environment.yml
	# pip can't find mkl packages for some reason?
	python -m pip list --format=freeze | grep -v $(BROKEN_PACKAGES_GREP_PATTERN) > requirements.txt

.PHONY: save-env
save-env:
	conda env export > environment.yml
	# pip can't find mkl packages for some reason?
	python -m pip list --format=freeze | grep -v $(BROKEN_PACKAGES_GREP_PATTERN) > requirements.txt
	rm -rf env
	python -m venv env
	. env/bin/activate && python -m pip install -r requirements.txt
	. env/bin/activate && python -m pip list --format=freeze > requirements.txt

