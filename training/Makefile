
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
	$(PYTHON) -m pip list --format=freeze | grep -v $(BROKEN_PACKAGES_GREP_PATTERN) > requirements.txt

ENV_DIR=tmp-env

.PHONY: save-env
save-env:
	conda env export > environment.yml
	# pip can't find mkl packages for some reason?
	$(PYTHON) -m pip list --format=freeze | grep -v $(BROKEN_PACKAGES_GREP_PATTERN) > requirements.txt
	rm -rf "$(ENV_DIR)"
	$(PYTHON) -m venv "$(ENV_DIR)"
	. "$(ENV_DIR)/bin/activate" && python -m pip install -r requirements.txt
	. "$(ENV_DIR)/bin/activate" && python -m pip list --format=freeze > requirements.txt
	rm -rf "$(ENV_DIR)"

.PHONY: git-lint
git-lint:
	git ls-files "*.py" | xargs flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
