
PYTHON?=python

CHECK_PY_FILES := \
	analysis_utils.py \
	coq_export_utils.py \
	training_utils.py \
	#

.PHONY: check
check::

define add_target
$(1)::
	$(2)
endef

$(foreach f,$(CHECK_PY_FILES),$(eval $(call add_target,check,$(PYTHON) $(f))))

ANALYZE_PY_FILES := \
	train_max_of_2.py \
	proof_max2_01_exhaustive.py \
	#

.PHONY: analyze
analyze::

$(foreach f,$(ANALYZE_PY_FILES),$(eval $(call add_target,analyze,$(PYTHON) $(f) --fail-if-cant-load)))

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
