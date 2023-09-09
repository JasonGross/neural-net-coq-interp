
PYTHON?=python

CHECK_PY_FILES := \
	analysis_utils.py \
	coq_export_utils.py \
	training_utils.py \
	#

.PHONY: check
check:
	$(PYTHON) $(CHECK_PY_FILES)

.PHONY: save-env
save-env:
	conda list -e > requirements.txt
	conda env export > environment.yml
