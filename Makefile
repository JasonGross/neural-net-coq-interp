# KNOWNTARGETS will not be passed along to Makefile.coq
KNOWNTARGETS := Makefile.coq
# KNOWNFILES will not get implicit targets from the final rule, and so
# depending on them won't invoke the submake
# Warning: These files get declared as PHONY, so any targets depending
# on them always get rebuilt
KNOWNFILES   := Makefile _CoqProject

.DEFAULT_GOAL := invoke-coqmakefile

Makefile.coq: Makefile _CoqProject
	$(COQBIN)coq_makefile -f _CoqProject -o Makefile.coq

SRC_DIR := theories
MOD_NAME := NeuralNetInterp
SORT_COQPROJECT = sed 's,[^/]*/,~&,g' | env LC_COLLATE=C sort | sed 's,~,,g'
EXISTING_COQPROJECT_CONTENTS_SORTED:=$(shell cat _CoqProject 2>&1 | $(SORT_COQPROJECT))
WARNINGS_PLUS := +implicit-core-hint-db,+implicits-in-term,+non-reversible-notation,+deprecated-intros-until-0,+deprecated-focus,+unused-intro-pattern,+variable-collision,+unexpected-implicit-declaration,+omega-is-deprecated,+deprecated-instantiate-syntax,+non-recursive,+undeclared-scope,+deprecated-hint-rewrite-without-locality,+deprecated-hint-without-locality,+deprecated-instance-without-locality,+deprecated-typeclasses-transparency-without-locality
WARNINGS_MINUS := -ltac2-missing-notation-var
WARNINGS := $(WARNINGS_PLUS),$(WARNINGS_MINUS),unsupported-attributes
ifneq (,$(wildcard .git))
FINDER := git ls-files '*.v'
else
FINDER := find $(SRC_DIR) -type f -name '*.v' | grep -v '\#'
endif
COQPROJECT_CMD:=(printf -- '-R $(SRC_DIR) $(MOD_NAME)\n'; printf -- '-arg -w -arg $(WARNINGS)\n'; $(FINDER) | $(SORT_COQPROJECT))
NEW_COQPROJECT_CONTENTS_SORTED:=$(shell $(COQPROJECT_CMD) | $(SORT_COQPROJECT))

#This is not required, but ulimit -s unlimited
#OCAMLOPTFLAGS?=-linscan
#export OCAMLOPTFLAGS

ifneq ($(EXISTING_COQPROJECT_CONTENTS_SORTED),$(NEW_COQPROJECT_CONTENTS_SORTED))
.PHONY: _CoqProject
_CoqProject:
	$(COQPROJECT_CMD) > $@
endif

_CoqProject:

invoke-coqmakefile: Makefile.coq
	. etc/ensure_stack_limit.sh; \
	 $(MAKE) --no-print-directory -f Makefile.coq $(filter-out $(KNOWNTARGETS),$(MAKECMDGOALS))

.PHONY: invoke-coqmakefile $(KNOWNFILES)

####################################################################
##                      Your targets here                         ##
####################################################################

# This should be the last rule, to handle any targets not declared above
%: invoke-coqmakefile
	@true
