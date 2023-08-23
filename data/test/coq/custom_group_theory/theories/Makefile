# all:
# 	coqc -R . algb grpdefn.v
# 	coqc -R . algb grpthm.v
# 	coqc -R . algb Morph.v
# 	coqc -R . algb Morph_cmd.v
# 	coqc -R . algb grpthm_cmd.v

all: Makefile.coq
	+make -f Makefile.coq all

clean: Makefile.coq
	+make -f Makefile.coq clean
	rm -f Makefile.coq

Makefile.coq: Make
	$(COQBIN)coq_makefile -f _CoqProject -o Makefile.coq

Make: ;

%: Makefile.coq
	+make -f Makefile.coq $@

.PHONY: all clean