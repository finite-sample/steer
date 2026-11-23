PAPER = steer

.PHONY: all clean watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib working-paper.sty working-paper-apsr.bst
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(PAPER).tex

clean:
	latexmk -C
	rm -f $(PAPER).bbl $(PAPER).blg
