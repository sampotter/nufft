STEM = proposal

all:
	pdflatex $(STEM).tex
	bibtex $(STEM)
	pdflatex $(STEM).tex
	pdflatex $(STEM).tex

clean:
	rm -rf *.aux
	rm -rf *.bbl
	rm -rf *.blg
	rm -rf *.log
	rm -rf *.pdf
