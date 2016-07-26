# TODO: add targets for generating eps files
# TODO: add target for generating writeup

M_FILE_DIR = ../src/m

all: writeup

clean:
	rm -rf auto
	rm -rf *.aux
	rm -rf *.bbl
	rm -rf *.blg
	rm -rf *.eps
	rm -rf *.log
	rm -rf *.out
	rm -rf *.pdf
	rm -rf *.pdf_tex
	rm -rf *.toc

writeup: plots writeup.pdf

plots:
	$(MAKE) -C $(M_FILE_DIR)
	cp $(M_FILE_DIR)/*.eps .

writeup.pdf: writeup.tex
	pdflatex writeup.tex
	bibtex writeup
	pdflatex writeup.tex
	pdflatex writeup.tex
