# TODO: add targets for generating eps files
# TODO: add target for generating paper

M_FILE_DIR = ../src/m

all: paper

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

paper: plots paper.pdf

plots:
	$(MAKE) -C $(M_FILE_DIR)
	cp $(M_FILE_DIR)/*.eps .

paper.pdf: paper.tex
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex
