all: paper

clean:
	rm -rf auto
	rm -rf *.aux
	rm -rf *.bbl
	rm -rf *.blg
	rm -rf *.eps
	rm -rf *.log
	rm -rf *.out
	# rm -rf *.pdf
	rm -rf *.pdf_tex
	rm -rf *.toc

paper: paper.pdf

TEX_FILES = \
	paper.tex \
	abstract.tex \
	appendix.tex \
	complexity.tex \
	error-analysis.tex \
	fmm.tex \
	future-work.tex \
	introduction.tex \
	nufft.tex \
	numerical-results.tex \
	periodic-summation.tex

paper.pdf: paper.tex 
	pdflatex paper.tex
	# bibtex paper
	# pdflatex paper.tex
	# pdflatex paper.tex
