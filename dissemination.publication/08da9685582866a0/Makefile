PAPER := paper

LATEX := latex
PDFLATEX := pdflatex
BIBTEX := bibtex

RM := /bin/rm -rf

FILES := \
	paper.tex \
	paper_abstract.tex \
	paper_appendix.tex

TABLES := \
	tables/accuracy_top1-50000.tex

# PDF.
pdf: ${PAPER}.pdf
${PAPER}.pdf: ${FILES} ${TABLES}
	${PDFLATEX} ${PAPER} && ${BIBTEX} ${PAPER} && ${PDFLATEX} ${PAPER} && ${PDFLATEX} ${PAPER}
xpdf: ${PAPER}.pdf
	@xpdf $^
evince: ${PAPER}.pdf
	@evince $^

clean:
	$(RM) $(addprefix ${PAPER}, .pdf .dvi .xcp .log .aux .blg .bbl .out)
	$(RM) *~* comment.cut
	$(RM) $(addprefix figures/, *-eps-converted-to.pdf )
