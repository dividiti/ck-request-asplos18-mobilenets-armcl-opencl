#!/bin/bash
source ./_clean.sh

echo "Compiling article ..."

pdflatex paper

bibtex paper

pdflatex paper
pdflatex paper

echo "Launching..."
evince paper.pdf
