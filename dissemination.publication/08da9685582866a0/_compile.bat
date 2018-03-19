call _clean.bat

echo Preprocessing tex paper sources via CK: paper.tex

call ck preprocess # --doc=paper.tex
if %errorlevel% neq 0 (
  echo.
  echo Error: CK preprocessing failed!
  exit /b 1
)

echo Compiling paper ...

pdflatex paper

bibtex paper

pdflatex paper
pdflatex paper

start paper.pdf
