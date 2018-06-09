call _clean.bat

rem echo Preprocessing tex paper sources via CK: paper.tex
rem 
rem call ck preprocess # --doc=paper.tex
rem if %errorlevel% neq 0 (
rem  echo.
rem  echo Error: CK preprocessing failed!
rem  exit /b 1
rem )

echo Compiling paper ...

pdflatex paper

bibtex paper

pdflatex paper
pdflatex paper

start paper.pdf
