TARGET=paper

all: 
	pdflatex -file-line-error -shell-escape $(TARGET)

bib: $(TARGET).tex
	-for i in 1 2 3; do \
		bibtex $(TARGET); \
		pdflatex -interaction nonstopmode $(TARGET) > /dev/null; \
	done

clean:
	rm \
	    $(TARGET).aux \
	    $(TARGET).bbl \
	    $(TARGET).blg \
	    $(TARGET).log

%.pdf: %.eps
	epstopdf $<

spell:
	aspell --lang=en --mode=tex check $(TARGET).tex
