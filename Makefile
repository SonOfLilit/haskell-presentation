.SUFFIXES: .lhs .html

PANDOC-S5 = pandoc -f markdown+lhs --mathml -t s5 --standalone --offline

all: Presentation.html

.lhs.html:
	$(PANDOC-S5) $< > $@
