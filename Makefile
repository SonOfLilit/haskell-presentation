.SUFFIXES: .lhs .html

PANDOC-S5 = pandoc -f markdown+lhs -t s5 --standalone

all: Presentation.html

.lhs.html:
	$(PANDOC-S5) $< > $@
