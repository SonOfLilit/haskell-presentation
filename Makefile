.SUFFIXES: .lhs .html

PANDOC-S5 = pandoc -w s5 --standalone

all: Presentation.html

.lhs.html:
	$(PANDOC-S5) $< > $@
