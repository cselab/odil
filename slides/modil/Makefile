PANDOC = pandoc
REVEALJS = ../
FILTER = ../filter.lua
TEMPLATE = ../template.revealjs

default: index.html

index.html: index.md $(FILTER) $(TEMPLATE)
	$(PANDOC) \
		-t revealjs \
		-f markdown-auto_identifiers \
		--katex \
		--wrap none \
		-V revealjs-url="$(REVEALJS)" \
		--lua-filter $(FILTER) \
		--template $(TEMPLATE) \
		"$<" -o "$@" 

clean: mostlyclean
	$(RM) index.html

.SUFFIXES:
.PHONY: default
