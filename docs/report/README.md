# Peptide Self-Assembly Evidence Extraction — Report Draft

This directory contains a self-contained LaTeX report draft synthesized from the project documentation (`README.md`, `AGENTS.md`, `docs/project_notes/`, `docs/personal_notes/`, and `docs/seminar_notes/`).

## Opening in ShareLaTeX / Overleaf

Because all `.tex` section files, `references.bib`, and `figures/` are contained directly within `docs/report/` using relative paths, you can open this project in ShareLaTeX or Overleaf in two ways:

1. **Upload as a ZIP archive**:
   ```bash
   cd docs/report
   zip -r report.zip main.tex references.bib sections/ figures/
   ```
   Upload `report.zip` in ShareLaTeX/Overleaf (**New Project -> Upload Project**) and set `main.tex` as the main document.
2. **GitHub Sync**: Connect the repository in Overleaf/ShareLaTeX and set the root/main document to `docs/report/main.tex` (or import `docs/report/`).

The template uses standard `pdflatex` + `bibtex` and only common TeX Live packages (`tikz`, `booktabs`, `longtable`, `listings`, `natbib`, `hyperref`, `subcaption`).

## Compiling Locally

```bash
cd docs/report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
