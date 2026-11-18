"""Monta o notebook a partir das fontes em `fonte/` (um arquivo .py por seção, formato percent do jupytext).

As fontes são a única cópia editável do notebook. O .ipynb é sempre gerado por este script.

Uso:
    python scripts/montar_notebook.py                       # gera sem saídas
    python scripts/montar_notebook.py --saidas notebook/MovieLens25M_KDD.ipynb
        # reaproveita as saídas das células de código cujo texto não mudou
    python scripts/montar_notebook.py --destino outro.ipynb
"""
import argparse
import sys
from pathlib import Path

import jupytext
import nbformat

RAIZ = Path(__file__).resolve().parents[1]
FONTE = RAIZ / "fonte"
PADRAO = RAIZ / "notebook" / "MovieLens25M_KDD.ipynb"


def ler_fontes():
    arquivos = sorted(FONTE.glob("[0-9][0-9]_*.py"))
    if not arquivos:
        sys.exit(f"Nenhuma fonte encontrada em {FONTE}")
    celulas = []
    for arquivo in arquivos:
        parte = jupytext.reads(arquivo.read_text(encoding="utf-8"), fmt="py:percent")
        celulas.extend(parte.cells)
    return arquivos, celulas


def montar(saidas=None):
    arquivos, celulas = ler_fontes()
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    anteriores = {}
    if saidas is not None:
        for c in nbformat.read(saidas, as_version=4).cells:
            if c.cell_type == "code":
                anteriores[c.source.strip()] = c
    sem_saida = []
    for i, c in enumerate(celulas):
        if c.cell_type == "markdown":
            nova = nbformat.v4.new_markdown_cell(c.source)
        else:
            nova = nbformat.v4.new_code_cell(c.source)
            antiga = anteriores.get(c.source.strip())
            if antiga is not None:
                nova.outputs = antiga.outputs
                nova.execution_count = antiga.execution_count
            elif saidas is not None:
                sem_saida.append(c.source.splitlines()[0][:70] if c.source else f"célula {i}")
        nova.id = f"c{i:03d}"
        nb.cells.append(nova)
    nbformat.validate(nb)
    return nb, arquivos, sem_saida


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saidas", type=Path, help="notebook executado de onde copiar as saídas das células inalteradas")
    p.add_argument("--destino", type=Path, default=PADRAO)
    a = p.parse_args()
    nb, arquivos, sem_saida = montar(a.saidas)
    a.destino.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, a.destino)
    codigo = sum(c.cell_type == "code" for c in nb.cells)
    print(f"{a.destino.relative_to(RAIZ) if a.destino.is_relative_to(RAIZ) else a.destino}: "
          f"{len(nb.cells)} células ({codigo} de código) a partir de {len(arquivos)} fontes")
    if sem_saida:
        print(f"{len(sem_saida)} células de código mudaram e estão sem saída (é preciso executar de novo):")
        for s in sem_saida:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
