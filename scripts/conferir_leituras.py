"""Confere se os números citados nas células de texto aparecem nas saídas de um notebook executado.

Uso: python scripts/conferir_leituras.py execucoes/real.ipynb [--anterior notebook/MovieLens25M_KDD.ipynb]

Para cada célula de texto (leituras, tabelas, roteiro), extrai os números escritos (1.409; 68,58%; 0,922) e procura
cada um, normalizado, nas saídas de texto de todas as células de código do notebook executado. Lista os que não
aparecem. Com --anterior, marca os que apareciam nas saídas da execução anterior: esses quase certamente mudaram e
precisam ser atualizados no texto. Números de seções, anos e limiares fixos aparecem como falso alarme; a lista é
um guia para revisão, não uma regra.
"""
import argparse
import re
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from montar_notebook import montar  # noqa: E402

NUMERO = re.compile(r"(?<![\w.,])[−-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|(?<![\w.,])[−-]?\d+(?:,\d+)?")


def normalizar_texto(numero):
    """1.409 → 1409; 68,58 → 68.58; −0,2 → -0.2."""
    n = numero.replace("−", "-")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", n):
        n = n.replace(".", "")
    return n.replace(",", ".")


def saidas(caminho):
    nb = nbformat.read(caminho, as_version=4)
    partes = []
    for c in nb.cells:
        for o in c.get("outputs", []) if c.cell_type == "code" else []:
            if o.output_type == "stream":
                partes.append(o.text)
            elif "text/plain" in o.get("data", {}):
                partes.append(o.data["text/plain"])
    texto = "\n".join(partes).replace(",", "")
    return set(re.findall(r"-?\d+(?:\.\d+)?", texto))


def presente(numero, numeros_saida):
    """O texto arredonda: 68.58 casa com 68.584, 0.92 com 0.922 e 5453 com 5452.713."""
    if numero in numeros_saida:
        return True
    casas = len(numero.split(".")[1]) if "." in numero else 0
    alvo = float(numero)
    return any(abs(float(s) - alvo) <= 0.5 * 10 ** -casas + 1e-9 for s in numeros_saida)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("executado")
    p.add_argument("--anterior")
    a = p.parse_args()
    atuais = saidas(a.executado)
    anteriores = saidas(a.anterior) if a.anterior else set()
    nb, _, _ = montar()
    total = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown":
            continue
        faltam = []
        for bruto in NUMERO.findall(c.source):
            n = normalizar_texto(bruto)
            if len(n.replace("-", "").replace(".", "")) < 2 or presente(n, atuais):
                continue
            faltam.append(f"{bruto}{' (estava na anterior)' if a.anterior and presente(n, anteriores) else ''}")
        if faltam:
            total += len(faltam)
            print(f"[{i:3d}] {c.source.splitlines()[0][:70]}")
            print("      " + "; ".join(dict.fromkeys(faltam)))
    print(f"\n{total} números do texto não aparecem nas saídas de {a.executado}")


if __name__ == "__main__":
    main()
