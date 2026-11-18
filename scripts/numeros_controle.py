"""Extrai os números de controle de um notebook executado e, com dois notebooks, compara as execuções.

Uso:
    python scripts/numeros_controle.py notebook/MovieLens25M_KDD.ipynb
    python scripts/numeros_controle.py execucoes/mac.ipynb notebook/MovieLens25M_KDD.ipynb

Com dois notebooks, mostra os números de controle lado a lado e lista as células de código cuja saída de texto
mudou. Serve para conferir que a execução no Mac e a no Colab dão os mesmos resultados.
"""
import re
import sys

import nbformat

# (nome, expressão regular com um grupo para o valor)
CONTROLES = [
    ("avaliações", r"Células com nota: ([\d,]+)"),
    ("usuários", r"Matriz usuário × filme: ([\d,]+) usuários"),
    ("filmes avaliados", r"usuários × ([\d,]+) filmes avaliados"),
    ("média podada", r"Média podada \(33% por cauda; [\d,]+ mantidas\)\s+([\d.]+)"),
    ("SSE do V-Optimal", r"SSE total = ([\d.]+)"),
    ("η década × média por filme", r"Razão de correlação η=([\d.]+)"),
    ("Pearson média × positivas", r"nota_media × prop_positivas: Pearson=([\d.]+)"),
    ("acerto kNN da imputação (%)", r"kNN \(k=15\)\s+([\d.]+)"),
    ("filmes com gênero imputado", r"Filmes com gênero imputado: ([\d,]+)"),
    ("filmes que continuam Unknown", r"continuam como Unknown: ([\d,]+)"),
    ("nota idêntica após tratamento", r"Distribuição da nota idêntica antes e depois do tratamento: (\w+)"),
    ("variância retida PCA base (%)", r"Matriz candidata no espaço PCA: \(\d+, \d+\); variância retida = ([\d.]+)%"),
    ("usuários atípicos", r"Atípicos \(b e c ao mesmo tempo\): ([\d,]+) usuários"),
    ("Hopkins PCA (percentis 1–99)", r"PCA \(7\)\s+todos\s+percentis 1–99\s+\d+\s+([\d.]+)"),
    ("variância retida PCA revisado (%)", r"PCA do vetor revisado: d′ = \d+ \(Kaiser-Guttman, ([\d.]+)%"),
]


def textos_por_celula(caminho):
    nb = nbformat.read(caminho, as_version=4)
    saida = {}
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        partes = []
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                partes.append(o.text)
            elif o.output_type == "error":
                partes.append(f"ERRO {o.ename}: {o.evalue}")
            elif "text/plain" in o.get("data", {}) and "image/png" not in o.get("data", {}):
                partes.append(o.data["text/plain"])
        saida[c.source.splitlines()[0] if c.source else ""] = "".join(partes)
    return saida


def controles(textos):
    tudo = "\n".join(textos.values())
    valores = {}
    for nome, padrao in CONTROLES:
        m = re.search(padrao, tudo)
        valores[nome] = m.group(1) if m else "—"
    return valores


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    a = textos_por_celula(sys.argv[1])
    va = controles(a)
    if len(sys.argv) == 2:
        for nome, valor in va.items():
            print(f"{nome:38s} {valor}")
        return
    b = textos_por_celula(sys.argv[2])
    vb = controles(b)
    iguais = 0
    print(f"{'número de controle':38s} {'1º notebook':>16s} {'2º notebook':>16s}")
    for nome in va:
        marca = "" if va[nome] == vb[nome] else "  <-- diferente"
        iguais += va[nome] == vb[nome]
        print(f"{nome:38s} {va[nome]:>16s} {vb[nome]:>16s}{marca}")
    print(f"\n{iguais} de {len(va)} números de controle iguais")
    comuns = [k for k in a if k in b]
    mudaram = [k for k in comuns if a[k] != b[k]]
    print(f"Células de código em comum: {len(comuns)}; com saída de texto diferente: {len(mudaram)}")
    for k in mudaram:
        print(f"  - {k[:80]}")
    so_um = [k for k in a if k not in b] + [k for k in b if k not in a]
    if so_um:
        print(f"Células presentes em só um dos notebooks: {len(so_um)}")


if __name__ == "__main__":
    main()
