"""Acerto da imputação de gênero por faixa de avaliadores (decisão D2).

Uso: python analises/imputacao_por_faixa.py

Nos filmes com gênero conhecido, o gênero é escondido e previsto. Acerto = gênero previsto está entre os
verdadeiros. Validação cruzada de 5 partes sobre os filmes com 10 ou mais avaliadores (ordem fixa por movieId,
semente 42). Cada parte treina num modelo com 4/5 desses filmes e testa: (a) na parte que ficou de fora, para as
faixas de 10 em diante; (b) em todos os filmes com menos de 10 avaliadores, que nunca entram no treino. Assim todas
as faixas têm média e desvio-padrão entre as 5 partes.

Métodos: chutar Drama (moda), contagem com peso 1/N^α (α = 0; 0,5; 1; N = filmes do gênero no catálogo de treino)
e kNN (k = 15) sobre o lift padronizado do público.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from comum import FAIXAS, atributos_knn, carregar_movielens, na_faixa, prever_knn, publico_por_genero, rotulo_faixa

movies, R, Y, generos = carregar_movielens()
S, n = publico_por_genero(R, Y)
conhecido = (Y.sum(1) > 0) & (S.sum(1) > 0)
X = atributos_knn(S, conhecido)
drama = generos.index("Drama")
base = np.where(conhecido & (n >= 10))[0]  # já em ordem de movieId (ordem do arquivo)
pequenos = np.where(conhecido & (n < 10))[0]

metodos = ["chutar Drama", "1/N^0 (sem peso)", "1/√N", "1/N", "kNN"]
acertos = {(f, m): [] for f in FAIXAS for m in metodos}
escolha_1n = np.zeros(len(generos))
for treino, teste in KFold(5, shuffle=True, random_state=42).split(base):
    tr, te = base[treino], base[teste]
    alvo = np.concatenate([te, pequenos])
    N = Y[tr].sum(0)
    previsoes = {"chutar Drama": np.full(len(alvo), drama), "kNN": prever_knn(X[tr], Y[tr], X[alvo])[0]}
    for alfa, nome in [(0, "1/N^0 (sem peso)"), (0.5, "1/√N"), (1, "1/N")]:
        previsoes[nome] = (S[alvo] / N ** alfa).argmax(1)
    escolha_1n += np.bincount(previsoes["1/N"], minlength=len(generos))
    for a, b in FAIXAS:
        sel = na_faixa(n[alvo], a, b)
        for m in metodos:
            acertos[((a, b), m)].append(100 * Y[alvo[sel], previsoes[m][sel]].mean())

linhas = []
for a, b in FAIXAS:
    sel = conhecido & na_faixa(n, a, b)
    linha = {"avaliadores": rotulo_faixa(a, b), "filmes com gênero": int(sel.sum())}
    for m in metodos:
        v = np.array(acertos[((a, b), m)])
        linha[m] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
    v_knn, v_moda = np.array(acertos[((a, b), "kNN")]), np.array(acertos[((a, b), "chutar Drama")])
    linha["kNN − chute (p.p.)"] = round((v_knn - v_moda).mean(), 1)
    linhas.append(linha)
tabela = pd.DataFrame(linhas)
pd.set_option("display.width", 220)
print("Acerto (%) por faixa de avaliadores: média ± desvio-padrão entre as 5 partes")
print(tabela.to_string(index=False))

passa = [(a, b) for (a, b), r in zip(FAIXAS, linhas) if r["kNN − chute (p.p.)"] >= 10]
minimo = None
for i, (a, b) in enumerate(FAIXAS):  # menor faixa a partir da qual todas as superiores também passam
    if all(f in passa for f in FAIXAS[i:]):
        minimo = a
        break
print(f"\nCritério: preencher as faixas em que o kNN supera o chute por pelo menos 10 p.p., contíguas até a maior.")
print(f"Mínimo de avaliadores pelo critério: {minimo}")
sem = (Y.sum(1) == 0)
for a, b in FAIXAS:
    sel = sem & na_faixa(n, a, b)
    print(f"  filmes sem gênero com {rotulo_faixa(a, b)} avaliadores: {sel.sum():,} ({int(n[sel].sum()):,} avaliações)")
print(f"  filmes sem gênero sem nenhuma avaliação: {(sem & (n == 0)).sum():,}")

print("\nGênero escolhido pela contagem com peso 1/N (soma das 5 partes):")
ordem = np.argsort(-escolha_1n)[:5]
print("  " + "; ".join(f"{generos[g]} {100 * escolha_1n[g] / escolha_1n.sum():.1f}%" for g in ordem))
print(f"  filmes por gênero no catálogo: " + "; ".join(f"{generos[g]} {int(Y[:, g].sum())}" for g in ordem))
