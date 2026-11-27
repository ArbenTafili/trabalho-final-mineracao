"""Preencher o gênero dos filmes sem gênero mantendo a distribuição? (pedido do aluno, 26/11/2026; D8)

Uso: python analises/preencher_genero_distribuicao.py

O aluno perguntou se uma estratégia como a contagem 1/√N, mesmo errando filme a filme, não manteria a distribuição
dos gêneros. Teste: nas populações parecidas com os 240 filmes que ficam sem gênero (curtas do IMDb e títulos que no
IMDb só têm gêneros fora do MovieLens), o gênero conhecido é escondido e preenchido por cada estratégia. Mede-se:
- a distância entre a distribuição de gêneros verdadeira desses filmes (pelas avaliações) e a do preenchimento
  (distância de variação total, em pontos percentuais);
- o erro no perfil de cada usuário afetado: diferença entre as proporções fracionárias de gênero calculadas com os
  gêneros verdadeiros e com os preenchidos (ou deixando esses filmes sem gênero, fora das proporções), em p.p.
"Acerto por filme" das versões fracionárias é sempre perto de 100%, porque todo gênero recebe algum peso; não
serve para compará-las.
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import sparse
from comum import carregar_movielens, publico_por_genero, RAIZ
movies, R, Y, generos = carregar_movielens()
keep = [i for i, g in enumerate(generos) if g != "IMAX"]
generos = [generos[i] for i in keep]; Y = Y[:, keep]
S, n = publico_por_genero(R, Y)
rec = pd.read_csv(RAIZ / "dados/imdb/imdb_recorte_movielens.tsv.gz", sep="\t", dtype=str, na_values="\\N", keep_default_na=False, quoting=3)
movies["tconst"] = movies["imdbId"].map(lambda i: f"tt{int(i):07d}")
tipo = movies.merge(rec[["tconst", "titleType", "genres"]].rename(columns={"genres": "g_imdb"}), on="tconst", how="left")
assert len(tipo) == len(movies)
CONV = set(generos) | {"Family"}
so_fora = tipo["g_imdb"].map(lambda s: pd.notna(s) and not any(g in CONV for g in s.split(","))).to_numpy()
conhecido = Y.sum(1) > 0
Yfrac = np.divide(Y, Y.sum(1, keepdims=True), out=np.zeros_like(Y), where=Y.sum(1, keepdims=True) > 0)
Rt = R.T.tocsr()  # usuário × filme

def metodos(alvo, treino):
    N = Y[treino].sum(0)
    peso = S[alvo] / np.sqrt(np.maximum(N, 1))
    um = lambda idx: np.eye(len(generos))[idx]
    moda = Y[alvo].sum(0).argmax()  # gênero mais comum da população (referência otimista: usa o gabarito)
    prior = Yfrac[treino].sum(0) / Yfrac[treino].sum()
    norm = lambda M: np.divide(M, M.sum(1, keepdims=True), out=np.full_like(M, 1 / M.shape[1]), where=M.sum(1, keepdims=True) > 0)
    return {
        "deixar sem gênero (marcar)": np.zeros((len(alvo), len(generos))),
        "1/√N, só o gênero mais votado": um(peso.argmax(1)),
        "1/√N fracionário (todos os gêneros, pelo peso)": norm(peso),
        "público sem peso, fracionário": norm(S[alvo].astype(float)),
        "gênero mais comum da população": um(np.full(len(alvo), moda)),
        "distribuição do catálogo (todos iguais)": np.tile(prior, (len(alvo), 1)),
    }

for nome, pop in [("curtas do IMDb", conhecido & (tipo["titleType"] == "short").to_numpy() & (n > 0)),
                  ("títulos só com gêneros fora do MovieLens", conhecido & so_fora & (n > 0))]:
    alvo = np.where(pop)[0]; treino = np.where(conhecido & ~pop)[0]
    verdade = Yfrac[alvo]
    T = (n[alvo, None] * verdade).sum(0); T /= T.sum()
    usuarios = np.unique(R[alvo].nonzero()[1])
    base_true = Rt @ Yfrac  # perfil fracionário com os gêneros verdadeiros
    print(f"\n=== {nome}: {len(alvo):,} filmes, {int(n[alvo].sum()):,} avaliações, {len(usuarios):,} usuários afetados ===")
    linhas = []
    for m, M in metodos(alvo, treino).items():
        acerto = 100 * (verdade * (M > 0)).sum(1).astype(bool)[M.sum(1) > 0].mean() if (M.sum(1) > 0).any() else np.nan
        Mix = (n[alvo, None] * M).sum(0)
        tv = 0.5 * np.abs(T - Mix / Mix.sum()).sum() if Mix.sum() > 0 else np.nan
        Yalt = Yfrac.copy(); Yalt[alvo] = M
        Ualt = Rt @ Yalt
        A = np.asarray(base_true[usuarios]); B = np.asarray(Ualt[usuarios])
        A = A / A.sum(1, keepdims=True); B = B / np.maximum(B.sum(1, keepdims=True), 1e-12)
        dist = 0.5 * np.abs(A - B).sum(1)
        linhas.append((m, acerto, 100 * tv, 100 * dist.mean(), 100 * np.percentile(dist, 99), 100 * dist.max()))
    print(pd.DataFrame(linhas, columns=["método", "acerto por filme %", "distância na distribuição geral (p.p.)",
                                        "erro médio no perfil do usuário (p.p.)", "p99 (p.p.)", "máximo (p.p.)"]).round(3).to_string(index=False))
    print("Distribuição verdadeira (5 maiores):", dict(sorted(zip(generos, (100 * T).round(1)), key=lambda x: -x[1])[:5]))
