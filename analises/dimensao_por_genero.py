"""Dimensionalidade intrínseca e d′ do PCA por público de gênero (estudo pedido pelo aluno, 26/11/2026; prévia).

Uso: python analises/dimensao_por_genero.py

Pergunta: "para este gênero, quantos componentes são necessários?". O público de um gênero são os 10% de usuários
com maior proporção fracionária naquele grupo (13 públicos de cerca de 16 mil usuários; um usuário pode estar em
mais de um). Espaço: vetor revisado aproximado (7 atributos derivados, 12 grupos mantidos e `raros`), montado em
pandas sem as fusões da seção 2, com min-max global (a mesma régua para todos os públicos); o PCA é centralizado e
tem a covariância calculada dentro de cada público. Em cada público: 10 amostras de 2.000 usuários, com e sem
pareamento por atividade (400 por quintil global de log(avaliações)); D2 pela contagem de pares, MLE (k de 10 a 20),
IQR da LID com k = 20, d′ por broken-stick, Kaiser em escala, 80% e ⌈D2⌉ + 1, e a fração do 1º componente.
Linhas de base: usuários sorteados entre todos e "placebos" (os 10% mais altos em nota média e em atividade).
Regra fixada antes: um gênero só "precisa de outro número de componentes" se a faixa do seu d′ não se sobrepõe à
dos usuários sorteados e a diferença passa da que o placebo produz.
Refazer quando o vetor revisado da proposta 4 (D7) estiver pronto.
"""
import math, time
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from comum import ML, SEM_GENERO

N, B, SEED = 2000, 10, 42
GRUPO = {"Animation": "Animation/Children", "Children": "Animation/Children", "IMAX": None}
RAROS = {"War", "Musical", "Documentary", "Western", "Film-Noir"}
t0 = time.time()
f = pd.read_csv(ML / "movies.csv"); f = f[f.genres != SEM_GENERO].copy()
f["ano"] = f.title.str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)
fg = f.assign(g=f.genres.str.split("|")).explode("g")[["movieId","g"]]
fg["g"] = fg.g.map(lambda x: GRUPO.get(x, x)); fg = fg.dropna().drop_duplicates()
fg["k"] = fg.groupby("movieId")["g"].transform("size")
g17 = sorted(fg.g.unique()); gi = {g:i for i,g in enumerate(g17)}
r = pd.read_csv(ML / "ratings.csv", usecols=["userId","movieId","rating"], dtype={"userId":"int32","movieId":"int32","rating":"float32"})
r = r[r.movieId.isin(set(f.movieId))]
fm = r.groupby("movieId")["rating"].agg(media_f="mean", n_f="size")
fm["log_n"] = np.log10(fm.n_f); fm = fm.join(f.set_index("movieId")["ano"])
rr = r.join(fm, on="movieId")
rr["vies"] = rr.rating - rr.media_f
u = rr.groupby("userId").agg(nota_media=("rating","mean"), nota_dp=("rating","std"), n=("rating","size"),
                              vies_usuario=("vies","mean"), popularidade_media=("log_n","mean"), ano_medio=("ano","mean"))
u["ano_medio"] = u.ano_medio.fillna(u.ano_medio.median()); u["atividade_log"] = np.log10(u.n)
w = r[["userId","movieId"]].merge(fg[["movieId","g","k"]], on="movieId")
w["w"] = 1.0 / w.k
F = w.groupby(["userId","g"])["w"].sum().unstack(fill_value=0.0).reindex(u.index, fill_value=0.0)
F = F.div(F.sum(axis=1), axis=0)
p = F.to_numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    u["diversidade"] = -np.where(p > 0, p*np.log(p), 0).sum(1) / np.log(len(g17))
mantidos = [g for g in g17 if g not in RAROS]
frac = F[mantidos].add_prefix("frac_"); frac["frac_raros"] = F[sorted(RAROS)].sum(axis=1)
X = pd.concat([u[["nota_media","nota_dp","vies_usuario","atividade_log","popularidade_media","ano_medio","diversidade"]], frac], axis=1).fillna(0.0)
mm = ((X - X.min()) / (X.max() - X.min())).to_numpy()
cols = list(X.columns); d = mm.shape[1]
dom = frac.idxmax(axis=1).str.replace("frac_", "")
print(f"Vetor revisado aproximado: {len(X):,} usuários × {d} atributos; montado em {time.time()-t0:.0f} s")
print("Gênero dominante (% dos usuários):", (100*dom.value_counts(normalize=True)).round(2).to_dict())

def estimar(Z):
    dist = pdist(Z); dist = dist[dist > 0]
    raios = np.quantile(dist, np.logspace(-3, -0.7, 25)); C = np.array([(dist <= x).mean() for x in raios])
    d2 = float(np.polyfit(np.log(raios), np.log(C), 1)[0])
    viz = np.maximum(NearestNeighbors(n_neighbors=21).fit(Z).kneighbors(Z)[0][:, 1:], 1e-12)
    mle = float(np.mean([np.mean((k-1) / np.maximum(np.log(viz[:, [k-1]] / viz[:, :k-1]).sum(1), 1e-12)) for k in range(10, 21)]))
    lid20 = 19 / np.maximum(np.log(viz[:, [19]] / viz[:, :19]).sum(1), 1e-12)
    c = Z - Z.mean(0); lam = np.sort(np.linalg.eigvalsh(np.cov(c, rowvar=False)))[::-1]; esc = lam / lam.sum()
    bs = np.array([sum(1/j for j in range(i, d+1))/d for i in range(1, d+1)]); acima = esc > bs
    return {"D2": d2, "MLE": mle, "LID20_IQR": float(np.subtract(*np.percentile(lid20, [75, 25]))),
            "broken": int(np.where(acima)[0].max()+1) if acima.any() else 0, "kaiser": int((lam/lam.mean() >= 1).sum()),
            "p80": int(np.searchsorted(np.cumsum(esc), .8)+1), "D2+1": math.ceil(d2)+1, "var_CP1": 100*esc[0]}

rng = np.random.default_rng(SEED)
idx_all = np.arange(len(X))
q_ativ = pd.qcut(u.atividade_log, 5, labels=False).to_numpy()
def amostras(membros, pareado=False):
    out = []
    for _ in range(B):
        if pareado:
            por = [membros[q_ativ[membros] == q] for q in range(5)]
            m = min(len(x) for x in por)
            if m < N // 5: return None, m
            s = np.concatenate([rng.choice(x, N//5, replace=False) for x in por])
        else:
            s = rng.choice(membros, N, replace=False)
        out.append(estimar(mm[s]))
    return pd.DataFrame(out), None

publicos = {"aleatório (todos)": idx_all}
top = 0.10
for g in mantidos + ["raros"]:
    v = frac["frac_" + g].to_numpy(); publicos[f"top10% {g}"] = idx_all[v >= np.quantile(v, 1 - top)]
for a in ["nota_media", "atividade_log"]:
    v = X[a].to_numpy(); publicos[f"placebo top10% {a}"] = idx_all[v >= np.quantile(v, 1 - top)]
for g, cnt in dom.value_counts().items():
    if cnt >= 5 * N: publicos[f"dominante {g}"] = idx_all[(dom == g).to_numpy()]

linhas = []
for nome, membros in publicos.items():
    res, _ = amostras(membros)
    resp, mfalta = amostras(membros, pareado=True)
    for tipo, R in [("sem pareamento", res), ("pareado por atividade", resp)]:
        if R is None:
            linhas.append({"público": nome, "amostra": tipo, "n_público": len(membros), "obs": f"quintil com só {mfalta}"}); continue
        lin = {"público": nome, "amostra": tipo, "n_público": len(membros), "aval_mediana": int(np.median(u.n.to_numpy()[membros]))}
        for c in ["D2", "MLE", "LID20_IQR", "var_CP1"]:
            lin[c] = f"{R[c].median():.2f} [{R[c].min():.2f}–{R[c].max():.2f}]"
            lin[c + "_m"], lin[c + "_v"] = R[c].mean(), R[c].var()
        for c in ["broken", "kaiser", "p80", "D2+1"]:
            lin[c] = f"{int(R[c].median())} [{R[c].min()}–{R[c].max()}]"
            lin[c + "_m"], lin[c + "_v"] = R[c].mean(), R[c].var()
        linhas.append(lin)
    print(f"  {nome}: {time.time()-t0:.0f} s", flush=True)
T = pd.DataFrame(linhas)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 30)
mostrar = ["público","amostra","n_público","aval_mediana","D2","MLE","LID20_IQR","var_CP1","broken","kaiser","p80","D2+1"]
for tipo in ["sem pareamento", "pareado por atividade"]:
    print(f"\n=== {tipo}: n = {N} por amostra, {B} amostras; mediana [mín–máx] ===")
    print(T[T.amostra == tipo][[c for c in mostrar if c in T.columns]].fillna("").to_string(index=False))
    sub = T[(T.amostra == tipo) & T.público.str.startswith("top10%")].dropna(subset=["D2_m"])
    for c in ["D2", "MLE", "broken", "kaiser", "p80", "D2+1"]:
        entre, dentro = sub[c + "_m"].var(), sub[c + "_v"].mean()
        print(f"  {c}: amplitude das médias entre gêneros {sub[c+'_m'].max()-sub[c+'_m'].min():.2f}; "
              f"razão variância entre/dentro (por amostra) = {entre / dentro if dentro > 0 else float('inf'):.1f}")
