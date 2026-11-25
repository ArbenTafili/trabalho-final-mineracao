# %% [markdown]
# ## 3. Transformação
#
# A transformação prepara a representação que P1 usará:
#
# - **3.1** — construção do perfil por usuário: as avaliações são convertidas em proporções;
# - **3.2** — seleção de características;
# - **3.3** — normalização;
# - **3.4** — transformação de características por PCA e escolha de d′;
# - **3.5** — efeito da dimensionalidade sobre as distâncias e dimensionalidade intrínseca;
# - **3.6** — visualização do espaço transformado;
# - **3.7** — preparação para o agrupamento: valores atípicos e tendência de agrupamento;
# - **3.8** — vetor revisado: atributos derivados, proporções fracionárias e limite de gêneros;
# - **3.9** — solução indexada para buscas por similaridade;
# - **3.10** — resultado da transformação e entrada de P1.
#
# Todas as células usam os dados tratados da Seção 2.
#
# ### 3.1 Perfil por usuário

# %%
# [3.1] Perfil por usuário a partir dos dados pré-processados
# prop_avaliacoes_total = avaliações do usuário / avaliações tratadas;
# prop_positivas = avaliações do usuário com nota >= LIMIAR_POSITIVA / avaliações do usuário;
# prop_<gênero> = avaliações do usuário em filmes do gênero / avaliações do usuário (gêneros se sobrepõem).
LIMIAR_POSITIVA = 4.0
usuarios_base_p1 = (ratings_clean.groupBy("userId").agg(
        F.count("*").alias("n_avaliacoes"),
        F.avg("rating").alias("nota_media"),
        F.stddev("rating").alias("nota_dp"),
        F.avg(F.when(F.col("rating") >= LIMIAR_POSITIVA, 1.0).otherwise(0.0)).alias("prop_positivas"))
    .withColumn("prop_avaliacoes_total", F.col("n_avaliacoes") / F.lit(linhas_ratings)))
lista_generos = [r["genre"] for r in movie_genres.select("genre").distinct().orderBy("genre").collect()]
sufixo = {g: g.lower().replace("-", "_") for g in lista_generos}
contagens_genero = ratings_genres.groupBy("userId").agg(*[
    F.sum(F.when(F.col("genre") == g, 1).otherwise(0)).alias("n_" + sufixo[g]) for g in lista_generos])
perfil_completo = usuarios_base_p1.join(contagens_genero, "userId", "left").fillna(0)
for g in lista_generos:
    perfil_completo = perfil_completo.withColumn("prop_" + sufixo[g], F.col("n_" + sufixo[g]) / F.col("n_avaliacoes"))
cols_prop = ["prop_" + sufixo[g] for g in lista_generos]
atributos_estruturados = ["prop_avaliacoes_total", "nota_media", "nota_dp", "prop_positivas"] + cols_prop

# Ordem fixa por userId: as amostras com semente (3.5 a 3.9) sorteiam os mesmos usuários em qualquer máquina.
perfil_pd = perfil_completo.select("userId", "n_avaliacoes", *atributos_estruturados).orderBy("userId").toPandas()
assert len(lista_generos) == 19 and "Unknown" not in lista_generos  # filmes sem gênero excluídos em 2.2-d
assert len(perfil_pd) == perfil_pd.userId.nunique() == n_usuarios_total
assert perfil_pd[atributos_estruturados].notna().all().all()
assert perfil_pd[cols_prop].ge(0).all().all() and perfil_pd[cols_prop].le(1).all().all()
assert abs(perfil_pd["prop_avaliacoes_total"].sum() - 1) < 1e-10
print(f"Perfil: {len(perfil_pd):,} usuários × {len(atributos_estruturados)} atributos "
      "(userId e n_avaliacoes ficam fora da matriz e servem para identificar e descrever grupos).")
print(perfil_pd[atributos_estruturados].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

# %% [markdown]
# **Leitura dos resultados.** Em média, 44,3% das avaliações de um usuário são de filmes de Drama, 34,8% de Comedy e 31,4% de Action. Gêneros como Documentary (0,8%) e Film-Noir (0,9%) têm proporções pequenas. As proporções de gênero de um usuário somam mais que 1 porque os gêneros se sobrepõem.

# %% [markdown]
# #### Histogramas dos atributos do perfil
#
# Todos os atributos do perfil aparecem como histograma, com a média e média ± 1 desvio-padrão marcadas. O número de avaliações e a participação no total usam eixo logarítmico, por causa da assimetria. A tabela registra média, desvio-padrão, mediana, assimetria e a porcentagem de usuários com valor zero.

# %%
# [3.1-b] Histogramas dos atributos do perfil, com média e desvio-padrão
atributos_histograma = ["n_avaliacoes"] + atributos_estruturados
fig, axes = plt.subplots(5, 5, figsize=(26, 22))
for ax, atributo in zip(axes.flat, atributos_histograma):
    escala_log = atributo in ("n_avaliacoes", "prop_avaliacoes_total")
    rotulo = "eixo log" if escala_log else ("proporção das avaliações do usuário" if atributo.startswith("prop_") else "nota")
    histograma_media_dp(ax, perfil_pd[atributo], atributo, bins=40, log_x=escala_log,
                        cor="#d18b24" if escala_log else "#4682a9", rotulo_x=rotulo)
for ax in axes.flat[len(atributos_histograma):]:
    ax.axis("off")
fig.suptitle(f"Atributos do perfil por usuário (n = {len(perfil_pd):,}); média e ± 1 DP em vermelho", y=1.005)
plt.tight_layout()
plt.show()
resumo_perfil = perfil_pd[atributos_histograma].agg(["mean", "std", "median", "skew"]).T
resumo_perfil["%_zeros"] = 100 * (perfil_pd[atributos_histograma] == 0).mean()
print(resumo_perfil.round(4).to_string())

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Volume:** o número de avaliações por usuário tem assimetria de 16,3 (média 153,8, mediana 71); a participação no total herda a mesma forma.
# - **Nota:** a nota média tem assimetria −0,56 (média 3,68); o DP da nota é zero para 0,15% dos usuários, que deram sempre a mesma nota.
# - **Gêneros comuns** (Drama, Comedy, Action, Thriller): distribuições próximas de simétricas, com menos de 0,4% de zeros.
# - **Gêneros raros:** a maioria dos usuários tem proporção zero em Documentary (62,8%), Film-Noir (58,3%), Western (32,3%), IMAX (26,8%) e Musical (22,4%). A cauda dos poucos usuários que avaliam esses gêneros explica os valores z extremos de 3.3 e os atípicos de 3.7.
#
# **Consequência:** os gêneros raros pedem tratamento próprio. O vetor revisado (3.8) os agrupa, e a atividade entra em escala logarítmica.

# %% [markdown]
# ### 3.2 Seleção de características
#
# O trabalho usa a abordagem de **filtro**, porque ela independe do método de aprendizado. *Wrappers* e *embedded* dependem de um algoritmo e acrescentariam um segundo viés.
#
# O CFS procura atributos correlacionados com a classe e pouco correlacionados entre si. Sem classe, só a segunda parte se aplica: os pares são ordenados pelo valor absoluto da correlação, em ordem decrescente. Fisher e ReliefF exigem classes. Depois de P1, eles podem usar os grupos encontrados como classe, mas para **descrever** quais atributos separam os grupos, não para escolher a entrada.
#
# **Critério registrado:** |Pearson| e |Spearman| ≥ 0,9 indicam redundância forte. O limiar é uma escolha explícita e pode ser revisto. Em cada par redundante, mantém-se o atributo que usa a escala completa da nota.

# %%
# [3.2] Seleção de características por filtro: redundância entre atributos
LIMIAR_REDUNDANCIA = 0.9
pearson_perfil = perfil_pd[atributos_estruturados].corr(method="pearson")
spearman_perfil = perfil_pd[atributos_estruturados].corr(method="spearman")
ranking = pd.DataFrame(
    [(a, b, pearson_perfil.loc[a, b], spearman_perfil.loc[a, b])
     for i, a in enumerate(atributos_estruturados) for b in atributos_estruturados[i + 1:]],
    columns=["atributo_1", "atributo_2", "pearson", "spearman"])
ranking["abs_pearson"] = ranking["pearson"].abs()
ranking = ranking.sort_values("abs_pearson", ascending=False).reset_index(drop=True)
print("Ranking dos pares por |Pearson| (10 primeiros):")
print(ranking.head(10).round(3).to_string(index=False))

redundantes = ranking[(ranking["pearson"].abs() >= LIMIAR_REDUNDANCIA) & (ranking["spearman"].abs() >= LIMIAR_REDUNDANCIA)]
print(f"Pares com |Pearson| e |Spearman| >= {LIMIAR_REDUNDANCIA}:")
print(redundantes.round(3).to_string(index=False) if len(redundantes) else "nenhum")

decisoes = {
    "prop_positivas": "removido por redundância com nota_media; depende de um limiar de nota ainda a validar",
    "prop_avaliacoes_total": "fora da distância: mede atividade, é muito assimétrico e dominaria a distância; descreve grupos",
}
pares_sem_decisao = redundantes[~redundantes["atributo_1"].isin(decisoes) & ~redundantes["atributo_2"].isin(decisoes)]
if len(pares_sem_decisao):
    print("ATENÇÃO: pares redundantes sem decisão registrada:")
    print(pares_sem_decisao.to_string(index=False))
atributos_p1 = [a for a in atributos_estruturados if a not in decisoes]
for atributo, motivo in decisoes.items():
    print(f"- {atributo}: {motivo}")
print(f"Entrada candidata de P1 ({len(atributos_p1)} atributos):", ", ".join(atributos_p1))

fig, ax = plt.subplots(figsize=(11, 9))
imagem = ax.imshow(pearson_perfil, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(atributos_estruturados)), atributos_estruturados, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(atributos_estruturados)), atributos_estruturados, fontsize=8)
ax.set_title("Pearson entre os atributos do perfil de usuário")
fig.colorbar(imagem, ax=ax, shrink=.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Normalização: padronização × re-escala
#
# A escala muda o peso de cada atributo na distância e no erro do PCA. Duas opções são comparadas:
#
# - **Padronização z-score:** subtrai a média e divide pelo desvio-padrão. Todos os atributos passam a ter a mesma variância, então gêneros raros com pouca variação são ampliados.
# - **Re-escala min-max:** leva cada atributo para [0, 1] e preserva diferenças de variância entre atributos.
#
# A coluna `peso_var` mostra a variância do atributo dividida pela soma das variâncias. Ela indica a participação média do atributo na distância euclidiana ao quadrado entre dois usuários.

# %%
# [3.3] Normalização: padronização z-score × re-escala min-max
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X_p1_original = perfil_pd[atributos_p1].astype(float)
assert X_p1_original.shape == (n_usuarios_total, 21)
assert np.isfinite(X_p1_original.to_numpy()).all()

scaler_p1 = StandardScaler()
X_p1_padronizado = pd.DataFrame(scaler_p1.fit_transform(X_p1_original), columns=atributos_p1, index=perfil_pd.index)
X_p1_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X_p1_original), columns=atributos_p1, index=perfil_pd.index)
assert np.allclose(X_p1_padronizado.mean(), 0, atol=1e-10)
assert np.allclose(X_p1_padronizado.std(ddof=0), 1, atol=1e-10)
assert X_p1_minmax.min().ge(-1e-12).all() and X_p1_minmax.max().le(1 + 1e-12).all()

comparacao_escalas = pd.DataFrame({
    "media_original": X_p1_original.mean(),
    "dp_original": X_p1_original.std(ddof=0),
    "z_min": X_p1_padronizado.min(),
    "z_max": X_p1_padronizado.max(),
    "peso_var_z_%": 100 * X_p1_padronizado.var(ddof=0) / X_p1_padronizado.var(ddof=0).sum(),
    "peso_var_minmax_%": 100 * X_p1_minmax.var(ddof=0) / X_p1_minmax.var(ddof=0).sum(),
}).round(4)
print("Matrizes candidatas:", X_p1_padronizado.shape, "(z-score) e", X_p1_minmax.shape, "(min-max)")
print(comparacao_escalas.to_string())
extremos = (X_p1_padronizado.abs() > 10).sum()
print("Usuários com |z| > 10 por atributo:", extremos[extremos > 0].to_dict())

# %% [markdown]
# ### 3.4 PCA e escolha de d′
#
# O PCA é aplicado à matriz padronizada, que já está centrada e em escala comparável. Nesse caso, a covariância equivale à matriz de correlação.
#
# Critérios para escolher d′:
#
# - **Scree plot** com variância explicada e acumulada; o limiar de 80% é um parâmetro registrado;
# - **Broken-stick:** retém os componentes iniciais cuja variância supera a esperada para autovalores aleatórios;
# - **Kaiser-Guttman:** retém autovalores ≥ 1 em dados padronizados.
#
# Cada critério tem seu próprio viés. A matriz PCA é uma alternativa a comparar em P1, não uma substituição obrigatória dos 21 atributos.

# %%
# [3.4] PCA sobre a matriz padronizada e critérios para d'
from sklearn.decomposition import PCA

pca_completo = PCA(svd_solver="full").fit(X_p1_padronizado)
autovalores = pca_completo.explained_variance_
p = len(autovalores)
prop_var = pca_completo.explained_variance_ratio_
acumulada = np.cumsum(prop_var)
broken_stick = np.array([sum(1 / i for i in range(k, p + 1)) / p for k in range(1, p + 1)])

LIMIAR_VARIANCIA = 0.80
d_kaiser = int((autovalores >= 1).sum())
acima_broken = prop_var > broken_stick
d_broken = p if acima_broken.all() else int(np.argmin(acima_broken))
d_variancia = int(np.searchsorted(acumulada, LIMIAR_VARIANCIA) + 1)

tabela_pca = pd.DataFrame({
    "componente": np.arange(1, p + 1), "autovalor": autovalores, "variancia_%": 100 * prop_var,
    "acumulada_%": 100 * acumulada, "broken_stick_%": 100 * broken_stick}).round(4)
print(tabela_pca.to_string(index=False))
print(f"d' por Kaiser-Guttman (autovalor >= 1): {d_kaiser}")
print(f"d' por broken-stick: {d_broken}")
print(f"d' para {LIMIAR_VARIANCIA:.0%} da variância acumulada: {d_variancia}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(range(1, p + 1), autovalores, "o-", color="#397461", label="autovalores")
axes[0].axhline(1, ls="--", color="gray", label="Kaiser-Guttman = 1")
axes[0].set(title="Scree plot", xlabel="Componente", ylabel="Autovalor")
axes[0].legend()
axes[1].bar(range(1, p + 1), 100 * prop_var, color="#397461", label="variância explicada")
axes[1].plot(range(1, p + 1), 100 * broken_stick, "o-", color="#d18b24", label="broken-stick")
axes[1].plot(range(1, p + 1), 100 * acumulada, "s--", color="#4682a9", label="acumulada")
axes[1].axhline(100 * LIMIAR_VARIANCIA, ls=":", color="gray")
axes[1].set(title="Variância explicada por componente (%)", xlabel="Componente", ylabel="%")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# Candidato registrado: Kaiser-Guttman, por ser o critério estatístico definido para dados padronizados.
d_pca = d_kaiser
assert 1 <= d_pca <= p
cargas = pd.DataFrame(pca_completo.components_[:d_pca].T, index=atributos_p1,
                      columns=[f"CP{k}" for k in range(1, d_pca + 1)])
print("Atributos com maior carga absoluta em cada componente retido:")
for cp in cargas.columns:
    principais = cargas[cp].abs().sort_values(ascending=False).head(4).index
    print(f"{cp}: " + ", ".join(f"{a} ({cargas.loc[a, cp]:+.2f})" for a in principais))
X_p1_pca = pd.DataFrame(pca_completo.transform(X_p1_padronizado)[:, :d_pca], columns=cargas.columns, index=perfil_pd.index)
print(f"Matriz candidata no espaço PCA: {X_p1_pca.shape}; variância retida = {100 * acumulada[d_pca - 1]:.2f}%")
# Alternativa para comparação nas seções seguintes: os componentes que somam 80% da variância.
X_p1_pca80 = pd.DataFrame(pca_completo.transform(X_p1_padronizado)[:, :d_variancia],
                          columns=[f"CP{k}" for k in range(1, d_variancia + 1)], index=perfil_pd.index)
print(f"Alternativa para comparação: PCA com {d_variancia} componentes ({100 * acumulada[d_variancia - 1]:.2f}% da variância)")

# %% [markdown]
# ### 3.5 Dimensionalidade e concentração de distâncias
#
# Em alta dimensionalidade, as distâncias tendem a se concentrar: a menor e a maior ficam proporcionalmente próximas, o que prejudica métodos baseados em distância. Como P1 usará distâncias entre usuários, a célula mede a distribuição de pares de distância em uma amostra fixa de 2.000 usuários. As medidas usadas são:
#
# - **Contraste relativo** = (máx − mín) / mín;
# - **Variância relativa** = desvio-padrão / média das distâncias.
#
# Valores menores indicam mais concentração. As medidas são comparadas nos três espaços candidatos.

# %%
# [3.5] Concentração de distâncias nos espaços candidatos
from scipy.spatial.distance import pdist

N_AMOSTRA_DIST = 2000
idx_dist = perfil_pd.sample(n=N_AMOSTRA_DIST, random_state=42).index
espacos = {
    f"z-score ({X_p1_padronizado.shape[1]} atributos)": X_p1_padronizado,
    f"min-max ({X_p1_minmax.shape[1]} atributos)": X_p1_minmax,
    f"PCA ({X_p1_pca.shape[1]} componentes)": X_p1_pca,
    f"PCA 80% ({X_p1_pca80.shape[1]} componentes)": X_p1_pca80,
}
linhas_dist, distancias = [], {}
for nome, X in espacos.items():
    d = pdist(X.loc[idx_dist].to_numpy(), metric="euclidean")
    d = d[d > 0]  # pares idênticos não entram no contraste relativo
    distancias[nome] = d
    linhas_dist.append((nome, len(d), d.mean(), d.std(), d.min(), d.max(),
                        (d.max() - d.min()) / d.min(), d.std() / d.mean()))
tabela_dist = pd.DataFrame(linhas_dist, columns=["espaço", "pares", "média", "DP", "mín", "máx",
                                                 "contraste_relativo", "variancia_relativa"])
print(f"Amostra fixa de {N_AMOSTRA_DIST:,} usuários (semente 42); distância euclidiana")
print(tabela_dist.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4.5))
for nome, d in distancias.items():
    ax.hist(d / d.mean(), bins=80, histtype="step", density=True, linewidth=1.4, label=nome)
ax.set(title="Distribuição de pares de distância (normalizada pela média)",
       xlabel="distância / média das distâncias", ylabel="densidade")
ax.legend()
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Dimensionalidade intrínseca
#
# A dimensionalidade intrínseca estima quantas dimensões efetivas descrevem a estrutura dos dados, independentemente do número de atributos. O estimador de máxima verossimilhança de Levina–Bickel usa as distâncias aos k vizinhos mais próximos (k = 10 a 20) em uma amostra fixa. Valores bem menores que 22 indicam atributos correlacionados e reforçam a redução por PCA.

# %%
# [3.5-b] Dimensionalidade intrínseca estimada por máxima verossimilhança (Levina–Bickel)
from sklearn.neighbors import NearestNeighbors

def dimensao_intrinseca_mle(X, k1=10, k2=20):
    distancias_viz, _ = NearestNeighbors(n_neighbors=k2 + 1).fit(X).kneighbors(X)
    distancias_viz = np.maximum(distancias_viz[:, 1:], 1e-12)  # descarta o próprio ponto
    estimativas = []
    for k in range(k1, k2 + 1):
        soma_logs = np.log(distancias_viz[:, [k - 1]] / distancias_viz[:, :k - 1]).sum(axis=1)
        estimativas.append(np.mean((k - 1) / np.maximum(soma_logs, 1e-12)))
    return float(np.mean(estimativas)), [float(e) for e in estimativas]

N_AMOSTRA_ID = 10_000
idx_id = perfil_pd.sample(n=N_AMOSTRA_ID, random_state=42).index
linhas_id = []
for nome, X in [(f"z-score ({X_p1_padronizado.shape[1]} atributos)", X_p1_padronizado),
                 (f"PCA ({X_p1_pca.shape[1]} componentes)", X_p1_pca),
                 (f"PCA 80% ({X_p1_pca80.shape[1]} componentes)", X_p1_pca80)]:
    media_id, por_k = dimensao_intrinseca_mle(X.loc[idx_id].to_numpy())
    linhas_id.append((nome, X.shape[1], media_id, min(por_k), max(por_k)))
tabela_id = pd.DataFrame(linhas_id, columns=["espaço", "dimensões", "ID_MLE (média k=10..20)", "mín", "máx"])
print(f"Amostra fixa de {N_AMOSTRA_ID:,} usuários (semente 42)")
print(tabela_id.round(2).to_string(index=False))

# %% [markdown]
# **Leitura dos resultados.** A dimensionalidade intrínseca estimada é de cerca de 13 no espaço padronizado de 21 atributos (12,7 a 14,1, conforme k) de cerca de 6 no espaço PCA de 6 componentes (6,4) e de cerca de 8 no PCA com 80% da variância, de 9 componentes (8,4). Os 21 atributos têm redundância, mas a estrutura dos dados não cabe em duas ou três dimensões.
#
# **Consequência:** o PCA com 6 componentes (66,84% da variância) é uma redução útil para as distâncias, mas descarta parte da estrutura. O critério de 80% da variância (9 componentes, 80,07%) fica mais próximo da dimensionalidade intrínseca. Ele entra como alternativa ao lado do PCA com 6 componentes nas comparações de 3.5 a 3.9 e é testado em P1. Na concentração de distâncias (3.5), ele separa perto de longe um pouco menos que o PCA com 6 componentes (variância relativa de 0,417 contra 0,426; contraste relativo de 92,57 contra 110,66) e ainda mais que os espaços de 21 atributos.

# %% [markdown]
# ### 3.6 Visualização do espaço transformado
#
# Projeções 2D de uma amostra fixa de usuários:
#
# - **PCA:** os dois primeiros componentes;
# - **MDS:** preserva as distâncias entre pares, calculado em 500 usuários por custo computacional;
# - **t-SNE:** preserva vizinhanças locais, aplicado sobre o espaço PCA;
# - **coordenadas paralelas:** os 21 atributos padronizados.
#
# Antes de P1 não há grupos. Por isso, a cor indica o grupo da nota média do usuário, só como referência visual.

# %%
# [3.6] Visualização do espaço transformado: PCA 2D, MDS, t-SNE e coordenadas paralelas
from sklearn.manifold import MDS, TSNE

idx_vis = perfil_pd.sample(n=3000, random_state=7).index
grupo_media_vis = np.ceil(perfil_pd.loc[idx_vis, "nota_media"]).clip(1, 5).astype(int).to_numpy()
pca_2d = X_p1_pca.loc[idx_vis, ["CP1", "CP2"]].to_numpy()
tsne_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X_p1_pca.loc[idx_vis].to_numpy())
tsne_2d_80 = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X_p1_pca80.loc[idx_vis].to_numpy())
idx_mds = idx_vis[:500]
mds_2d = MDS(n_components=2, n_init=1, max_iter=300, random_state=42).fit_transform(X_p1_padronizado.loc[idx_mds].to_numpy())

fig, axes = plt.subplots(1, 4, figsize=(26, 6))
projecoes = [
    (axes[0], pca_2d, grupo_media_vis, f"PCA: CP1 × CP2 ({len(idx_vis):,} usuários)"),
    (axes[1], mds_2d, grupo_media_vis[:len(idx_mds)], f"MDS: {X_p1_padronizado.shape[1]} atributos z ({len(idx_mds)} usuários)"),
    (axes[2], tsne_2d, grupo_media_vis, f"t-SNE: PCA com {X_p1_pca.shape[1]} componentes"),
    (axes[3], tsne_2d_80, grupo_media_vis, f"t-SNE: PCA 80% ({X_p1_pca80.shape[1]} componentes)"),
]
for ax, coords, cor, titulo in projecoes:
    pontos = ax.scatter(coords[:, 0], coords[:, 1], c=cor, cmap="viridis", s=6, alpha=.6, vmin=1, vmax=5)
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(pontos, ax=axes, shrink=.8, label="Grupo da nota média do usuário (1 a 5)")
plt.show()

# Coordenadas paralelas: 300 usuários, 21 atributos padronizados (limitados a [−3, 3] apenas no gráfico).
paralelas = X_p1_padronizado.loc[idx_vis[:300]].clip(-3, 3).copy()
paralelas["grupo_nota_media"] = grupo_media_vis[:300].astype(str)
fig, ax = plt.subplots(figsize=(20, 6))
pd.plotting.parallel_coordinates(paralelas.sort_values("grupo_nota_media"), "grupo_nota_media", ax=ax,
                                 colormap="viridis", alpha=.35, linewidth=.8)
ax.set_title("Coordenadas paralelas dos atributos padronizados (300 usuários; cor = grupo da nota média)")
ax.tick_params(axis="x", rotation=60)
ax.set_ylabel("valor z (limitado a [−3, 3])")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Projeções 2D:** PCA, MDS e t-SNE mostram uma nuvem contínua de usuários, sem grupos isolados. O t-SNE sobre o PCA com 9 componentes mostra o mesmo quadro: uma nuvem contínua, com poucos aglomerados pequenos na borda.
# - **Nota média:** a cor não organiza a estrutura principal. Usuários com médias diferentes aparecem misturados, com poucas regiões locais de médias mais baixas ou mais altas no t-SNE.
# - **Coordenadas paralelas:** a variação entre usuários está sobretudo nas proporções de gênero.
#
# **Consequência:** espera-se que um agrupamento produza regiões de um contínuo, definidas principalmente pela composição de gêneros, e não grupos naturalmente separados.

# %% [markdown]
# ### 3.7 Preparação para o agrupamento: valores atípicos e tendência de agrupamento
#
# Antes de P1, duas verificações no espaço transformado.
#
# **Valores atípicos.** O k-means e outros métodos baseados em médias são sensíveis a objetos muito afastados, que deslocam os centróides. Atípicos podem ser objetos legítimos, e não erros: aqui seriam usuários com gosto muito concentrado em poucos gêneros. Três critérios são calculados:
#
# - **(a) univariado:** |z| > 3 em cada atributo padronizado;
# - **(b) multivariado:** distância de Mahalanobis ao centro no espaço PCA, comparada ao quantil 0,999 da χ² com 7 graus de liberdade (referência sob normalidade multivariada);
# - **(c) vizinhança:** distância ao 10º vizinho mais próximo acima do percentil 99.
#
# Um usuário é marcado como atípico quando está longe do centro (b) **e** isolado dos vizinhos (c). Quem está longe do centro, mas tem vizinhos próximos, pode formar um grupo de nicho e não é marcado. Nenhum usuário é removido: a marcação `atipico` permite comparar P1 com e sem esses usuários.

# %%
# [3.7-a] Valores atípicos no espaço do perfil: |z| por atributo, Mahalanobis e distância ao 10º vizinho
from scipy import stats
from sklearn.neighbors import NearestNeighbors

n_usuarios = len(X_p1_padronizado)

# (a) Univariado: |z| > 3 em cada atributo padronizado (regra indicativa: vários atributos são assimétricos).
LIMIAR_Z = 3
acima_z = X_p1_padronizado.abs() > LIMIAR_Z
algum_z = acima_z.any(axis=1)
tabela_z = pd.DataFrame({
    "usuarios_|z|>3": acima_z.sum(),
    "%_usuarios": 100 * acima_z.mean(),
    "assimetria": X_p1_original.skew(),
}).sort_values("usuarios_|z|>3", ascending=False)
print(f"(a) |z| > {LIMIAR_Z} em pelo menos um dos {X_p1_padronizado.shape[1]} atributos: "
      f"{algum_z.sum():,} de {n_usuarios:,} usuários ({100 * algum_z.mean():.2f}%)")
print(tabela_z.head(10).round(3).to_string())

# (b) Multivariado: distância de Mahalanobis ao centro no espaço PCA (cada componente dividido pela sua variância).
X_pca_np = X_p1_pca.to_numpy()
centrado = X_pca_np - X_pca_np.mean(axis=0)
d2_mahal = (centrado ** 2 / centrado.var(axis=0)).sum(axis=1)
GL = X_pca_np.shape[1]
LIMIAR_D2 = stats.chi2.ppf(0.999, df=GL)
longe_centro = d2_mahal > LIMIAR_D2
print(f"\n(b) Mahalanobis² > χ²(0,999; {GL} gl) = {LIMIAR_D2:.2f}: {longe_centro.sum():,} usuários "
      f"({100 * longe_centro.mean():.2f}%); sob normalidade multivariada seriam esperados {0.001 * n_usuarios:,.0f} (0,10%)")

# (c) Vizinhança: distância ao 10º vizinho mais próximo no espaço PCA (a coluna 0 é o próprio usuário).
K_VIZ = 10
dist_viz, _ = NearestNeighbors(n_neighbors=K_VIZ + 1, n_jobs=-1).fit(X_pca_np).kneighbors(X_pca_np)
dist_k = dist_viz[:, K_VIZ]
LIMIAR_K = np.quantile(dist_k, 0.99)
isolado = dist_k > LIMIAR_K
print(f"\n(c) Distância ao {K_VIZ}º vizinho > percentil 99 ({LIMIAR_K:.3f}): {isolado.sum():,} usuários (1% por construção); "
      f"mediana da distância = {np.median(dist_k):.3f}")

# Marcação: atípico = longe do centro (b) e isolado dos vizinhos (c) ao mesmo tempo.
atipico = longe_centro & isolado
perfil_pd["atipico"] = atipico
print(f"\nAtípicos (b e c ao mesmo tempo): {atipico.sum():,} usuários ({100 * atipico.mean():.2f}%)")
print(f"Longe do centro, mas com vizinhos próximos (possível nicho, não marcado): {(longe_centro & ~isolado).sum():,}")
print(f"Isolados, mas perto do centro (não marcados): {(isolado & ~longe_centro).sum():,}")

atributo_dominante = X_p1_padronizado[atipico].abs().idxmax(axis=1).value_counts()
print("\nAtributo com maior |z| em cada usuário atípico (10 mais frequentes):")
print(atributo_dominante.head(10).to_string())
comparacao_atipicos = pd.DataFrame({
    "atípicos (mediana)": perfil_pd.loc[atipico, ["n_avaliacoes", "nota_media", "nota_dp"]].median(),
    "demais (mediana)": perfil_pd.loc[~atipico, ["n_avaliacoes", "nota_media", "nota_dp"]].median(),
})
print("\nAtípicos × demais usuários:")
print(comparacao_atipicos.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
axes[0].hist(np.log10(d2_mahal), bins=80, color="#4682a9")
axes[0].axvline(np.log10(LIMIAR_D2), color="crimson", ls="--", label=f"χ²(0,999; {GL} gl) = {LIMIAR_D2:.1f}")
axes[0].set(title="Mahalanobis² ao centro (espaço PCA)", xlabel="log10(D²)", ylabel="usuários")
axes[0].legend()
axes[1].hist(np.log10(np.maximum(dist_k, 1e-6)), bins=80, color="#4682a9")
axes[1].axvline(np.log10(LIMIAR_K), color="crimson", ls="--", label=f"percentil 99 = {LIMIAR_K:.2f}")
axes[1].set(title=f"Distância ao {K_VIZ}º vizinho mais próximo (espaço PCA)", xlabel="log10(distância)", ylabel="usuários")
axes[1].legend()
rng_atip = np.random.default_rng(42)
idx_demais = rng_atip.choice(np.flatnonzero(~atipico), size=5000, replace=False)
idx_atipicos = np.flatnonzero(atipico)
axes[2].scatter(X_pca_np[idx_demais, 0], X_pca_np[idx_demais, 1], s=4, alpha=.3, color="grey", label="5.000 demais (amostra)")
axes[2].scatter(X_pca_np[idx_atipicos, 0], X_pca_np[idx_atipicos, 1], s=7, alpha=.7, color="crimson",
                label=f"atípicos ({len(idx_atipicos):,})")
axes[2].set(title="Atípicos no plano CP1 × CP2", xlabel="CP1", ylabel="CP2")
axes[2].legend(markerscale=3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **(a) |z| > 3:** 31.263 usuários (19,23%) passam do limite em pelo menos um atributo. Nos dez atributos com mais casos, a fração vai de 1,30% a 2,07% (IMAX, Animation, Children, Sci-Fi e Horror lideram), bem acima dos 0,27% esperados numa normal. As proporções de gêneros raros são muito assimétricas: 12,0 em Documentary e 9,6 em Film-Noir. Com caudas tão longas, a regra univariada marca a cauda de cada atributo, não casos isolados, e não serve sozinha como critério.
# - **(b) Mahalanobis:** 3.563 usuários (2,19%) passam do quantil 0,999 da χ² com 6 graus de liberdade, cerca de 22 vezes os 163 esperados sob normalidade. A nuvem de usuários tem caudas mais pesadas que uma normal multivariada.
# - **(c) Vizinhança:** o limite do percentil 99 (2,00) é 2,8 vezes a mediana da distância ao 10º vizinho (0,709).
# - **Atípicos marcados:** 1.293 usuários (0,80%) estão longe do centro e isolados ao mesmo tempo. Outros 2.270 estão longe do centro, mas têm vizinhos próximos (possíveis nichos), e só 333 estão isolados perto do centro.
# - **Quem são:** em quase todos, o atributo mais extremo é um gênero raro: Documentary (222 usuários), Film-Noir (215), Western (183) e War (118). Eles avaliam menos filmes (mediana de 35 contra 71) e dão notas um pouco mais altas (mediana da média 3,88 contra 3,70). Com poucas avaliações, poucos filmes de um gênero raro bastam para gerar uma proporção extrema.
# - **Gráfico:** no plano CP1 × CP2 os atípicos se espalham pela borda, mas também se sobrepõem ao centro, porque se afastam sobretudo nos componentes 4 a 6, que carregam Mystery, Western, Film-Noir, War e Documentary.
#
# **Limitação:** os limiares (quantil 0,999 e percentil 99) são escolhas registradas, e a referência χ² supõe normalidade, que os dados não seguem.
#
# **Consequência:** os 1.293 usuários são mantidos e ficam marcados em `perfil_pd["atipico"]`, para que P1 seja executado com e sem eles. Com a exclusão dos filmes sem gênero (2.2-d), `Unknown` já não existe; os valores extremos vêm agora dos gêneros raros de verdade, que o vetor revisado (3.8) agrupa.

# %% [markdown]
# #### Tendência de agrupamento
#
# Um algoritmo de agrupamento sempre devolve grupos, mesmo em dados sem estrutura. A estatística de Hopkins verifica, antes de P1, se os usuários se distribuem de forma diferente de pontos uniformes no mesmo domínio:
#
# - **u:** distância de um ponto uniforme ao usuário real mais próximo;
# - **w:** distância de um usuário real sorteado ao seu vizinho real mais próximo;
# - **H = Σu / (Σu + Σw):** H ≈ 0,5 indica dados indistinguíveis de pontos uniformes; H próximo de 1 indica dados concentrados em regiões.
#
# O cálculo usa 1.000 pontos por repetição e 10 repetições, nos dois espaços candidatos, com e sem os atípicos. Como os valores extremos alargam o domínio mín–máx e inflam H, o domínio dos percentis 1–99 é usado como versão robusta. Um controle com dados uniformes de mesmo tamanho deve dar H ≈ 0,5.
#
# **Limite da medida:** H mede concentração, não separação. Um único aglomerado denso também dá H alto. A separação entre grupos só pode ser avaliada em P1.

# %%
# [3.7-b] Tendência de agrupamento: estatística de Hopkins contra pontos uniformes no mesmo domínio
from sklearn.neighbors import NearestNeighbors

M_HOPKINS, REPETICOES_HOPKINS = 1000, 10


def hopkins(X, dominio="min-max", m=M_HOPKINS, repeticoes=REPETICOES_HOPKINS, semente=42):
    """H = Σu / (Σu + Σw).
    u: distância de um ponto uniforme ao usuário real mais próximo; w: distância de um usuário real sorteado
    ao vizinho real mais próximo. H ≈ 0,5: dados indistinguíveis de uniformes; H → 1: dados concentrados."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    vizinhos = NearestNeighbors(n_neighbors=2).fit(X)
    if dominio == "min-max":
        inferior, superior = X.min(axis=0), X.max(axis=0)
    else:  # percentis 1 a 99: domínio menos influenciado pelos valores extremos
        inferior, superior = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    rng = np.random.default_rng(semente)
    valores = []
    for _ in range(repeticoes):
        sorteados = rng.choice(n, size=m, replace=False)
        w = vizinhos.kneighbors(X[sorteados], n_neighbors=2)[0][:, 1]
        u = vizinhos.kneighbors(rng.uniform(inferior, superior, size=(m, d)), n_neighbors=1)[0][:, 0]
        valores.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(valores)), float(np.std(valores)), u, w


sem_atipicos = ~perfil_pd["atipico"].to_numpy()
ROTULO_PCA, ROTULO_Z = f"PCA ({X_p1_pca.shape[1]})", f"z-score ({X_p1_padronizado.shape[1]})"
ROTULO_PCA80 = f"PCA 80% ({X_p1_pca80.shape[1]})"
cenarios_hopkins = [
    (ROTULO_PCA, "todos", X_p1_pca.to_numpy()),
    (ROTULO_PCA, "sem atípicos", X_p1_pca.to_numpy()[sem_atipicos]),
    (ROTULO_Z, "todos", X_p1_padronizado.to_numpy()),
    (ROTULO_Z, "sem atípicos", X_p1_padronizado.to_numpy()[sem_atipicos]),
    (ROTULO_PCA80, "todos", X_p1_pca80.to_numpy()),
    (ROTULO_PCA80, "sem atípicos", X_p1_pca80.to_numpy()[sem_atipicos]),
]
linhas_hopkins = []
for espaco, populacao, X in cenarios_hopkins:
    for dominio in ["min-max", "percentis 1–99"]:
        h_medio, h_dp, u, w = hopkins(X, dominio)
        linhas_hopkins.append((espaco, populacao, dominio, len(X), h_medio, h_dp))
        if (espaco, populacao, dominio) == (ROTULO_PCA, "todos", "percentis 1–99"):
            u_vis, w_vis = u, w

# Controle: dados uniformes com o mesmo tamanho, no domínio dos percentis 1–99 de cada espaço.
for espaco, X in [(ROTULO_PCA, X_p1_pca.to_numpy()), (ROTULO_Z, X_p1_padronizado.to_numpy()), (ROTULO_PCA80, X_p1_pca80.to_numpy())]:
    X_uniforme = np.random.default_rng(0).uniform(np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0), size=X.shape)
    h_medio, h_dp, _, _ = hopkins(X_uniforme, "min-max")
    linhas_hopkins.append((espaco, "controle: dados uniformes", "min-max", len(X_uniforme), h_medio, h_dp))

tabela_hopkins = pd.DataFrame(linhas_hopkins, columns=["espaço", "população", "domínio dos pontos uniformes", "n",
                                                       "H médio", "DP"])
print(f"Estatística de Hopkins: m = {M_HOPKINS:,} pontos por repetição, {REPETICOES_HOPKINS} repetições (semente 42); "
      "distância euclidiana")
print(tabela_hopkins.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.8))
log_w, log_u = np.log10(np.maximum(w_vis, 1e-3)), np.log10(np.maximum(u_vis, 1e-3))
bins = np.linspace(min(log_w.min(), log_u.min()), max(log_w.max(), log_u.max()), 50)
ax.hist(log_w, bins=bins, alpha=.6, color="#4682a9", label="usuário real → vizinho real mais próximo (w)")
ax.hist(log_u, bins=bins, alpha=.6, color="#e08a3c", label="ponto uniforme → usuário real mais próximo (u)")
ax.set(title=f"Distância ao vizinho mais próximo no espaço PCA: usuários × pontos uniformes (m = {M_HOPKINS:,})",
       xlabel="log10(distância)", ylabel="pontos")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Controle:** com dados uniformes, H = 0,500 nos três espaços, o que confirma o cálculo.
# - **Domínio mín–máx:** H = 0,960 no PCA e 0,953 no z-score. Esses valores são inflados pelos extremos, que alargam o domínio dos pontos uniformes: sem os 1.293 atípicos, caem para 0,930 e 0,932.
# - **Domínio robusto (percentis 1–99):** H = 0,705 no PCA com 6 componentes, 0,718 no PCA com 9 e 0,767 no z-score; sem os atípicos, 0,696, 0,709 e 0,762. O desvio-padrão entre repetições é no máximo 0,0040.
# - **Distâncias:** no histograma, a distância de um usuário ao vizinho real mais próximo (w) fica, em geral, abaixo da distância de um ponto uniforme ao usuário mais próximo (u), mas as duas distribuições se sobrepõem em parte.
#
# **Interpretação:** os usuários estão mais concentrados do que pontos uniformes, e isso não é efeito dos atípicos, porque retirá-los quase não muda o H robusto. Mas a tendência é moderada: uma referência usual é H > 0,75 para tendência clara, e o espaço PCA fica abaixo dela. Junto com as projeções de 3.6, o quadro é de um núcleo denso com caudas, e não de grupos naturalmente separados.
#
# **Consequência:** vale agrupar, porque os dados não são uniformes, mas deve-se esperar separação fraca entre grupos. A validação de P1 precisa comparar os índices com os de partições de dados uniformes no mesmo domínio.

# %% [markdown]
# ### 3.8 Vetor revisado: atributos derivados e limite de gêneros
#
# O vetor base (3.1–3.2) tem três limitações observadas nas seções anteriores:
#
# - as proporções de gênero de um usuário somam mais que 1, porque os gêneros se sobrepõem;
# - gêneros raros geram os valores extremos e a maior parte dos atípicos (3.3 e 3.7);
# - a atividade do usuário ficou fora da distância.
#
# O vetor revisado mantém a lógica de proporções com denominador explícito e acrescenta:
#
# - **grupos do perfil:** o IMAX sai, porque é formato de exibição, e não gênero (quase todo filme IMAX tem outro gênero); Animation e Children formam um grupo só, `Animation/Children`, porque têm o mesmo público em grande parte. Os rótulos dos dados não mudam: a junção vale só neste vetor;
# - **proporção fracionária:** cada avaliação distribui peso 1/k entre os k grupos do filme, e o vetor de gêneros soma 1;
# - **limite de gêneros:** grupos com menos de 2% das avaliações formam o grupo `raros`;
# - **atributos derivados:** viés do usuário (nota menos a média do filme), atividade em escala log, popularidade média e ano médio dos filmes avaliados, e diversidade de gêneros (entropia normalizada);
# - **gênero dominante:** atributo categórico, usado só para descrever.
#
# A seleção por filtro usa o mesmo limiar de 3.2: em cada par redundante, sai o atributo com maior |Pearson| médio com os demais.

# %%
# [3.8-a] Vetor revisado: atributos derivados, proporções fracionárias e limite de gêneros
LIMITE_GENERO = 0.02  # gêneros com menos de 2% das avaliações (contagem fracionária) formam o grupo "raros"

filmes_info = (ratings_clean.groupBy("movieId").agg(F.count("*").alias("n_filme"), F.avg("rating").alias("media_filme"))
    .join(movies_clean.select("movieId", "genres", "ano"), "movieId")
    .withColumn("ano_lancamento", F.col("ano").cast("double"))  # ano do título ou, sem ele, do IMDb (2.2-b)
    .withColumn("log_n_filme", F.log10(F.col("n_filme")))
    .select("movieId", "n_filme", "media_filme", "ano_lancamento", "log_n_filme")
    .cache())
derivados_pd = (ratings_clean.select("userId", "movieId", "rating")
    .join(F.broadcast(filmes_info), "movieId")
    .groupBy("userId").agg(
        F.avg(F.col("rating") - F.col("media_filme")).alias("vies_usuario"),
        F.avg("log_n_filme").alias("popularidade_media"),
        F.avg("ano_lancamento").alias("ano_medio_lancamento"))
    .toPandas())
# Grupos do perfil (decisão D5): IMAX é formato de exibição, não gênero, e sai; Animation e Children viram um grupo só.
GRUPO_DO_GENERO = {"IMAX": None, "Animation": "Animation/Children", "Children": "Animation/Children"}
grupo_do = lambda g: GRUPO_DO_GENERO.get(g, g)
grupos_perfil = sorted({grupo_do(g) for g in lista_generos} - {None})
coluna_grupo = {gr: "frac_" + gr.lower().replace("-", "_").replace("/", "_") for gr in grupos_perfil}
mapa_grupo = spark.createDataFrame([(g, grupo_do(g)) for g in lista_generos if grupo_do(g)], ["genre", "grupo"])
filme_grupo = movie_genres.select("movieId", "genre").join(F.broadcast(mapa_grupo), "genre").select("movieId", "grupo").distinct()
grupos_por_filme = filme_grupo.groupBy("movieId").agg(F.count("*").alias("n_grupos_filme"))
filme_grupo = filme_grupo.join(grupos_por_filme, "movieId").cache()
so_imax = movies_clean.join(grupos_por_filme, "movieId", "left_anti").select("movieId")
n_so_imax = so_imax.count()
aval_so_imax = ratings_clean.join(F.broadcast(so_imax), "movieId").count()
assert len(grupos_perfil) == len(lista_generos) - 2 and not any("/" in c for c in coluna_grupo.values())
print(f"Grupos do perfil: {len(grupos_perfil)} (os {len(lista_generos)} gêneros, sem IMAX, com Animation e Children juntos)")
print(f"Filmes que só têm IMAX: {n_so_imax}, com {aval_so_imax:,} avaliações; essas avaliações não entram nas proporções de gênero")


def pesos_do_filme(generos):
    """Regra conferível à mão: cada avaliação do filme distribui peso 1/k entre os k grupos distintos do filme."""
    grupos = sorted({grupo_do(g) for g in generos.split("|")} - {None})
    return {gr: 1 / len(grupos) for gr in grupos}


assert pesos_do_filme("Action|IMAX") == {"Action": 1.0}
assert pesos_do_filme("IMAX") == {}
assert pesos_do_filme("Animation|Children|Comedy") == {"Animation/Children": 0.5, "Comedy": 0.5}
assert pesos_do_filme("Animation") == {"Animation/Children": 1.0}

# Sobreposição de Animation e Children, nos dois sentidos (avaliações) e nos filmes (Jaccard).
flags = (movie_genres.filter(F.col("genre").isin("Animation", "Children"))
         .groupBy("movieId").agg(F.collect_set("genre").alias("gs"))
         .join(filmes_info.select("movieId", "n_filme"), "movieId", "left").fillna({"n_filme": 0}).toPandas())
tem_a, tem_c = flags["gs"].map(lambda s: "Animation" in s), flags["gs"].map(lambda s: "Children" in s)
print(f"Avaliações de Animation em filmes também Children: {100 * flags.loc[tem_a & tem_c, 'n_filme'].sum() / flags.loc[tem_a, 'n_filme'].sum():.1f}%; "
      f"de Children em filmes também Animation: {100 * flags.loc[tem_a & tem_c, 'n_filme'].sum() / flags.loc[tem_c, 'n_filme'].sum():.1f}%; "
      f"Jaccard nos filmes: {(tem_a & tem_c).sum() / (tem_a | tem_c).sum():.3f}")

# Exemplo calculado: um filme com Animation e Children e um com IMAX (Toy Story e Avatar, se estiverem no catálogo).
catalogo_ex = (movies_clean.select("movieId", "title", "genres").join(filmes_info.select("movieId", "n_filme"), "movieId")
               .orderBy(F.desc("n_filme"), "movieId").toPandas())
exemplos = []
for titulo, condicao in [("Toy Story (1995)", lambda s: "Animation" in s and "Children" in s),
                         ("Avatar (2009)", lambda s: "IMAX" in s and "|" in s)]:
    achados = catalogo_ex[catalogo_ex["title"] == titulo]
    exemplos.append(achados.iloc[0] if len(achados) else catalogo_ex[catalogo_ex["genres"].map(condicao)].iloc[0])
pesos_spark = filme_grupo.filter(F.col("movieId").isin([int(e["movieId"]) for e in exemplos])).toPandas()
print("\nExemplo calculado (peso de cada avaliação do filme):")
for e in exemplos:
    generos = e["genres"].split("|")
    v8 = pesos_do_filme(e["genres"])
    no_spark = pesos_spark[pesos_spark["movieId"] == e["movieId"]]
    assert set(no_spark["grupo"]) == set(v8) and np.allclose(1 / no_spark["n_grupos_filme"], 1 / len(v8))
    print(f"  {e['title']} ({e['genres']}): v7, por gênero: 1/{len(generos)} para cada um; "
          f"v8, por grupo: " + ", ".join(f"{gr} 1/{len(v8)}" for gr in v8))

# Proporção fracionária: cada avaliação distribui peso 1/k entre os k grupos do filme; o vetor de gêneros soma 1.
frac_pd = (ratings_clean.select("userId", "movieId")
    .join(F.broadcast(filme_grupo), "movieId")
    .groupBy("userId").pivot("grupo", grupos_perfil).agg(F.sum(1.0 / F.col("n_grupos_filme")))
    .toPandas().fillna(0.0)
    .rename(columns=coluna_grupo))
cols_frac = [coluna_grupo[gr] for gr in grupos_perfil]
massa_genero = frac_pd[cols_frac].sum() / frac_pd[cols_frac].sum().sum()  # participação de cada grupo nas avaliações
perfil_rev_pd = (perfil_pd[["userId", "n_avaliacoes", "nota_media", "nota_dp", "atipico"]]
                 .merge(derivados_pd, on="userId", how="left").merge(frac_pd, on="userId", how="left"))
assert len(perfil_rev_pd) == len(perfil_pd)
assert (perfil_rev_pd["userId"].to_numpy() == perfil_pd["userId"].to_numpy()).all()
perfil_rev_pd.index = perfil_pd.index
peso_total = perfil_rev_pd[cols_frac].sum(axis=1)  # = avaliações do usuário menos as de filmes que só têm IMAX
assert (peso_total > 0).all()
perfil_rev_pd[cols_frac] = perfil_rev_pd[cols_frac].div(peso_total, axis=0)
assert np.allclose(perfil_rev_pd[cols_frac].sum(axis=1), 1.0, atol=1e-9)

# Limite de gêneros: participação de cada grupo no total de avaliações, com contagem fracionária.
destino_com = lambda limite: {gr: "mantido" if massa_genero[coluna_grupo[gr]] >= limite else "agrupado em raros"
                               for gr in grupos_perfil}
destino = destino_com(LIMITE_GENERO)
tabela_generos = (pd.DataFrame({"grupo": grupos_perfil,
                                "participação_%": [100 * massa_genero[coluna_grupo[gr]] for gr in grupos_perfil],
                                "destino": [destino[gr] for gr in grupos_perfil],
                                **{f"com limite de {lim:.1%}".replace(".", ","): [destino_com(lim)[gr] for gr in grupos_perfil]
                                   for lim in (0.015, 0.025)}})
                  .sort_values("participação_%", ascending=False))
print(f"\nParticipação de cada grupo nas avaliações (contagem fracionária; limite = {100 * LIMITE_GENERO:.0f}%; "
      "sensibilidade com 1,5% e 2,5%):")
print(tabela_generos.round(3).to_string(index=False))
for lim in (0.015, 0.025):
    mudam = [gr for gr in grupos_perfil if destino_com(lim)[gr] != destino[gr]]
    print(f"Com limite de {100 * lim:.1f}%: mudam de lado {mudam if mudam else 'nenhum'}".replace(".", ",", 1))
generos_mantidos = [gr for gr in grupos_perfil if destino[gr] == "mantido"]
generos_raros = [gr for gr in grupos_perfil if destino[gr] == "agrupado em raros"]
perfil_rev_pd["frac_raros"] = perfil_rev_pd[[coluna_grupo[gr] for gr in generos_raros]].sum(axis=1)

# Atributos derivados.
perfil_rev_pd["atividade_log"] = np.log10(perfil_rev_pd["n_avaliacoes"])
p_generos = perfil_rev_pd[cols_frac].to_numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    entropia = -np.where(p_generos > 0, p_generos * np.log(p_generos), 0.0).sum(axis=1)
perfil_rev_pd["diversidade_generos"] = entropia / np.log(len(cols_frac))
sem_ano = perfil_rev_pd["ano_medio_lancamento"].isna()
print(f"\nUsuários sem nenhum filme com ano no título: {int(sem_ano.sum())}")
if sem_ano.any():
    perfil_rev_pd.loc[sem_ano, "ano_medio_lancamento"] = perfil_rev_pd["ano_medio_lancamento"].median()
perfil_rev_pd["genero_dominante"] = (perfil_rev_pd[[coluna_grupo[gr] for gr in generos_mantidos] + ["frac_raros"]]
                                     .idxmax(axis=1).str.replace("frac_", "", regex=False))

ATRIBUTOS_DERIVADOS = ["vies_usuario", "atividade_log", "popularidade_media", "ano_medio_lancamento", "diversidade_generos"]
candidatos_rev = (["nota_media", "nota_dp"] + ATRIBUTOS_DERIVADOS
                  + [coluna_grupo[gr] for gr in generos_mantidos] + ["frac_raros"])
print("\nAtributos derivados (por usuário):")
print("  vies_usuario = média de (nota do usuário − média do filme): rigor do usuário sem o efeito dos filmes escolhidos")
print("  atividade_log = log10(avaliações do usuário); popularidade_media = média de log10(avaliações dos filmes avaliados)")
print("  ano_medio_lancamento = média do ano dos filmes avaliados; "
      f"diversidade_generos = entropia das proporções / log({len(cols_frac)})")
print(perfil_rev_pd[ATRIBUTOS_DERIVADOS].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

# Seleção por filtro, com o mesmo limiar de 3.2: em cada par redundante sai o atributo de maior |Pearson| médio com os demais.
pearson_rev = perfil_rev_pd[candidatos_rev].corr(method="pearson")
spearman_rev = perfil_rev_pd[candidatos_rev].corr(method="spearman")
atributos_rev, removidos_rev = list(candidatos_rev), {}
while True:
    p_abs = pearson_rev.loc[atributos_rev, atributos_rev].abs().to_numpy(copy=True)
    s_abs = spearman_rev.loc[atributos_rev, atributos_rev].abs().to_numpy(copy=True)
    np.fill_diagonal(p_abs, 0.0)
    np.fill_diagonal(s_abs, 0.0)
    redundante = (p_abs >= LIMIAR_REDUNDANCIA) & (s_abs >= LIMIAR_REDUNDANCIA)
    if not redundante.any():
        break
    i, j = np.unravel_index(np.argmax(np.where(redundante, p_abs, 0.0)), p_abs.shape)
    media_abs = p_abs.sum(axis=1) / (len(atributos_rev) - 1)
    sai, fica = (atributos_rev[i], atributos_rev[j]) if media_abs[i] >= media_abs[j] else (atributos_rev[j], atributos_rev[i])
    removidos_rev[sai] = (f"redundante com {fica} (Pearson {pearson_rev.loc[sai, fica]:.3f}; "
                          f"Spearman {spearman_rev.loc[sai, fica]:.3f}); maior |Pearson| médio com os demais")
    atributos_rev.remove(sai)
pares_rev = pd.DataFrame([(a, b, pearson_rev.loc[a, b], spearman_rev.loc[a, b])
                          for k, a in enumerate(candidatos_rev) for b in candidatos_rev[k + 1:]],
                         columns=["atributo_1", "atributo_2", "pearson", "spearman"])
pares_rev = pares_rev.reindex(pares_rev["pearson"].abs().sort_values(ascending=False).index)
print(f"\nPares mais associados entre os {len(candidatos_rev)} candidatos:")
print(pares_rev.head(8).round(3).to_string(index=False))
print("Removidos por redundância:", removidos_rev if removidos_rev else "nenhum")
print(f"Vetor revisado ({len(atributos_rev)} atributos):", ", ".join(atributos_rev))
print("Gênero dominante (descreve; fica fora da distância):",
      perfil_rev_pd["genero_dominante"].value_counts(normalize=True).mul(100).round(2).head(8).to_dict())

colunas_hist = candidatos_rev
n_linhas = int(np.ceil(len(colunas_hist) / 5))
fig, axes = plt.subplots(n_linhas, 5, figsize=(26, 4.4 * n_linhas))
for ax, coluna in zip(axes.flat, colunas_hist):
    histograma_media_dp(ax, perfil_rev_pd[coluna], coluna, bins=40,
                        cor="#7656a5" if coluna in ATRIBUTOS_DERIVADOS else "#4682a9",
                        rotulo_x="proporção fracionária" if coluna.startswith("frac_") else "")
for ax in list(axes.flat)[len(colunas_hist):]:
    ax.axis("off")
fig.suptitle("Candidatos do vetor revisado (roxo = atributos derivados); média e ± 1 DP em vermelho", y=1.005)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Grupos do perfil:** 17 grupos (os 19 gêneros, sem IMAX e com Animation e Children juntos). Participação fracionária: Drama 20,8%, Comedy 17,1%, Action 9,6%, Thriller 9,2%, Adventure 7,3% e Romance 6,8% lideram. Doze grupos ficam acima do limite de 2%.
# - **Animation/Children:** com 3,34%, o grupo fica acima do limite. Na v7, Animation (1,66%) caía em `raros`, junto com Film-Noir e Documentary, e Children (2,38%) ficava sozinho. O grupo tem menos que a soma dos dois (4,03%): contado uma vez só por filme, ele passa a outros gêneros o peso que antes ia duas vezes para o mesmo público. No exemplo, cada avaliação de Toy Story dava 1/5 a cada um dos 5 gêneros e passa a dar 1/4 a cada um dos 4 grupos.
# - **Sobreposição:** 77,6% das avaliações de Animation são de filmes também Children, mas só 59,6% das de Children são de filmes também Animation, e o Jaccard nos filmes é 0,21. Os dois não são equivalentes, e a correlação entre usuários (0,82 na 3.2) fica abaixo do limiar de redundância. A junção é uma escolha de taxonomia: o grupo descreve o público de animação e infantil.
# - **IMAX:** sai do perfil, porque é formato de exibição. A fração dele vai para os outros gêneros do filme (Avatar: de 1/4 para 1/3 em Action, Adventure e Sci-Fi). Só 1 filme tem apenas IMAX, com 27 avaliações, que ficam fora das proporções.
# - **Gêneros raros:** War, Musical, Documentary, Western e Film-Noir somam 5,27% das avaliações e formam o grupo `raros`.
# - **Sensibilidade do limite:** com 1,5%, só War muda de lado (fica sozinho); com 2,5%, só Mystery (vai para `raros`). Nenhum outro grupo muda entre 1,5% e 2,5%, e o limite fica em 2%.
# - **Atributos derivados:**
#   - viés do usuário: média 0,08 e DP 0,44 (de −3,42 a 2,44). Em média, os usuários dão notas próximas da média de cada filme, mas alguns são sistematicamente rigorosos ou generosos;
#   - atividade em log: média 1,92 (cerca de 83 avaliações), DP 0,44;
#   - popularidade média: 4,04, ou seja, o filme típico visto por um usuário tem cerca de 11 mil avaliações;
#   - ano médio dos filmes vistos: 1994, com DP de 7,3 anos;
#   - diversidade de gêneros: 0,80, com DP 0,06.
# - **Redundância:** o par mais associado é nota média × viés do usuário (Pearson 0,903; Spearman 0,881). Como só um dos dois critérios passa de 0,9, os dois ficam. Nenhum atributo foi removido, e o vetor revisado tem 20 atributos.
# - **Gênero dominante:** Drama em 54,3% dos usuários, Comedy em 28,0% e Action em 8,2%.
#
# **Limitação:** o limite de 2% e os grupos são escolhas registradas. A animação adulta (anime, por exemplo) entra no grupo de animação e infantil, porque só 21% dos filmes têm os dois gêneros; os rótulos dos dados continuam separados.

# %% [markdown]
# #### Comparação entre o vetor base e o vetor revisado
#
# Os dois vetores são comparados com os mesmos critérios e amostras das seções anteriores: PCA com Kaiser-Guttman, broken-stick e 80% da variância (3.4); contraste e variância relativa na amostra de 2.000 usuários (3.5); dimensionalidade intrínseca na amostra de 10.000 (3.5-b); atípicos e estatística de Hopkins com as regras de 3.7. O vetor base continua disponível, e a escolha da entrada de P1 é registrada em 3.10.

# %%
# [3.8-b] Vetor base × vetor revisado: PCA, concentração de distâncias, dimensionalidade, atípicos e tendência
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy import stats

X_rev_original = perfil_rev_pd[atributos_rev].astype(float)
X_rev_padronizado = pd.DataFrame(StandardScaler().fit_transform(X_rev_original), columns=atributos_rev, index=perfil_pd.index)
pca_rev = PCA(svd_solver="full").fit(X_rev_padronizado)
autovalores_rev = pca_rev.explained_variance_
variancia_rev = 100 * pca_rev.explained_variance_ratio_
n_atr = len(atributos_rev)
broken_rev = 100 * np.array([sum(1.0 / j for j in range(i, n_atr + 1)) / n_atr for i in range(1, n_atr + 1)])
d_rev = int((autovalores_rev >= 1).sum())
d_broken_rev = int(np.argmax(variancia_rev <= broken_rev)) if (variancia_rev <= broken_rev).any() else n_atr
d_80_rev = int(np.searchsorted(np.cumsum(variancia_rev), 80) + 1)
X_rev_pca = pd.DataFrame(pca_rev.transform(X_rev_padronizado)[:, :d_rev],
                         columns=[f"CP{k}" for k in range(1, d_rev + 1)], index=perfil_pd.index)
print(f"PCA do vetor revisado: d′ = {d_rev} (Kaiser-Guttman, {np.cumsum(variancia_rev)[d_rev - 1]:.2f}% da variância); "
      f"broken-stick = {d_broken_rev}; 80% da variância = {d_80_rev}")
cargas_rev = pd.DataFrame(pca_rev.components_[:d_rev].T, index=atributos_rev, columns=X_rev_pca.columns)
for cp in cargas_rev.columns:
    principais = cargas_rev[cp].abs().sort_values(ascending=False).head(4).index
    print(f"  {cp} ({variancia_rev[int(cp[2:]) - 1]:.2f}%): " + ", ".join(f"{a} ({cargas_rev.loc[a, cp]:+.2f})" for a in principais))


def atipicos_no_espaco(X):
    """Mesma regra de 3.7-a: Mahalanobis² acima do quantil 0,999 da χ² e distância ao 10º vizinho acima do percentil 99."""
    X = np.asarray(X, dtype=float)
    centrado = X - X.mean(axis=0)
    d2 = (centrado ** 2 / centrado.var(axis=0)).sum(axis=1)
    dist_k10 = NearestNeighbors(n_neighbors=K_VIZ + 1, n_jobs=-1).fit(X).kneighbors(X)[0][:, K_VIZ]
    return (d2 > stats.chi2.ppf(0.999, df=X.shape[1])) & (dist_k10 > np.quantile(dist_k10, 0.99))


perfil_rev_pd["atipico_rev"] = atipicos_no_espaco(X_rev_pca)
# Alternativa registrada em 3.4: o PCA com os componentes que somam 80% da variância, nos dois vetores. Só comparação.
X_rev_pca80 = pd.DataFrame(pca_rev.transform(X_rev_padronizado)[:, :d_80_rev],
                           columns=[f"CP{k}" for k in range(1, d_80_rev + 1)], index=perfil_pd.index)
espacos_comparacao = [
    ("base", "z-score", X_p1_padronizado, np.nan, np.nan),
    ("base", "PCA", X_p1_pca, 100 * acumulada[d_pca - 1], int(perfil_pd["atipico"].sum())),
    ("revisado", "z-score", X_rev_padronizado, np.nan, np.nan),
    ("revisado", "PCA", X_rev_pca, np.cumsum(variancia_rev)[d_rev - 1], int(perfil_rev_pd["atipico_rev"].sum())),
    ("base", "PCA 80%", X_p1_pca80, 100 * acumulada[d_variancia - 1], int(atipicos_no_espaco(X_p1_pca80).sum())),
    ("revisado", "PCA 80%", X_rev_pca80, np.cumsum(variancia_rev)[d_80_rev - 1], int(atipicos_no_espaco(X_rev_pca80).sum())),
]
linhas_comp = []
for vetor, espaco, X, var_retida, n_atipicos in espacos_comparacao:
    d = pdist(X.loc[idx_dist].to_numpy(), metric="euclidean")
    d = d[d > 0]
    id_media, _ = dimensao_intrinseca_mle(X.loc[idx_id].to_numpy())
    h_medio, h_dp, _, _ = hopkins(X.to_numpy(), "percentis 1–99")
    linhas_comp.append((vetor, espaco, X.shape[1], var_retida, (d.max() - d.min()) / d.min(), d.std() / d.mean(),
                        id_media, h_medio, n_atipicos))
tabela_comparacao = pd.DataFrame(linhas_comp, columns=["vetor", "espaço", "dimensões", "variância retida %",
                                                       "contraste relativo", "variância relativa", "ID (MLE)",
                                                       "Hopkins (1–99)", "atípicos"])
print("\nComparação nas mesmas amostras de 3.5 (2.000 usuários), 3.5-b (10.000) e 3.7 (Hopkins, m = 1.000, 10 repetições):")
print(tabela_comparacao.round(4).to_string(index=False))
ambos = int((perfil_pd["atipico"] & perfil_rev_pd["atipico_rev"]).sum())
uniao = int((perfil_pd["atipico"] | perfil_rev_pd["atipico_rev"]).sum())
print(f"Atípicos nos dois vetores: {ambos:,} (Jaccard = {ambos / uniao:.3f})")
print("Atributo dominante dos atípicos do vetor revisado:",
      X_rev_padronizado[perfil_rev_pd["atipico_rev"].to_numpy()].abs().idxmax(axis=1).value_counts().head(6).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
pos = np.arange(1, n_atr + 1)
axes[0].bar(pos, variancia_rev, color="#7656a5", label="variância explicada (%)")
axes[0].plot(pos, broken_rev, "o--", color="#d18b24", label="broken-stick (%)")
axes[0].axvline(d_rev + 0.5, color="crimson", ls=":", label=f"Kaiser: d′ = {d_rev}")
axes[0].set(title="Scree plot do vetor revisado", xlabel="componente", ylabel="% da variância")
axes[0].legend()
imagem = axes[1].imshow(pearson_rev.loc[atributos_rev, atributos_rev], vmin=-1, vmax=1, cmap="coolwarm")
axes[1].set_xticks(range(n_atr), atributos_rev, rotation=70, ha="right", fontsize=8)
axes[1].set_yticks(range(n_atr), atributos_rev, fontsize=8)
axes[1].set_title("Pearson entre os atributos do vetor revisado")
fig.colorbar(imagem, ax=axes[1], shrink=.8)
dominante_vis = perfil_rev_pd.loc[idx_vis, "genero_dominante"]
principais_dom = dominante_vis.value_counts().head(6).index
for rotulo in list(principais_dom) + ["outros"]:
    mascara = (dominante_vis.isin(principais_dom) == False) if rotulo == "outros" else (dominante_vis == rotulo)
    pontos = X_rev_pca.loc[idx_vis][mascara.to_numpy()]
    axes[2].scatter(pontos["CP1"], pontos["CP2"], s=6, alpha=.55, label=f"{rotulo} ({mascara.sum()})")
axes[2].set(title=f"Vetor revisado: CP1 × CP2 ({len(idx_vis):,} usuários; cor = gênero dominante)", xlabel="CP1", ylabel="CP2")
axes[2].legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **PCA do vetor revisado:** 7 componentes por Kaiser-Guttman retêm 73,94% da variância, contra 66,84% com 6 componentes no vetor base. O broken-stick retém 3 componentes (contra 2), sinal de mais estrutura nos primeiros eixos.
# - **Componentes:**
#   - CP1 (20,9%): ação, aventura e ficção contra romance;
#   - CP2 (14,7%): crime e suspense contra animação/infantil e fantasia;
#   - CP3 (11,7%): nota média e viés do usuário;
#   - CP4 (8,4%): atividade contra popularidade, ou seja, usuários muito ativos veem filmes menos populares;
#   - CP5 (7,6%): gêneros raros, em usuários que veem filmes mais antigos;
#   - CP6 (5,7%): ano médio dos filmes vistos;
#   - CP7 (5,0%): atividade e comédia contra mistério e horror.
# - **Distâncias:** no espaço PCA, o contraste relativo cai de 110,66 para 45,18 e a variância relativa, de 0,426 para 0,369: as distâncias ficam um pouco mais concentradas no vetor revisado. O contraste relativo depende só do par mais próximo e do mais distante da amostra, por isso oscila mais; a variância relativa usa todos os pares.
# - **Dimensionalidade intrínseca:** 7,2 no PCA revisado e 12,0 no z revisado, contra 6,4 e 13,2 no vetor base.
# - **Atípicos:** 1.171 no vetor revisado, contra 1.293 no base (−9%). Só 412 são atípicos nos dois (Jaccard 0,20). No vetor revisado, os atípicos se concentram em horror (327 usuários).
# - **Hopkins (domínio robusto):** 0,68 no PCA revisado e 0,77 no z revisado, contra 0,70 e 0,77 no base. A tendência de agrupamento continua moderada.
# - **Antes e depois dos grupos do perfil (transparência):** sem os grupos (v7), o PCA revisado tinha 6 componentes (69,13% da variância), 1.078 atípicos e Hopkins de 0,68. Com IMAX fora e Animation/Children juntos, passa a 7 componentes (73,94%), 1.171 atípicos e o mesmo Hopkins. A decisão dos grupos é de taxonomia e não depende desta comparação.
# - **PCA com 80% da variância (9 componentes, só comparação):**
#
#   | | base, 6 | base, 9 | revisado, 7 | revisado, 9 |
#   |---|---:|---:|---:|---:|
#   | variância retida | 66,84% | 80,07% | 73,94% | 82,37% |
#   | variância relativa das distâncias | 0,426 | 0,417 | 0,369 | 0,354 |
#   | Hopkins (1–99) | 0,705 | 0,718 | 0,684 | 0,685 |
#   | atípicos | 1.293 | 1.422 | 1.171 | 1.351 |
#
#   Com 9 componentes, os dois vetores guardam mais informação e a tendência de agrupamento sobe um pouco no vetor base. Em troca, as distâncias ficam um pouco menos nítidas e aparecem mais atípicos. As diferenças são pequenas nos dois sentidos, e a escolha entre 6 ou 7 e 9 componentes fica para P1, pela qualidade dos grupos.
#
# **Interpretação:** o vetor revisado guarda mais variância, gera menos atípicos e acrescenta eixos interpretáveis (atividade, popularidade e época), ao custo de um componente a mais. As distâncias ficam um pouco mais concentradas, e a tendência de agrupamento fica próxima da do vetor base.
#
# **Consequência:** o PCA do vetor revisado passa a ser a entrada principal de P1, pelos ganhos de representação: proporções que somam 1, sem contagem dupla de gêneros, com atividade e popularidade. O vetor base fica como comparação, também para medir o efeito da concentração de distâncias um pouco maior.

# %% [markdown]
# ### 3.9 Solução indexada para buscas por similaridade
#
# P1 depende de buscas por similaridade: o DBSCAN faz buscas por abrangência (todos os vizinhos dentro de um raio), a escolha desse raio usa a distância ao k-ésimo vizinho, e o kNN busca os k mais próximos. Com 162.541 usuários, a varredura linear compara cada consulta com todos os objetos. A célula compara, nos quatro espaços candidatos:
#
# - **varredura linear:** referência exata;
# - **kd-tree e ball-tree:** índices exatos por particionamento do espaço;
# - **pivôs:** pela desigualdade triangular, |d(q,p) − d(o,p)| ≤ d(q,o). Se esse limite inferior passa do raio, o objeto é descartado sem que a distância seja calculada. Os pivôs são escolhidos por variância máxima das distâncias e por seleção incremental;
# - **HNSW:** índice aproximado baseado em grafo, avaliado pelo recall (fração dos 10 vizinhos verdadeiros recuperados).
#
# Os tempos dependem da máquina, então a comparação vale entre métodos na mesma execução. A fração de distâncias evitadas pelos pivôs não depende da máquina.

# %%
# [3.9] Solução indexada: varredura linear, kd-tree, ball-tree, pivôs e índice aproximado (HNSW)
import importlib.util, subprocess, sys, time
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

if importlib.util.find_spec("faiss") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"], check=False)
try:
    import faiss
except ImportError:
    faiss = None
    print("faiss indisponível: o índice aproximado (HNSW) não será medido.")

K_INDICE, N_CONSULTAS_KNN, N_CONSULTAS_RAIO, N_CONSULTAS_PIVO, P_MAX = 10, 1000, 200, 100, 16
espacos_indice = {
    f"base PCA ({X_p1_pca.shape[1]})": X_p1_pca.to_numpy(),
    f"base z ({X_p1_padronizado.shape[1]})": X_p1_padronizado.to_numpy(),
    f"revisado PCA ({X_rev_pca.shape[1]})": X_rev_pca.to_numpy(),
    f"revisado z ({X_rev_padronizado.shape[1]})": X_rev_padronizado.to_numpy(),
    f"base PCA 80% ({X_p1_pca80.shape[1]})": X_p1_pca80.to_numpy(),
    f"revisado PCA 80% ({X_rev_pca80.shape[1]})": X_rev_pca80.to_numpy(),
}
rng_indice = np.random.default_rng(7)
consultas_idx = rng_indice.choice(len(perfil_pd), size=N_CONSULTAS_KNN, replace=False)


def cronometrar(funcao):
    inicio = time.perf_counter()
    resultado = funcao()
    return resultado, time.perf_counter() - inicio


linhas_indice, linhas_pivos, raios = [], [], {}
for nome, X in espacos_indice.items():
    X = np.ascontiguousarray(X, dtype=np.float64)
    X_consulta, X_raio = X[consultas_idx], X[consultas_idx[:N_CONSULTAS_RAIO]]
    # Varredura linear: compara a consulta com todos os objetos; é a referência exata.
    nn_linear, t_constr = cronometrar(lambda: NearestNeighbors(n_neighbors=K_INDICE + 1, algorithm="brute").fit(X))
    (dist_exata, _), t_knn = cronometrar(lambda: nn_linear.kneighbors(X_consulta))
    raio = float(np.median(dist_exata[:, K_INDICE]))  # raio da busca por abrangência = mediana da distância ao 10º vizinho
    raios[nome] = raio
    limite_k = dist_exata[:, [K_INDICE]] + 1e-6
    vizinhos_raio, t_raio = cronometrar(lambda: nn_linear.radius_neighbors(X_raio, radius=raio, return_distance=False))
    linhas_indice.append((nome, "varredura linear", t_constr, t_knn, t_raio, 1.0))
    for algoritmo in ("kd_tree", "ball_tree"):
        nn_arvore, t_constr = cronometrar(lambda: NearestNeighbors(n_neighbors=K_INDICE + 1, algorithm=algoritmo).fit(X))
        (dist_arvore, _), t_knn = cronometrar(lambda: nn_arvore.kneighbors(X_consulta))
        _, t_raio = cronometrar(lambda: nn_arvore.radius_neighbors(X_raio, radius=raio, return_distance=False))
        recall = float(np.mean(dist_arvore[:, 1:] <= limite_k))
        linhas_indice.append((nome, algoritmo.replace("_", "-"), t_constr, t_knn, t_raio, recall))
    if faiss is not None:
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        indice_hnsw = faiss.IndexHNSWFlat(X32.shape[1], 32)
        indice_hnsw.hnsw.efConstruction = 40
        _, t_constr = cronometrar(lambda: indice_hnsw.add(X32))
        for ef in (16, 64, 256):
            indice_hnsw.hnsw.efSearch = ef
            (dist_hnsw, _), t_knn = cronometrar(lambda: indice_hnsw.search(X32[consultas_idx], K_INDICE + 1))
            dist_hnsw = np.sqrt(np.maximum(dist_hnsw, 0.0))  # o faiss devolve a distância ao quadrado
            recall = float(np.mean(dist_hnsw[:, 1:] <= limite_k + 1e-4))
            linhas_indice.append((nome, f"HNSW aproximado (efSearch={ef})", t_constr, t_knn, np.nan, recall))

    # Pivôs: |d(q,p) − d(o,p)| ≤ d(q,o) pela desigualdade triangular; se o limite inferior passa do raio, o objeto
    # é descartado sem calcular d(q,o).
    candidatos = rng_indice.choice(len(X), size=200, replace=False)
    a, b = rng_indice.choice(len(X), size=3000), rng_indice.choice(len(X), size=3000)
    dist_a, dist_b = cdist(X[a], X[candidatos]), cdist(X[b], X[candidatos])
    validos = np.linalg.norm(X[a] - X[b], axis=1) > 0
    limite_inferior_cand = np.abs(dist_a - dist_b)[validos]
    pivos_variancia = candidatos[np.argsort(dist_a.var(axis=0))[::-1][:P_MAX]]  # variância máxima das distâncias
    escolhidos, limite_atual = [], np.zeros(limite_inferior_cand.shape[0])
    for _ in range(P_MAX):  # seleção incremental: cada novo pivô maximiza o limite inferior médio dos pares
        ganho = np.maximum(limite_atual[:, None], limite_inferior_cand).mean(axis=0)
        ganho[escolhidos] = -np.inf
        j = int(np.argmax(ganho))
        escolhidos.append(j)
        limite_atual = np.maximum(limite_atual, limite_inferior_cand[:, j])
    pivos_incremental = candidatos[escolhidos]
    for estrategia, pivos in (("variância máxima", pivos_variancia), ("seleção incremental", pivos_incremental)):
        dist_objetos = cdist(X, X[pivos])  # N × P: calculado uma única vez, na construção do índice
        dist_consultas = cdist(X[consultas_idx[:N_CONSULTAS_PIVO]], X[pivos])
        descartados = np.zeros(P_MAX)
        for linha in dist_consultas:
            limite = np.maximum.accumulate(np.abs(dist_objetos - linha), axis=1)
            descartados += (limite > raio).mean(axis=0)
        descartados /= len(dist_consultas)
        for p in (1, 2, 4, 8, 16):
            linhas_pivos.append((nome, estrategia, p, 100 * descartados[p - 1]))

tabela_indice = pd.DataFrame(linhas_indice, columns=["espaço", "método", "construção (s)",
                                                     f"kNN: {N_CONSULTAS_KNN} consultas (s)",
                                                     f"raio: {N_CONSULTAS_RAIO} consultas (s)", "recall@10"])
tabela_pivos = pd.DataFrame(linhas_pivos, columns=["espaço", "estratégia", "pivôs", "% de distâncias evitadas"])
print(f"Busca kNN (k = {K_INDICE}) e por abrangência (raio = mediana da distância ao 10º vizinho), "
      f"{len(perfil_pd):,} usuários; tempos em segundos neste ambiente:")
print(tabela_indice.round(4).to_string(index=False))
print("\nRaio da busca por abrangência em cada espaço:", {k: round(v, 4) for k, v in raios.items()})
print("\nFiltragem por pivôs (busca por abrangência; % dos objetos descartados pelo limite inferior):")
print(tabela_pivos.pivot_table(index=["espaço", "estratégia"], columns="pivôs",
                               values="% de distâncias evitadas").round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(19, 5.5))
for (nome, estrategia), parte in tabela_pivos.groupby(["espaço", "estratégia"], sort=False):
    axes[0].plot(parte["pivôs"], parte["% de distâncias evitadas"], "o-" if estrategia == "seleção incremental" else "s--",
                 label=f"{nome} — {estrategia}")
axes[0].set(title="Distâncias evitadas pelos pivôs na busca por abrangência", xlabel="número de pivôs",
            ylabel="% dos objetos descartados", xscale="log", ylim=(0, 100))
axes[0].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
axes[0].legend(fontsize=7.5)
axes[0].grid(alpha=.2)
tempos = tabela_indice.pivot_table(index="método", columns="espaço", values=f"kNN: {N_CONSULTAS_KNN} consultas (s)", sort=False)
tempos.plot.bar(ax=axes[1], logy=True, rot=20)
axes[1].set(title=f"Tempo de {N_CONSULTAS_KNN} buscas kNN (k = {K_INDICE})", ylabel="segundos (escala log)", xlabel="")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Espaços PCA (6 ou 7 dimensões):** a kd-tree é o índice exato mais rápido: 0,17 s para 1.000 buscas kNN no PCA revisado, contra 0,53 s da varredura linear. A ball-tree não ganha da varredura.
# - **Espaços z (20 ou 21 dimensões):** as árvores ficam muito mais lentas que a varredura linear (kd-tree 5,0 a 6,4 s; ball-tree 12,0 a 12,6 s; varredura 0,6 s). É o efeito da dimensionalidade: os particionamentos deixam de descartar regiões.
# - **HNSW:** 1.000 buscas em 0,013 a 0,026 s, com recall@10 de 0,98 a 1,00 já com efSearch = 16. O custo fica na construção (5,0 a 8,9 s).
# - **PCA com 80% da variância (9 componentes):** a kd-tree perde a vantagem. Nas 1.000 buscas kNN, leva 0,33 s no vetor base e 0,54 s no revisado, contra 0,57 s e 0,56 s da varredura; nas buscas por abrangência, fica 2 a 4 vezes mais lenta que a varredura (0,12 e 0,22 s, contra 0,05 e 0,06 s). Com 16 pivôs, a busca por abrangência descarta 99,2% (base) e 99,0% (revisado) das distâncias. O HNSW continua rápido (0,026 e 0,015 s, recall@10 de 1,00).
# - **Pivôs:** com 16 pivôs de seleção incremental, a busca por abrangência descarta 99,9% das distâncias no PCA base e 99,8% no PCA revisado, contra 92,2% (z revisado) e 89,2% (z base). Na maioria dos casos, a seleção incremental supera a de variância máxima, sobretudo nos espaços z: com 2 pivôs no z revisado, cerca de 59% contra 46%.
#
# **Limitação:** os tempos dependem da máquina (Colab gratuito); a fração de distâncias evitadas não.
#
# **Consequência:** P1 usa o espaço PCA, onde os índices funcionam. As buscas exatas (DBSCAN e distância ao k-ésimo vizinho) usam kd-tree ou pivôs com seleção incremental, e o HNSW atende buscas aproximadas. Se P1 usar o PCA com 9 componentes, as buscas exatas por abrangência (DBSCAN) devem usar pivôs ou a varredura, e não a kd-tree.

# %% [markdown]
# ### 3.10 Resultado da transformação e entrada de P1
#
# | Objeto | Conteúdo | Uso em P1 |
# |---|---|---|
# | `perfil_pd` | `userId`, `n_avaliacoes` e 23 atributos derivados | identificar e descrever os grupos |
# | `X_p1_padronizado` | 21 atributos (média, DP e 19 proporções de gênero), z-score | entrada candidata interpretável |
# | `X_p1_minmax` | os mesmos 21 atributos em [0, 1] | alternativa de escala |
# | `X_p1_pca` | 6 componentes principais (critério de Kaiser-Guttman, 66,84%) | entrada candidata com dimensionalidade reduzida |
# | `X_p1_pca80`, `X_rev_pca80` | 9 componentes, 80% da variância, nos dois vetores (80,07% e 82,37%) | alternativa a testar em P1 |
# | `perfil_pd["atipico"]` | 1.293 usuários longe do centro e isolados dos vizinhos (3.7) | executar P1 com e sem esses usuários |
# | `perfil_rev_pd` | atributos derivados, proporções fracionárias, gênero dominante e `atipico_rev` (1.171 usuários) | descrever os grupos do vetor revisado |
# | `X_rev_padronizado` | 20 atributos do vetor revisado, z-score | entrada alternativa interpretável |
# | `X_rev_pca` | 7 componentes do vetor revisado (Kaiser-Guttman, 73,94%) | **entrada principal de P1** |
#
# **Resultados desta execução:**
#
# - **Seleção de características:** apenas `nota_media × prop_positivas` atingiu o limiar de redundância (Pearson 0,922; Spearman 0,946). O segundo par mais associado, `prop_animation × prop_children` (0,823 / 0,804), ficou abaixo do limiar, então os dois atributos foram mantidos.
# - **Escalas:**
#   - no z-score, cada atributo tem o mesmo peso na variância total (4,76%), mas gêneros raros geram valores extremos: máximos de 47,1 em Documentary, 38,9 em Western e 38,2 em Film-Noir;
#   - no min-max, os gêneros mais frequentes dominam a distância: Action pesa 11,54% e Drama 10,12%, contra 0,23% de Documentary.
# - **PCA:**
#   - d′ = 6 por Kaiser-Guttman (66,84% da variância);
#   - d′ = 2 por broken-stick;
#   - d′ = 9 para 80% da variância acumulada.
#   - Os dois primeiros componentes explicam 39,68% e opõem, respectivamente, perfis infantis/fantasia e ação/ficção científica contra romance/comédia.
# - **Concentração de distâncias** (2.000 usuários, distância euclidiana):
#
#   | Espaço | Contraste relativo | Variância relativa |
#   |---|---:|---:|
#   | z-score, 21 atributos | 33,22 | 0,3740 |
#   | min-max, 21 atributos | 22,24 | 0,3590 |
#   | PCA, 6 componentes | 110,66 | 0,4259 |
#   | PCA 80%, 9 componentes | 92,57 | 0,4173 |
#
#   Os espaços PCA foram os que menos concentraram as distâncias (o de 6 componentes um pouco menos que o de 9), e o min-max, o que mais concentrou.
#
# - **Dimensionalidade intrínseca:** cerca de 13 no espaço z de 21 atributos. As projeções 2D mostram uma nuvem contínua, sem grupos isolados.
# - **Valores atípicos:** 1.293 usuários (0,80%) estão longe do centro e isolados; em geral, têm poucas avaliações concentradas em gêneros raros (Documentary, Film-Noir, Western e War). Foram mantidos e marcados.
# - **Tendência de agrupamento:** estatística de Hopkins de 0,705 (PCA) e 0,767 (z-score) no domínio robusto, contra 0,500 nos dados uniformes de controle. Os usuários são mais concentrados que dados uniformes, mas a tendência é moderada.
# - **Vetor revisado (3.8):** proporções fracionárias, IMAX fora, Animation e Children num grupo só, gêneros raros agrupados, sem contagem dupla e com cinco atributos derivados. Retém 73,94% da variância em 7 componentes (o base, 66,84% em 6), tem 9% menos atípicos (1.171) e Hopkins de 0,68 no PCA, com distâncias um pouco mais concentradas.
# - **Solução indexada (3.9):** no espaço PCA, a kd-tree faz 1.000 buscas kNN em 0,17 s (varredura: 0,53 s), 16 pivôs descartam 99,9% das distâncias e o HNSW chega a recall de 100%. Nos espaços z, as árvores ficam mais lentas que a varredura.
#
# **Consequência para P1:** a entrada principal é `X_rev_pca`, o PCA do vetor revisado. Ele guarda mais variância, gera menos atípicos, não depende da contagem dupla de gêneros e trata animação e infantil como um público só. O PCA do vetor base e as matrizes z-score ficam como comparação, e o PCA com 80% da variância (9 componentes, nos dois vetores) é a alternativa a testar: guarda mais informação, mas tem mais atípicos e distâncias um pouco menos nítidas (3.8-b). As buscas de vizinhos usam kd-tree ou pivôs no espaço PCA, e HNSW quando uma resposta aproximada basta. P1 deve ser executado com e sem os usuários atípicos, e seus índices devem ser comparados com os de partições de dados uniformes.
