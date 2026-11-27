# %% [markdown]
# ## 3. Transformação
#
# A transformação prepara a representação que P1 usará:
#
# - **3.1** — construção do perfil por usuário: as avaliações são convertidas em proporções, e o desvio-padrão da nota é medido em cada gênero;
# - **3.2** — seleção de características;
# - **3.3** — normalização;
# - **3.4** — transformação de características por PCA e escolha de d′ por quatro critérios;
# - **3.5** — efeito da dimensionalidade sobre as distâncias e dimensionalidade intrínseca, global e no público de cada gênero;
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
assert len(lista_generos) == 18 and not {"Unknown", "IMAX"} & set(lista_generos)  # 2.2 (IMAX) e 2.2-d (sem gênero)
assert len(perfil_pd) == perfil_pd.userId.nunique() == n_usuarios_total
assert perfil_pd[atributos_estruturados].notna().all().all()
assert perfil_pd[cols_prop].ge(0).all().all() and perfil_pd[cols_prop].le(1).all().all()
assert abs(perfil_pd["prop_avaliacoes_total"].sum() - 1) < 1e-10

# dp_<gênero> = desvio-padrão das notas do usuário nos filmes do gênero; só existe com 2 ou mais avaliações no gênero.
MIN_AVAL_DP = 2
cols_dp = ["dp_" + sufixo[g] for g in lista_generos]
dp_genero_pd = (ratings_genres.groupBy("userId", "genre")
    .agg(F.count("*").alias("n"), F.stddev("rating").alias("dp"))
    .filter(F.col("n") >= MIN_AVAL_DP)
    .groupBy("userId").pivot("genre", lista_generos).agg(F.first("dp"))
    .toPandas()
    .rename(columns={g: "dp_" + sufixo[g] for g in lista_generos}))
perfil_pd = perfil_pd.merge(dp_genero_pd, on="userId", how="left")
assert len(perfil_pd) == n_usuarios_total and perfil_pd[cols_dp].stack().ge(0).all()
# Conferência num caso: o desvio de Drama do primeiro usuário, recalculado das notas dele.
notas_u0 = np.array([r["rating"] for r in ratings_genres.filter(
    (F.col("userId") == int(perfil_pd["userId"].iloc[0])) & (F.col("genre") == "Drama")).select("rating").collect()])
if len(notas_u0) >= MIN_AVAL_DP:
    assert np.isclose(perfil_pd["dp_drama"].iloc[0], notas_u0.std(ddof=1))
    print(f"Conferência: usuário {int(perfil_pd['userId'].iloc[0])}, {len(notas_u0)} notas de Drama, "
          f"desvio-padrão {notas_u0.std(ddof=1):.4f} = dp_drama {perfil_pd['dp_drama'].iloc[0]:.4f}")

print(f"Perfil: {len(perfil_pd):,} usuários × {len(atributos_estruturados)} atributos, mais {len(cols_dp)} desvios-padrão "
      "por gênero (userId e n_avaliacoes ficam fora da matriz e servem para identificar e descrever grupos).")
print(perfil_pd[atributos_estruturados].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())
print(f"\nDesvio-padrão da nota por gênero (usuários com {MIN_AVAL_DP} ou mais avaliações no gênero):")
resumo_dp = perfil_pd[cols_dp].describe().T[["count", "mean", "std", "min", "50%", "max"]]
resumo_dp.insert(1, "%_usuarios", 100 * resumo_dp["count"] / len(perfil_pd))
print(resumo_dp.sort_values("count", ascending=False).round(4).to_string())

# %% [markdown]
# **Leitura dos resultados.** Em média, 44,3% das avaliações de um usuário são de filmes de Drama, 34,8% de Comedy e 31,4% de Action. Gêneros como Documentary (0,8%) e Film-Noir (0,9%) têm proporções pequenas. As proporções de gênero de um usuário somam mais que 1 porque os gêneros se sobrepõem. O desvio-padrão da nota por gênero existe para 99,9% dos usuários em Drama e para só 21,7% em Documentary e 23,6% em Film-Noir; a conferência recalcula o desvio de Drama do primeiro usuário a partir das 53 notas dele.

# %% [markdown]
# #### Histogramas dos atributos do perfil
#
# Todos os atributos do perfil aparecem como histograma, com a média e média ± 1 desvio-padrão marcadas. O número de avaliações e a participação no total usam eixo logarítmico, por causa da assimetria. Os desvios-padrão da nota em cada gênero (em verde) só existem para quem tem 2 ou mais avaliações no gênero; o n de cada painel mostra quantos usuários têm o valor. A tabela registra média, desvio-padrão, mediana, assimetria, a porcentagem de usuários com valor zero e a porcentagem com valor.

# %%
# [3.1-b] Histogramas dos atributos do perfil, com média e desvio-padrão
atributos_histograma = ["n_avaliacoes"] + atributos_estruturados + cols_dp
n_linhas = int(np.ceil(len(atributos_histograma) / 6))
fig, axes = plt.subplots(n_linhas, 6, figsize=(30, 4.3 * n_linhas))
for ax, atributo in zip(axes.flat, atributos_histograma):
    escala_log = atributo in ("n_avaliacoes", "prop_avaliacoes_total")
    if atributo in cols_dp:
        cor, rotulo = "#3f8f6b", "desvio-padrão da nota no gênero"
    else:
        cor = "#d18b24" if escala_log else "#4682a9"
        rotulo = "eixo log" if escala_log else ("proporção das avaliações do usuário" if atributo.startswith("prop_") else "nota")
    histograma_media_dp(ax, perfil_pd[atributo], atributo, bins=40, log_x=escala_log, cor=cor, rotulo_x=rotulo)
for ax in axes.flat[len(atributos_histograma):]:
    ax.axis("off")
fig.suptitle(f"Atributos do perfil por usuário (n = {len(perfil_pd):,}); média e ± 1 DP em vermelho; "
             "desvio-padrão da nota por gênero em verde", y=1.005)
plt.tight_layout()
plt.show()
resumo_perfil = perfil_pd[atributos_histograma].agg(["mean", "std", "median", "skew"]).T
resumo_perfil["%_zeros"] = 100 * (perfil_pd[atributos_histograma] == 0).mean()
resumo_perfil["%_com_valor"] = 100 * perfil_pd[atributos_histograma].notna().mean()
print(resumo_perfil.round(4).to_string())

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Volume:** o número de avaliações por usuário tem assimetria de 16,3 (média 153,8, mediana 71); a participação no total herda a mesma forma.
# - **Nota:** a nota média tem assimetria −0,56 (média 3,68); o DP da nota é zero para 0,15% dos usuários, que deram sempre a mesma nota.
# - **Gêneros comuns** (Drama, Comedy, Action, Thriller): distribuições próximas de simétricas, com menos de 0,4% de zeros.
# - **Gêneros raros:** a maioria dos usuários tem proporção zero em Documentary (62,8%) e Film-Noir (58,3%), e uma parte grande, em Western (32,3%) e Musical (22,4%). A cauda dos poucos usuários que avaliam esses gêneros explica os valores z extremos de 3.3.
# - **Desvio-padrão da nota por gênero (em verde):** a média vai de 0,63 (Film-Noir) a 0,93 (Comedy), sempre abaixo ou igual ao desvio geral médio (0,93): dentro de um gênero, o usuário varia as notas um pouco menos do que no conjunto. Os valores zero, de quem deu a mesma nota a todos os filmes do gênero, chegam a 6,6% em Animation e 6,3% em War e Musical, quase sempre em usuários com 2 ou 3 avaliações no gênero. A distribuição é assimétrica à direita (assimetria de 0,42 a 1,25).
#
# **Consequência:** os gêneros raros pedem tratamento próprio. O vetor revisado (3.8) os agrupa, e a atividade entra em escala logarítmica. O desvio por gênero falta para muitos usuários nos gêneros raros e é instável com poucas avaliações; por isso entra no vetor revisado numa forma relativa e definida para todos (3.8).

# %% [markdown]
# ### 3.2 Seleção de características
#
# O trabalho usa a abordagem de **filtro**, porque ela independe do método de aprendizado. *Wrappers* e *embedded* dependem de um algoritmo e acrescentariam um segundo viés.
#
# O CFS procura atributos correlacionados com a classe e pouco correlacionados entre si. Sem classe, só a segunda parte se aplica: os pares são ordenados pelo valor absoluto da correlação, em ordem decrescente. Fisher e ReliefF exigem classes. Depois de P1, eles podem usar os grupos encontrados como classe, mas para **descrever** quais atributos separam os grupos, não para escolher a entrada.
#
# **Critério registrado:** |Pearson| e |Spearman| ≥ 0,8 indicam redundância forte. O limiar é uma escolha explícita (na versão anterior era 0,9). Dois atributos têm decisão própria, registrada no código: `prop_positivas`, que repete a nota média e depende de um limiar de nota positiva ainda a validar, e `prop_avaliacoes_total`, que mede atividade. Nos demais pares redundantes, a regra é automática: sai o atributo com maior |Pearson| médio com os demais, porque é o que mais repete o resto do perfil.
#
# Os desvios-padrão da nota por gênero (3.1) descrevem o perfil, mas não entram nesta seleção nem no vetor base: faltam para quem tem menos de 2 avaliações no gênero (até 78% dos usuários nos gêneros raros). Eles entram no vetor revisado (3.8), numa forma definida para todos os usuários.

# %%
# [3.2] Seleção de características por filtro: redundância entre atributos
LIMIAR_REDUNDANCIA = 0.8
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
atributos_p1 = [a for a in atributos_estruturados if a not in decisoes]
# Pares redundantes sem decisão registrada: sai o atributo com maior |Pearson| médio com os demais atributos que restam.
while True:
    restantes = redundantes[redundantes["atributo_1"].isin(atributos_p1) & redundantes["atributo_2"].isin(atributos_p1)]
    if not len(restantes):
        break
    a, b = restantes.iloc[0][["atributo_1", "atributo_2"]]
    media_abs = (pearson_perfil.loc[atributos_p1, atributos_p1].abs().sum() - 1) / (len(atributos_p1) - 1)
    sai, fica = (a, b) if media_abs[a] >= media_abs[b] else (b, a)
    decisoes[sai] = (f"redundante com {fica} (Pearson {pearson_perfil.loc[sai, fica]:.3f}; Spearman "
                     f"{spearman_perfil.loc[sai, fica]:.3f}); maior |Pearson| médio com os demais "
                     f"({media_abs[sai]:.3f} contra {media_abs[fica]:.3f})")
    atributos_p1.remove(sai)
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
# **Leitura dos resultados.** Com o limiar de 0,8, dois pares passam nos dois critérios: `nota_media × prop_positivas` (Pearson 0,922; Spearman 0,946), que tem decisão própria, e `prop_animation × prop_children` (0,823 e 0,804). No segundo, sai `prop_children`, que tem o maior |Pearson| médio com os demais (0,267 contra 0,227): Animation fica e representa o público de animação e infantil no vetor base. O terceiro par mais associado, `prop_action × prop_adventure` (0,674 e 0,694), fica longe do limiar. O vetor base tem 19 atributos: nota média, desvio-padrão da nota e 17 proporções de gênero.

# %% [markdown]
# ### 3.3 Normalização: padronização × re-escala
#
# A escala muda o peso de cada atributo na distância e no erro do PCA. Duas opções são comparadas:
#
# - **Padronização z-score:** subtrai a média e divide pelo desvio-padrão. Todos os atributos passam a ter a mesma variância, então gêneros raros com pouca variação são ampliados.
# - **Re-escala min-max:** leva cada atributo para [0, 1] e preserva diferenças de variância entre atributos.
#
# A coluna `peso_var` mostra a variância do atributo dividida pela soma das variâncias. Ela indica a participação média do atributo na distância euclidiana ao quadrado entre dois usuários.
#
# **Escala usada a partir de 3.4: min-max.** O z-score dá a gêneros raros, com pouca variação, o mesmo peso dos gêneros comuns e cria valores extremos (a tabela abaixo mostra máximos de dezenas de desvios-padrão). O min-max mantém cada atributo em [0, 1] e preserva a diferença de variância entre eles. O z-score fica só como comparação nesta célula e no critério univariado |z| de 3.7-a.

# %%
# [3.3] Normalização: padronização z-score × re-escala min-max
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X_p1_original = perfil_pd[atributos_p1].astype(float)
assert X_p1_original.shape == (n_usuarios_total, len(atributos_p1)) and "prop_positivas" not in atributos_p1
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
# **Leitura dos resultados.** No z-score, cada atributo pesa 5,26% da variância total, mas os gêneros raros ganham valores extremos: máximos de 47,1 desvios-padrão em Documentary, 38,9 em Western e 38,2 em Film-Noir, e há 854 casos de |z| > 10 (usuário e gênero), quase todos em gêneros raros e médios (Horror 193, Film-Noir 160, Documentary 137). No min-max, os gêneros frequentes pesam mais na distância (Action 12,62%, Drama 11,07%, Comedy 10,04%) e os raros quase nada (Documentary 0,25%, Western 0,36%, Film-Noir 0,38%). É essa diferença de variância que o PCA em min-max preserva.

# %% [markdown]
# ### 3.4 PCA e escolha de d′
#
# O PCA é aplicado à matriz min-max. Cada atributo é centralizado na média, e os componentes são os autovetores da matriz de covariância, ordenados pelos autovalores. Os autovalores **em escala** (cada um dividido pela soma de todos) dizem que fração da variância total cada componente guarda e somam 1.
#
# O número de componentes d′ é escolhido por quatro critérios, de três tipos:
#
# - **gráfico:** scree plot com a variância acumulada; d′ é o menor número de componentes que soma 80% da variância (limiar registrado);
# - **estatístico, broken-stick:** compara cada autovalor em escala com o esperado se a variância fosse repartida ao acaso (distribuição broken-stick) e retém os componentes iniciais que ficam acima da curva. Se autovalores do fim também ficarem acima, isso é registrado à parte: no fim, a curva tende a zero mais depressa que os autovalores de atributos fracos e quase independentes, e esses cruzamentos não indicam estrutura;
# - **estatístico, Kaiser-Guttman:** retém os autovalores maiores ou iguais à média dos autovalores, que é o limiar unitário em escala;
# - **dimensão intrínseca:** estimada pela distribuição das distâncias entre usuários, sem depender dos eixos do PCA. A dimensão de correlação (fractal) D2 é a inclinação de log C(r) × log r, em que C(r) é a fração dos pares de usuários a uma distância de até r. O ρ-score é μ² / (2σ²), com μ e σ a média e o desvio-padrão das distâncias. Em cada estimador, d′ = ⌈D⌉.
#
# **Regra fixada antes da execução:** o PCA principal usa d′ = ⌈D2⌉, porque a dimensão intrínseca mede a estrutura dos dados, e não só a variância dos eixos. O broken-stick, o mais compacto, é a alternativa nas seções seguintes. Os quatro critérios são comparados nas distâncias (3.5), e a escolha final entre os espaços é de P1.

# %%
# [3.4] PCA sobre a matriz min-max e os critérios para d′
import math
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

LIMIAR_VARIANCIA, N_AMOSTRA_D2, N_AMOSTRA_RHO, SEMENTES_RHO = 0.80, 5000, 2000, range(5)
FAIXA_D2 = (-3.0, -0.7)  # raios do quantil 0,1% ao 20% das distâncias: região de escala do ajuste


def dimensao_correlacao(X, semente=42, n=N_AMOSTRA_D2, faixa=FAIXA_D2):
    """D2: inclinação de log C(r) × log r numa amostra fixa; C(r) = fração dos pares com distância até r."""
    X = np.asarray(X, dtype=float)
    amostra = X[np.random.default_rng(semente).choice(len(X), min(n, len(X)), replace=False)]
    dist = np.sort(pdist(amostra))
    dist = dist[dist > 0]
    raios = np.quantile(dist, np.logspace(faixa[0], faixa[1], 25))
    C = np.searchsorted(dist, raios, side="right") / len(dist)
    return float(np.polyfit(np.log(raios), np.log(C), 1)[0]), raios, C


def rho_score(X, sementes=SEMENTES_RHO, n=N_AMOSTRA_RHO):
    """ρ = μ² / (2σ²) dos pares de distância; mediana entre amostras fixas."""
    X = np.asarray(X, dtype=float)
    valores = []
    for s in sementes:
        dist = pdist(X[np.random.default_rng(s).choice(len(X), min(n, len(X)), replace=False)])
        dist = dist[dist > 0]
        valores.append(float(dist.mean() ** 2 / (2 * dist.var())))
    return float(np.median(valores)), valores


def criterios_pca(X_minmax):
    """PCA da matriz min-max (centralizada; autovalores da covariância) e o d′ de cada critério."""
    pca = PCA(svd_solver="full").fit(X_minmax)
    autoval = pca.explained_variance_
    d = len(autoval)
    escala = autoval / autoval.sum()
    acum = np.cumsum(escala)
    broken = np.array([sum(1 / j for j in range(i, d + 1)) / d for i in range(1, d + 1)])
    acima = escala > broken
    d2, raios, C = dimensao_correlacao(X_minmax)
    rho, rhos = rho_score(X_minmax)
    criterios = {
        "gráfico: 80% da variância": int(np.searchsorted(acum, LIMIAR_VARIANCIA) + 1),
        "broken-stick": max(1, int(np.argmax(~acima)) if (~acima).any() else d),  # componentes iniciais acima da curva
        "Kaiser-Guttman": int((autoval / autoval.mean() >= 1).sum()),
        "dimensão intrínseca: D2": min(d, math.ceil(d2)),
        "dimensão intrínseca: ρ-score": min(d, math.ceil(rho)),
    }
    return {"pca": pca, "autoval": autoval, "escala": escala, "acum": acum, "broken": broken, "acima": acima,
            "d2": d2, "raios": raios, "C": C, "rho": rho, "rhos": rhos, "criterios": criterios}


CORES_CRITERIO = {"gráfico: 80% da variância": "#4682a9", "broken-stick": "#d18b24", "Kaiser-Guttman": "#7656a5",
                  "dimensão intrínseca: D2": "crimson", "dimensão intrínseca: ρ-score": "#3f8f6b"}


def grafico_criterios(ax_scree, ax_acum, res, titulo):
    """Scree plot com broken-stick e Kaiser-Guttman, e variância acumulada com o d′ de cada critério."""
    d = len(res["escala"])
    pos = np.arange(1, d + 1)
    ax_scree.bar(pos, 100 * res["escala"], color="#9db8cc", label="autovalor em escala")
    ax_scree.plot(pos, 100 * res["broken"], "o-", ms=3, color="#d18b24", label="broken-stick")
    ax_scree.axhline(100 / d, ls="--", color="#7656a5", label="Kaiser-Guttman: média dos autovalores")
    ax_acum.plot(pos, 100 * res["acum"], "s-", ms=3, color="#4682a9", label="variância acumulada")
    ax_acum.axhline(100 * LIMIAR_VARIANCIA, ls=":", color="gray")
    for nome, v in res["criterios"].items():
        ax_scree.axvline(v + 0.5, color=CORES_CRITERIO[nome], ls=":", lw=1.6)
        ax_acum.plot(v, 100 * res["acum"][v - 1], "o", color=CORES_CRITERIO[nome], ms=9, alpha=.8,
                     label=f"{nome}: d′ = {v}")
    ax_scree.set(title=f"Scree plot: {titulo}", xlabel="componente (linhas pontilhadas: d′ de cada critério)",
                 ylabel="% da variância")
    ax_acum.set(title=f"Variância acumulada e d′ de cada critério: {titulo}", xlabel="componentes", ylabel="%",
                ylim=(0, 102))
    ax_scree.legend(fontsize=8)
    ax_acum.legend(fontsize=8, loc="lower right")
    for ax in (ax_scree, ax_acum):
        ax.grid(alpha=.2)


def imprimir_criterios(res):
    d = len(res["escala"])
    tabela = pd.DataFrame({"componente": np.arange(1, d + 1), "autovalor": res["autoval"],
                           "em escala (%)": 100 * res["escala"], "acumulada (%)": 100 * res["acum"],
                           "broken-stick (%)": 100 * res["broken"], "acima do broken-stick": res["acima"],
                           "autovalor / média": res["autoval"] / res["autoval"].mean()})
    print(tabela.round(4).to_string(index=False))
    print(f"\nD2 = {res['d2']:.2f}; ρ-score = {res['rho']:.2f} (mediana de {len(res['rhos'])} amostras: "
          + ", ".join(f"{v:.2f}" for v in res["rhos"]) + ")")
    print("d′ por critério:")
    for nome, v in res["criterios"].items():
        print(f"  {nome}: {v} ({100 * res['acum'][v - 1]:.2f}% da variância)")
    cauda = [int(i) + 1 for i in np.flatnonzero(res["acima"]) if i + 1 > res["criterios"]["broken-stick"]]
    if cauda:
        print(f"  broken-stick: autovalores do fim também acima da curva: {cauda} (a regra do último acima daria {max(cauda)})")
    else:
        print("  broken-stick: nenhum autovalor do fim acima da curva (as regras dos primeiros acima e do último acima coincidem)")


res_base = criterios_pca(X_p1_minmax)
pca_completo, acumulada = res_base["pca"], res_base["acum"]
print(f"PCA sobre a matriz min-max ({X_p1_minmax.shape[1]} atributos), centralizada; autovalores da covariância")
imprimir_criterios(res_base)

# Sensibilidade da dimensão de correlação à faixa de raios do ajuste e à semente da amostra.
sens_d2 = pd.DataFrame([(f"quantis {100 * 10 ** f[0]:.2g}% a {100 * 10 ** f[1]:.2g}%", s,
                         dimensao_correlacao(X_p1_minmax, s, faixa=f)[0])
                        for f in [FAIXA_D2, (-3.5, -1.0), (-2.5, -0.5)] for s in (42, 7, 123)],
                       columns=["faixa de raios", "semente", "D2"])
print("\nSensibilidade de D2 (faixa de raios × semente da amostra de 5.000 usuários):")
print(sens_d2.pivot(index="faixa de raios", columns="semente", values="D2").round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(25, 5.8))
grafico_criterios(axes[0], axes[1], res_base, "vetor base")
ajuste = np.polyfit(np.log(res_base["raios"]), np.log(res_base["C"]), 1)
axes[2].loglog(res_base["raios"], res_base["C"], "o", color="crimson", ms=4, label="C(r) observado")
axes[2].loglog(res_base["raios"], np.exp(np.polyval(ajuste, np.log(res_base["raios"]))), "-", color="gray",
               label=f"reta ajustada: inclinação D2 = {res_base['d2']:.2f}")
axes[2].set(title="Dimensão de correlação: log C(r) × log r", xlabel="raio r",
            ylabel="C(r): fração dos pares com distância ≤ r")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=.2, which="both")
plt.tight_layout()
plt.show()

# Regra fixada antes: PCA principal com d′ = ⌈D2⌉; alternativa compacta: broken-stick.
d_pca, d_bs = res_base["criterios"]["dimensão intrínseca: D2"], res_base["criterios"]["broken-stick"]
assert 1 <= d_pca <= len(acumulada) and 1 <= d_bs <= len(acumulada)
print(f"\nd′ escolhido: {d_pca} componentes pela dimensão intrínseca (D2 = {res_base['d2']:.2f}); "
      f"alternativa compacta: {d_bs} pelo broken-stick")
cargas = pd.DataFrame(pca_completo.components_[:d_pca].T, index=atributos_p1,
                      columns=[f"CP{k}" for k in range(1, d_pca + 1)])
print("Atributos com maior carga absoluta em cada componente retido:")
for cp in cargas.columns:
    principais = cargas[cp].abs().sort_values(ascending=False).head(4).index
    print(f"{cp} ({100 * res_base['escala'][int(cp[2:]) - 1]:.2f}%): "
          + ", ".join(f"{a} ({cargas.loc[a, cp]:+.2f})" for a in principais))
projecao_base = pca_completo.transform(X_p1_minmax)
X_p1_pca = pd.DataFrame(projecao_base[:, :d_pca], columns=cargas.columns, index=perfil_pd.index)
X_p1_pca_bs = pd.DataFrame(projecao_base[:, :d_bs], columns=[f"CP{k}" for k in range(1, d_bs + 1)], index=perfil_pd.index)
print(f"Matriz candidata no espaço PCA: {X_p1_pca.shape}; variância retida = {100 * acumulada[d_pca - 1]:.2f}%")
print(f"Alternativa compacta (broken-stick): {X_p1_pca_bs.shape}; variância retida = {100 * acumulada[d_bs - 1]:.2f}%")

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Autovalores:** os três primeiros componentes guardam 33,36%, 20,70% e 10,85% da variância (64,91% juntos), e só eles ficam acima da curva broken-stick, o terceiro por pouco (10,85% contra 10,78%). Do 4º em diante, todos ficam abaixo, inclusive no fim: com min-max e covariância, os últimos autovalores não voltam a cruzar a curva.
# - **d′ de cada critério:**
#
#   | Critério | d′ | variância retida |
#   |---|---:|---:|
#   | estatístico: broken-stick | 3 | 64,91% |
#   | estatístico: Kaiser-Guttman (autovalor ≥ média) | 4 | 72,46% |
#   | dimensão intrínseca: ρ-score = 3,95 | 4 | 72,46% |
#   | gráfico: 80% da variância | 6 | 82,07% |
#   | dimensão intrínseca: D2 = 6,17 | 7 | 85,73% |
#
#   No Kaiser-Guttman, o 4º autovalor vale 1,43 vez a média, e o 5º, 0,97. O ρ-score varia de 3,87 a 4,14 entre as 5 amostras; a mediana (3,95) dá 4, mas uma amostra isolada poderia dar 5.
# - **Sensibilidade de D2:** a semente da amostra muda D2 em no máximo 0,08. A faixa de raios do ajuste muda mais: 6,9 com raios menores (quantis de 0,032% a 10%) e 5,5 com raios maiores (0,32% a 32%), ou seja, ⌈D2⌉ entre 6 e 7. Em raios pequenos, o ruído acrescenta dimensões; em raios grandes, o limite do domínio [0, 1] as reduz. A faixa registrada fica no meio. No gráfico, os pontos de log C(r) × log r formam uma curva suave, mais inclinada nos raios pequenos e menos nos grandes; por isso D2 depende da faixa, como mostra a tabela de sensibilidade.
# - **Componentes do PCA principal (7):**
#   - CP1 (33,4%): Action, Adventure e Sci-Fi contra Drama;
#   - CP2 (20,7%): Comedy contra Drama, Thriller e Crime;
#   - CP3 (10,8%): desvio-padrão da nota contra nota média, ou seja, quem varia mais as notas contra quem dá notas altas;
#   - CP4 (7,5%): Comedy contra desvio-padrão da nota e Drama;
#   - CP5 (5,1%): Horror contra Action; CP6 (4,5%): Sci-Fi contra Crime, Animation e Fantasy; CP7 (3,7%): nota média e desvio-padrão contra Romance e Thriller.
#
# **Interpretação:** os critérios vão de 3 a 7 componentes. Os estatísticos e o ρ-score são os mais compactos (3 e 4); o gráfico e D2 guardam mais (6 e 7). Pela regra fixada, o PCA principal tem 7 componentes (85,73% da variância), e o do broken-stick, com 3 (64,91%), é a alternativa compacta. Com o min-max, 7 componentes guardam muito mais variância do que o PCA do z-score guardava com 6 (66,84% na versão anterior), porque os gêneros raros, de pouca variância, quase não pesam.

# %% [markdown]
# ### 3.5 Dimensionalidade e concentração de distâncias
#
# Em alta dimensionalidade, as distâncias tendem a se concentrar: a menor e a maior ficam proporcionalmente próximas, o que prejudica métodos baseados em distância. Como P1 usará distâncias entre usuários, a célula mede a distribuição de pares de distância em uma amostra fixa de 2.000 usuários, na matriz min-max e no PCA com o d′ de cada critério de 3.4 (critérios que dão o mesmo d′ dividem o espaço). As medidas usadas são:
#
# - **Contraste relativo** = (máx − mín) / mín;
# - **Variância relativa** = desvio-padrão / média das distâncias.
#
# Valores menores indicam mais concentração.

# %%
# [3.5] Concentração de distâncias: min-max e o PCA com o d′ de cada critério
N_AMOSTRA_DIST = 2000
idx_dist = perfil_pd.sample(n=N_AMOSTRA_DIST, random_state=42).index.to_numpy()


def espacos_por_criterio(res, X_minmax):
    """A matriz min-max e o PCA com o d′ de cada critério (critérios com o mesmo d′ dividem o espaço)."""
    projecao = res["pca"].transform(X_minmax)
    por_d = {}
    for nome, v in res["criterios"].items():
        por_d.setdefault(v, []).append(nome)
    espacos = {f"min-max ({X_minmax.shape[1]} atributos)": np.asarray(X_minmax, dtype=float)}
    for v in sorted(por_d):
        espacos[f"PCA ({v}): " + "; ".join(por_d[v])] = projecao[:, :v]
    return espacos


espacos = espacos_por_criterio(res_base, X_p1_minmax)
linhas_dist, distancias = [], {}
for nome, X in espacos.items():
    d = pdist(X[idx_dist], metric="euclidean")
    d = d[d > 0]  # pares idênticos não entram no contraste relativo
    distancias[nome] = d
    linhas_dist.append((nome, X.shape[1], len(d), d.mean(), d.std(), d.min(), d.max(),
                        (d.max() - d.min()) / d.min(), d.std() / d.mean()))
tabela_dist = pd.DataFrame(linhas_dist, columns=["espaço", "dimensões", "pares", "média", "DP", "mín", "máx",
                                                 "contraste_relativo", "variancia_relativa"])
print(f"Amostra fixa de {N_AMOSTRA_DIST:,} usuários (semente 42); distância euclidiana")
print(tabela_dist.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
cores_pca = iter(["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"])
for nome, d in distancias.items():
    ax.hist(d / d.mean(), bins=80, histtype="step", density=True, linewidth=1.5, label=nome,
            color="tab:orange" if nome.startswith("min-max") else next(cores_pca))
ax.set(title="Distribuição de pares de distância (normalizada pela média): min-max e PCA com o d′ de cada critério",
       xlabel="distância / média das distâncias", ylabel="densidade")
ax.legend(fontsize=8)
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# | Espaço | Dimensões | Contraste relativo | Variância relativa |
# |---|---:|---:|---:|
# | min-max | 19 | 23,10 | 0,3550 |
# | PCA: broken-stick | 3 | 651,43 | 0,4750 |
# | PCA: Kaiser-Guttman e ρ-score | 4 | 176,25 | 0,4358 |
# | PCA: 80% da variância | 6 | 73,95 | 0,4065 |
# | PCA: dimensão intrínseca D2 | 7 | 72,23 | 0,3905 |
#
# - **Ordem:** quanto menos componentes, menos concentradas as distâncias: a variância relativa sobe de 0,355 no min-max para 0,475 com 3 componentes. No gráfico, a curva do min-max (laranja) é a mais estreita em torno da média, e as curvas do PCA se alargam à medida que d′ diminui.
# - **Contraste relativo:** depende só do par mais próximo e do mais distante. Com 3 componentes, o par mais próximo fica quase idêntico (distância 0,0026), e o contraste chega a 651; por isso a variância relativa, que usa todos os pares, é a medida principal.
#
# **Consequência:** cada critério troca nitidez das distâncias por informação. O PCA do broken-stick separa melhor perto de longe, mas descarta 35% da variância; o PCA principal (7 componentes) guarda 85,73% e concentra um pouco menos as distâncias que o min-max.

# %% [markdown]
# #### Dimensionalidade intrínseca global
#
# A dimensionalidade intrínseca estima quantas dimensões efetivas descrevem a estrutura dos dados, independentemente do número de atributos. Além da dimensão de correlação D2 e do ρ-score de 3.4, a célula usa o estimador de máxima verossimilhança de Levina–Bickel (MLE), que parte das distâncias aos k vizinhos mais próximos (k = 10 a 20). Os três são medidos na matriz min-max e em cada espaço PCA de 3.5, em amostras fixas. Num espaço PCA que guarda a estrutura, a dimensão intrínseca fica perto da do min-max; num espaço que corta estrutura, ela cai.

# %%
# [3.5-b] Dimensionalidade intrínseca global: D2, ρ-score e máxima verossimilhança (Levina–Bickel)
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
idx_id = perfil_pd.sample(n=N_AMOSTRA_ID, random_state=42).index.to_numpy()
linhas_id = []
for nome, X in espacos.items():
    media_id, por_k = dimensao_intrinseca_mle(X[idx_id])
    linhas_id.append((nome, X.shape[1], dimensao_correlacao(X)[0], rho_score(X)[0], media_id, min(por_k), max(por_k)))
tabela_id = pd.DataFrame(linhas_id, columns=["espaço", "dimensões", "D2", "ρ-score", "MLE (média k=10..20)",
                                             "MLE mín", "MLE máx"])
print(f"D2: amostra fixa de {N_AMOSTRA_D2:,} usuários; ρ-score: mediana de {len(SEMENTES_RHO)} amostras de "
      f"{N_AMOSTRA_RHO:,}; MLE: amostra fixa de {N_AMOSTRA_ID:,} (semente 42)")
print(tabela_id.round(2).to_string(index=False))

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Min-max (19 atributos):** D2 = 6,17, ρ-score = 3,95 e MLE = 11,94 (de 11,41 a 12,70, conforme k). Os três medem em escalas diferentes: o ρ-score usa a dispersão de todas as distâncias, D2 os raios pequenos e médios, e o MLE os 10 a 20 vizinhos mais próximos, onde a variação fina entre usuários acrescenta dimensões. Todos ficam bem abaixo dos 19 atributos.
# - **Espaços PCA:** as estimativas caem com d′. No PCA principal (7), D2 = 4,89 e MLE = 7,23; no do broken-stick (3), D2 = 2,83 e MLE = 3,30. O PCA com 7 componentes guarda a maior parte da estrutura de escala média, mas a estrutura fina medida pelo MLE cai de 11,9 para 7,2: parte da variação local fica nos componentes descartados.
#
# **Consequência:** a dimensão intrínseca global confirma que os dados não cabem em 2 ou 3 dimensões. O PCA principal fica perto da dimensão de correlação do min-max (7 componentes para D2 = 6,17), como a regra pretendia.

# %% [markdown]
# #### Dimensionalidade intrínseca local: o público de cada gênero
#
# A dimensionalidade intrínseca global resume todos os usuários. A local pergunta se o público de um gênero precisa de mais ou menos componentes que o conjunto. O desenho evita três armadilhas:
#
# - **definição do público:** os 10% de usuários com maior proporção de avaliações no gênero. Assim, todo público tem o mesmo tamanho, e um usuário pode estar em mais de um. O gênero dominante não serve, porque deixa Drama com mais da metade dos usuários e gêneros pequenos com poucas dezenas;
# - **tamanho e atividade:** as estimativas dependem do número de pontos e da atividade (quem avalia pouco tem proporções mais ruidosas). Cada estimativa usa 2.000 usuários do público, um quinto em cada quintil de atividade da população, repetida em 5 amostras;
# - **efeito de qualquer seleção:** escolher os 10% mais altos num atributo já muda a dimensão. Duas referências medem isso: usuários sorteados entre todos e um placebo, os 10% de maior nota média, que não tem relação com gênero.
#
# A régua é a mesma para todos: a matriz min-max global de 3.3. Em cada amostra: D2, MLE, d′ = ⌈D2⌉ e a fração da variância no primeiro componente de um PCA feito dentro da amostra. **Regra fixada antes:** um gênero só precisa de outro número de componentes se a faixa do seu ⌈D2⌉ entre as amostras não se sobrepõe à dos usuários sorteados e a diferença de D2 para os sorteados passa da do placebo.

# %%
# [3.5-c] Dimensionalidade intrínseca local: o público de cada gênero, com amostras pareadas por atividade
N_LOCAL, AMOSTRAS_LOCAL, TOPO = 2000, 5, 0.10
X_local = X_p1_minmax.to_numpy()
quintil_ativ = pd.qcut(np.log10(perfil_pd["n_avaliacoes"]), 5, labels=False, duplicates="drop").to_numpy()


def amostra_pareada(membros, rng, n=N_LOCAL):
    """n usuários do público, a mesma quantidade em cada quintil global de atividade."""
    por_quintil = [membros[quintil_ativ[membros] == q] for q in np.unique(quintil_ativ)]
    k = min(n // len(por_quintil), min(len(x) for x in por_quintil))
    return np.concatenate([rng.choice(x, k, replace=False) for x in por_quintil])


def medidas_locais(Z):
    d2 = dimensao_correlacao(Z, n=len(Z))[0]
    return d2, dimensao_intrinseca_mle(Z)[0], math.ceil(d2), 100 * PCA().fit(Z).explained_variance_ratio_[0]


todos = np.arange(len(perfil_pd))
publicos = {"usuários sorteados (referência)": todos}
for g in lista_generos:
    v = perfil_pd["prop_" + sufixo[g]].to_numpy()
    publicos[g] = todos[(v >= np.quantile(v, 1 - TOPO)) & (v > 0)]
v = perfil_pd["nota_media"].to_numpy()
publicos["placebo: 10% de maior nota média"] = todos[v >= np.quantile(v, 1 - TOPO)]
rng_local = np.random.default_rng(42)
linhas_local = []
for nome, membros in publicos.items():
    amostras = [amostra_pareada(membros, rng_local) for _ in range(AMOSTRAS_LOCAL)]
    m = np.array([medidas_locais(X_local[a]) for a in amostras])
    linhas_local.append((nome, len(membros), len(amostras[0]), int(np.median(perfil_pd["n_avaliacoes"].to_numpy()[membros])),
                         np.median(m[:, 0]), m[:, 0].min(), m[:, 0].max(), np.median(m[:, 1]),
                         int(np.median(m[:, 2])), int(m[:, 2].min()), int(m[:, 2].max()), np.median(m[:, 3])))
tabela_local = pd.DataFrame(linhas_local, columns=["público", "usuários", "por amostra", "mediana de avaliações",
                                                   "D2", "D2 mín", "D2 máx", "MLE", "d′ = ⌈D2⌉", "d′ mín", "d′ máx",
                                                   "% no 1º componente"])
print(f"Público = 10% com maior proporção no gênero; {AMOSTRAS_LOCAL} amostras pareadas por atividade; "
      "medianas entre as amostras; matriz min-max global")
print(tabela_local.round(2).to_string(index=False))

ref, placebo = tabela_local.iloc[0], tabela_local.iloc[-1]
generos_local = tabela_local.iloc[1:-1]
diferentes = generos_local[((generos_local["d′ mín"] > ref["d′ máx"]) | (generos_local["d′ máx"] < ref["d′ mín"]))
                           & ((generos_local["D2"] - ref["D2"]).abs() > abs(placebo["D2"] - ref["D2"]))]
print(f"\nReferência: d′ = {ref['d′ = ⌈D2⌉']} ({ref['d′ mín']} a {ref['d′ máx']}); D2 = {ref['D2']:.2f}. "
      f"Efeito do placebo em D2: {placebo['D2'] - ref['D2']:+.2f}. "
      f"D2 dos gêneros: de {generos_local['D2'].min():.2f} ({generos_local.loc[generos_local['D2'].idxmin(), 'público']}) "
      f"a {generos_local['D2'].max():.2f} ({generos_local.loc[generos_local['D2'].idxmax(), 'público']}).")
print("Gêneros que precisam de outro número de componentes, pela regra fixada:",
      ", ".join(diferentes["público"]) if len(diferentes) else "nenhum")

fig, ax = plt.subplots(figsize=(12, 6.5))
ordem = tabela_local.sort_values("D2").reset_index(drop=True)
for i, linha in ordem.iterrows():
    cor = "gray" if linha["público"] == ref["público"] else ("#d18b24" if linha["público"] == placebo["público"] else "crimson")
    ax.errorbar(linha["D2"], i, xerr=[[linha["D2"] - linha["D2 mín"]], [linha["D2 máx"] - linha["D2"]]], fmt="o", color=cor, capsize=3)
    ax.text(linha["D2 máx"] + 0.03, i, f"d′ = {linha['d′ = ⌈D2⌉']}", va="center", fontsize=8)
ax.axvspan(ref["D2 mín"], ref["D2 máx"], color="gray", alpha=.15, label="faixa dos usuários sorteados")
ax.set_yticks(range(len(ordem)), ordem["público"])
ax.set(title="Dimensão de correlação D2 no público de cada gênero (mediana e faixa entre as amostras)", xlabel="D2")
ax.legend(fontsize=8)
ax.grid(axis="x", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Número de componentes:** o público de todos os 18 gêneros precisa de 7 componentes (d′ = ⌈D2⌉), como os usuários sorteados, nas 5 amostras de cada um. Western é o único com uma amostra em 6 (D2 de 5,92 a 6,03). Pela regra fixada, **nenhum gênero precisa de outro número de componentes**.
# - **Tamanho das diferenças:** D2 vai de 5,97 (Western) a 6,57 (Mystery), contra 6,20 nos usuários sorteados; no gráfico, 17 dos 18 públicos ficam um pouco à direita da faixa dos sorteados, e só Western fica à esquerda. O placebo, os 10% de maior nota média, muda D2 em −0,36. As diferenças entre gêneros (de −0,23 a +0,37) têm o mesmo tamanho do efeito de selecionar 10% dos usuários por qualquer atributo.
# - **O que muda de um gênero para outro é a forma:** a fração da variância no primeiro componente vai de 27,0% (Mystery), 27,4% (Sci-Fi) e 27,6% (Romance) a 35,0% (War), 35,7% (Drama) e 37,5% (Adventure), contra 34,0% nos sorteados. No público de Adventure, Drama e War, um eixo só concentra mais a variação; no de Mystery, Sci-Fi e Romance, ela se espalha por mais direções.
# - **Atividade:** a mediana de avaliações vai de 38 (Thriller) a 79 (Horror). O pareamento por quintil de atividade evita que essa diferença apareça como diferença de dimensão.
#
# **Consequência:** para o número de componentes, a análise local confirma o d′ = 7 global: não é preciso um d′ por gênero. A caracterização de cada gênero vem da forma (o peso do primeiro eixo) e, depois de P1, dos grupos. A versão "depois do agrupamento", com a dimensão dentro de cada grupo de P1 comparada com subconjuntos sorteados do mesmo tamanho, entra na validação de P1.

# %% [markdown]
# ### 3.6 Visualização do espaço transformado
#
# Projeções 2D de uma amostra fixa de usuários:
#
# - **PCA:** os dois primeiros componentes;
# - **MDS:** preserva as distâncias entre pares, calculado em 500 usuários por custo computacional;
# - **t-SNE:** preserva vizinhanças locais, aplicado sobre o PCA principal e sobre a alternativa do broken-stick;
# - **coordenadas paralelas:** os atributos do vetor base em min-max, em 100 usuários, para que as linhas possam ser seguidas.
#
# Antes de P1 não há grupos. Por isso, a cor indica o grupo da nota média do usuário, só como referência visual.

# %%
# [3.6] Visualização do espaço transformado: PCA 2D, MDS, t-SNE e coordenadas paralelas
from sklearn.manifold import MDS, TSNE

idx_vis = perfil_pd.sample(n=3000, random_state=7).index
grupo_media_vis = np.ceil(perfil_pd.loc[idx_vis, "nota_media"]).clip(1, 5).astype(int).to_numpy()
pca_2d = X_p1_pca.loc[idx_vis, ["CP1", "CP2"]].to_numpy()
tsne_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X_p1_pca.loc[idx_vis].to_numpy())
tsne_2d_bs = TSNE(n_components=2, perplexity=30, init="pca" if X_p1_pca_bs.shape[1] >= 2 else "random",  # PCA exige 2+ eixos
                  random_state=42).fit_transform(X_p1_pca_bs.loc[idx_vis].to_numpy())
idx_mds = idx_vis[:500]
mds_2d = MDS(n_components=2, n_init=1, max_iter=300, random_state=42).fit_transform(X_p1_minmax.loc[idx_mds].to_numpy())

fig, axes = plt.subplots(1, 4, figsize=(26, 6))
projecoes = [
    (axes[0], pca_2d, grupo_media_vis, f"PCA: CP1 × CP2 ({len(idx_vis):,} usuários)"),
    (axes[1], mds_2d, grupo_media_vis[:len(idx_mds)], f"MDS: {X_p1_minmax.shape[1]} atributos min-max ({len(idx_mds)} usuários)"),
    (axes[2], tsne_2d, grupo_media_vis, f"t-SNE: PCA com {X_p1_pca.shape[1]} componentes (D2)"),
    (axes[3], tsne_2d_bs, grupo_media_vis, f"t-SNE: PCA com {X_p1_pca_bs.shape[1]} componentes (broken-stick)"),
]
for ax, coords, cor, titulo in projecoes:
    pontos = ax.scatter(coords[:, 0], coords[:, 1], c=cor, cmap="viridis", s=6, alpha=.6, vmin=1, vmax=5)
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(pontos, ax=axes, shrink=.8, label="Grupo da nota média do usuário (1 a 5)")
plt.show()

# Coordenadas paralelas: 100 usuários da mesma amostra, atributos do vetor base em min-max ([0, 1]).
N_PARALELAS = 100
paralelas = X_p1_minmax.loc[idx_vis[:N_PARALELAS]].copy()
paralelas["grupo_nota_media"] = grupo_media_vis[:N_PARALELAS].astype(str)
fig, ax = plt.subplots(figsize=(20, 6))
pd.plotting.parallel_coordinates(paralelas.sort_values("grupo_nota_media"), "grupo_nota_media", ax=ax,
                                 colormap="viridis", alpha=.5, linewidth=.9)
ax.set_title(f"Coordenadas paralelas dos atributos em min-max ({N_PARALELAS} usuários; cor = grupo da nota média)")
ax.tick_params(axis="x", rotation=60)
ax.set_ylabel("valor min-max (0 a 1)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Projeções 2D:** PCA, MDS e t-SNE mostram uma nuvem contínua de usuários, sem grupos isolados. O t-SNE sobre o PCA do broken-stick (3 componentes) forma filamentos dentro da nuvem, efeito comum do t-SNE em poucas dimensões, mas também sem grupos separados.
# - **Nota média:** a cor não organiza a estrutura principal. Nos dois t-SNE, as médias mais altas (amarelo) ficam mais na borda, e há manchas locais de médias perto de 3 no interior.
# - **Coordenadas paralelas (100 usuários, min-max):** as linhas se separam sobretudo na nota média e nas proporções de Comedy, Drama e Thriller; nos gêneros raros (Documentary, Film-Noir, Western), quase todas ficam perto de 0. Os usuários de média alta (amarelo) começam no alto do primeiro eixo, mas se misturam aos demais nos eixos de gênero.
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
# - **(b) multivariado:** distância de Mahalanobis ao centro no espaço PCA principal, comparada ao quantil 0,999 da χ² com d′ graus de liberdade (referência sob normalidade multivariada);
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
# - **(a) |z| > 3:** 28.962 usuários (17,82%) passam do limite em pelo menos um atributo. Nos dez atributos com mais casos, a fração vai de 1,19% a 1,86% (Animation, Sci-Fi, Horror, Musical e Fantasy lideram), bem acima dos 0,27% esperados numa normal. As proporções de gêneros raros são muito assimétricas: 12,0 em Documentary e 9,6 em Film-Noir. Com caudas tão longas, a regra univariada marca a cauda de cada atributo, não casos isolados, e não serve sozinha como critério.
# - **(b) Mahalanobis:** no PCA principal (7 componentes), 3.899 usuários (2,40%) passam do quantil 0,999 da χ² com 7 graus de liberdade (24,32), cerca de 24 vezes os 163 esperados sob normalidade. A nuvem de usuários tem caudas mais pesadas que uma normal multivariada.
# - **(c) Vizinhança:** o limite do percentil 99 (0,211) é 2,4 vezes a mediana da distância ao 10º vizinho (0,088).
# - **Atípicos marcados:** 1.156 usuários (0,71%) estão longe do centro e isolados ao mesmo tempo. Outros 2.743 estão longe do centro, mas têm vizinhos próximos (possíveis nichos), e só 470 estão isolados perto do centro.
# - **Quem são:** o atributo mais extremo é, em geral, um gênero de frequência média ou a nota: Horror (214 usuários), Animation (150), desvio-padrão da nota (145), Sci-Fi (125) e nota média (85). Eles avaliam menos filmes (mediana de 30 contra 71) e dão notas um pouco mais altas (mediana da média 3,78 contra 3,70). Com poucas avaliações, poucos filmes de um gênero bastam para gerar uma proporção extrema.
# - **Mudança em relação ao z-score:** antes, os gêneros raros dominavam os atípicos (Documentary, Film-Noir, Western e War), porque o z-score os ampliava. No min-max, esses gêneros quase não pesam, e os atípicos passam a ser os usuários extremos em gêneros de peso médio e na forma de dar notas.
# - **Gráfico:** no plano CP1 × CP2 os atípicos se espalham pela borda, mas também se sobrepõem ao centro, porque se afastam também nos componentes 3 a 7, que carregam a nota, Horror, Sci-Fi e Animation.
#
# **Limitação:** os limiares (quantil 0,999 e percentil 99) são escolhas registradas, e a referência χ² supõe normalidade, que os dados não seguem.
#
# **Consequência:** os 1.156 usuários são mantidos e ficam marcados em `perfil_pd["atipico"]`, para que P1 seja executado com e sem eles.

# %% [markdown]
# #### Tendência de agrupamento
#
# Um algoritmo de agrupamento sempre devolve grupos, mesmo em dados sem estrutura. A estatística de Hopkins verifica, antes de P1, se os usuários se distribuem de forma diferente de pontos uniformes no mesmo domínio:
#
# - **u:** distância de um ponto uniforme ao usuário real mais próximo;
# - **w:** distância de um usuário real sorteado ao seu vizinho real mais próximo;
# - **H = Σu / (Σu + Σw):** H ≈ 0,5 indica dados indistinguíveis de pontos uniformes; H próximo de 1 indica dados concentrados em regiões.
#
# O cálculo usa 1.000 pontos por repetição e 10 repetições, no PCA principal, na matriz min-max e no PCA do broken-stick, com e sem os atípicos. Como os valores extremos alargam o domínio mín–máx e inflam H, o domínio dos percentis 1–99 é usado como versão robusta. Um controle com dados uniformes de mesmo tamanho deve dar H ≈ 0,5.
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
ROTULO_PCA, ROTULO_MM = f"PCA ({X_p1_pca.shape[1]})", f"min-max ({X_p1_minmax.shape[1]})"
ROTULO_BS = f"PCA broken-stick ({X_p1_pca_bs.shape[1]})"
cenarios_hopkins = [
    (ROTULO_PCA, "todos", X_p1_pca.to_numpy()),
    (ROTULO_PCA, "sem atípicos", X_p1_pca.to_numpy()[sem_atipicos]),
    (ROTULO_MM, "todos", X_p1_minmax.to_numpy()),
    (ROTULO_MM, "sem atípicos", X_p1_minmax.to_numpy()[sem_atipicos]),
    (ROTULO_BS, "todos", X_p1_pca_bs.to_numpy()),
    (ROTULO_BS, "sem atípicos", X_p1_pca_bs.to_numpy()[sem_atipicos]),
]
linhas_hopkins = []
for espaco, populacao, X in cenarios_hopkins:
    for dominio in ["min-max", "percentis 1–99"]:
        h_medio, h_dp, u, w = hopkins(X, dominio)
        linhas_hopkins.append((espaco, populacao, dominio, len(X), h_medio, h_dp))
        if (espaco, populacao, dominio) == (ROTULO_PCA, "todos", "percentis 1–99"):
            u_vis, w_vis = u, w

# Controle: dados uniformes com o mesmo tamanho, no domínio dos percentis 1–99 de cada espaço.
for espaco, X in [(ROTULO_PCA, X_p1_pca.to_numpy()), (ROTULO_MM, X_p1_minmax.to_numpy()), (ROTULO_BS, X_p1_pca_bs.to_numpy())]:
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
ax.set(title=f"Distância ao vizinho mais próximo no PCA principal: usuários × pontos uniformes (m = {M_HOPKINS:,})",
       xlabel="log10(distância)", ylabel="pontos")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Controle:** com dados uniformes, H = 0,500 nos três espaços, o que confirma o cálculo.
# - **Domínio mín–máx:** H = 0,895 no PCA principal (7), 0,926 no min-max (19) e 0,946 no PCA do broken-stick (3). Esses valores são inflados pelos extremos, que alargam o domínio dos pontos uniformes: sem os 1.156 atípicos, caem para 0,881, 0,925 e 0,936.
# - **Domínio robusto (percentis 1–99):** H = 0,678 no PCA principal, 0,752 no min-max e 0,654 no PCA do broken-stick; sem os atípicos, 0,674, 0,748 e 0,654. O desvio-padrão entre repetições é no máximo 0,012.
# - **Distâncias:** no histograma, a distância de um usuário ao vizinho real mais próximo (w) fica, em geral, abaixo da distância de um ponto uniforme ao usuário mais próximo (u), mas as duas distribuições se sobrepõem em parte.
#
# **Interpretação:** os usuários estão mais concentrados do que pontos uniformes, e isso não é efeito dos atípicos, porque retirá-los quase não muda o H robusto. Mas a tendência é moderada: uma referência usual é H > 0,75 para tendência clara; o min-max fica no limite, e os espaços PCA ficam abaixo dela. Junto com as projeções de 3.6, o quadro é de um núcleo denso com caudas, e não de grupos naturalmente separados.
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
# - **grupos do perfil:** Animation e Children formam um grupo só, `Animation/Children`, porque têm o mesmo público em grande parte. Os rótulos dos dados não mudam: a junção vale só neste vetor. O IMAX já saiu na limpeza (2.2), por ser formato de exibição;
# - **proporção fracionária:** cada avaliação distribui peso 1/k entre os k grupos do filme, e o vetor de gêneros soma 1;
# - **limite de gêneros:** grupos com menos de 2% das avaliações formam o grupo `raros`;
# - **atributos derivados:** viés do usuário (nota menos a média do filme), atividade em escala log, popularidade média e ano médio dos filmes avaliados, e diversidade de gêneros (entropia normalizada);
# - **desvio-padrão da nota em cada grupo, na forma relativa:** quanto o usuário varia mais ou menos as notas naquele grupo do que no geral, `dprel_<grupo>` = n / (n + 5) · (desvio no grupo − desvio geral), com n as avaliações do usuário no grupo, e 0 com menos de 2 avaliações. O fator n / (n + 5) puxa para 0 quem tem poucas avaliações no grupo, porque um desvio calculado com 2 ou 3 notas é instável. A forma relativa evita repetir o desvio geral: o desvio bruto por gênero, completado com o desvio geral onde falta, fica muito correlacionado com `nota_dp`. Os grupos são os 12 mantidos e `raros`, que têm mais avaliações por usuário que os gêneros isolados;
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
# Grupos do perfil: Animation e Children viram um grupo só (o IMAX já saiu na 2.2).
GRUPO_DO_GENERO = {"Animation": "Animation/Children", "Children": "Animation/Children"}
grupo_do = lambda g: GRUPO_DO_GENERO.get(g, g)
grupos_perfil = sorted({grupo_do(g) for g in lista_generos} - {None})
coluna_grupo = {gr: "frac_" + gr.lower().replace("-", "_").replace("/", "_") for gr in grupos_perfil}
mapa_grupo = spark.createDataFrame([(g, grupo_do(g)) for g in lista_generos if grupo_do(g)], ["genre", "grupo"])
filme_grupo = movie_genres.select("movieId", "genre").join(F.broadcast(mapa_grupo), "genre").select("movieId", "grupo").distinct()
grupos_por_filme = filme_grupo.groupBy("movieId").agg(F.count("*").alias("n_grupos_filme"))
filme_grupo = filme_grupo.join(grupos_por_filme, "movieId").cache()
assert movies_clean.join(grupos_por_filme, "movieId", "left_anti").limit(1).count() == 0  # todo filme tem um grupo
assert len(grupos_perfil) == len(lista_generos) - 1 and not any("/" in c for c in coluna_grupo.values())
print(f"Grupos do perfil: {len(grupos_perfil)} (os {len(lista_generos)} gêneros, com Animation e Children juntos)")


def pesos_do_filme(generos):
    """Regra conferível à mão: cada avaliação do filme distribui peso 1/k entre os k grupos distintos do filme."""
    grupos = sorted({grupo_do(g) for g in generos.split("|")} - {None})
    return {gr: 1 / len(grupos) for gr in grupos}


assert pesos_do_filme("Action|Adventure") == {"Action": 0.5, "Adventure": 0.5}
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

# Exemplo calculado: um filme com Animation e Children (Toy Story, se estiver no catálogo).
catalogo_ex = (movies_clean.select("movieId", "title", "genres").join(filmes_info.select("movieId", "n_filme"), "movieId")
               .orderBy(F.desc("n_filme"), "movieId").toPandas())
exemplos = []
for titulo, condicao in [("Toy Story (1995)", lambda s: "Animation" in s and "Children" in s)]:
    achados = catalogo_ex[catalogo_ex["title"] == titulo]
    exemplos.append(achados.iloc[0] if len(achados) else catalogo_ex[catalogo_ex["genres"].map(condicao)].iloc[0])
pesos_spark = filme_grupo.filter(F.col("movieId").isin([int(e["movieId"]) for e in exemplos])).toPandas()
print("\nExemplo calculado (peso de cada avaliação do filme):")
for e in exemplos:
    generos = e["genres"].split("|")
    v8 = pesos_do_filme(e["genres"])
    no_spark = pesos_spark[pesos_spark["movieId"] == e["movieId"]]
    assert set(no_spark["grupo"]) == set(v8) and np.allclose(1 / no_spark["n_grupos_filme"], 1 / len(v8))
    print(f"  {e['title']} ({e['genres']}): por gênero, 1/{len(generos)} para cada um; "
          f"por grupo, " + ", ".join(f"{gr} 1/{len(v8)}" for gr in v8))

# Proporção fracionária: cada avaliação distribui peso 1/k entre os k grupos do filme; o vetor de gêneros soma 1.
frac_pd = (ratings_clean.select("userId", "movieId")
    .join(F.broadcast(filme_grupo), "movieId")
    .groupBy("userId").pivot("grupo", grupos_perfil).agg(F.sum(1.0 / F.col("n_grupos_filme")))
    .toPandas().fillna(0.0)
    .rename(columns=coluna_grupo))
cols_frac = [coluna_grupo[gr] for gr in grupos_perfil]
massa_genero = frac_pd[cols_frac].sum() / frac_pd[cols_frac].sum().sum()  # participação de cada grupo nas avaliações
perfil_rev_pd = (perfil_pd[["userId", "n_avaliacoes", "nota_media", "nota_dp", "atipico"]].reset_index(drop=True)
                 .merge(derivados_pd, on="userId", how="left").merge(frac_pd, on="userId", how="left"))
assert len(perfil_rev_pd) == len(perfil_pd)
assert (perfil_rev_pd["userId"].to_numpy() == perfil_pd["userId"].to_numpy()).all()
perfil_rev_pd.index = perfil_pd.index
peso_total = perfil_rev_pd[cols_frac].sum(axis=1)  # = avaliações do usuário (todo filme tem pelo menos um grupo)
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

# Desvio-padrão relativo por grupo (12 mantidos e raros): n/(n + M) · (desvio no grupo − desvio geral); 0 com n < 2.
# Uma avaliação conta no grupo se o filme tem algum gênero dele (contagem inteira, sem a fração 1/k).
M_ENCOLHIMENTO = 5
grupos_dp = generos_mantidos + ["raros"]
coluna_dprel = {gr: "dprel_" + ("raros" if gr == "raros" else coluna_grupo[gr][len("frac_"):]) for gr in grupos_dp}
cols_dprel = [coluna_dprel[gr] for gr in grupos_dp]
mapa_final = spark.createDataFrame([(gr, gr if destino[gr] == "mantido" else "raros") for gr in grupos_perfil],
                                   ["grupo", "grupo_final"])
filme_grupo_final = filme_grupo.join(F.broadcast(mapa_final), "grupo").select("movieId", "grupo_final").distinct()
dp_grupo_pd = (ratings_clean.select("userId", "movieId", "rating")
    .join(F.broadcast(filme_grupo_final), "movieId")
    .groupBy("userId", "grupo_final").agg(F.count("*").alias("n"), F.stddev("rating").alias("dp_grupo"))
    .toPandas()
    .merge(perfil_pd[["userId", "nota_dp"]], on="userId"))
dp_grupo_pd["dprel"] = np.where(dp_grupo_pd["n"] >= 2, dp_grupo_pd["n"] / (dp_grupo_pd["n"] + M_ENCOLHIMENTO)
                                * (dp_grupo_pd["dp_grupo"] - dp_grupo_pd["nota_dp"]), 0.0)
dprel_pd = (dp_grupo_pd.pivot(index="userId", columns="grupo_final", values="dprel")
            .reindex(index=perfil_rev_pd["userId"], columns=grupos_dp).fillna(0.0).rename(columns=coluna_dprel))
perfil_rev_pd = perfil_rev_pd.join(pd.DataFrame(dprel_pd.to_numpy(), columns=cols_dprel, index=perfil_rev_pd.index))
# Conferência num caso: Drama do primeiro usuário, recalculado das notas dele.
u0 = int(perfil_rev_pd["userId"].iloc[0])
notas_u0 = np.array([r["rating"] for r in ratings_clean.filter(F.col("userId") == u0)
                     .join(filme_grupo_final.filter(F.col("grupo_final") == "Drama"), "movieId").select("rating").collect()])
if "Drama" in grupos_dp and len(notas_u0) >= 2:
    esperado = len(notas_u0) / (len(notas_u0) + M_ENCOLHIMENTO) * (notas_u0.std(ddof=1) - perfil_rev_pd["nota_dp"].iloc[0])
    assert np.isclose(perfil_rev_pd["dprel_drama"].iloc[0], esperado)
    print(f"\nConferência do desvio relativo: usuário {u0}, Drama: n = {len(notas_u0)}, desvio no grupo "
          f"{notas_u0.std(ddof=1):.4f}, desvio geral {perfil_rev_pd['nota_dp'].iloc[0]:.4f} → "
          f"{len(notas_u0)}/({len(notas_u0)} + {M_ENCOLHIMENTO}) × ({notas_u0.std(ddof=1):.4f} − "
          f"{perfil_rev_pd['nota_dp'].iloc[0]:.4f}) = {esperado:.4f}")
print("Usuários com 2 ou mais avaliações em cada grupo (%):",
      {gr: round(100 * (dp_grupo_pd.loc[dp_grupo_pd["grupo_final"] == gr, "n"] >= 2).sum() / len(perfil_rev_pd), 1)
       for gr in grupos_dp})

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
                  + [coluna_grupo[gr] for gr in generos_mantidos] + ["frac_raros"] + cols_dprel)
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
                        cor="#7656a5" if coluna in ATRIBUTOS_DERIVADOS else ("#3f8f6b" if coluna in cols_dprel else "#4682a9"),
                        rotulo_x="proporção fracionária" if coluna.startswith("frac_") else
                        ("desvio no grupo − desvio geral (encolhido)" if coluna in cols_dprel else ""))
for ax in list(axes.flat)[len(colunas_hist):]:
    ax.axis("off")
fig.suptitle("Candidatos do vetor revisado (roxo = atributos derivados; verde = desvio relativo por grupo); "
             "média e ± 1 DP em vermelho", y=1.005)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Grupos do perfil:** 17 grupos (os 18 gêneros, com Animation e Children juntos). Participação fracionária: Drama 20,8%, Comedy 17,1%, Action 9,6%, Thriller 9,2%, Adventure 7,3% e Romance 6,8% lideram. Doze grupos ficam acima do limite de 2%.
# - **Animation/Children:** com 3,34%, o grupo fica acima do limite. Na v7, Animation (1,66%) caía em `raros`, junto com Film-Noir e Documentary, e Children (2,38%) ficava sozinho. O grupo tem menos que a soma dos dois (4,03%): contado uma vez só por filme, ele passa a outros gêneros o peso que antes ia duas vezes para o mesmo público. No exemplo, cada avaliação de Toy Story dava 1/5 a cada um dos 5 gêneros e passa a dar 1/4 a cada um dos 4 grupos.
# - **Sobreposição:** 77,6% das avaliações de Animation são de filmes também Children, mas só 59,6% das de Children são de filmes também Animation, e o Jaccard nos filmes é 0,21. Os dois não são equivalentes, mas a correlação entre usuários (Pearson 0,823; Spearman 0,804) passa do limiar de 0,8, e por isso `prop_children` sai do vetor base (3.2). Aqui, em vez de descartar um dos dois, a junção guarda os dois públicos num grupo só: é uma escolha de taxonomia.
# - **IMAX:** já saiu na limpeza (2.2). A fração que ele levava vai para os outros gêneros do filme (Avatar: de 1/4 para 1/3 em Action, Adventure e Sci-Fi), e o único filme só com IMAX recebeu gênero do IMDb (2.2-b); todo filme tem pelo menos um grupo.
# - **Gêneros raros:** War, Musical, Documentary, Western e Film-Noir somam 5,27% das avaliações e formam o grupo `raros`.
# - **Sensibilidade do limite:** com 1,5%, só War muda de lado (fica sozinho); com 2,5%, só Mystery (vai para `raros`). Nenhum outro grupo muda entre 1,5% e 2,5%, e o limite fica em 2%.
# - **Atributos derivados:**
#   - viés do usuário: média 0,08 e DP 0,44 (de −3,42 a 2,44). Em média, os usuários dão notas próximas da média de cada filme, mas alguns são sistematicamente rigorosos ou generosos;
#   - atividade em log: média 1,92 (cerca de 83 avaliações), DP 0,44;
#   - popularidade média: 4,04, ou seja, o filme típico visto por um usuário tem cerca de 11 mil avaliações;
#   - ano médio dos filmes vistos: 1994, com DP de 7,3 anos;
#   - diversidade de gêneros: 0,80, com DP 0,06.
# - **Desvio relativo por grupo:** o desvio no grupo existe (2 ou mais avaliações) para 71,2% dos usuários em Horror, 83,6% em Mystery, 83,8% em Animation/Children, 93,6% em `raros` e mais de 89% nos demais grupos; nos outros casos, o atributo vale 0. Na conferência, o usuário 1 tem 53 notas de Drama, desvio de 1,0477 no grupo e 1,0042 no geral: 53/58 × 0,0435 = 0,0397.
# - **Redundância:** o par mais associado é nota média × viés do usuário (Pearson 0,903; Spearman 0,881), que passa do limiar de 0,8. Sai `nota_media`, que tem o maior |Pearson| médio com os demais; o viés guarda o rigor do usuário sem o efeito dos filmes que ele escolheu. Os desvios relativos não são redundantes com nada. O vetor revisado tem 32 atributos: 6 de nota e de comportamento (desvio-padrão geral da nota, viés, atividade, popularidade, ano médio e diversidade), 13 proporções fracionárias e 13 desvios relativos.
# - **Gênero dominante:** Drama em 54,3% dos usuários, Comedy em 28,0% e Action em 8,2%.
#
# **Limitação:** o limite de 2% e os grupos são escolhas registradas. A animação adulta (anime, por exemplo) entra no grupo de animação e infantil, porque só 21% dos filmes têm os dois gêneros; os rótulos dos dados continuam separados.

# %% [markdown]
# #### Comparação entre o vetor base e o vetor revisado
#
# O vetor revisado passa pelo mesmo PCA de 3.4 (min-max, covariância) e pelos mesmos critérios de d′, com a mesma regra: o PCA principal usa d′ = ⌈D2⌉, e o broken-stick é a alternativa compacta. Os dois vetores são comparados nas mesmas amostras das seções anteriores: concentração de distâncias (3.5, 2.000 usuários), dimensionalidade intrínseca (3.5-b: D2 em 5.000 e MLE em 10.000), atípicos e estatística de Hopkins com as regras de 3.7. O vetor base continua disponível, e a escolha da entrada de P1 é registrada em 3.10.

# %%
# [3.8-b] Vetor base × vetor revisado: PCA, concentração de distâncias, dimensionalidade, atípicos e tendência
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy import stats

X_rev_original = perfil_rev_pd[atributos_rev].astype(float)
X_rev_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X_rev_original), columns=atributos_rev, index=perfil_pd.index)
res_rev = criterios_pca(X_rev_minmax)
pca_rev, acumulada_rev = res_rev["pca"], res_rev["acum"]
print(f"PCA do vetor revisado sobre a matriz min-max ({X_rev_minmax.shape[1]} atributos), centralizada; "
      "autovalores da covariância")
imprimir_criterios(res_rev)
d_rev, d_bs_rev = res_rev["criterios"]["dimensão intrínseca: D2"], res_rev["criterios"]["broken-stick"]
projecao_rev = pca_rev.transform(X_rev_minmax)
X_rev_pca = pd.DataFrame(projecao_rev[:, :d_rev], columns=[f"CP{k}" for k in range(1, d_rev + 1)], index=perfil_pd.index)
X_rev_pca_bs = pd.DataFrame(projecao_rev[:, :d_bs_rev], columns=[f"CP{k}" for k in range(1, d_bs_rev + 1)],
                            index=perfil_pd.index)
print(f"\nPCA do vetor revisado: d′ = {d_rev} (dimensão intrínseca, {100 * acumulada_rev[d_rev - 1]:.2f}% da variância); "
      f"alternativa compacta: {d_bs_rev} pelo broken-stick ({100 * acumulada_rev[d_bs_rev - 1]:.2f}%)")
cargas_rev = pd.DataFrame(pca_rev.components_[:d_rev].T, index=atributos_rev, columns=X_rev_pca.columns)
for cp in cargas_rev.columns:
    principais = cargas_rev[cp].abs().sort_values(ascending=False).head(4).index
    print(f"  {cp} ({100 * res_rev['escala'][int(cp[2:]) - 1]:.2f}%): "
          + ", ".join(f"{a} ({cargas_rev.loc[a, cp]:+.2f})" for a in principais))


def atipicos_no_espaco(X):
    """Mesma regra de 3.7-a: Mahalanobis² acima do quantil 0,999 da χ² e distância ao 10º vizinho acima do percentil 99."""
    X = np.asarray(X, dtype=float)
    centrado = X - X.mean(axis=0)
    d2 = (centrado ** 2 / centrado.var(axis=0)).sum(axis=1)
    dist_k10 = NearestNeighbors(n_neighbors=K_VIZ + 1, n_jobs=-1).fit(X).kneighbors(X)[0][:, K_VIZ]
    return (d2 > stats.chi2.ppf(0.999, df=X.shape[1])) & (dist_k10 > np.quantile(dist_k10, 0.99))


perfil_rev_pd["atipico_rev"] = atipicos_no_espaco(X_rev_pca)
espacos_comparacao = [
    ("base", "min-max", X_p1_minmax, np.nan, np.nan),
    ("base", "PCA (D2)", X_p1_pca, 100 * acumulada[d_pca - 1], int(perfil_pd["atipico"].sum())),
    ("base", "PCA (broken-stick)", X_p1_pca_bs, 100 * acumulada[d_bs - 1], int(atipicos_no_espaco(X_p1_pca_bs).sum())),
    ("revisado", "min-max", X_rev_minmax, np.nan, np.nan),
    ("revisado", "PCA (D2)", X_rev_pca, 100 * acumulada_rev[d_rev - 1], int(perfil_rev_pd["atipico_rev"].sum())),
    ("revisado", "PCA (broken-stick)", X_rev_pca_bs, 100 * acumulada_rev[d_bs_rev - 1],
     int(atipicos_no_espaco(X_rev_pca_bs).sum())),
]
linhas_comp = []
for vetor, espaco, X, var_retida, n_atipicos in espacos_comparacao:
    Xn = X.to_numpy()
    d = pdist(Xn[idx_dist], metric="euclidean")
    d = d[d > 0]
    id_media, _ = dimensao_intrinseca_mle(Xn[idx_id])
    h_medio, h_dp, _, _ = hopkins(Xn, "percentis 1–99")
    linhas_comp.append((vetor, espaco, X.shape[1], var_retida, (d.max() - d.min()) / d.min(), d.std() / d.mean(),
                        dimensao_correlacao(Xn)[0], id_media, h_medio, n_atipicos))
tabela_comparacao = pd.DataFrame(linhas_comp, columns=["vetor", "espaço", "dimensões", "variância retida %",
                                                       "contraste relativo", "variância relativa", "D2", "ID (MLE)",
                                                       "Hopkins (1–99)", "atípicos"])
print("\nComparação nas mesmas amostras de 3.5 (2.000 usuários), 3.5-b (D2: 5.000; MLE: 10.000) e 3.7 "
      "(Hopkins, m = 1.000, 10 repetições):")
print(tabela_comparacao.round(4).to_string(index=False))
ambos = int((perfil_pd["atipico"] & perfil_rev_pd["atipico_rev"]).sum())
uniao = int((perfil_pd["atipico"] | perfil_rev_pd["atipico_rev"]).sum())
print(f"Atípicos nos dois vetores: {ambos:,} (Jaccard = {ambos / uniao:.3f})")

# Transparência: o mesmo vetor revisado sem os desvios relativos por grupo, com os mesmos critérios e amostras.
sem_dprel = [a for a in atributos_rev if a not in cols_dprel]
X_rev_sem_dp = pd.DataFrame(MinMaxScaler().fit_transform(perfil_rev_pd[sem_dprel].astype(float)), columns=sem_dprel,
                            index=perfil_pd.index)
res_sem_dp = criterios_pca(X_rev_sem_dp)
d_sem_dp = res_sem_dp["criterios"]["dimensão intrínseca: D2"]
X_sem_dp_pca = res_sem_dp["pca"].transform(X_rev_sem_dp)[:, :d_sem_dp]
d = pdist(X_sem_dp_pca[idx_dist], metric="euclidean")
d = d[d > 0]
print(f"\nVetor revisado sem os desvios relativos ({len(sem_dprel)} atributos): D2 = {res_sem_dp['d2']:.2f}; d′ por critério: "
      + "; ".join(f"{nome} {v}" for nome, v in res_sem_dp["criterios"].items())
      + f". PCA (D2) com {d_sem_dp} componentes: {100 * res_sem_dp['acum'][d_sem_dp - 1]:.2f}% da variância; "
      f"variância relativa das distâncias {d.std() / d.mean():.4f}; Hopkins (1–99) {hopkins(X_sem_dp_pca, 'percentis 1–99')[0]:.4f}; "
      f"atípicos {int(atipicos_no_espaco(X_sem_dp_pca).sum()):,}")
z_rev = (X_rev_original - X_rev_original.mean()) / X_rev_original.std(ddof=0)  # só para descrever, como em 3.7-a
print("Atributo dominante dos atípicos do vetor revisado (maior |z|):",
      z_rev[perfil_rev_pd["atipico_rev"].to_numpy()].abs().idxmax(axis=1).value_counts().head(6).to_dict())

fig, axes = plt.subplots(2, 2, figsize=(25, 13))
grafico_criterios(axes[0, 0], axes[0, 1], res_rev, "vetor revisado")
n_atr = len(atributos_rev)
imagem = axes[1, 0].imshow(pearson_rev.loc[atributos_rev, atributos_rev], vmin=-1, vmax=1, cmap="coolwarm")
axes[1, 0].set_xticks(range(n_atr), atributos_rev, rotation=70, ha="right", fontsize=7)
axes[1, 0].set_yticks(range(n_atr), atributos_rev, fontsize=7)
axes[1, 0].set_title("Pearson entre os atributos do vetor revisado")
fig.colorbar(imagem, ax=axes[1, 0], shrink=.8)
dominante_vis = perfil_rev_pd.loc[idx_vis, "genero_dominante"]
principais_dom = dominante_vis.value_counts().head(6).index
for rotulo in list(principais_dom) + ["outros"]:
    mascara = (~dominante_vis.isin(principais_dom)) if rotulo == "outros" else (dominante_vis == rotulo)
    pontos = X_rev_pca.loc[idx_vis][mascara.to_numpy()]
    axes[1, 1].scatter(pontos["CP1"], pontos["CP2"], s=6, alpha=.55, label=f"{rotulo} ({mascara.sum()})")
axes[1, 1].set(title=f"Vetor revisado: CP1 × CP2 ({len(idx_vis):,} usuários; cor = gênero dominante)",
               xlabel="CP1", ylabel="CP2")
axes[1, 1].legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Critérios no vetor revisado (32 atributos):** broken-stick 4 componentes (48,10% da variância), Kaiser-Guttman 8 (64,29%), ρ-score 7,88 → 8, D2 = 8,25 → 9 (67,32%) e 80% da variância → 14. Pela regra fixada, o PCA principal tem 9 componentes, e o do broken-stick, 4.
# - **O "fim estranho" do broken-stick volta neste vetor:** os autovalores 24 a 31 ficam acima da curva, e a regra do último acima daria 31 componentes, quase todos. A causa são os 13 desvios relativos: colunas fracas e quase independentes (no mapa de correlação, o bloco de baixo à direita é quase todo neutro), que formam um piso de autovalores pequenos e parecidos, enquanto a curva broken-stick cai para perto de zero. Esses cruzamentos não indicam estrutura, e por isso a regra usa os componentes iniciais acima da curva. O último autovalor é zero porque as 13 proporções fracionárias somam 1.
# - **Componentes do PCA revisado:**
#   - CP1 (18,4%): Adventure, Action e Sci-Fi contra Drama;
#   - CP2 (11,4%): atividade, contra popularidade dos filmes vistos;
#   - CP3 (10,5%): Thriller, Crime e atividade contra Comedy;
#   - CP4 (7,8%): desvio-padrão geral da nota contra viés;
#   - CP5 (5,1%): Comedy e Thriller contra Drama;
#   - CP6 (3,9%) e CP7 (3,8%): os desvios relativos (variar mais as notas em aventura, ficção e fantasia; em fantasia e raros, contra mistério e suspense);
#   - CP8 (3,5%) e CP9 (3,0%): ano médio dos filmes vistos.
# - **Comparação** (mesmas amostras):
#
#   | | base, min-max | base, PCA 7 | base, PCA 3 | revisado, min-max | revisado, PCA 9 | revisado, PCA 4 |
#   |---|---:|---:|---:|---:|---:|---:|
#   | variância retida | — | 85,73% | 64,91% | — | 67,32% | 48,10% |
#   | variância relativa das distâncias | 0,355 | 0,391 | 0,475 | 0,250 | 0,310 | 0,393 |
#   | D2 | 6,17 | 4,89 | 2,83 | 8,25 | 5,90 | 3,61 |
#   | ID (MLE) | 11,9 | 7,2 | 3,3 | 17,6 | 8,8 | 4,3 |
#   | Hopkins (1–99) | 0,752 | 0,678 | 0,654 | 0,692 | 0,656 | 0,671 |
#   | atípicos | — | 1.156 | 612 | — | 1.348 | 603 |
#
# - **Atípicos:** 1.348 no vetor revisado, contra 1.156 no base; só 377 são atípicos nos dois (Jaccard 0,18). No revisado, o atributo mais extremo é Horror (128 usuários) ou um desvio relativo (Comedy 78, Thriller 77, Drama 76).
# - **Efeito dos desvios relativos (transparência):** sem eles, o vetor revisado tem 19 atributos, D2 = 6,41 e d′ = 7, que guardam 80,34% da variância; a variância relativa das distâncias é 0,344, o Hopkins 0,681 e há 1.070 atípicos. Com eles, são 9 componentes para 67,32%, distâncias mais concentradas (0,310), Hopkins 0,656 e 1.348 atípicos.
#
# **Interpretação:** o vetor revisado ganha representação: proporções que somam 1, atividade, popularidade, época e, agora, a forma como o usuário varia as notas em cada grupo. Mas os desvios relativos acrescentam muitas direções fracas: o vetor precisa de mais componentes para guardar menos variância, e as distâncias ficam mais concentradas. O vetor base em min-max, com 19 atributos, é o que mostra a tendência de agrupamento mais alta (0,752) e o PCA principal do vetor base é o que guarda mais variância (85,73% em 7 componentes).
#
# **Consequência:** a entrada principal registrada continua o PCA do vetor revisado (9 componentes), pelos ganhos de representação. O custo medido dos desvios relativos fica registrado, e P1 compara essa entrada com o PCA do vetor base (7 componentes) e com o vetor revisado sem os desvios (7 componentes), pela qualidade dos grupos.

# %% [markdown]
# ### 3.9 Solução indexada para buscas por similaridade
#
# P1 depende de buscas por similaridade: o DBSCAN faz buscas por abrangência (todos os vizinhos dentro de um raio), a escolha desse raio usa a distância ao k-ésimo vizinho, e o kNN busca os k mais próximos. Com 162.541 usuários, a varredura linear compara cada consulta com todos os objetos. A célula compara, nos seis espaços candidatos (PCA principal, matriz min-max e PCA do broken-stick, nos dois vetores):
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
    f"base min-max ({X_p1_minmax.shape[1]})": X_p1_minmax.to_numpy(),
    f"revisado PCA ({X_rev_pca.shape[1]})": X_rev_pca.to_numpy(),
    f"revisado min-max ({X_rev_minmax.shape[1]})": X_rev_minmax.to_numpy(),
    f"base PCA broken-stick ({X_p1_pca_bs.shape[1]})": X_p1_pca_bs.to_numpy(),
    f"revisado PCA broken-stick ({X_rev_pca_bs.shape[1]})": X_rev_pca_bs.to_numpy(),
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
# - **PCA principal do vetor base (7 dimensões):** a kd-tree é o índice exato mais rápido nas buscas kNN: 0,28 s para 1.000 buscas, contra 0,74 s da varredura linear. Nas buscas por abrangência, as duas empatam (0,085 s). A ball-tree não ganha da varredura (2,55 s).
# - **PCA principal do vetor revisado (9 dimensões):** a kd-tree perde a vantagem: 0,90 s contra 0,81 s da varredura nas buscas kNN, e 0,23 s contra 0,12 s nas buscas por abrangência.
# - **PCA do broken-stick (3 e 4 dimensões):** é onde os índices mais ganham: a kd-tree faz as 1.000 buscas kNN em 0,03 s, contra 0,75 a 0,95 s da varredura (25 a 32 vezes mais rápida), e as buscas por abrangência em 0,008 a 0,009 s.
# - **Espaços min-max (19 e 32 atributos):** as árvores ficam muito mais lentas que a varredura linear (no vetor base, kd-tree 4,9 s e ball-tree 15,2 s; no revisado, 23,7 s e 30,3 s; varredura de 1,0 a 1,1 s). É o efeito da dimensionalidade: os particionamentos deixam de descartar regiões.
# - **HNSW:** 1.000 buscas em 0,02 a 0,06 s com efSearch = 16, com recall@10 de 0,99 ou mais, exceto no min-max revisado (0,958; 0,993 com efSearch = 64). O custo fica na construção (7 a 18 s).
# - **Pivôs:** com 16 pivôs de seleção incremental, a busca por abrangência descarta 99,6% das distâncias no PCA base, 99,2% no PCA revisado e 99,98% nos espaços do broken-stick, contra 95,3% no min-max base e só 67,1% no min-max revisado, de 32 atributos. Na maioria dos casos, a seleção incremental supera a de variância máxima (com 8 pivôs no min-max revisado, 60,2% contra 50,0%).
#
# **Limitação:** os tempos dependem da máquina (Colab gratuito); a fração de distâncias evitadas não.
#
# **Consequência:** P1 usa os espaços PCA, onde os índices funcionam. No PCA principal do vetor base e nos espaços do broken-stick, as buscas exatas (DBSCAN e distância ao k-ésimo vizinho) usam a kd-tree; no PCA revisado, de 9 dimensões, as buscas exatas por abrangência usam pivôs com seleção incremental ou a varredura, e não a kd-tree. O HNSW atende as buscas aproximadas. Nos espaços min-max, só a varredura ou o HNSW.

# %% [markdown]
# ### 3.10 Resultado da transformação e entrada de P1
#
# | Objeto | Conteúdo | Uso em P1 |
# |---|---|---|
# | `perfil_pd` | `userId`, `n_avaliacoes`, 22 atributos do perfil e 18 desvios-padrão da nota por gênero | identificar e descrever os grupos |
# | `X_p1_minmax` | 19 atributos do vetor base (nota média, desvio-padrão e 17 proporções de gênero), em min-max | entrada candidata interpretável |
# | `X_p1_padronizado` | os mesmos 19 atributos em z-score | só comparação (3.3) e critério univariado de atípicos (3.7-a) |
# | `X_p1_pca` | 7 componentes principais do vetor base (dimensão intrínseca D2 = 6,17; 85,73%) | entrada candidata com dimensionalidade reduzida |
# | `X_p1_pca_bs`, `X_rev_pca_bs` | 3 e 4 componentes, pelo broken-stick (64,91% e 48,10%) | alternativa compacta a testar em P1 |
# | `perfil_pd["atipico"]` | 1.156 usuários longe do centro e isolados dos vizinhos (3.7) | executar P1 com e sem esses usuários |
# | `perfil_rev_pd` | atributos derivados, proporções fracionárias, desvios relativos por grupo, gênero dominante e `atipico_rev` (1.348 usuários) | descrever os grupos do vetor revisado |
# | `X_rev_minmax` | 32 atributos do vetor revisado, em min-max | entrada alternativa interpretável |
# | `X_rev_pca` | 9 componentes do vetor revisado (D2 = 8,25; 67,32%) | **entrada principal de P1** |
#
# **Resultados desta execução:**
#
# - **Pré-processamento que chega aqui:** o IMAX saiu da lista de gêneros na limpeza (2.2), e o perfil tem 18 gêneros.
# - **Perfil:** além das proporções, o desvio-padrão da nota em cada gênero, que existe para 21,7% (Documentary) a 99,9% (Drama) dos usuários.
# - **Seleção de características (limiar 0,8):** saem `prop_positivas` (decisão própria) e `prop_children`, redundante com `prop_animation` (Pearson 0,823; Spearman 0,804); o vetor base tem 19 atributos.
# - **Escalas:** o min-max passa a ser a base do PCA. No z-score, os gêneros raros chegam a 47,1 desvios-padrão (Documentary); no min-max, os gêneros frequentes pesam mais na distância (Action 12,62%, contra 0,25% de Documentary).
# - **PCA (min-max, covariância), d′ por critério:** broken-stick 3 (64,91%), Kaiser-Guttman 4 (72,46%), ρ-score 4, 80% da variância 6 (82,07%) e dimensão de correlação D2 = 6,17 → 7 (85,73%). Pela regra fixada, o PCA principal tem 7 componentes. Os dois primeiros explicam 54,06% e opõem ação, aventura e ficção a drama (CP1) e comédia a drama, suspense e crime (CP2).
# - **Concentração de distâncias** (2.000 usuários, distância euclidiana):
#
#   | Espaço | Contraste relativo | Variância relativa |
#   |---|---:|---:|
#   | min-max, 19 atributos | 23,10 | 0,3550 |
#   | PCA, 3 componentes (broken-stick) | 651,43 | 0,4750 |
#   | PCA, 4 componentes (Kaiser-Guttman e ρ-score) | 176,25 | 0,4358 |
#   | PCA, 6 componentes (80% da variância) | 73,95 | 0,4065 |
#   | PCA, 7 componentes (D2) | 72,23 | 0,3905 |
#
#   Quanto menos componentes, menos concentradas as distâncias, e mais variância descartada.
#
# - **Dimensionalidade intrínseca:** no min-max, D2 = 6,17, ρ-score = 3,95 e MLE = 11,94. Por gênero (3.5-c), o público de todos os 18 gêneros precisa de 7 componentes, como os usuários sorteados; as diferenças de D2 entre gêneros têm o tamanho do efeito de qualquer seleção de 10% dos usuários, e o que muda é a forma (o 1º componente guarda de 27,0% a 37,5% da variância do público).
# - **Valores atípicos:** 1.156 usuários (0,71%) estão longe do centro e isolados; em geral, são extremos em Horror, Animation ou na forma de dar notas, com poucas avaliações. Foram mantidos e marcados.
# - **Tendência de agrupamento:** estatística de Hopkins de 0,678 (PCA principal), 0,752 (min-max) e 0,654 (PCA do broken-stick) no domínio robusto, contra 0,500 nos dados uniformes de controle. Os usuários são mais concentrados que dados uniformes, mas a tendência é moderada.
# - **Vetor revisado (3.8):** proporções fracionárias, Animation e Children num grupo só, gêneros raros agrupados, cinco atributos derivados e o desvio relativo da nota em cada grupo; sai `nota_media`, redundante com o viés. Com 32 atributos, o PCA principal tem 9 componentes (67,32%); o broken-stick dá 4, e o fim do scree plot volta a cruzar a curva por causa dos desvios relativos. Sem os desvios, o mesmo vetor teria 7 componentes com 80,34%, distâncias menos concentradas e Hopkins um pouco maior.
# - **Solução indexada (3.9):** no PCA principal do vetor base, a kd-tree faz 1.000 buscas kNN em 0,28 s (varredura: 0,74 s); no PCA revisado, de 9 dimensões, perde a vantagem (0,90 s contra 0,81 s). Com 16 pivôs, as buscas por abrangência descartam 99,6% (base) e 99,2% (revisado) das distâncias, e o HNSW responde em centésimos de segundo com recall de 0,99 ou mais. Nos espaços min-max, as árvores ficam muito mais lentas que a varredura (até 30 s contra 1 s com 32 atributos).
#
# **Consequência para P1:** a entrada principal registrada é `X_rev_pca`, o PCA do vetor revisado, pelos ganhos de representação: proporções que somam 1, sem contagem dupla de gêneros, com atividade, popularidade, época e a forma como o usuário varia as notas em cada grupo. Como os desvios relativos cobram um preço medido (menos variância por componente e distâncias mais concentradas), P1 compara essa entrada com o PCA do vetor base (7 componentes) e com o vetor revisado sem os desvios (7 componentes), e com as alternativas do broken-stick. As buscas de vizinhos usam os índices medidos em 3.9. P1 deve ser executado com e sem os usuários atípicos, e seus índices devem ser comparados com os de partições de dados uniformes.
