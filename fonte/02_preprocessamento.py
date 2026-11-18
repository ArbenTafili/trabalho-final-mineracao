# %% [markdown]
# ## 2. Pré-processamento
#
# O objetivo é preparar os dados-alvo para a mineração. A etapa cobre:
#
# - qualidade dos dados: ruídos, outliers, valores ausentes, inconsistências e redundâncias;
# - integração das tabelas por atributos comuns;
# - eliminação de atributos irrelevantes;
# - amostragem.
#
# As transformações de atributos ficam na Seção 3.
#
# ### 2.1 Qualidade: completude, unicidade, validade e integridade

# %%
# [2.1] Auditoria de qualidade dos dados-alvo
VALID_RATINGS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

print("=== Completude: quantidade de nulos por atributo ===")
for nome, df in datasets.items():
    if nome == "tags":  # fora dos dados-alvo nesta fase
        continue
    nulos = df.agg(*[
        F.sum(F.when(F.col(coluna).isNull(), 1).otherwise(0)).alias(coluna)
        for coluna in df.columns
    ])
    print(f"\n{nome}:")
    nulos.show(truncate=False)

def resumo_duplicatas(df, chaves):
    repetidos = df.groupBy(*chaves).count().filter(F.col("count") > 1)
    linha = repetidos.agg(
        F.count("*").alias("chaves_duplicadas"),
        F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("linhas_excedentes"),
    ).first()
    return linha["chaves_duplicadas"], linha["linhas_excedentes"]

print("\n=== Unicidade no grão esperado ===")
duplicatas_por_tabela = {}
for nome, df, chaves in [
    ("ratings", ratings_df, ["userId", "movieId"]),
    ("movies", movies_df, ["movieId"]),
    ("links", links_df, ["movieId"]),
]:
    chaves_dup, linhas_excedentes = resumo_duplicatas(df, chaves)
    duplicatas_por_tabela[nome] = linhas_excedentes
    print(f"{nome}: {chaves_dup} chaves duplicadas; {linhas_excedentes} linhas excedentes")

notas_invalidas = ratings_df.filter(
    F.col("rating").isNull() | (~F.col("rating").isin(*VALID_RATINGS))
).count()
filmes_sem_genero = movies_df.filter(
    F.col("genres").isNull()
    | (F.trim(F.col("genres")) == "")
    | (F.col("genres") == "(no genres listed)")
).count()
avaliacoes_orfas = ratings_df.join(
    movies_df.select("movieId").dropDuplicates(), on="movieId", how="left_anti"
).count()
print("\n=== Validade e integridade ===")
print("Notas inválidas:", notas_invalidas)
print("Filmes sem gênero informado:", filmes_sem_genero)
print("Avaliações sem filme correspondente:", avaliacoes_orfas)

# %% [markdown]
# ### 2.2 Limpeza e tratamento
#
# Para valores ausentes, há quatro alternativas: ignorar, adaptar o algoritmo, descartar ou substituir/estimar.
#
# | Ocorrência | Classificação | Decisão |
# |---|---|---|
# | `ratings`: 0 nulos, 0 duplicatas e 0 notas inválidas | qualidade verificada | nenhuma avaliação removida; a regra de deduplicação (mais recente pelo `timestamp`) fica no código para reprodutibilidade |
# | 5.062 filmes `(no genres listed)` | valor ausente em atributo usado no perfil | rótulo `Unknown`; os que têm pelo menos 10 avaliações recebem gênero imputado (2.2-b) |
# | 107 links sem `tmdbId` | valor ausente em atributo irrelevante | ignorar: links não entram na análise |
# | Tags | fora dos dados-alvo | não tratadas nesta fase |
# | Notas 0,5 e 1,0 fora do whisker | outlier legítimo | manter |
# | Pares usuário–filme não avaliados | não são valores ausentes | não imputar; preencher criaria avaliações artificiais |

# %%
# [2.2] Limpeza determinística e conversão de timestamps
ratings_clean = (
    ratings_df
    .filter(
        F.col("userId").isNotNull()
        & F.col("movieId").isNotNull()
        & F.col("rating").isin(*VALID_RATINGS)
    )
)

# A auditoria evita ordenar 25 milhões de linhas quando não há duplicatas.
if duplicatas_por_tabela["ratings"]:
    ordem_avaliacao = Window.partitionBy("userId", "movieId").orderBy(
        F.col("timestamp").desc_nulls_last()
    )
    ratings_clean = (
        ratings_clean
        .withColumn("_ordem", F.row_number().over(ordem_avaliacao))
        .filter(F.col("_ordem") == 1)
        .drop("_ordem")
    )

ratings_clean = ratings_clean.withColumn(
    "rating_datetime", F.from_unixtime("timestamp").cast("timestamp")
).cache()
ratings_clean.count()  # materializa o cache para as estatísticas seguintes

movies_clean = (
    movies_df
    .filter(F.col("movieId").isNotNull() & F.col("title").isNotNull())
    .dropDuplicates(["movieId"])
    .withColumn(
        "genres",
        F.when(
            F.col("genres").isNull()
            | (F.trim(F.col("genres")) == "")
            | (F.col("genres") == "(no genres listed)"),
            F.lit("Unknown"),
        ).otherwise(F.col("genres")),
    )
)

links_clean = (
    links_df
    .filter(F.col("movieId").isNotNull())
    .dropDuplicates(["movieId"])
)

clean_datasets = {
    "ratings_clean": ratings_clean,
    "movies_clean": movies_clean,
    "links_clean": links_clean,
}
for nome, df in clean_datasets.items():
    df.createOrReplaceTempView(nome)
print("Views limpas criadas:", ", ".join(clean_datasets))

# %% [markdown]
# #### Imputação do gênero dos filmes sem gênero
#
# O gênero é usado no perfil, e os filmes sem gênero com avaliações levam essas avaliações para `Unknown`, que não representa preferência. Em vez de dar a moda (Drama) a todos, o gênero é estimado a partir de quem avaliou o filme:
#
# - **atributos do filme:** a média do perfil de gêneros dos seus avaliadores, calculada sem o próprio filme (para o gênero dele não vazar), dividida pela média geral (lift);
# - **métodos comparados:** moda, gênero de maior lift, Naive Bayes e kNN com 15 vizinhos, em validação cruzada de 5 partes nos filmes com gênero conhecido;
# - **medida:** acerto quando o gênero previsto está entre os gêneros verdadeiros do filme. O erro do Naive Bayes serve de taxa de ruído de referência;
# - **regra:** só filmes com pelo menos 10 avaliações recebem gênero, e só se o melhor método superar a moda por pelo menos 10 pontos percentuais. Os demais continuam `Unknown`.

# %%
# [2.2-b] Imputação do gênero dos filmes sem gênero, a partir do perfil de quem os avaliou
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

MIN_AVALIADORES = 10  # perfil de avaliadores só é confiável com pelo menos 10 avaliações do filme
K_IMPUTACAO = 15
generos_reais = sorted(r["genre"] for r in movies_clean.select(F.explode(F.split("genres", r"\|")).alias("genre"))
                       .distinct().collect() if r["genre"] != "Unknown")
cols_g = ["g_" + g.lower().replace("-", "_") for g in generos_reais]

# Vetor fracionário de cada filme (1/k para cada um dos seus k gêneros); filmes sem gênero têm vetor nulo.
filmes_vetor = (movies_clean.select("movieId", F.split("genres", r"\|").alias("gs"))
    .withColumn("conhecido", F.when(F.array_contains("gs", "Unknown"), 0).otherwise(1))
    .withColumn("k", F.size("gs"))
    .select("movieId", "conhecido",
            *[F.when(F.array_contains("gs", g), 1.0 / F.col("k")).otherwise(0.0).alias(c) for g, c in zip(generos_reais, cols_g)]))
avaliacoes_vetor = ratings_clean.select("userId", "movieId").join(F.broadcast(filmes_vetor), "movieId")
totais_usuario = avaliacoes_vetor.groupBy("userId").agg(F.sum("conhecido").alias("n_conhecidos"),
                                                        *[F.sum(c).alias("u_" + c) for c in cols_g])
# Perfil de cada avaliador SEM o próprio filme (deixa-um-fora), para o gênero do filme não vazar para o seu perfil.
denominador = F.col("n_conhecidos") - F.col("conhecido")
perfil_avaliadores_pd = (avaliacoes_vetor.join(totais_usuario, "userId")
    .where(denominador > 0)
    .select("movieId", *[F.when(denominador > 0, (F.col("u_" + c) - F.col(c)) / denominador).alias(c) for c in cols_g])
    .groupBy("movieId").agg(F.count("*").alias("n_avaliadores"), *[F.avg(c).alias(c) for c in cols_g])
    .join(movies_clean.select("movieId", "title", "genres"), "movieId")
    .toPandas())
perfil_avaliadores_pd["lista_generos"] = perfil_avaliadores_pd["genres"].str.split("|")

conhecidos = perfil_avaliadores_pd[(perfil_avaliadores_pd["genres"] != "Unknown")
                                   & (perfil_avaliadores_pd["n_avaliadores"] >= MIN_AVALIADORES)].reset_index(drop=True)
alvos = perfil_avaliadores_pd[(perfil_avaliadores_pd["genres"] == "Unknown")
                              & (perfil_avaliadores_pd["n_avaliadores"] >= MIN_AVALIADORES)].reset_index(drop=True)
sem_genero_avaliados = perfil_avaliadores_pd[perfil_avaliadores_pd["genres"] == "Unknown"]
print(f"Filmes sem gênero com avaliações: {len(sem_genero_avaliados):,}; com pelo menos {MIN_AVALIADORES} avaliações: "
      f"{len(alvos):,} ({int(alvos['n_avaliadores'].sum()):,} de {int(sem_genero_avaliados['n_avaliadores'].sum()):,} avaliações)")
print(f"Filmes com gênero conhecido usados para validar: {len(conhecidos):,}")

# Atributos: preferência dos avaliadores por gênero em relação à média (lift), padronizada.
media_generos = conhecidos[cols_g].mean()
lift_conhecidos = (conhecidos[cols_g] / media_generos).to_numpy()
padronizador = StandardScaler().fit(lift_conhecidos)
X_conhecidos = padronizador.transform(lift_conhecidos)
binarizador = MultiLabelBinarizer(classes=generos_reais)
Y_conhecidos = binarizador.fit_transform(conhecidos["lista_generos"])
pos_drama = generos_reais.index("Drama")

# Validação cruzada em 5 partes: acerto = gênero previsto está entre os gêneros verdadeiros do filme.
acertos = {"moda (sempre Drama)": [], "maior lift": [], "Naive Bayes": [], f"kNN (k={K_IMPUTACAO})": []}
faixa_teste = []
for treino, teste in KFold(n_splits=5, shuffle=True, random_state=42).split(X_conhecidos):
    linhas = np.arange(len(teste))
    Y_teste = Y_conhecidos[teste]
    acertos["moda (sempre Drama)"].append(Y_teste[:, pos_drama])
    acertos["maior lift"].append(Y_teste[linhas, lift_conhecidos[teste].argmax(axis=1)])
    nb = OneVsRestClassifier(GaussianNB()).fit(X_conhecidos[treino], Y_conhecidos[treino])
    acertos["Naive Bayes"].append(Y_teste[linhas, nb.predict_proba(X_conhecidos[teste]).argmax(axis=1)])
    _, vizinhos = NearestNeighbors(n_neighbors=K_IMPUTACAO).fit(X_conhecidos[treino]).kneighbors(X_conhecidos[teste])
    votos = Y_conhecidos[treino][vizinhos].sum(axis=1)
    acertos[f"kNN (k={K_IMPUTACAO})"].append(Y_teste[linhas, votos.argmax(axis=1)])
    faixa_teste.append(conhecidos.loc[teste, "n_avaliadores"].to_numpy())
faixa_teste = np.concatenate(faixa_teste)
faixas = [("10 a 49", (faixa_teste >= 10) & (faixa_teste < 50)), ("50 ou mais", faixa_teste >= 50)]
linhas_val = []
for metodo, lista in acertos.items():
    acerto = np.concatenate(lista)
    linhas_val.append((metodo, 100 * acerto.mean(), 100 * (1 - acerto.mean()),
                       *[100 * acerto[m].mean() for _, m in faixas]))
validacao = pd.DataFrame(linhas_val, columns=["método", "acerto_%", "erro_%"] + [f"acerto_% ({n} avaliações)" for n, _ in faixas])
print("\nValidação cruzada (5 partes) nos filmes com gênero conhecido:")
print(validacao.round(2).to_string(index=False))
print(f"Erro do Naive Bayes (taxa de ruído de referência): {validacao.set_index('método').loc['Naive Bayes', 'erro_%']:.2f}%")

# Decisão: imputar com o melhor método só se ele superar a moda por pelo menos 10 pontos percentuais.
modelos = validacao[~validacao["método"].str.startswith("moda")].sort_values("acerto_%", ascending=False)
melhor = modelos.iloc[0]
ganho = melhor["acerto_%"] - validacao.set_index("método").loc["moda (sempre Drama)", "acerto_%"]
imputar = ganho >= 10 and len(alvos) > 0
print(f"\nMelhor método: {melhor['método']} ({melhor['acerto_%']:.2f}% de acerto; ganho de {ganho:.2f} p.p. sobre a moda). "
      f"Imputar: {'sim' if imputar else 'não'}")

if imputar:
    X_alvos = padronizador.transform((alvos[cols_g] / media_generos).to_numpy())
    if melhor["método"].startswith("kNN"):
        _, vizinhos = NearestNeighbors(n_neighbors=K_IMPUTACAO).fit(X_conhecidos).kneighbors(X_alvos)
        pontuacao = Y_conhecidos[vizinhos].sum(axis=1) / K_IMPUTACAO
    elif melhor["método"] == "Naive Bayes":
        pontuacao = OneVsRestClassifier(GaussianNB()).fit(X_conhecidos, Y_conhecidos).predict_proba(X_alvos)
    else:
        pontuacao = (alvos[cols_g] / media_generos).to_numpy()
    alvos["genero_imputado"] = [generos_reais[i] for i in pontuacao.argmax(axis=1)]
    alvos["confianca"] = pontuacao.max(axis=1)
    imputados = spark.createDataFrame(alvos[["movieId", "genero_imputado"]].astype({"movieId": "int64"}))
    movies_clean = (movies_clean.join(F.broadcast(imputados), "movieId", "left")
        .withColumn("genero_foi_imputado", F.col("genero_imputado").isNotNull())
        .withColumn("genres", F.coalesce(F.col("genero_imputado"), F.col("genres")))
        .drop("genero_imputado"))
else:
    movies_clean = movies_clean.withColumn("genero_foi_imputado", F.lit(False))
movies_clean = movies_clean.cache()
movies_clean.createOrReplaceTempView("movies_clean")
n_imputados = movies_clean.filter("genero_foi_imputado").count()
print(f"Filmes com gênero imputado: {n_imputados:,}; continuam como Unknown: "
      f"{movies_clean.filter(F.col('genres') == 'Unknown').count():,}")
if imputar:
    print("\nGêneros imputados:", alvos["genero_imputado"].value_counts().to_dict())
    print("Exemplos (filmes sem gênero mais avaliados):")
    print(alvos.sort_values("n_avaliadores", ascending=False)[["title", "n_avaliadores", "genero_imputado", "confianca"]]
          .head(12).round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].barh(validacao["método"], validacao["acerto_%"], color=["#bbbbbb", "#4682a9", "#7656a5", "#d18b24"])
    axes[0].set(title="Acerto na validação cruzada (gênero previsto entre os verdadeiros)", xlabel="% de acerto", xlim=(0, 100))
    axes[0].grid(axis="x", alpha=.2)
    contagem = alvos["genero_imputado"].value_counts().sort_values()
    axes[1].barh(contagem.index, contagem.values, color="#d18b24")
    axes[1].set(title=f"Gênero imputado aos {len(alvos):,} filmes sem gênero", xlabel="filmes")
    axes[1].grid(axis="x", alpha=.2)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Validação:** em 24.010 filmes com gênero conhecido, o kNN acerta 83,4%, contra 47,3% da moda (sempre Drama): ganho de 36,1 pontos percentuais. O Naive Bayes acerta 69,6%, e seu erro, de 30,4%, é a taxa de ruído de referência. O gênero de maior lift acerta 39,7%, abaixo da moda: o lift divide pela média geral do gênero, o que amplia os gêneros raros.
# - **Suporte:** o acerto do kNN sobe de 78,1% nos filmes com 10 a 49 avaliações para 87,9% nos filmes com 50 ou mais. O acerto cresce com o número de avaliações, e por isso a imputação exige pelo menos 10.
# - **Resultado:** 320 filmes recebem gênero, e 4.742 continuam `Unknown` (494 sem avaliação e 4.248 com 1 a 9 avaliações). Os gêneros imputados mais frequentes são Drama (122 filmes), Comedy (56), Documentary (51) e Animation (27).
# - **Exemplos:** entre os filmes sem gênero mais avaliados, Whiplash recebe Drama, Hereditary recebe Horror, Pitch Perfect 3 recebe Comedy e Cosmos recebe Documentary. A confiança é a fração dos 15 filmes vizinhos que têm o gênero imputado; nos exemplos, vai de 0,60 a 1,00.
#
# **Limitação:** cada filme recebe um único gênero, o mais provável, enquanto os gêneros do catálogo se sobrepõem (as proporções de 1.6 somam mais de 100%). O acerto mede se o gênero previsto está entre os verdadeiros, não se a lista completa foi recuperada.
#
# **Consequência:** a imputação tira de `Unknown` 63% das avaliações de filmes sem gênero. A coluna `genero_foi_imputado` de `movies_clean` permite repetir qualquer análise sem os filmes imputados.

# %% [markdown]
# ### 2.3 Integração das tabelas
#
# Avaliações e filmes são integrados pela chave `movieId`. A junção deve preservar uma linha por avaliação. A expansão por gênero multiplica linhas de forma esperada e só é usada nas análises por gênero.

# %%
# [2.3-a] Comparação entre dados-alvo e dados tratados; liberação do cache da seleção
linhas_removidas = total_avaliacoes - ratings_clean.count()
print("Avaliações removidas na limpeza:", linhas_removidas)
if linhas_removidas == 0:
    print("Os resultados de 1.5–1.8 valem também para as avaliações tratadas; "
          "nos gêneros, os filmes sem gênero passam a `Unknown` ou recebem gênero imputado (2.2-b); 2.6 mostra o efeito.")
_ = ratings_sel.unpersist()  # ratings_clean já está materializado em cache (2.2)

# %%
# [2.3-b] Integração: joins no grão esperado e expansão por gênero
movie_genres = (
    movies_clean
    .withColumn("genre", F.explode(F.split(F.col("genres"), r"\|")))
    .select("movieId", "title", "genre")
)

ratings_movies = ratings_clean.join(
    movies_clean.select("movieId", "title", "genres"),
    on="movieId",
    how="inner",
)

ratings_genres = ratings_clean.join(
    movie_genres,
    on="movieId",
    how="inner",
)

movie_genres.createOrReplaceTempView("movie_genres")
ratings_movies.createOrReplaceTempView("ratings_movies")
ratings_genres.createOrReplaceTempView("ratings_genres")

linhas_ratings = ratings_clean.count()
linhas_ratings_movies = ratings_movies.count()
linhas_ratings_genres = ratings_genres.count()

print("Avaliações limpas:", linhas_ratings)
print("Avaliações após join com filmes:", linhas_ratings_movies)
print("Linhas após expansão por gênero:", linhas_ratings_genres)
print("Cobertura do join ratings-movies:", round(linhas_ratings_movies / linhas_ratings * 100, 4), "%")

assert linhas_ratings_movies == linhas_ratings, (
    "O join com movies alterou o grão das avaliações. Verifique chaves órfãs ou duplicadas."
)

print("Views derivadas criadas: movie_genres, ratings_movies e ratings_genres")

# %%
# [2.3-c] Verificações finais do pré-processamento
contagens_finais = spark.createDataFrame([
    ("usuários com avaliações", ratings_clean.select("userId").distinct().count()),
    ("filmes avaliados", ratings_clean.select("movieId").distinct().count()),
    ("filmes no catálogo", movies_clean.select("movieId").distinct().count()),
    ("gêneros", movie_genres.select("genre").distinct().count()),
], ["medida", "valor"])

contagens_finais.show(truncate=False)

assert ratings_clean.filter(
    F.col("userId").isNull()
    | F.col("movieId").isNull()
    | (~F.col("rating").isin(*VALID_RATINGS))
).limit(1).count() == 0

assert movies_clean.groupBy("movieId").count().filter(F.col("count") > 1).limit(1).count() == 0

print("Verificações concluídas: dados limpos, notas válidas e joins no grão esperado.")

# %% [markdown]
# ### 2.4 Amostragem
#
# Todas as medidas usam a base completa. As amostras servem apenas aos gráficos que ficariam pesados com 25 milhões de pontos: boxplot/violin com cerca de 100 mil avaliações e matriz de dispersão com 3.000 usuários. A amostra precisa ser representativa. Por isso, a célula compara as proporções e os momentos da amostra com os da população.

# %%
# [2.4] Representatividade das amostras usadas nos gráficos
populacao_pct = rating_distribution.select("rating", "proporcao_pct").toPandas().set_index("rating")["proporcao_pct"]
amostra_pct = amostra_pd["rating"].value_counts(normalize=True).mul(100).reindex(populacao_pct.index, fill_value=0)
comparacao_notas = pd.DataFrame({"populacao_%": populacao_pct, "amostra_%": amostra_pct.round(4)})
comparacao_notas["diferenca_pp"] = (comparacao_notas["amostra_%"] - comparacao_notas["populacao_%"]).round(4)
print(f"Amostra de notas: n={len(amostra_pd):,} (aleatória simples, semente 42)")
print(comparacao_notas.to_string())
print(f"Média: população={estatisticas_notas['media']:.4f}; amostra={amostra_pd['rating'].mean():.4f}")
print(f"DP: população={estatisticas_notas['desvio_padrao']:.4f}; amostra={amostra_pd['rating'].std(ddof=1):.4f}")
print(f"Maior diferença absoluta de proporção: {comparacao_notas['diferenca_pp'].abs().max():.4f} ponto percentual")

comparacao_usuarios = pd.DataFrame({
    "media_populacao": usuarios_pd[cols_grafico].mean(),
    "media_amostra": amostra_scatter.mean(),
    "dp_populacao": usuarios_pd[cols_grafico].std(ddof=1),
    "dp_amostra": amostra_scatter.std(ddof=1),
}).round(4)
print(f"Usuários: população n={len(usuarios_pd):,}; amostra da matriz de dispersão n={len(amostra_scatter):,}")
print(comparacao_usuarios.to_string())

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Amostra de notas:** as 100.099 notas diferem da população em no máximo 0,12 ponto percentual (nota 4,5). A média é 3,5329, contra 3,5339, e o desvio-padrão é igual (1,0607).
# - **Amostra de usuários:** nos 3.000 usuários, médias e desvios ficam próximos dos da população (por exemplo, `nota_media` 3,670 contra 3,679).
#
# As amostras são representativas para os gráficos em que foram usadas.

# %% [markdown]
# ### 2.5 Valores ausentes
#
# A tabela de avaliações não tem nulos (2.1). Os ausentes relevantes são as combinações usuário × filme sem avaliação. A célula mede a esparsidade dessa matriz e testa, num recorte de 500 usuários sorteados e dos 300 filmes mais avaliados, a alternativa de substituir os ausentes pela média da categoria (primeiro gênero do filme) ou pela média do filme. O efeito é medido no desvio-padrão das notas e na correlação entre usuários, comparada à correlação calculada só nos filmes que os dois avaliaram.

# %%
# [2.5] Valores ausentes: esparsidade da matriz usuário × filme e efeito de imputar pela média
n_usuarios_av = ratings_clean.select("userId").distinct().count()
n_filmes_av = ratings_clean.select("movieId").distinct().count()
celulas_matriz = n_usuarios_av * n_filmes_av
densidade = linhas_ratings / celulas_matriz
print(f"Matriz usuário × filme: {n_usuarios_av:,} usuários × {n_filmes_av:,} filmes avaliados = {celulas_matriz:,} células")
print(f"Células com nota: {linhas_ratings:,} ({100 * densidade:.4f}%); "
      f"sem nota: {celulas_matriz - linhas_ratings:,} ({100 * (1 - densidade):.4f}%)")
avaliacoes_por_usuario = ratings_clean.groupBy("userId").count().toPandas()["count"]
cobertura_pct = 100 * avaliacoes_por_usuario / n_filmes_av
print("Porcentagem dos filmes avaliados que cada usuário avaliou (mín, Q1, mediana, Q3, máx):",
      cobertura_pct.quantile([0, .25, .5, .75, 1]).round(4).tolist())

# Recorte para medir o efeito da imputação: 500 usuários sorteados e os 300 filmes mais avaliados.
N_USUARIOS_DEMO, N_FILMES_DEMO = 500, 300
ids_usuarios = np.sort(ratings_clean.select("userId").distinct().toPandas()["userId"].to_numpy())
usuarios_demo = np.sort(np.random.default_rng(42).choice(ids_usuarios, size=N_USUARIOS_DEMO, replace=False))
filmes_demo = (ratings_clean.groupBy("movieId").count()
               .orderBy(F.desc("count"), F.asc("movieId")).limit(N_FILMES_DEMO).cache())
ordem_filmes = filmes_demo.toPandas()["movieId"].tolist()
recorte = (ratings_clean.filter(F.col("userId").isin(usuarios_demo.tolist()))
           .join(F.broadcast(filmes_demo.select("movieId")), "movieId")
           .select("userId", "movieId", "rating").toPandas())
matriz_demo = (recorte.pivot_table(index="userId", columns="movieId", values="rating")
               .reindex(index=usuarios_demo, columns=ordem_filmes))

# "Categoria" do filme = primeiro gênero listado; a média da categoria usa todas as avaliações da base.
genero_principal = movies_clean.select("movieId", F.split(F.col("genres"), r"\|").getItem(0).alias("genero_principal"))
media_categoria = (ratings_clean.join(genero_principal, "movieId")
                   .groupBy("genero_principal").agg(F.avg("rating").alias("media_categoria")))
categoria_filmes = (genero_principal.join(filmes_demo.select("movieId"), "movieId")
                    .join(media_categoria, "genero_principal").toPandas().set_index("movieId")["media_categoria"])
media_filmes_demo = (ratings_clean.join(filmes_demo.select("movieId"), "movieId").groupBy("movieId")
                     .agg(F.avg("rating").alias("media_filme")).toPandas().set_index("movieId")["media_filme"])
imputada_categoria = matriz_demo.fillna(categoria_filmes.reindex(ordem_filmes))
imputada_filme = matriz_demo.fillna(media_filmes_demo.reindex(ordem_filmes))
assert imputada_categoria.notna().all().all() and imputada_filme.notna().all().all()

observadas = matriz_demo.notna().to_numpy()
sem_nota_no_recorte = int((observadas.sum(axis=1) == 0).sum())
triangulo = np.triu_indices(N_USUARIOS_DEMO, 1)
corr_coavaliados = matriz_demo.T.corr(min_periods=10).to_numpy()[triangulo]
corr_coavaliados = corr_coavaliados[~np.isnan(corr_coavaliados)]
with np.errstate(invalid="ignore", divide="ignore"):
    corr_categoria = np.corrcoef(imputada_categoria.to_numpy())[triangulo]
    corr_filme = np.corrcoef(imputada_filme.to_numpy())[triangulo]
corr_categoria, corr_filme = corr_categoria[~np.isnan(corr_categoria)], corr_filme[~np.isnan(corr_filme)]
linhas_imputacao = [
    ("só células observadas", int(observadas.sum()), np.nanmean(matriz_demo.to_numpy()), np.nanstd(matriz_demo.to_numpy(), ddof=1),
     len(corr_coavaliados), corr_coavaliados.mean(), 100 * (corr_coavaliados > 0.9).mean()),
    ("imputação pela média da categoria", observadas.size, imputada_categoria.to_numpy().mean(), imputada_categoria.to_numpy().std(ddof=1),
     len(corr_categoria), corr_categoria.mean(), 100 * (corr_categoria > 0.9).mean()),
    ("imputação pela média do filme", observadas.size, imputada_filme.to_numpy().mean(), imputada_filme.to_numpy().std(ddof=1),
     len(corr_filme), corr_filme.mean(), 100 * (corr_filme > 0.9).mean()),
]
tabela_imputacao = pd.DataFrame(linhas_imputacao, columns=["variante", "células usadas", "média", "DP",
                                                           "pares de usuários", "Pearson médio", "% pares com r > 0,9"])
print(f"\nRecorte: {N_USUARIOS_DEMO} usuários × {N_FILMES_DEMO} filmes mais avaliados; "
      f"{100 * observadas.mean():.2f}% das células têm nota; {sem_nota_no_recorte} usuários não avaliaram nenhum desses filmes.")
print("Correlação entre usuários: só observadas = pares com pelo menos 10 filmes avaliados em comum.")
print(tabela_imputacao.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(17, 5))
histograma_media_dp(axes[0], cobertura_pct, "Porcentagem dos filmes avaliados por usuário", log_x=True,
                    rotulo_x="% dos filmes avaliados (eixo log)")
bordas = np.linspace(-1, 1, 41)
for valores, rotulo, cor in [(corr_coavaliados, "só filmes avaliados em comum", "#4682a9"),
                             (corr_categoria, "imputação pela média da categoria", "#b65d5d"),
                             (corr_filme, "imputação pela média do filme", "#d18b24")]:
    axes[1].hist(valores, bins=bordas, weights=np.full(len(valores), 100 / len(valores)), histtype="step", lw=1.8,
                 color=cor, label=f"{rotulo} (média {valores.mean():.2f})")
axes[1].set(title=f"Pearson entre pares de usuários no recorte ({N_USUARIOS_DEMO} × {N_FILMES_DEMO})",
            xlabel="correlação de Pearson", ylabel="% dos pares")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Esparsidade:** a matriz usuário × filme tem 9.597.558.427 células, das quais só 25.000.095 (0,26%) têm nota. O usuário mediano avaliou 0,12% dos filmes; o mais ativo, 54,5%.
# - **Recorte de teste:** mesmo entre os 300 filmes mais avaliados, só 17,2% das células têm nota.
# - **Efeito da imputação:**
#   - o DP das notas cai de 1,02 (só notas reais) para 0,45 com a média da categoria e 0,52 com a média do filme: a variação real é reduzida à metade;
#   - a correlação média entre usuários sobe de 0,12 (filmes avaliados em comum) para 0,19 com a média da categoria e 0,54 com a média do filme. Os usuários passam a parecer semelhantes porque compartilham os valores preenchidos, não porque avaliaram igual.
#
# **Limitação:** o teste usa um recorte de 500 usuários e dos filmes mais populares, onde a matriz é muito menos vazia que no total.
#
# **Consequência:** os ausentes não são preenchidos. O perfil de 3.1 usa apenas os filmes avaliados por cada usuário, que é a alternativa de ignorar os valores ausentes. Uma imputação pela média do perfil só faz sentido depois de P1, quando os segmentos existirem.

# %% [markdown]
# ### 2.6 Distribuições depois do tratamento
#
# Nenhuma avaliação foi removida. Por isso, só se repetem os testes que o tratamento pode alterar:
#
# - **(a)** a distribuição da nota, que deve ser idêntica;
# - **(b)** filmes, avaliações, média e desvio-padrão por gênero, antes e depois da imputação;
# - **(c)** as notas dos filmes que receberam gênero, comparadas às do gênero inteiro;
# - **(d)** o qui-quadrado gênero × grupo de nota, com o V de Cramér antes e depois.
#
# Volume, tempo, década e correlações por filme não dependem do gênero. As correlações por usuário mudam só nas avaliações dos filmes imputados, cuja quantidade aparece em (b).

# %%
# [2.6] Distribuições depois do tratamento: o que muda com a imputação de gênero
from scipy import stats

# (a) Nota: nenhuma avaliação foi removida, então a distribuição precisa ser idêntica à de 1.5.
dist_nota_tratada = ratings_clean.groupBy("rating").count().toPandas().set_index("rating")["count"].sort_index()
dist_nota_alvo = rating_distribution.select("rating", "count").toPandas().set_index("rating")["count"].sort_index()
print(f"(a) Distribuição da nota idêntica antes e depois do tratamento: {dist_nota_tratada.equals(dist_nota_alvo)} "
      f"({int(dist_nota_tratada.sum()):,} avaliações)")

# (b) Gêneros: filmes, avaliações, média e DP antes (dados-alvo) e depois (dados tratados).
antes = dist_generos.assign(genre=dist_generos["genre"].replace({SEM_GENERO: "Unknown"}))[
    ["genre", "filmes", "avaliacoes", "media", "dp"]].set_index("genre")
depois = (ratings_genres.groupBy("genre")
    .agg(F.count("*").alias("avaliacoes"), F.avg("rating").alias("media"), F.stddev("rating").alias("dp"))
    .join(movie_genres.groupBy("genre").agg(F.count("*").alias("filmes")), "genre", "right")
    .toPandas().fillna({"avaliacoes": 0}).set_index("genre"))
comparacao_generos = antes.merge(depois, left_index=True, right_index=True, how="left", suffixes=("_antes", "_depois"))
comparacao_generos["Δ_filmes"] = comparacao_generos["filmes_depois"] - comparacao_generos["filmes_antes"]
comparacao_generos["Δ_avaliacoes"] = comparacao_generos["avaliacoes_depois"] - comparacao_generos["avaliacoes_antes"]
comparacao_generos["Δ_media"] = comparacao_generos["media_depois"] - comparacao_generos["media_antes"]
comparacao_generos = comparacao_generos.sort_values("Δ_avaliacoes", ascending=False)
print("\n(b) Gêneros antes × depois (Unknown = (no genres listed) nos dados-alvo):")
print(comparacao_generos[["filmes_antes", "filmes_depois", "Δ_filmes", "avaliacoes_antes", "avaliacoes_depois",
                          "Δ_avaliacoes", "media_antes", "media_depois", "Δ_media", "dp_antes", "dp_depois"]]
      .round(4).to_string())

# (c) Avaliações dos filmes imputados: nota média por gênero imputado, comparada à do gênero inteiro.
imputados_notas = (ratings_clean.join(movies_clean.filter("genero_foi_imputado").select("movieId", "genres"), "movieId")
    .groupBy("genres").agg(F.count("*").alias("avaliacoes"), F.avg("rating").alias("media_imputados"),
                           F.stddev("rating").alias("dp_imputados")).toPandas())
if len(imputados_notas):
    imputados_notas = imputados_notas.merge(depois[["media"]].rename(columns={"media": "media_do_genero"}),
                                            left_on="genres", right_index=True)
    print("\n(c) Notas dos filmes que receberam gênero, comparadas à média do gênero:")
    print(imputados_notas.sort_values("avaliacoes", ascending=False).round(3).to_string(index=False))

# (d) Gênero × grupo de nota: mesmo qui-quadrado e V de Cramér de 1.8-d, agora nos dados tratados.
contagens_tratadas = (ratings_genres.withColumn("grupo_nota", F.ceil("rating").cast("int"))
    .groupBy("genre", "grupo_nota").count().toPandas()
    .pivot(index="genre", columns="grupo_nota", values="count").reindex(columns=[1, 2, 3, 4, 5]).fillna(0))
linhas_v = []
for genero, contagens in contagens_tratadas.iterrows():
    dentro = contagens.to_numpy(dtype=float)
    fora = total_por_grupo - dentro
    qui2, p_qui, _, _ = stats.chi2_contingency(np.vstack([dentro, fora]), correction=False)
    linhas_v.append((genero, int(dentro.sum()), qui2, p_qui, np.sqrt(qui2 / total_por_grupo.sum())))
testes_genero_tratado = pd.DataFrame(linhas_v, columns=["gênero", "avaliações", "qui2", "p", "V_cramer_depois"])
v_antes = testes_genero[["gênero", "V_cramer"]].rename(columns={"V_cramer": "V_cramer_antes"})
v_antes["gênero"] = v_antes["gênero"].replace({SEM_GENERO: "Unknown"})
testes_genero_tratado = testes_genero_tratado.merge(v_antes, on="gênero", how="left")
testes_genero_tratado["Δ_V"] = testes_genero_tratado["V_cramer_depois"] - testes_genero_tratado["V_cramer_antes"]
print("\n(d) Gênero × grupo de nota nos dados tratados (V de Cramér antes e depois):")
print(testes_genero_tratado.sort_values("V_cramer_depois", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
mudou = comparacao_generos[comparacao_generos["Δ_avaliacoes"] != 0]
axes[0].barh(mudou.index, mudou["Δ_avaliacoes"], color=["#d18b24" if v < 0 else "#4682a9" for v in mudou["Δ_avaliacoes"]])
axes[0].set(title="Avaliações que mudaram de gênero com a imputação", xlabel="Δ avaliações (depois − antes)")
axes[0].grid(axis="x", alpha=.2)
ordem_v = testes_genero_tratado.sort_values("V_cramer_depois")
pos = np.arange(len(ordem_v))
axes[1].barh(pos - 0.2, ordem_v["V_cramer_antes"], height=0.4, color="#bbbbbb", label="antes (dados-alvo)")
axes[1].barh(pos + 0.2, ordem_v["V_cramer_depois"], height=0.4, color="#4682a9", label="depois (dados tratados)")
axes[1].set_yticks(pos, ordem_v["gênero"])
axes[1].set(title="V de Cramér: gênero × grupo de nota", xlabel="V de Cramér")
axes[1].legend()
axes[1].grid(axis="x", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **(a) Nota:** a distribuição das 25.000.095 notas é idêntica antes e depois do tratamento, porque nenhuma avaliação foi removida. Os histogramas e resumos de 1.5 continuam válidos.
# - **(b) Gêneros:** `Unknown` perde 320 filmes e 16.758 avaliações (de 26.627 para 9.869). As avaliações vão sobretudo para Drama (+5.479), Adventure (+2.608) e Comedy (+2.597). Em todos os gêneros, a média muda no máximo 0,0002 e o desvio-padrão no máximo 0,0008 (Documentary). A média de `Unknown` cai de 3,33 para 3,06: os filmes que continuam sem gênero são os menos vistos e têm notas mais baixas.
# - **(c) Filmes imputados:** suas notas nem sempre seguem a média do gênero. Em Drama e Comedy ficam próximas (3,67 contra 3,68; 3,43 contra 3,42); em Horror ficam acima (3,49 contra 3,29); em Action, Children e Animation, abaixo (2,94 contra 3,47; 2,93 contra 3,43; 3,28 contra 3,62). A imputação usa quem avaliou o filme, não a nota. Como as avaliações imputadas são no máximo 0,37% das avaliações de um gênero (Documentary), as médias dos gêneros não mudam.
# - **(d) Associação gênero × nota:** o V de Cramér praticamente não muda. A maior variação é a de `Unknown` (de 0,0051 para 0,0077), que perdeu os filmes mais avaliados. Drama continua o gênero mais associado à nota (V = 0,119), e todas as associações seguem fracas.
#
# **Consequência:** o tratamento não altera as distribuições estudadas na Seção 1. O efeito da imputação fica restrito a `Unknown`, que passa a 0,04% das avaliações, e chega ao perfil por usuário em 3.1.
