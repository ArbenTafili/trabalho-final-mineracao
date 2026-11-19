# %% [markdown]
# ## 1. Seleção de dados
#
# A seleção define os **dados-alvo** e os conhece antes de qualquer tratamento: estrutura de instâncias e atributos, tipos e domínios, e exploração preliminar com estatísticas descritivas e visualização. A exploração é feita sobre as tabelas como foram carregadas, porque ela orienta a escolha do pré-processamento e do método de aprendizado.
#
# ### 1.1 Obtenção e carregamento

# %%
# [1.1-a] Obtenção dos arquivos do MovieLens 25M (cópia local ou espelho documentado)
from pathlib import Path
import importlib.util, os, subprocess, sys
for pacote in ('pyspark', 'huggingface_hub', 'matplotlib', 'pandas'):
    if importlib.util.find_spec(pacote) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pacote])
DATA_DIR = Path(os.environ.get('MOVIELENS_DIR', 'ml-25m'))  # execução local: pasta indicada na variável
arquivos = ['movies.csv', 'ratings.csv', 'tags.csv', 'links.csv']
if not all((DATA_DIR / nome).is_file() for nome in arquivos):
    from huggingface_hub import hf_hub_download
    raiz = Path('movielens_mirror')
    DATA_DIR = raiz / 'movielens_25m' / 'ml-25m'
    for nome in arquivos:
        hf_hub_download(repo_id='cephcyn/2026_01_22_datasets', repo_type='dataset', filename=f'movielens_25m/ml-25m/{nome}', local_dir=str(raiz))
assert all((DATA_DIR / nome).is_file() for nome in arquivos)
print('MovieLens 25M:', DATA_DIR.resolve())

# %%
# [1.1-b] SparkSession e importações
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType,
    LongType, DoubleType, StringType
)

spark = (
    SparkSession.builder
    .appName("MovieLensKDD")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("Spark inicializado:", spark.version)

# %%
# [1.1-c] Carregamento com esquemas explícitos
movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
])

ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", DoubleType(), True),
    StructField("timestamp", LongType(), True),
])

tags_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("tag", StringType(), True),
    StructField("timestamp", LongType(), True),
])

links_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("imdbId", LongType(), True),
    StructField("tmdbId", LongType(), True),
])

# Os CSV do MovieLens escapam aspas dentro do texto duplicando-as ("My Friend ""A"" (2018)"), como no padrão CSV.
# O Spark usa barra invertida por padrão; sem escape='"', 11 títulos ficariam com aspas sobrando e sem o ano no fim.
movies_df = spark.read.csv(str(DATA_DIR / "movies.csv"), header=True, schema=movies_schema, escape='"')
ratings_df = spark.read.csv(str(DATA_DIR / "ratings.csv"), header=True, schema=ratings_schema)
tags_df = spark.read.csv(str(DATA_DIR / "tags.csv"), header=True, schema=tags_schema, escape='"')
links_df = spark.read.csv(str(DATA_DIR / "links.csv"), header=True, schema=links_schema)

datasets = {
    "movies": movies_df,
    "ratings": ratings_df,
    "tags": tags_df,
    "links": links_df,
}

print("Tabelas carregadas:", ", ".join(datasets))

# %% [markdown]
# ### 1.2 Estrutura: tabelas, instâncias, atributos e grão
#
# O MovieLens é relacional. Para a mineração, os dados precisam ser representados como uma tabela de instâncias (linhas) e atributos (colunas). Por isso, primeiro se registra o grão de cada tabela e depois se monta a visão usada na exploração (1.4).

# %%
# [1.2] Estrutura das tabelas: linhas, colunas, esquemas e exemplos
for nome, df in datasets.items():
    df.createOrReplaceTempView(f"{nome}_raw")
    df.createOrReplaceTempView(nome)

perfil_tabelas = []
for nome, df in datasets.items():
    total_linhas = df.count()
    perfil_tabelas.append((nome, total_linhas, len(df.columns)))

perfil_df = spark.createDataFrame(
    perfil_tabelas,
    ["tabela", "linhas", "colunas"]
)

perfil_df.orderBy("tabela").show(truncate=False)

for nome, df in datasets.items():
    print(f"\n--- {nome.upper()} ---")
    df.printSchema()
    df.show(3, truncate=False)

# %% [markdown]
# ### 1.3 Tipos de atributos e domínios
#
# | Tabela | Instância (grão) | Atributo | Tipo | Domínio | Uso |
# |---|---|---|---|---|---|
# | `ratings` | uma nota de um usuário para um filme | `userId`, `movieId` | nominal (identificador) | discreto | chave; não entra em médias nem correlações |
# | | | `rating` | ordinal | discreto, 10 valores de 0,5 a 5,0 | medida observada principal |
# | | | `timestamp` | intervalar (tempo) | discreto, segundos Unix | contexto temporal; convertido no pré-processamento |
# | `movies` | um filme do catálogo | `movieId` | nominal | discreto | chave |
# | | | `title` | texto | texto livre com ano entre parênteses | ano de lançamento extraído do título |
# | | | `genres` | nominal multirrótulo | 20 rótulos, incluindo `(no genres listed)` | consultas por gênero e perfil P1 |
# | `tags` | uma tag de um usuário para um filme | `tag` | texto livre | alta cardinalidade | fora dos dados-alvo nesta fase; pode entrar no fim do trabalho |
# | `links` | identificadores externos de um filme | `imdbId`, `tmdbId` | nominal | discreto | ligação externa; atributo irrelevante para P1/P2 |
#
# - **Classe/meta:** não existe.
# - **Cardinalidade e dimensionalidade:** 25.000.095 avaliações × 4 atributos na tabela principal. Para P1, a dimensionalidade será definida pelos atributos derivados por usuário (Seção 3).

# %% [markdown]
# ### 1.4 Dados-alvo e visão desnormalizada para exploração
#
# A visão a seguir junta avaliações e filmes **sem tratamento**. Ela é usada apenas para a exploração desta seção. A integridade do relacionamento é auditada em 2.1 e as tabelas tratadas são reconstruídas em 2.3.

# %%
# [1.4] Dados-alvo da seleção: tabelas escolhidas e visão avaliação × filme × gênero
ratings_sel = ratings_df.cache()
movies_sel = movies_df
total_avaliacoes = ratings_sel.count()  # materializa o cache para a exploração

movie_genres_sel = (
    movies_sel
    .withColumn("genre", F.explode(F.split(F.col("genres"), r"\|")))
    .select("movieId", "title", "genre")
)
ratings_movies_sel = ratings_sel.join(movies_sel.select("movieId", "title", "genres"), on="movieId", how="inner")
ratings_genres_sel = ratings_sel.join(movie_genres_sel, on="movieId", how="inner")
linhas_ratings_genres_sel = ratings_genres_sel.count()

print("Avaliações nos dados-alvo:", total_avaliacoes)
print("Linhas da visão avaliação × gênero (um filme pode ter vários gêneros):", linhas_ratings_genres_sel)
print("Tags e links ficam fora dos dados-alvo: não são usados em P1 nem em P2 nesta fase.")

# %% [markdown]
# ### 1.5 Exploração univariada da nota
#
# A ordem de apresentação é:
#
# 1. proporção e moda;
# 2. média, mediana e média podada;
# 3. quartis e percentis;
# 4. boxplot e violin plot;
# 5. intervalo, variância e desvio-padrão;
# 6. histogramas equi-width, equi-depth e V-Optimal.

# %%
# [1.5-a] Proporção, moda, localização, dispersão e formato da nota
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

resumo_notas = ratings_sel.agg(
    F.count("*").alias("avaliacoes"),
    F.countDistinct("userId").alias("usuarios"),
    F.countDistinct("movieId").alias("filmes_avaliados"),
    F.round(F.avg("rating"), 4).alias("media"),
    F.expr("percentile_approx(rating, 0.5, 10000)").alias("mediana"),
    F.round(F.stddev("rating"), 4).alias("desvio_padrao"),
    F.min("rating").alias("minimo"),
    F.max("rating").alias("maximo"),
)
print("Filmes no catálogo:", movies_sel.count())
resumo_notas.show(truncate=False)

# Frequência e proporção de cada valor observado; denominador = total de avaliações.
rating_distribution = (
    ratings_sel.groupBy("rating").count()
    .withColumn("proporcao_pct", F.round(F.col("count") / F.lit(total_avaliacoes) * 100, 4))
    .orderBy("rating")
)
rating_distribution.show(truncate=False)
dist_pd = rating_distribution.select("rating", "count").orderBy("rating").toPandas()
moda = float(dist_pd.loc[dist_pd["count"].idxmax(), "rating"])

resumo_extra = ratings_sel.agg(
    F.expr("percentile_approx(rating, array(0.25, 0.5, 0.75), 10000)").alias("quartis"),
    F.variance("rating").alias("variancia"),
    F.skewness("rating").alias("assimetria"),
    F.kurtosis("rating").alias("excesso_curtose"),
).first()
estatisticas_notas = resumo_notas.first()
intervalo = estatisticas_notas["maximo"] - estatisticas_notas["minimo"]

# Média podada: ordena as notas e remove 33% das observações em cada cauda.
valores = dist_pd["rating"].astype(float).tolist()
freqs = dist_pd["count"].astype(int).tolist()
N = int(sum(freqs))

def media_podada(valores, freqs, proporcao):
    remover = int(sum(freqs) * proporcao)
    inicio, fim = remover, sum(freqs) - remover
    acumulado = soma = mantidos = 0
    for valor, freq in zip(valores, freqs):
        esquerda, direita = acumulado, acumulado + freq
        qtd = max(0, min(direita, fim) - max(esquerda, inicio))
        soma += valor * qtd
        mantidos += qtd
        acumulado = direita
    return soma / mantidos, mantidos

media_podada_33, n_podado = media_podada(valores, freqs, 0.33)

medidas_nota = pd.DataFrame([
    ("Moda", moda),
    ("Média", estatisticas_notas["media"]),
    ("Mediana", estatisticas_notas["mediana"]),
    (f"Média podada (33% por cauda; {n_podado:,} mantidas)", round(media_podada_33, 4)),
    ("Q1; Q2; Q3 (aproximados)", "; ".join(f"{q:.1f}" for q in resumo_extra["quartis"])),
    ("Intervalo (máx − mín)", intervalo),
    ("Variância amostral", round(resumo_extra["variancia"], 4)),
    ("Desvio-padrão amostral", estatisticas_notas["desvio_padrao"]),
    ("Assimetria", round(resumo_extra["assimetria"], 4)),
    ("Excesso de curtose", round(resumo_extra["excesso_curtose"], 4)),
], columns=["Medida", "Valor"])
print(medidas_nota.to_string(index=False))

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Localização:** a moda é 4,0 e a mediana 3,5; a média (3,5339) fica entre as duas.
# - **Formato:** a assimetria negativa (−0,70) indica cauda à esquerda: notas baixas são menos frequentes e mais espalhadas. Por isso a média podada (3,6813), que descarta 33% de cada cauda, fica acima da média. O excesso de curtose (0,22) é próximo de zero, ou seja, caudas semelhantes às de uma normal.
# - **Dispersão:** o desvio-padrão de 1,06 ponto, numa escala de 4,5 pontos, indica variação moderada.
#
# **Limitação:** a nota é ordinal; média e desvio-padrão a tratam como intervalar, prática usual mas que deve ser lembrada.
#
# **Consequência:** as notas ≥ 4,0 somam 49,81% das avaliações, o que faz desse valor um limiar natural de nota positiva para P2.

# %% [markdown]
# #### Boxplot e violin plot
#
# O boxplot resume quartis e extremos. O violin plot acrescenta a forma da distribuição. Os dois gráficos usam uma amostra aleatória simples com semente fixa, sem truncamento por `limit`. A representatividade dessa amostra é verificada em 2.4.

# %%
# [1.5-b] Boxplot e violin plot da nota em amostra aleatória simples
FRACAO_AMOSTRA_NOTAS = 0.004  # cerca de 100 mil avaliações
# Sorteio pelo hash de (userId, movieId, semente 42): cada avaliação tem a mesma chance de entrar e a amostra é a
# mesma em qualquer máquina, porque não depende de como o Spark divide os dados.
sorteio = F.abs(F.xxhash64("userId", "movieId", F.lit(42))) % 1_000_000
amostra_pd = ratings_sel.where(sorteio < FRACAO_AMOSTRA_NOTAS * 1_000_000).select("rating").toPandas()

import inspect

def horizontal(metodo):  # Matplotlib >= 3.10 usa orientation; versões anteriores usam vert
    return {"orientation": "horizontal"} if "orientation" in inspect.signature(metodo).parameters else {"vert": False}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].boxplot(amostra_pd["rating"], patch_artist=True, **horizontal(axes[0].boxplot))
axes[0].set_title(f"Boxplot da nota — amostra n={len(amostra_pd):,}")
axes[0].set_xlabel("Nota")
axes[1].violinplot(amostra_pd["rating"], showmedians=True, **horizontal(axes[1].violinplot))
axes[1].set_title("Violin plot da nota — mesma amostra")
axes[1].set_xlabel("Nota")
for ax in axes:
    ax.grid(axis="x", alpha=.2)
plt.tight_layout()
plt.show()

q1, q3 = amostra_pd["rating"].quantile([0.25, 0.75])
limite_inferior = q1 - 1.5 * (q3 - q1)
print(f"Amostra: Q1={q1}, Q3={q3}; limite inferior do whisker = {limite_inferior}")
print("Notas abaixo desse limite pertencem à escala válida: são outliers legítimos e não são removidas.")

# %% [markdown]
# **Leitura dos resultados.** Na amostra, Q1 = 3,0, Q3 = 4,0 e o limite inferior do whisker é 1,5. As notas 0,5 e 1,0 (4,68% das avaliações) aparecem como outliers pela regra do boxplot, mas pertencem à escala válida e são mantidas. O violin plot mostra mais densidade nos valores inteiros (4,0: 26,6%; 3,0: 19,6%; 5,0: 14,4%) do que nos meios pontos vizinhos.

# %% [markdown]
# #### Histogramas
#
# - **Cinco grupos de nota:** cada grupo reúne duas notas adjacentes da escala original. O histograma mostra média e desvio-padrão no título.
# - **Equi-width:** três intervalos de mesma largura entre 0,5 e 5,0. Amplitude = 4,5; largura = 1,5; baixa = [0,5; 2,0); média = [2,0; 3,5); alta = [3,5; 5,0].
# - **Equi-depth:** procura quantidades semelhantes por faixa. Como só há dez valores possíveis e muitos empates, faixas exatamente iguais não são garantidas.
# - **V-Optimal:** escolhe as faixas contíguas que minimizam a soma dos quadrados dentro das faixas.

# %%
# [1.5-c] Histograma de cinco grupos de nota e histograma equi-width
from pyspark.ml.feature import Bucketizer
import matplotlib.pyplot as plt

# Três intervalos equi-width (mesma largura entre 0,5 e 5,0).
equi_width = Bucketizer(
    splits=[0.5, 2.0, 3.5, 5.000001], inputCol="rating", outputCol="bucket", handleInvalid="skip"
)
rating_buckets = (
    equi_width.transform(ratings_sel.select("rating"))
    .groupBy("bucket").count()
    .withColumn("faixa", F.when(F.col("bucket") == 0, "Baixa [0,5; 2,0)")
        .when(F.col("bucket") == 1, "Média [2,0; 3,5)").otherwise("Alta [3,5; 5,0]"))
    .withColumn("proporcao_pct", F.round(F.col("count") / F.lit(total_avaliacoes) * 100, 4))
    .orderBy("bucket")
)
rating_buckets.show(truncate=False)

# Cinco classes de nota; cada classe reúne dois valores adjacentes da escala original.
histograma_5_notas = (
    ratings_sel.withColumn("grupo_nota", F.ceil("rating").cast("int"))
    .groupBy("grupo_nota").count()
    .withColumn("proporcao_pct", F.round(F.col("count") / F.lit(total_avaliacoes) * 100, 4))
    .orderBy("grupo_nota")
)
assert histograma_5_notas.count() == 5
assert histograma_5_notas.agg(F.sum("count")).first()[0] == total_avaliacoes
histograma_5_notas.show(truncate=False)

# Os dois gráficos mostram proporção sobre o total de avaliações válidas.
eq_pd = rating_buckets.toPandas()
h5_pd = histograma_5_notas.toPandas()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(eq_pd["faixa"], eq_pd["proporcao_pct"], color="#397461")
axes[0].set_title("Distribuição equi-width")
axes[0].set_ylabel("Proporção das avaliações (%)")
axes[0].tick_params(axis="x", rotation=12)
axes[1].bar(h5_pd["grupo_nota"].astype(str), h5_pd["proporcao_pct"], color="#d18b24")
estatisticas_notas = resumo_notas.first()
media_nota_geral = float(estatisticas_notas["media"])
dp_nota_geral = float(estatisticas_notas["desvio_padrao"])
rotulos_grupos = ["1\n(0,5 e 1,0)", "2\n(1,5 e 2,0)", "3\n(2,5 e 3,0)", "4\n(3,5 e 4,0)", "5\n(4,5 e 5,0)"]
axes[1].set_title(f"Distribuição das notas — média={media_nota_geral:.4f}; DP={dp_nota_geral:.4f}")
axes[1].set_xlabel("Grupo de nota (cada grupo reúne duas notas originais)")
axes[1].set_ylabel("Proporção das avaliações (%)")
axes[1].set_xticks(range(5), rotulos_grupos)
for ax in axes: ax.grid(axis="y", alpha=.2)
plt.tight_layout()
plt.show()

# %%
# [1.5-d] Histograma equi-depth por quantis
from pyspark.ml.feature import QuantileDiscretizer

# O QuantileDiscretizer tenta criar faixas com o MESMO número de avaliações em cada uma.
# Vamos dividir em 3 grupos (tercis)
discretizer = QuantileDiscretizer(numBuckets=3, inputCol="rating", outputCol="bucket_depth", relativeError=0.01)

# O modelo encontra os cortes ideais
model = discretizer.fit(ratings_sel)
df_equidepth = model.transform(ratings_sel)

print("Limites encontrados pelo Equi-depth:", model.getSplits())

equidepth_hist = (
    df_equidepth.groupBy("bucket_depth")
    .count()
    .withColumn("proporcao_pct", F.round((F.col("count") / F.lit(total_avaliacoes)) * 100, 4))
    .orderBy("bucket_depth")
)

equidepth_hist.show(truncate=False)

# %%
# [1.5-e] Histograma V-Optimal com três faixas e comparação dos três histogramas
from itertools import combinations

def custo_v(i, j):
    n = sum(freqs[i:j])
    media = sum(valores[k] * freqs[k] for k in range(i, j)) / n
    return sum(freqs[k] * (valores[k] - media) ** 2 for k in range(i, j))

melhor = min((sum(custo_v(a, b) for a, b in zip((0,) + cortes, cortes + (len(valores),))), cortes)
             for cortes in combinations(range(1, len(valores)), 2))
_, cortes_v = melhor
limites_v = (0,) + cortes_v + (len(valores),)
linhas_v = []
for a, b in zip(limites_v, limites_v[1:]):
    n = sum(freqs[a:b])
    linhas_v.append((f"{valores[a]:.1f}–{valores[b-1]:.1f}", n, round(100 * n / N, 4), round(custo_v(a, b), 1)))
v_optimal_pd = pd.DataFrame(linhas_v, columns=["Faixa de notas", "n", "%", "SSE dentro da faixa"])
print("V-Optimal (3 faixas; ótimo exato sobre os 10 valores discretos); SSE total =", round(melhor[0], 1))
print(v_optimal_pd.to_string(index=False))

comparacao_histogramas = pd.DataFrame({
    "equi-width (%)": eq_pd.sort_values("bucket")["proporcao_pct"].to_numpy(),
    "equi-depth (%)": equidepth_hist.toPandas().sort_values("bucket_depth")["proporcao_pct"].to_numpy(),
    "V-Optimal (%)": v_optimal_pd["%"].to_numpy(),
}, index=["faixa 1", "faixa 2", "faixa 3"])
print("Proporção das avaliações em cada faixa (denominador = total de avaliações):")
print(comparacao_histogramas.to_string())

# %% [markdown]
# #### Comparação visual dos histogramas e diagrama circular
#
# Os três histogramas de três faixas são apresentados lado a lado, com a média e o desvio-padrão da nota. O diagrama circular mostra a frequência relativa dos cinco grupos de nota.

# %%
# [1.5-f] Comparação visual dos histogramas de três faixas e diagrama circular dos cinco grupos
eqd_pd = equidepth_hist.toPandas().sort_values("bucket_depth")
cortes_eqd = model.getSplits()
rotulos_eqd = [f"[{a:g}; {b:g})".replace("-inf", "−∞").replace("inf", "∞") for a, b in zip(cortes_eqd[:-1], cortes_eqd[1:])]
paineis = [
    ("Equi-width", ["[0,5; 2,0)", "[2,0; 3,5)", "[3,5; 5,0]"], eq_pd.sort_values("bucket")["proporcao_pct"].to_numpy()),
    ("Equi-depth", rotulos_eqd, eqd_pd["proporcao_pct"].to_numpy()),
    ("V-Optimal", v_optimal_pd["Faixa de notas"].tolist(), v_optimal_pd["%"].to_numpy()),
]
fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
for ax, (titulo, rotulos, valores) in zip(axes[:3], paineis):
    ax.bar(range(len(valores)), valores, color="#397461", edgecolor="#1f2d28")
    ax.set_xticks(range(len(valores)), rotulos)
    for x, v in enumerate(valores):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title(f"{titulo} (3 faixas)\nmédia={media_nota_geral:.4f}; DP={dp_nota_geral:.4f}", fontsize=10)
    ax.set_ylabel("Proporção das avaliações (%)")
    ax.set_ylim(0, 72)
    ax.grid(axis="y", alpha=.2)
axes[3].pie(h5_pd["proporcao_pct"], labels=[f"Grupo {g}" for g in h5_pd["grupo_nota"]], autopct="%.1f%%",
            startangle=90, counterclock=False,
            colors=["#b65d5d", "#d18b24", "#c9b458", "#4682a9", "#397461"])
axes[3].set_title("Cinco grupos de nota — frequência relativa")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Cinco grupos:** o grupo 4 (notas 3,5 e 4,0) concentra 39,27% das avaliações e o grupo 1 (0,5 e 1,0) apenas 4,68%.
# - **Equi-width:** coloca 62,52% das avaliações na faixa alta [3,5; 5,0].
# - **Equi-depth:** não consegue faixas de mesmo tamanho (17,89%, 32,30% e 49,81%), porque há apenas dez valores possíveis e muitos empates.
# - **V-Optimal:** forma as faixas 0,5–2,0, 2,5–3,5 e 4,0–5,0 (12,84%, 37,35% e 49,81%), que minimizam a soma dos quadrados dentro das faixas. Sua última fronteira (4,0) coincide com o limiar de nota positiva usado no perfil.
#
# **Consequência:** a distribuição se concentra em notas altas. A proporção de notas positivas de um usuário deve ser lida em relação a essa base de quase 50%.

# %% [markdown]
# ### 1.6 Exploração univariada categórica e volume por objeto
#
# - **Gêneros:** frequência, proporção e moda no catálogo. Os rótulos aparecem como na fonte, incluindo `(no genres listed)`.
# - **Volume:** quantidade de avaliações por usuário e por filme, e filmes do catálogo sem avaliação.

# %%
# [1.6] Gêneros no catálogo e volume de avaliações por usuário e por filme
total_filmes = movies_sel.count()
genres_freq = (
    movie_genres_sel.groupBy("genre").count()
    .withColumn("proporcao_pct", F.round(F.col("count") / F.lit(total_filmes) * 100, 4))
    .orderBy(F.col("count").desc())
)
genres_freq.show(25, truncate=False)
print("Denominador = filmes do catálogo; gêneros se sobrepõem, então a soma passa de 100%.")
print("Moda do atributo gênero:", genres_freq.first()["genre"])

for rotulo, chave in [("usuário", "userId"), ("filme", "movieId")]:
    suporte = ratings_sel.groupBy(chave).count()
    q = suporte.approxQuantile("count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.0)  # erro 0: quantis exatos
    print(f"Avaliações por {rotulo} (mín, Q1, mediana, Q3, máx): {q}")

filmes_sem_avaliacao = movies_sel.join(ratings_sel.select("movieId").distinct(), "movieId", "left_anti").count()
print(f"Filmes do catálogo sem nenhuma avaliação: {filmes_sem_avaliacao:,} de {total_filmes:,}")

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Gêneros:** Drama é a moda (41,02% dos filmes do catálogo), seguido de Comedy (27,03%). 5.062 filmes (8,11%) não têm gênero informado.
# - **Volume:**
#   - por usuário, a mediana é de 70 avaliações (máximo 32.202);
#   - por filme, a mediana é de 6 (máximo 81.491), e 25% dos filmes têm no máximo 2 avaliações;
#   - 3.376 filmes do catálogo não têm nenhuma avaliação.
#
# **Consequência:** médias por filme só são comparadas com suporte mínimo, e a atividade do usuário entra como proporção, não como contagem bruta.

# %% [markdown]
# #### Exploração temporal das avaliações
#
# O `timestamp` é intervalar: diferenças entre instantes fazem sentido. Aqui ele é convertido em ano da avaliação para descrever o volume por ano, a média e o desvio-padrão da nota por ano, e o uso de notas de meio ponto. A razão de correlação η mede quanto da variação das notas se associa ao ano. Também se mede o período de atividade de cada usuário, ou seja, os dias entre a primeira e a última avaliação.

# %%
# [1.6-b] Exploração temporal: ano da avaliação, nota média e uso de meio ponto
por_ano = (
    ratings_sel
    .withColumn("ano_avaliacao", F.year(F.from_unixtime("timestamp")))
    .groupBy("ano_avaliacao")
    .agg(F.count("*").alias("n"), F.avg("rating").alias("media"), F.stddev("rating").alias("dp"),
         F.avg(F.when((F.col("rating") * 2) % 2 == 1, 1.0).otherwise(0.0)).alias("prop_meio_ponto"))
    .orderBy("ano_avaliacao")
    .toPandas()
)
assert int(por_ano["n"].sum()) == total_avaliacoes
por_ano["proporcao_%"] = 100 * por_ano["n"] / por_ano["n"].sum()
por_ano["meio_ponto_%"] = 100 * por_ano["prop_meio_ponto"]
print(por_ano[["ano_avaliacao", "n", "proporcao_%", "media", "dp", "meio_ponto_%"]].round(4).to_string(index=False))
print(f"Período das avaliações: {int(por_ano.ano_avaliacao.min())}–{int(por_ano.ano_avaliacao.max())}")

# Razão de correlação η entre ano da avaliação (categoria) e nota (numérica), a partir das somas por ano.
media_geral_notas = float((por_ano["n"] * por_ano["media"]).sum() / por_ano["n"].sum())
ss_entre_anos = float((por_ano["n"] * (por_ano["media"] - media_geral_notas) ** 2).sum())
ss_total_notas = float(resumo_extra["variancia"]) * (total_avaliacoes - 1)
eta2_ano = ss_entre_anos / ss_total_notas
print(f"Razão de correlação ano da avaliação × nota: η={eta2_ano ** 0.5:.4f}; η²={eta2_ano:.4f}")

atividade = ratings_sel.groupBy("userId").agg(((F.max("timestamp") - F.min("timestamp")) / 86400.0).alias("dias"))
q_dias = atividade.approxQuantile("dias", [0.0, 0.25, 0.5, 0.75, 1.0], 0.0)  # erro 0: quantis exatos
print("Dias entre a primeira e a última avaliação de cada usuário (mín, Q1, mediana, Q3, máx):",
      [round(x, 1) for x in q_dias])

fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
axes[0].bar(por_ano["ano_avaliacao"].astype(int), por_ano["proporcao_%"], color="#397461")
axes[0].set(title="Avaliações por ano (% do total)", xlabel="Ano da avaliação", ylabel="% das avaliações")
axes[1].errorbar(por_ano["ano_avaliacao"], por_ano["media"], yerr=por_ano["dp"], fmt="o-", color="#d18b24",
                 ecolor="#bbbbbb", capsize=2)
axes[1].axhline(media_geral_notas, ls="--", color="gray", label=f"média geral = {media_geral_notas:.4f}")
axes[1].set(title=f"Média ± DP da nota por ano (η² = {eta2_ano:.4f})", xlabel="Ano da avaliação", ylabel="Nota",
            ylim=(0, 5.5))
axes[1].legend()
axes[2].plot(por_ano["ano_avaliacao"], por_ano["meio_ponto_%"], "s-", color="#4682a9")
axes[2].set(title="Avaliações com meio ponto (0,5; 1,5; ...) por ano", xlabel="Ano da avaliação",
            ylabel="% das avaliações do ano")
for ax in axes:
    ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Volume:** as avaliações vão de 1995 a 2019. O volume oscila por ano, com picos em 2000 (6,94%), 2005 (6,45%) e 2015–2017 (6,42% a 7,03%).
# - **Nota por ano:** a média anual varia pouco, de 3,43 (2004) a 3,65 (2013), sem contar 1995, que tem só 3 avaliações. O ano explica apenas 0,26% da variação das notas (η = 0,051; η² = 0,0026).
# - **Mudança de escala:** notas de meio ponto só aparecem a partir de 2003 (0% até 2002, 29,15% em 2003 e 39% a 44% depois). Nas avaliações anteriores a 2003, cada um dos cinco grupos de nota contém apenas a nota inteira correspondente.
# - **Período de atividade:** a mediana é 0 dia (Q3 = 38,3 dias), ou seja, mais da metade dos usuários fez todas as suas avaliações no mesmo dia.
#
# **Limitação:** para a maioria dos usuários, o perfil de P1 reflete uma única sessão de avaliação, não preferências acompanhadas ao longo do tempo.
#
# **Consequência:** o tempo não entra no perfil de P1, e a mudança de escala de 2003 é registrada na interpretação das notas de meio ponto.

# %% [markdown]
# #### Distribuição dos gêneros
#
# A distribuição dos filmes por gênero é a base para decidir o tratamento dos gêneros no pré-processamento. Para cada gênero, quatro medidas lado a lado:
#
# - porcentagem dos filmes do catálogo;
# - porcentagem das avaliações (um filme com vários gêneros conta em cada um);
# - avaliações por filme, que separa gêneros com muitos filmes, mas pouca audiência;
# - nota média ± desvio-padrão.
#
# `(no genres listed)` aparece em laranja: são filmes sem gênero, alguns com avaliações, que recebem gênero no pré-processamento (2.2-b). As tags ficam fora dos dados-alvo nesta fase: não são usadas em P1 nem em P2. A função de histograma definida aqui é usada nas seções seguintes.

# %%
# [1.6-c] Distribuição dos gêneros: filmes, avaliações, avaliações por filme e nota (média e desvio-padrão)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histograma_media_dp(ax, valores, titulo, bins=30, log_x=False, cor="#4682a9", rotulo_x=""):
    """Histograma em % dos objetos, com a média (linha contínua) e média ± 1 DP (linhas tracejadas)."""
    valores = pd.Series(valores).dropna().astype(float)
    media, dp = valores.mean(), valores.std(ddof=1)
    if log_x:
        positivos = valores[valores > 0]
        bordas = np.logspace(np.log10(positivos.min()), np.log10(positivos.max()), bins + 1)
        ax.set_xscale("log")
    else:
        bordas = bins
    ax.hist(valores, bins=bordas, weights=np.full(len(valores), 100 / len(valores)), color=cor, edgecolor="white")
    ax.axvline(media, color="crimson", lw=1.6, label=f"média = {media:.3g}")
    for lado in (media - dp, media + dp):
        if not log_x or lado > 0:
            ax.axvline(lado, color="crimson", lw=1.1, ls="--")
    ax.plot([], [], color="crimson", ls="--", label=f"média ± 1 DP (DP = {dp:.3g})")
    ax.set_title(f"{titulo}\nμ={media:.3g}; DP={dp:.3g}; n={len(valores):,}", fontsize=10)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("% dos objetos")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    return media, dp


SEM_GENERO = "(no genres listed)"
filmes_avaliados_sel = ratings_sel.select("movieId").distinct()
dist_generos = (ratings_genres_sel.groupBy("genre")
    .agg(F.count("*").alias("avaliacoes"), F.avg("rating").alias("media"), F.stddev("rating").alias("dp"))
    .join(genres_freq.select("genre", F.col("count").alias("filmes")), "genre", "right")
    .join(movie_genres_sel.join(filmes_avaliados_sel, "movieId").groupBy("genre")
          .agg(F.countDistinct("movieId").alias("filmes_avaliados")), "genre", "left")
    .toPandas().fillna({"avaliacoes": 0, "filmes_avaliados": 0}))
dist_generos["%_filmes"] = 100 * dist_generos["filmes"] / total_filmes
dist_generos["%_avaliacoes"] = 100 * dist_generos["avaliacoes"] / total_avaliacoes
dist_generos["avaliacoes_por_filme"] = dist_generos["avaliacoes"] / dist_generos["filmes"]
dist_generos["%_filmes_sem_avaliacao"] = 100 * (1 - dist_generos["filmes_avaliados"] / dist_generos["filmes"])
dist_generos = dist_generos.sort_values("%_filmes", ascending=True).reset_index(drop=True)
print("Denominadores: % filmes = filmes do gênero / 62.423 filmes; % avaliações = avaliações de filmes do gênero / "
      "25.000.095 avaliações. Gêneros se sobrepõem, então as somas passam de 100%.")
print(dist_generos.sort_values("%_avaliacoes", ascending=False)[
    ["genre", "filmes", "%_filmes", "avaliacoes", "%_avaliacoes", "avaliacoes_por_filme", "%_filmes_sem_avaliacao", "media", "dp"]
].round(3).to_string(index=False))

# Filmes sem gênero: quantos têm avaliações e quantas avaliações têm (recebem gênero em 2.2-b).
sem_genero_pd = (movies_sel.filter(F.col("genres") == SEM_GENERO).select("movieId")
                 .join(ratings_sel.groupBy("movieId").count(), "movieId", "left").fillna(0).toPandas())
for limite in (1, 10, 50):
    parte = sem_genero_pd[sem_genero_pd["count"] >= limite]
    print(f"Filmes sem gênero com pelo menos {limite} avaliação(ões): {len(parte):,} de {len(sem_genero_pd):,}; "
          f"{int(parte['count'].sum()):,} avaliações")

cores = ["#d18b24" if g == SEM_GENERO else "#4682a9" for g in dist_generos["genre"]]
fig, axes = plt.subplots(1, 4, figsize=(26, 8.5), sharey=True)
axes[0].barh(dist_generos["genre"], dist_generos["%_filmes"], color=cores)
axes[0].set(title="Filmes por gênero (% dos filmes do catálogo)", xlabel="% dos filmes")
axes[1].barh(dist_generos["genre"], dist_generos["%_avaliacoes"], color=cores)
axes[1].set(title="Avaliações por gênero (% das avaliações)", xlabel="% das avaliações")
axes[2].barh(dist_generos["genre"], dist_generos["avaliacoes_por_filme"], color=cores)
axes[2].set_xscale("log")
axes[2].set(title="Avaliações por filme do gênero (média)", xlabel="avaliações por filme (eixo log)")
axes[3].errorbar(dist_generos["media"], dist_generos["genre"], xerr=dist_generos["dp"], fmt="o", color="#397461",
                 ecolor="#bbbbbb", capsize=3)
axes[3].axvline(media_nota_geral, color="crimson", ls="--", label=f"média geral = {media_nota_geral:.2f}")
axes[3].set(title="Nota média ± DP por gênero", xlabel="nota (0,5 a 5,0)", xlim=(0.5, 5))
axes[3].legend(fontsize=9)
for ax in axes:
    ax.grid(axis="x", alpha=.2)
fig.suptitle("Distribuição dos gêneros; laranja = filmes sem gênero informado", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Filmes × avaliações:** a ordem dos gêneros muda conforme o denominador. Drama lidera nos dois (41,0% dos filmes e 43,9% das avaliações). Action tem 11,8% dos filmes, mas 29,8% das avaliações, e Adventure, 6,6% contra 23,3%. No sentido oposto, Documentary tem 9,0% dos filmes e só 1,3% das avaliações.
# - **Avaliações por filme:** separam os gêneros com muitos filmes pouco vistos. IMAX tem 5.453 avaliações por filme, Adventure 1.407 e Sci-Fi 1.203, contra 57,5 em Documentary. Western tem a maior fração de filmes sem nenhuma avaliação (17,4%).
# - **Gêneros com pouca participação nas avaliações:** Film-Noir (0,99%), Documentary (1,29%), Western (1,94%), Musical (3,86%) e IMAX (4,25%). Musical e IMAX têm mais avaliações por filme que Drama: a participação baixa vem de terem poucos filmes, não de filmes pouco vistos.
# - **Nota:** a média por gênero vai de 3,29 (Horror) a 3,93 (Film-Noir), com desvio-padrão entre 0,91 e 1,16. As distribuições dos gêneros se sobrepõem bastante.
# - **Filmes sem gênero:** `(no genres listed)` reúne 5.062 filmes (8,1% do catálogo), mas só 26.627 avaliações (0,11%), ou 5,3 por filme, a menor audiência entre todos os rótulos. A nota média é 3,33, com o maior desvio-padrão (1,16). Dos 5.062 filmes:
#   - 4.568 têm pelo menos uma avaliação;
#   - 320 têm pelo menos 10 avaliações e concentram 16.758 avaliações (63% das avaliações sem gênero);
#   - 38 têm pelo menos 50 avaliações.
#
# **Consequência:**
#
# - o gênero que falta é buscado no IMDb pela tabela `links` (2.2-b), o que vale também para os filmes com poucas avaliações; quem avaliou o filme serve para validar esse gênero (kNN, 2.2-c), o que só funciona quando o filme tem avaliadores suficientes;
# - os gêneros com pouca participação continuam como rótulos nos dados, porque são informação real do catálogo, e são agrupados no vetor revisado (3.8), para que não distorçam as distâncias.

# %% [markdown]
# ### 1.7 Consultas por gênero, filme e usuário
#
# Cada gênero é tratado como uma consulta: há um histograma de cinco grupos de nota para cada um dos 20 rótulos. A proporção de cada gráfico usa as avaliações daquele gênero como denominador, então as cinco faixas somam 100%. As tabelas também mostram a participação sobre as avaliações originais e sobre a expansão avaliação × gênero; essas participações podem passar de 100% no total, porque os gêneros se sobrepõem.
#
# As distribuições das médias e do número de avaliações por filme e por usuário usam a população de filmes ou de usuários como denominador. Os títulos mostram média e desvio-padrão entre objetos. Filmes só são ordenados por média com suporte mínimo de 50 avaliações.

# %%
# [1.7-a] Estatísticas por filme, gênero e usuário: média, desvio e proporções
estatisticas_filmes = (
    ratings_movies_sel
    .groupBy("movieId", "title")
    .agg(
        F.count("*").alias("quantidade_avaliacoes"),
        F.avg("rating").alias("media_avaliacoes"),
        F.stddev("rating").alias("desvio_padrao"),
    )
    .withColumn("proporcao_total", F.col("quantidade_avaliacoes") / F.lit(total_avaliacoes))
    .withColumn("participacao_total_pct", F.col("proporcao_total") * 100)
    .select("movieId", "title", "quantidade_avaliacoes",
            F.round("proporcao_total", 10).alias("proporcao_total"),
            F.round("participacao_total_pct", 8).alias("participacao_total_pct"),
            F.round("media_avaliacoes", 4).alias("media_avaliacoes"),
            F.round("desvio_padrao", 4).alias("desvio_padrao"))
)

melhores_filmes_com_suporte = (
    estatisticas_filmes
    .filter(F.col("quantidade_avaliacoes") >= 50)
    .orderBy(F.col("media_avaliacoes").desc(), F.col("quantidade_avaliacoes").desc())
)

# Cada filme pode ter mais de um gênero; aqui a proporção usa o total de linhas
# da tabela expandida avaliação × gênero, não o total de avaliações originais.
estatisticas_generos = (
    ratings_genres_sel
    .groupBy("genre")
    .agg(
        F.count("*").alias("quantidade_avaliacoes_genero"),
        F.avg("rating").alias("media_avaliacoes"),
        F.stddev("rating").alias("desvio_padrao"),
    )
    .withColumn("proporcao_da_expansao",
                F.col("quantidade_avaliacoes_genero") / F.lit(linhas_ratings_genres_sel))
    .withColumn("participacao_expansao_pct", F.col("proporcao_da_expansao") * 100)
    .withColumn("equivalente_sobre_avaliacoes_originais",
                F.col("quantidade_avaliacoes_genero") / F.lit(total_avaliacoes))
    .select("genre", "quantidade_avaliacoes_genero",
            F.round("proporcao_da_expansao", 6).alias("proporcao_da_expansao"),
            F.round("participacao_expansao_pct", 4).alias("participacao_expansao_pct"),
            F.round("equivalente_sobre_avaliacoes_originais", 10).alias("equivalente_sobre_avaliacoes_originais"),
            F.round("media_avaliacoes", 4).alias("media_avaliacoes"),
            F.round("desvio_padrao", 4).alias("desvio_padrao"))
    .orderBy(F.col("media_avaliacoes").desc())
)

estatisticas_usuarios = (
    ratings_sel
    .groupBy("userId")
    .agg(
        F.count("*").alias("quantidade_avaliacoes"),
        F.avg("rating").alias("media_avaliacoes"),
        F.stddev("rating").alias("desvio_padrao"),
    )
    .withColumn("proporcao_total", F.col("quantidade_avaliacoes") / F.lit(total_avaliacoes))
    .withColumn("participacao_total_pct", F.col("proporcao_total") * 100)
    .select("userId", "quantidade_avaliacoes",
            F.round("proporcao_total", 10).alias("proporcao_total"),
            F.round("participacao_total_pct", 8).alias("participacao_total_pct"),
            F.round("media_avaliacoes", 4).alias("media_avaliacoes"),
            F.round("desvio_padrao", 4).alias("desvio_padrao"))
)

print("Filmes com melhor média (mínimo 50 avaliações):")
melhores_filmes_com_suporte.show(20, truncate=False)
print("Média, desvio e proporção por gênero; a expansão por gêneros é o denominador:")
estatisticas_generos.show(30, truncate=False)
print("Média, desvio e participação de cada usuário no total das avaliações:")
estatisticas_usuarios.orderBy(F.desc("quantidade_avaliacoes")).show(20, truncate=False)
print("Em estatisticas_filmes/usuarios, proporcao_total = contagem / 25.000.095. "
      "Em estatisticas_generos, proporcao_da_expansao = contagem / 67.809.886; gêneros se sobrepõem.")

# %%
# [1.7-b] Histogramas das consultas por gênero, das médias e do volume por objeto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

generos_consulta = [r["genre"] for r in movie_genres_sel.select("genre").distinct().orderBy("genre").collect()]
ratings_por_genero = ratings_genres_sel.filter(F.col("genre").isin(generos_consulta))
consulta_5 = (ratings_por_genero.withColumn("grupo_nota", F.ceil("rating").cast("int"))
    .groupBy("genre", "grupo_nota").count())
consulta_totais = consulta_5.groupBy("genre").agg(F.sum("count").alias("total_genero"))
consulta_5_pd = (consulta_5.join(consulta_totais, "genre")
    .withColumn("proporcao_no_genero_pct", F.col("count") / F.col("total_genero") * 100)
    .withColumn("participacao_total_avaliacoes_pct", F.col("count") / F.lit(total_avaliacoes) * 100)
    .withColumn("participacao_expansao_pct", F.col("count") / F.lit(linhas_ratings_genres_sel) * 100)
    .toPandas())
indice_completo = pd.MultiIndex.from_product([generos_consulta, [1, 2, 3, 4, 5]], names=["genre", "grupo_nota"])
consulta_5_pd = consulta_5_pd.set_index(["genre", "grupo_nota"]).reindex(indice_completo, fill_value=0).reset_index()
resumo_consultas = (ratings_por_genero.groupBy("genre")
    .agg(F.count("*").alias("n_avaliacoes"), F.avg("rating").alias("media"), F.stddev("rating").alias("dp"))
    .withColumn("participacao_total_avaliacoes_pct", F.col("n_avaliacoes") / F.lit(total_avaliacoes) * 100)
    .withColumn("participacao_expansao_pct", F.col("n_avaliacoes") / F.lit(linhas_ratings_genres_sel) * 100)
    .orderBy("genre").toPandas())
assert len(generos_consulta) == 20
assert consulta_5_pd.groupby("genre")["grupo_nota"].nunique().eq(5).all()
assert consulta_5_pd.groupby("genre")["proporcao_no_genero_pct"].sum().sub(100).abs().max() < 1e-8
print("Cada histograma usa cinco grupos. Proporções dentro do gênero somam 100%.")
print("A participação no total original usa 25.000.095 avaliações; a participação na expansão usa 67.809.886 linhas avaliação×gênero.")
print("Gêneros se sobrepõem; suas participações no total original podem somar mais de 100%.")
print("Contagens por gênero e grupo de nota:")
print(consulta_5_pd.pivot(index="genre", columns="grupo_nota", values="count").fillna(0).astype(int).to_string())
print("Proporções internas por gênero (%):")
print(consulta_5_pd.pivot(index="genre", columns="grupo_nota", values="proporcao_no_genero_pct").fillna(0).round(2).to_string())
print("Média, DP e participação por consulta:")
print(resumo_consultas.to_string(index=False, formatters={"media": "{:.4f}".format, "dp": "{:.4f}".format,
    "participacao_total_avaliacoes_pct": "{:.4f}%".format, "participacao_expansao_pct": "{:.4f}%".format}))

fig, axes = plt.subplots(4, 5, figsize=(19, 13), sharey=True)
cores = ["#397461", "#d18b24", "#7656a5", "#4682a9", "#b65d5d"]
for ax, genero in zip(axes.flat, generos_consulta):
    dados = consulta_5_pd[consulta_5_pd.genre == genero].sort_values("grupo_nota")
    resumo = resumo_consultas[resumo_consultas.genre == genero].iloc[0]
    ax.bar(dados["grupo_nota"].astype(str), dados["proporcao_no_genero_pct"], color=cores, edgecolor="#333333", linewidth=.4)
    ax.set_title(f"{genero} — média {resumo.media:.2f}; DP {resumo.dp:.2f}; n={int(resumo.n_avaliacoes):,}", fontsize=9)
    ax.set_xticks(range(5), ["1", "2", "3", "4", "5"])
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
for ax in axes[:, 0]: ax.set_ylabel("Avaliações do gênero (%)")
fig.suptitle("Histogramas das consultas por gênero: cinco grupos de nota", y=1.01)
plt.tight_layout()
plt.show()

# Para filmes e usuários, cada barra mostra o percentual dos objetos naquela faixa.
medias_filme = estatisticas_filmes.select("media_avaliacoes").toPandas()["media_avaliacoes"].dropna()
medias_usuario = estatisticas_usuarios.select("media_avaliacoes").toPandas()["media_avaliacoes"].dropna()
contagens_filme = estatisticas_filmes.select("quantidade_avaliacoes").toPandas()["quantidade_avaliacoes"]
contagens_usuario = estatisticas_usuarios.select("quantidade_avaliacoes").toPandas()["quantidade_avaliacoes"]
for nome, valores in [("Média por filme", medias_filme), ("Média por usuário", medias_usuario)]:
    print(f"{nome}: média={valores.mean():.4f}; DP={valores.std(ddof=1):.4f}; n={len(valores):,}")
for nome, valores in [("Avaliações por filme", contagens_filme), ("Avaliações por usuário", contagens_usuario)]:
    print(f"{nome}: média={valores.mean():.2f}; DP={valores.std(ddof=1):.2f}; n={len(valores):,}")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, valores, nome, cor in [(axes[0,0], medias_filme, "Média das notas por filme", "#397461"),
                                (axes[0,1], medias_usuario, "Média das notas por usuário", "#397461")]:
    ax.hist(valores, bins=30, weights=np.full(len(valores), 100/len(valores)), color=cor, edgecolor="white")
    ax.set_title(f"{nome}\nμ={valores.mean():.3f}; DP={valores.std(ddof=1):.3f}; n={len(valores):,}")
    ax.set_xlabel("Nota média (0,5–5,0)"); ax.set_ylabel("Proporção de objetos (%)"); ax.grid(axis="y", alpha=.2)
for ax, valores, nome, cor in [(axes[1,0], contagens_filme, "Número de avaliações por filme", "#d18b24"),
                                (axes[1,1], contagens_usuario, "Número de avaliações por usuário", "#d18b24")]:
    ax.hist(valores, bins=30, weights=np.full(len(valores), 100/len(valores)), color=cor, edgecolor="white")
    ax.set_xscale("log")
    ax.set_title(f"{nome}\nμ={valores.mean():.2f}; DP={valores.std(ddof=1):.2f}; n={len(valores):,}")
    ax.set_xlabel("Avaliações por objeto (eixo x logarítmico)"); ax.set_ylabel("Proporção de objetos (%)"); ax.grid(axis="y", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Por gênero:**
#   - a média varia de 3,29 (Horror) a 3,93 (Film-Noir), cerca de 0,6 ponto entre os extremos;
#   - gêneros de nicho e clássicos (Film-Noir, War, Documentary) têm média maior e desvio-padrão menor; Horror, Comedy e Children, média menor e desvio maior;
#   - nos histogramas, o grupo 5 representa 35,34% das avaliações de Film-Noir e 18,06% das de Horror.
# - **Por filme (suporte ≥ 50 avaliações):** os primeiros lugares misturam documentários de natureza, como Planet Earth II (4,48; 1.124 avaliações), e clássicos muito avaliados, como The Shawshank Redemption (4,41; 81.482 avaliações).
# - **Média entre objetos:** a média das médias por filme (3,07) é bem menor que a média das médias por usuário (3,68). Cada objeto pesa igual: há muitos filmes pouco avaliados com médias baixas, enquanto os usuários avaliam sobretudo filmes populares e bem avaliados.
# - **Volume:** o desvio-padrão do número de avaliações supera a média (filmes: 423,4 ± 2.477,9; usuários: 153,8 ± 268,1). Isso indica cauda longa; por isso os histogramas de volume usam eixo logarítmico.

# %% [markdown]
# #### Consultas com filtro
#
# Além dos gêneros, cada consulta compara dois grupos de avaliações com o mesmo histograma de cinco grupos de nota:
#
# - filmes populares (1% mais avaliados) × filmes de nicho (até a mediana de avaliações por filme);
# - filmes antigos (antes de 1970) × recentes (2010 em diante), pelo ano do título;
# - avaliações feitas antes de 2003 × a partir de 2003, quando surgiram as notas de meio ponto;
# - usuários pouco ativos (25% com menos avaliações) × muito ativos (25% com mais);
# - filmes sem gênero informado × com gênero.
#
# Cada histograma usa as avaliações do grupo como denominador, e a legenda traz média, desvio-padrão e número de avaliações. A diferença entre os grupos também é medida pelo d de Cohen (diferença de médias dividida pelo desvio-padrão combinado), porque com milhões de avaliações qualquer diferença seria estatisticamente significativa.

# %%
# [1.7-c] Consultas com filtro: histogramas de cinco grupos de nota, com média e desvio-padrão
q_volume_filme = estatisticas_filmes.approxQuantile("quantidade_avaliacoes", [0.5, 0.99], 0.0)
q_volume_usuario = estatisticas_usuarios.approxQuantile("quantidade_avaliacoes", [0.25, 0.75], 0.0)
LIMITE_NICHO, LIMITE_POPULAR = q_volume_filme
LIMITE_POUCO_ATIVO, LIMITE_MUITO_ATIVO = q_volume_usuario
ANO_ANTIGO, ANO_RECENTE, ANO_MEIO_PONTO = 1970, 2010, 2003
print(f"Filmes de nicho: até {LIMITE_NICHO:.0f} avaliações (mediana); populares: {LIMITE_POPULAR:.0f} ou mais (1% mais avaliados)")
print(f"Usuários pouco ativos: até {LIMITE_POUCO_ATIVO:.0f} avaliações (Q1); muito ativos: {LIMITE_MUITO_ATIVO:.0f} ou mais (Q3)")

ROTULOS = {
    "popular": "populares (1% mais avaliados)", "nicho": "nicho (até a mediana)",
    "antigo": f"antigos (antes de {ANO_ANTIGO})", "recente": f"recentes ({ANO_RECENTE} em diante)",
    "antes": f"antes de {ANO_MEIO_PONTO}", "depois": f"{ANO_MEIO_PONTO} em diante",
    "pouco": "pouco ativos (25% menos)", "muito": "muito ativos (25% mais)",
    "sem": "sem gênero", "com": "com gênero",
}
filmes_consulta = (estatisticas_filmes.select("movieId", "quantidade_avaliacoes")
    .join(movies_sel.select("movieId", "title", "genres"), "movieId")
    .withColumn("ano_txt", F.regexp_extract(F.trim(F.col("title")), r"\((\d{4})\)$", 1))
    .withColumn("ano_lanc", F.when(F.col("ano_txt") != "", F.col("ano_txt").cast("int")))
    .select("movieId",
            F.when(F.col("quantidade_avaliacoes") >= LIMITE_POPULAR, ROTULOS["popular"])
             .when(F.col("quantidade_avaliacoes") <= LIMITE_NICHO, ROTULOS["nicho"])
             .otherwise("intermediário").alias("popularidade"),
            F.when(F.col("ano_lanc") < ANO_ANTIGO, ROTULOS["antigo"])
             .when(F.col("ano_lanc") >= ANO_RECENTE, ROTULOS["recente"])
             .when(F.col("ano_lanc").isNotNull(), "intermediário").otherwise("sem ano").alias("epoca"),
            F.when(F.col("genres") == "(no genres listed)", ROTULOS["sem"]).otherwise(ROTULOS["com"]).alias("genero_informado")))
usuarios_consulta = estatisticas_usuarios.select("userId",
    F.when(F.col("quantidade_avaliacoes") <= LIMITE_POUCO_ATIVO, ROTULOS["pouco"])
     .when(F.col("quantidade_avaliacoes") >= LIMITE_MUITO_ATIVO, ROTULOS["muito"])
     .otherwise("intermediário").alias("atividade"))
# Uma única passagem pelas avaliações: contagem, soma e soma dos quadrados por combinação de filtros e grupo de nota.
tabela_consultas = (ratings_sel
    .join(F.broadcast(filmes_consulta), "movieId")
    .join(usuarios_consulta, "userId")
    .withColumn("periodo", F.when(F.year(F.from_unixtime("timestamp")) < ANO_MEIO_PONTO, ROTULOS["antes"])
                .otherwise(ROTULOS["depois"]))
    .withColumn("grupo_nota", F.ceil("rating").cast("int"))
    .groupBy("popularidade", "epoca", "genero_informado", "atividade", "periodo", "grupo_nota")
    .agg(F.count("*").alias("n"), F.sum("rating").alias("soma"), F.sum(F.col("rating") * F.col("rating")).alias("soma_q"))
    .toPandas())
assert int(tabela_consultas["n"].sum()) == total_avaliacoes

DEFINICAO_CONSULTAS = [
    ("popularidade", ("popular", "nicho"), "Filmes populares × de nicho"),
    ("epoca", ("antigo", "recente"), "Filmes antigos × recentes (ano do título)"),
    ("periodo", ("antes", "depois"), f"Avaliações antes × depois de {ANO_MEIO_PONTO}"),
    ("atividade", ("pouco", "muito"), "Usuários pouco × muito ativos"),
    ("genero_informado", ("sem", "com"), "Filmes sem gênero × com gênero"),
]
linhas_consulta, efeitos = [], []
fig, axes = plt.subplots(1, 5, figsize=(26, 5.5), sharey=True)
for ax, (coluna, chaves, titulo) in zip(axes, DEFINICAO_CONSULTAS):
    resumo_grupos = []
    for j, chave in enumerate(chaves):
        parte = tabela_consultas[tabela_consultas[coluna] == ROTULOS[chave]]
        n = int(parte["n"].sum())
        media = parte["soma"].sum() / n
        dp = ((parte["soma_q"].sum() - n * media ** 2) / (n - 1)) ** 0.5
        hist = parte.groupby("grupo_nota")["n"].sum().reindex([1, 2, 3, 4, 5], fill_value=0) / n * 100
        ax.bar(np.arange(1, 6) + (j - 0.5) * 0.4, hist.values, width=0.4, color=("#4682a9", "#d18b24")[j],
               label=f"{ROTULOS[chave]}\nμ={media:.2f}; DP={dp:.2f}; n={n:,}")
        linhas_consulta.append((titulo, ROTULOS[chave], n, 100 * n / total_avaliacoes, media, dp, *hist.values))
        resumo_grupos.append((n, media, dp))
    (n1, m1, s1), (n2, m2, s2) = resumo_grupos
    dp_combinado = (((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) ** 0.5
    efeitos.append((titulo, m1 - m2, (m1 - m2) / dp_combinado))
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("grupo de nota (1 a 5)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=.2)
axes[0].set_ylabel("% das avaliações do grupo")
fig.suptitle("Consultas com filtro: distribuição dos cinco grupos de nota em cada par de grupos", y=1.02)
plt.tight_layout()
plt.show()

tabela_consulta = pd.DataFrame(linhas_consulta, columns=["consulta", "grupo", "n_avaliacoes", "%_do_total", "média", "DP",
                                                         "g1_%", "g2_%", "g3_%", "g4_%", "g5_%"])
print("Denominador de cada histograma = avaliações do grupo; %_do_total usa 25.000.095 avaliações.")
print(tabela_consulta.round(2).to_string(index=False))
print("\nDiferença de médias entre os dois grupos de cada consulta e d de Cohen (diferença / DP combinado):")
print(pd.DataFrame(efeitos, columns=["consulta", "diferença de médias", "d de Cohen"]).round(3).to_string(index=False))

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Popularidade:** é o filtro com maior efeito. Os filmes populares (1% mais avaliados, com 9.943 avaliações ou mais) têm média 3,72 e somam 47,6% de todas as avaliações; os de nicho (até 6 avaliações) têm média 2,98 e 10,2% de notas no grupo 1. O d de Cohen é 0,73, perto de um efeito grande.
# - **Época do filme:** filmes antigos (antes de 1970) têm média 3,82, contra 3,54 dos recentes (d = 0,28).
# - **Atividade do usuário:** os 25% menos ativos (até 36 avaliações) dão média 3,71; os 25% mais ativos (162 ou mais) dão 3,45 e respondem por 69,9% das avaliações (d = 0,24).
# - **Filmes sem gênero:** média 3,33, contra 3,53 dos demais (d = −0,20), em apenas 0,11% das avaliações.
# - **Antes × depois de 2003:** médias quase iguais (3,56 × 3,52; d = 0,03). A mudança de escala altera a forma do histograma (menos notas no grupo 3, mais no 4), não a média.
#
# **Limitação:** popularidade e nota se influenciam mutuamente: filmes bons atraem avaliações. A consulta descreve a associação, não a causa.
#
# **Consequência:** a popularidade dos filmes vistos é um atributo derivado relevante do usuário e entra no vetor revisado (3.8).

# %% [markdown]
# ### 1.8 Exploração multivariada
#
# No MovieLens, as colunas brutas são identificadores, texto, categorias ou a própria nota. Por isso, a análise multivariada usa atributos derivados por usuário e por filme, com denominadores explícitos, como orientado pelo professor.
#
# A ordem de apresentação é: covariância, razão de correlação, Pearson, Spearman e visualização por matriz de dispersão. A correlação não demonstra causalidade.
#
# Nesta etapa, o perfil de usuário usa um conjunto compacto de atributos. O perfil completo, com os 20 gêneros, é construído na transformação (3.1) a partir dos dados tratados.
#
# **Escala e proporção antes das matrizes.** Antes de cada matriz, todos os atributos são levados para [0, 1]: contagens divididas pelo máximo, nota pela posição na escala de 0,5 a 5,0, desvio-padrão pelo maior valor observado e ano entre o primeiro e o último ano. As proporções já estão nesse intervalo. Assim, a covariância fica comparável entre atributos; a covariância na escala original aparece como consulta. Pearson e Spearman não mudam com essas transformações lineares.

# %%
# [1.8-a] Covariância, Pearson e Spearman no grão de usuário e matriz de dispersão
# Denominadores: prop_avaliacoes_total = avaliações do usuário / total de avaliações;
# prop_positivas = avaliações do usuário com nota >= 4 / avaliações do usuário;
# prop_<gênero> = avaliações do usuário em filmes do gênero / avaliações do usuário.
usuarios_base = (ratings_sel.groupBy("userId").agg(
    F.count("*").alias("n_avaliacoes"), F.avg("rating").alias("nota_media"),
    F.stddev("rating").alias("nota_dp"),
    F.avg(F.when(F.col("rating") >= 4, 1.0).otherwise(0.0)).alias("prop_positivas"))
    .withColumn("prop_avaliacoes_total", F.col("n_avaliacoes") / F.lit(total_avaliacoes)))
generos_usuario = ratings_genres_sel.groupBy("userId").agg(*[
    F.sum(F.when(F.col("genre") == genero, 1).otherwise(0)).alias("n_" + genero.lower())
    for genero in ("Action", "Comedy", "Drama")])
usuarios_features = usuarios_base.join(generos_usuario, "userId", "left")
for genero in ("action", "comedy", "drama"):
    usuarios_features = usuarios_features.withColumn("prop_" + genero,
        F.coalesce(F.col("n_" + genero), F.lit(0)) / F.col("n_avaliacoes"))
atributos = ["prop_avaliacoes_total", "nota_media", "nota_dp", "prop_positivas",
             "prop_action", "prop_comedy", "prop_drama"]
usuarios_pd = usuarios_features.select("userId", *atributos).orderBy("userId").toPandas()  # ordem fixa para a amostra
n_usuarios_total = ratings_sel.select("userId").distinct().count()
assert len(usuarios_pd) == usuarios_pd.userId.nunique() == n_usuarios_total
assert usuarios_pd[atributos].notna().all().all()
assert usuarios_pd[[c for c in atributos if c.startswith("prop_")]].le(1).all().all()
assert abs(usuarios_pd["prop_avaliacoes_total"].sum() - 1) < 1e-10

covariancia = usuarios_pd[atributos].cov()
pearson = usuarios_pd[atributos].corr(method="pearson")
spearman = usuarios_pd[atributos].corr(method="spearman")
assert np.allclose(pearson.values, pearson.values.T)
assert np.allclose(spearman.values, spearman.values.T)
print("Usuários:", len(usuarios_pd))
print("Soma de prop_avaliacoes_total:", usuarios_pd["prop_avaliacoes_total"].sum())
# Escala e proporção antes da matriz: cada atributo vai para [0, 1] pelo seu domínio.
#   atividade_rel = avaliações do usuário / avaliações do usuário mais ativo (= prop_avaliacoes_total / máximo);
#   nota_media_rel = (nota_media − 0,5) / 4,5, a posição na escala de 0,5 a 5,0;
#   nota_dp_rel = nota_dp / maior DP observado; as proporções já estão em [0, 1].
usuarios_escala = pd.DataFrame({
    "atividade_rel": usuarios_pd["prop_avaliacoes_total"] / usuarios_pd["prop_avaliacoes_total"].max(),
    "nota_media_rel": (usuarios_pd["nota_media"] - 0.5) / 4.5,
    "nota_dp_rel": usuarios_pd["nota_dp"] / usuarios_pd["nota_dp"].max(),
    "prop_positivas": usuarios_pd["prop_positivas"],
    "prop_action": usuarios_pd["prop_action"],
    "prop_comedy": usuarios_pd["prop_comedy"],
    "prop_drama": usuarios_pd["prop_drama"],
})
assert usuarios_escala.min().min() >= 0 and usuarios_escala.max().max() <= 1 + 1e-12
print("Atributos de usuário em escala [0, 1]:")
print(usuarios_escala.describe().T[["mean", "std", "50%", "min", "max"]].round(4).to_string())
print("Covariância na escala [0, 1] (comparável entre atributos):")
print(usuarios_escala.cov().round(5).to_string())
print("Covariância na escala original (consulta; as unidades diferem entre atributos):")
print(covariancia.to_string(float_format=lambda x: f"{x:.6g}"))
fig, axes = plt.subplots(2, 4, figsize=(21, 8.5))
for ax, coluna in zip(axes.flat, usuarios_escala.columns):
    histograma_media_dp(ax, usuarios_escala[coluna], coluna, bins=40, rotulo_x="escala [0, 1]")
axes.flat[-1].axis("off")
fig.suptitle(f"Atributos de usuário em escala [0, 1], antes da matriz (n = {len(usuarios_escala):,})", y=1.01)
plt.tight_layout()
plt.show()
print("Pearson:\n", pearson.round(3).to_string())
print("Spearman:\n", spearman.round(3).to_string())
pares = [(a, b, pearson.loc[a, b], spearman.loc[a, b]) for i, a in enumerate(atributos) for b in atributos[i + 1:]]
for a, b, r, rho in sorted(pares, key=lambda x: abs(x[2]), reverse=True)[:5]:
    print(f"{a} × {b}: Pearson={r:.3f}; Spearman={rho:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, matriz, titulo in zip(axes, (pearson, spearman), ("Pearson", "Spearman")):
    imagem = ax.imshow(matriz, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(atributos)), atributos, rotation=55, ha="right")
    ax.set_yticks(range(len(atributos)), atributos)
    ax.set_title(titulo + " por usuário")
fig.colorbar(imagem, ax=axes, shrink=.75)
plt.show()

# Matriz de dispersão em amostra fixa de 3.000 usuários; a representatividade é verificada em 2.4.
cols_grafico = ["nota_media", "nota_dp", "prop_positivas", "prop_action", "prop_comedy", "prop_drama"]
amostra_scatter = usuarios_pd[cols_grafico].sample(n=min(3000, len(usuarios_pd)), random_state=42)
pd.plotting.scatter_matrix(amostra_scatter, figsize=(11, 11), diagonal="hist", alpha=0.15, s=5)
plt.suptitle("Matriz de dispersão de atributos de usuário — amostra fixa n=3.000", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Média × proporção positiva:** associação muito forte (Pearson 0,922; Spearman 0,946). As duas medem a generosidade do usuário.
# - **Composição de gêneros:** as proporções competem entre si: Action × Drama (−0,52) e Action × Comedy (−0,42). Quem avalia proporcionalmente mais ação avalia proporcionalmente menos drama e comédia; parte desse efeito vem do próprio uso de proporções.
# - **Média × desvio da nota:** −0,47. Usuários com média maior tendem a variar menos.
# - **Atividade:** `prop_avaliacoes_total` se associa pouco à média (−0,17 / −0,13); usuários muito ativos dão notas ligeiramente menores.
# - **Pearson × Spearman:** os valores são próximos, o que indica relações aproximadamente lineares.
#
# - **Escala [0, 1]:** a atividade tem média 0,0048 e mediana 0,0022. Quase todos os usuários ficam perto de zero, porque o mais ativo tem 32.202 avaliações. Na escala comparável, as maiores variâncias são da proporção positiva (0,040) e das proporções de gênero (0,018 a 0,022), e a maior covariância fora da diagonal é entre nota média e proporção positiva (0,0195), coerente com a correlação de 0,922. Na escala original, as variâncias iam de 1,1×10⁻¹⁰ a 0,23 e não permitiam essa comparação.

# %% [markdown]
# #### Covariância e correlações no grão de filme
#
# Cada linha representa um filme, com participação no total de avaliações, média, desvio-padrão e ano de lançamento. A participação é n_avaliacoes / total_avaliacoes. O ano é extraído do título. Filmes sem ano ou sem desvio-padrão calculável ficam fora desta matriz, e essa população é informada na saída.

# %%
# [1.8-b] Covariância, Pearson e Spearman no grão de filme
# A participação de avaliações usa como denominador o total global de avaliações válidas.
filmes_cov_spark = (
    ratings_movies_sel.groupBy("movieId", "title")
    .agg(
        F.count("*").alias("n_avaliacoes"),
        F.avg("rating").alias("media_nota"),
        F.stddev("rating").alias("dp_nota"),
    )
    .withColumn("prop_avaliacoes_total", F.col("n_avaliacoes") / F.lit(total_avaliacoes))
    .withColumn("ano_texto", F.regexp_extract(F.col("title"), r"\((\d{4})\)$", 1))
    .withColumn("ano_lancamento", F.when(F.col("ano_texto") != "", F.col("ano_texto").cast("double")))
    .select("movieId", "n_avaliacoes", "prop_avaliacoes_total", "media_nota", "dp_nota", "ano_lancamento")
)
filmes_cov_pd = filmes_cov_spark.toPandas().dropna()
atributos_filme_cov = ["prop_avaliacoes_total", "media_nota", "dp_nota", "ano_lancamento"]
assert len(filmes_cov_pd) == filmes_cov_pd["movieId"].nunique() > 1
assert filmes_cov_pd["prop_avaliacoes_total"].between(0, 1).all()
assert 0 < filmes_cov_pd["prop_avaliacoes_total"].sum() <= 1
print(f"Filmes incluídos (ano e DP disponíveis): {len(filmes_cov_pd):,}")
print(f"Denominador da proporção: {total_avaliacoes:,} avaliações válidas")
print("Matriz por filme usa a participação no total, a média, o DP e o ano; a contagem bruta fica para consulta.")
# Escala e proporção antes da matriz: cada atributo do filme vai para [0, 1].
ano_min, ano_max = filmes_cov_pd["ano_lancamento"].min(), filmes_cov_pd["ano_lancamento"].max()
filmes_escala = pd.DataFrame({
    "atividade_rel": filmes_cov_pd["prop_avaliacoes_total"] / filmes_cov_pd["prop_avaliacoes_total"].max(),
    "media_rel": (filmes_cov_pd["media_nota"] - 0.5) / 4.5,
    "dp_rel": filmes_cov_pd["dp_nota"] / filmes_cov_pd["dp_nota"].max(),
    "ano_rel": (filmes_cov_pd["ano_lancamento"] - ano_min) / (ano_max - ano_min),
})
assert filmes_escala.min().min() >= 0 and filmes_escala.max().max() <= 1 + 1e-12
print(f"Atributos de filme em escala [0, 1] (ano: {ano_min:.0f} = 0 e {ano_max:.0f} = 1):")
print(filmes_escala.describe().T[["mean", "std", "50%", "min", "max"]].round(4).to_string())
print("Covariância por filme na escala [0, 1] (comparável entre atributos):")
print(filmes_escala.cov().round(5).to_string())
print("Covariância por filme na escala original (consulta):")
print(filmes_cov_pd[atributos_filme_cov].cov().to_string(float_format=lambda x: f"{x:.6g}"))
fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
for ax, coluna in zip(axes, filmes_escala.columns):
    histograma_media_dp(ax, filmes_escala[coluna], coluna, bins=40, cor="#397461", rotulo_x="escala [0, 1]")
fig.suptitle(f"Atributos de filme em escala [0, 1], antes da matriz (n = {len(filmes_escala):,})", y=1.03)
plt.tight_layout()
plt.show()
print("Pearson por filme:")
print(filmes_cov_pd[atributos_filme_cov].corr(method="pearson").round(3).to_string())
print("Spearman por filme:")
print(filmes_cov_pd[atributos_filme_cov].corr(method="spearman").round(3).to_string())

# %% [markdown]
# **Leitura dos resultados.** Entre filmes, a participação no total de avaliações se associa positivamente à média (Pearson 0,132; Spearman 0,179): filmes mais avaliados tendem a ter média um pouco maior. O ano de lançamento quase não tem relação linear com a média (−0,027), mas a razão η a seguir mostra uma associação não linear por década.
#
# **Escala [0, 1]:** o ano tem a maior variância (0,0285) e a atividade a menor (0,0011; mediana 0,0001, porque o filme mais avaliado tem 81.491 avaliações). A maior covariância fora da diagonal é entre DP da nota e ano (0,0029): filmes mais recentes têm notas um pouco mais dispersas, o que confirma o Pearson de 0,128. Na escala original, a variância do ano (599) escondia todas as outras.

# %% [markdown]
# #### Razão de correlação η
#
# Mede a associação entre uma variável categórica (década de lançamento) e uma numérica (média por filme). Só entram filmes com pelo menos 50 avaliações e décadas com pelo menos 20 filmes. Cada filme tem o mesmo peso. O resultado é uma associação descritiva, não causal.

# %%
# [1.8-c] Razão de correlação η: década de lançamento × média por filme
MIN_AVALIACOES_FILME = 50
MIN_FILMES_DECADA = 20
estatisticas_filmes = estatisticas_filmes.cache()
filmes_com_suporte = estatisticas_filmes.filter(F.col('quantidade_avaliacoes') >= MIN_AVALIACOES_FILME)
total_com_suporte = filmes_com_suporte.count()
filmes_com_ano = (filmes_com_suporte
    .withColumn('ano_texto', F.regexp_extract(F.col('title'), r'\((\d{4})\)$', 1))
    .filter(F.col('ano_texto') != '')
    .withColumn('ano', F.col('ano_texto').cast('int'))
    .withColumn('decada', (F.floor(F.col('ano') / 10) * 10).cast('int'))
    .select('movieId', 'decada', 'media_avaliacoes'))
total_com_ano = filmes_com_ano.count()
contagens_decada = filmes_com_ano.groupBy('decada').count()
decadas_elegiveis = contagens_decada.filter(F.col('count') >= MIN_FILMES_DECADA).select('decada')
filmes_decada = filmes_com_ano.join(decadas_elegiveis, 'decada', 'inner').cache()
resumo_decada = (filmes_decada.groupBy('decada')
    .agg(F.count('*').alias('n_filmes'), F.avg('media_avaliacoes').alias('media_das_medias'))
    .orderBy('decada').toPandas())
assert len(resumo_decada) >= 2
n_filmes_analise = int(resumo_decada['n_filmes'].sum())
media_global = (resumo_decada['n_filmes'] * resumo_decada['media_das_medias']).sum() / n_filmes_analise
ss_entre = (resumo_decada['n_filmes'] * (resumo_decada['media_das_medias'] - media_global) ** 2).sum()
ss_total = filmes_decada.agg(F.sum((F.col('media_avaliacoes') - F.lit(float(media_global))) ** 2)).first()[0]
assert ss_total > 0
eta_quadrado = float(ss_entre / ss_total)
assert -1e-10 <= eta_quadrado <= 1 + 1e-10
eta = eta_quadrado ** 0.5
print('Filmes com >=50 avaliações:', total_com_suporte)
print('Destes, com ano no fim do título:', total_com_ano)
print('Filmes nas décadas com >=20 filmes:', n_filmes_analise)
print('Décadas incluídas:', len(resumo_decada))
print(f'Razão de correlação η={eta:.4f}; η²={eta_quadrado:.4f}')
print(resumo_decada.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(resumo_decada['decada'].astype(str), resumo_decada['media_das_medias'], color='#397461', edgecolor='#1f2d28')
ax.set(title='Média das avaliações por filme e década de lançamento', xlabel='Década extraída do título', ylabel='Média das notas por filme (0,5 a 5)', ylim=(0, 5))
ax.grid(axis='y', alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.** A década responde por 6,7% da variação das médias por filme (η = 0,2589; η² = 0,0670).
#
# - As médias caem de 3,62 (anos 1920) para 3,15 (anos 1980) e voltam a subir até 3,31 (anos 2010).
# - O padrão não é linear, o que explica o Pearson próximo de zero.
# - Uma explicação possível é a sobrevivência: filmes antigos que ainda recebem 50 ou mais avaliações tendem a ser clássicos.
#
# A associação é descritiva, não causal.

# %% [markdown]
# #### Testes estatísticos e tamanhos de efeito
#
# Os coeficientes descritivos são complementados por testes de hipótese:
#
# - **(a)** significância e intervalo de confiança de 95% das correlações por usuário;
# - **(b)** ANOVA de um fator e Kruskal-Wallis para a média por filme entre décadas, na mesma população da razão η;
# - **(c)** qui-quadrado de independência entre cada gênero e os cinco grupos de nota.
#
# No teste (c), cada avaliação é contada uma vez por teste (avaliações do gênero contra as demais), o que respeita a independência mesmo com gêneros multirrótulo. Com milhões de observações, valores-p muito pequenos são esperados mesmo para diferenças irrelevantes. Por isso, a leitura usa também o tamanho de efeito: r, η² e V de Cramér.

# %%
# [1.8-d] Testes estatísticos e tamanhos de efeito
from scipy import stats

# (a) Significância e intervalo de confiança de 95% das correlações por usuário (transformação z de Fisher).
n_u = len(usuarios_pd)
pares_teste = [("nota_media", "prop_positivas"), ("prop_action", "prop_drama"), ("nota_media", "nota_dp"),
               ("prop_action", "prop_comedy"), ("nota_dp", "prop_positivas"), ("prop_avaliacoes_total", "nota_media")]
linhas_corr = []
for a, b in pares_teste:
    r, p_r = stats.pearsonr(usuarios_pd[a], usuarios_pd[b])
    rho, p_rho = stats.spearmanr(usuarios_pd[a], usuarios_pd[b])
    z, ep = np.arctanh(r), 1 / np.sqrt(n_u - 3)
    ic_inf, ic_sup = np.tanh([z - 1.96 * ep, z + 1.96 * ep])
    linhas_corr.append((f"{a} × {b}", r, ic_inf, ic_sup, p_r, rho, p_rho))
testes_corr = pd.DataFrame(linhas_corr, columns=["par", "pearson", "ic95_inf", "ic95_sup", "p_pearson",
                                                 "spearman", "p_spearman"])
print(f"(a) Correlações por usuário (n={n_u:,}); H0: correlação populacional = 0")
print(testes_corr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# (b) Década × média por filme: ANOVA de um fator e Kruskal-Wallis, na mesma população da razão η.
decadas_pd = filmes_decada.select("decada", "media_avaliacoes").toPandas()
grupos_decada = [g["media_avaliacoes"].to_numpy() for _, g in decadas_pd.groupby("decada")]
f_anova, p_anova = stats.f_oneway(*grupos_decada)
h_kw, p_kw = stats.kruskal(*grupos_decada)
print(f"\n(b) Década × média por filme ({len(decadas_pd):,} filmes, {len(grupos_decada)} décadas); H0: mesma média/distribuição")
print(f"ANOVA: F={f_anova:.2f}; p={p_anova:.3g}   Kruskal-Wallis: H={h_kw:.2f}; p={p_kw:.3g}   (η² já calculado = {eta_quadrado:.4f})")

# (c) Gênero × grupo de nota: qui-quadrado de independência 2×5 por gênero
# (avaliações do gênero × demais avaliações) e V de Cramér como tamanho de efeito.
total_por_grupo = h5_pd.set_index("grupo_nota")["count"].reindex([1, 2, 3, 4, 5]).to_numpy(dtype=float)
tabela_genero = consulta_5_pd.pivot(index="genre", columns="grupo_nota", values="count").reindex(columns=[1, 2, 3, 4, 5]).fillna(0)
linhas_qui = []
for genero, contagens in tabela_genero.iterrows():
    dentro = contagens.to_numpy(dtype=float)
    fora = total_por_grupo - dentro
    qui2, p_qui, gl, _ = stats.chi2_contingency(np.vstack([dentro, fora]), correction=False)
    linhas_qui.append((genero, int(dentro.sum()), qui2, gl, p_qui, np.sqrt(qui2 / total_por_grupo.sum())))
testes_genero = pd.DataFrame(linhas_qui, columns=["gênero", "avaliações", "qui2", "gl", "p", "V_cramer"]).sort_values("V_cramer", ascending=False)
print(f"\n(c) Gênero × grupo de nota (n={int(total_por_grupo.sum()):,} avaliações por teste); H0: independência")
print(testes_genero.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(testes_genero["gênero"], testes_genero["V_cramer"], color="#4682a9")
ax.set(title="Tamanho de efeito da associação gênero × grupo de nota (V de Cramér)", ylabel="V de Cramér")
ax.tick_params(axis="x", rotation=60)
ax.grid(axis="y", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **(a)** Todas as correlações testadas diferem de zero, com valor-p abaixo da precisão numérica e intervalos de confiança estreitos (por exemplo, r = 0,922, IC 95% de 0,922 a 0,923). Com 162.541 usuários, até r = −0,17 é significativo; a relevância depende do tamanho de r.
# - **(b)** A ANOVA (F = 104,9) e o Kruskal-Wallis (H = 923,4) rejeitam a igualdade das médias por filme entre décadas (p ≈ 10⁻¹⁹⁰). O efeito é modesto (η² = 0,067).
# - **(c)** Os 20 testes gênero × grupo de nota rejeitam a independência, mas os efeitos são pequenos:
#   - o maior V de Cramér é o de Drama (0,119);
#   - Comedy, Crime e Horror ficam entre 0,06 e 0,08;
#   - os demais, abaixo de 0,06.
#
# **Conclusão:** com milhões de avaliações, a significância estatística é quase automática. A interpretação deve se apoiar no tamanho de efeito. As diferenças de nota entre gêneros e décadas existem, mas são pequenas.

# %% [markdown]
# ### 1.9 Fechamento da seleção: o que a exploração indica para o pré-processamento
#
# | Achado da exploração (Seção 1) | Consequência para o pré-processamento (Seção 2) |
# |---|---|
# | A nota ocupa os 10 valores válidos e é assimétrica à esquerda; 0,5 e 1,0 ficam abaixo do whisker | conferir domínio e nulos; manter notas baixas como outliers legítimos |
# | 5.062 filmes com `(no genres listed)`, parte deles com avaliações (1.6-c) | rótulo `Unknown`; preencher o gênero pelo IMDb (2.2-b) e validar com o público do filme (kNN, 2.2-c) |
# | `imdbId`/`tmdbId` são identificadores externos | não entram como atributos de P1/P2; o `imdbId` serve de chave para buscar no IMDb o gênero e o ano que faltam (2.2-b); ausências de `tmdbId` não exigem tratamento |
# | Filmes sem ano no fim do título (1.8-c) | ano preenchido pelo IMDb (2.2-b) para o perfil revisado; a análise por década da Seção 1 continua só com o ano do título |
# | Filmes do catálogo sem avaliação | ficam fora de P1/P2, pois não têm comportamento de avaliação |
# | Volume muito assimétrico por usuário e por filme | usar proporções com denominadores explícitos e suporte mínimo; contagem bruta só descreve grupos |
# | `nota_media` × `prop_positivas` muito associadas (Pearson 0,922; Spearman 0,946) | analisar redundância na seleção de características (3.2) |
# | Gêneros raros com proporções pequenas | comparar escalas antes de calcular distâncias (3.3) |
# | Notas de meio ponto só existem a partir de 2003 | registrar a mudança de escala; não exige tratamento |
# | Mais da metade dos usuários avaliou tudo em um único dia | o perfil descreve sobretudo uma sessão; o tempo fica fora do perfil |
# | Testes: diferenças significativas, mas pequenas (V de Cramér ≤ 0,12; η² ≤ 0,067) | interpretar pelo tamanho de efeito, não só pelo valor-p |
# | Consultas com filtro: filmes populares têm notas bem maiores que os de nicho (d de Cohen 0,73); os demais filtros têm efeito pequeno | considerar a popularidade dos filmes vistos como atributo derivado do usuário (3.8) |
# | Atributos com escalas muito diferentes nas matrizes (variâncias de 10⁻¹⁰ a 599) | comparar matrizes em escala [0, 1] e normalizar antes das distâncias (3.3) |
# | Gêneros com pouca participação nas avaliações ou com muitos filmes pouco vistos (1.6-c) | manter os rótulos nos dados; agrupá-los no vetor revisado (3.8), para que não distorçam as distâncias |
# | Tags descrevem poucos usuários e não respondem a P1 nem a P2 | ficam fora dos dados-alvo nesta fase |
#
# A seleção está documentada: fontes, instâncias, atributos, tipos, domínios, população e associações exploratórias. Tags e IDs externos não entram no perfil sem uma pergunta que os justifique.
