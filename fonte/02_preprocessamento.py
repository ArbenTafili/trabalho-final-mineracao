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
# | 5.062 filmes `(no genres listed)` | valor ausente em atributo usado no perfil | rótulo `Unknown`; gênero preenchido pelo IMDb, ligado pela tabela `links` (2.2-b); os que o IMDb não resolve continuam `Unknown` |
# | 412 filmes sem ano no título | valor ausente em atributo usado no perfil revisado (3.8) | ano preenchido pelo IMDb (2.2-b) |
# | 107 links sem `tmdbId` | valor ausente em atributo não usado | ignorar: dos links, só o `imdbId` é usado, como chave para o IMDb (2.2-b) |
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
# #### Gênero e ano que faltam: integração com o IMDb
#
# O gênero entra no perfil do usuário, e o ano entra no perfil revisado (3.8). No MovieLens, 5.062 filmes não têm gênero e 412 não têm ano no título. Os dois valores existem na realidade, só não foram registrados.
#
# A tabela `links` do próprio MovieLens traz o código IMDb de cada filme. O arquivo público do IMDb (`title.basics`) tem gênero e ano de lançamento por código. A integração das duas fontes pela chave é feita assim:
#
# - **fonte:** um recorte fixo do `title.basics` com os filmes do MovieLens, com o MD5 registrado, para que toda execução use os mesmos dados (o IMDb publica só a versão do dia);
# - **integridade da junção:** um código por filme e uma linha do IMDb por código (1:1); os filmes não encontrados são contados; títulos de TV são listados para conferência por título e ano;
# - **conversão de gêneros:** os 17 gêneros com o mesmo nome passam direto, `Family` vira `Children` e os gêneros que não existem no MovieLens (Biography, History, Music, Sport, Short etc.) são ignorados. Cada linha da conversão é justificada nos dois sentidos, nos filmes que têm gênero nas duas fontes;
# - **preenchimento:** só onde o MovieLens não tem o valor. As colunas originais ficam intactas (`genres_movielens`, `ano_titulo`), e as colunas `origem_genero` e `origem_ano` dizem de onde veio cada valor, para que qualquer análise possa excluir o que foi preenchido.

# %%
# [2.2-b] Integração com o IMDb: gênero e ano dos filmes que não os têm no MovieLens
import hashlib
import urllib.request
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

GENEROS_IGUAIS = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
CONVERSAO_IMDB = {**{g: g for g in GENEROS_IGUAIS}, "Family": "Children"}  # os demais gêneros do IMDb são ignorados

# Recorte fixo do title.basics com os filmes do MovieLens (scripts/recortar_imdb.py). Sem ele, baixa a versão do dia.
IMDB_RECORTE = Path(os.environ.get("IMDB_RECORTE", "imdb_recorte_movielens.tsv.gz"))
colunas_imdb = ["tconst", "titleType", "primaryTitle", "startYear", "genres"]
links_pd = links_clean.select("movieId", "imdbId").toPandas()
links_pd["tconst"] = "tt" + links_pd["imdbId"].astype("int64").astype(str).str.zfill(7)  # códigos de 8 dígitos ficam inteiros
if not IMDB_RECORTE.is_file():
    IMDB_RECORTE = Path("title.basics.tsv.gz")
    if not IMDB_RECORTE.is_file():
        urllib.request.urlretrieve("https://datasets.imdbws.com/title.basics.tsv.gz", IMDB_RECORTE)
imdb_pd = pd.read_csv(IMDB_RECORTE, sep="\t", usecols=colunas_imdb, dtype=str, na_values="\\N",
                      keep_default_na=False, quoting=3)
imdb_pd = imdb_pd[imdb_pd["tconst"].isin(set(links_pd["tconst"]))]
print(f"IMDb: {IMDB_RECORTE.name} (MD5 {hashlib.md5(IMDB_RECORTE.read_bytes()).hexdigest()}); "
      f"{len(imdb_pd):,} títulos ligados ao MovieLens")

# Integridade da junção: 1:1 pela tabela links.
assert not links_pd["tconst"].duplicated().any(), "dois filmes do MovieLens com o mesmo código IMDb"
assert not imdb_pd["tconst"].duplicated().any(), "código repetido no IMDb"
filmes_pd = (movies_clean.select("movieId", "title", "genres").toPandas()
             .merge(links_pd[["movieId", "imdbId", "tconst"]], on="movieId", how="left")
             .merge(imdb_pd, on="tconst", how="left", suffixes=("", "_imdb"))
             .sort_values("movieId").reset_index(drop=True))
assert len(filmes_pd) == filmes_pd["movieId"].nunique()
avaliacoes_por_filme = ratings_clean.groupBy("movieId").count().toPandas().set_index("movieId")["count"]
filmes_pd["avaliacoes"] = filmes_pd["movieId"].map(avaliacoes_por_filme).fillna(0).astype(int)
nao_encontrado = filmes_pd["titleType"].isna()
digitos = filmes_pd.loc[nao_encontrado, "imdbId"].dropna().astype("int64").astype(str).str.len()
print(f"Junção 1:1 pela tabela links. Encontrados no IMDb: {(~nao_encontrado).sum():,} de {len(filmes_pd):,} filmes; "
      f"não encontrados: {nao_encontrado.sum():,} ({filmes_pd.loc[nao_encontrado, 'avaliacoes'].sum():,} avaliações; "
      f"dígitos do código: {digitos.value_counts().sort_index().to_dict()})")

# Conversão de gêneros e justificativa nos dois sentidos (filmes com gênero nas duas fontes).
def converter_imdb(texto):
    return [] if pd.isna(texto) else sorted({CONVERSAO_IMDB[g] for g in texto.split(",") if g in CONVERSAO_IMDB})

filmes_pd["generos_imdb"] = filmes_pd["genres_imdb"].map(converter_imdb)
filmes_pd["generos_ml"] = filmes_pd["genres"].map(lambda s: [] if s == "Unknown" else [g for g in s.split("|") if g != "IMAX"])
ambos = filmes_pd[(filmes_pd["generos_ml"].str.len() > 0) & (filmes_pd["generos_imdb"].str.len() > 0)]
linhas_conv = []
for imdb_g, ml_g in CONVERSAO_IMDB.items():
    tem_ml = ambos["generos_ml"].map(lambda s: ml_g in s)
    tem_imdb = ambos["generos_imdb"].map(lambda s: ml_g in s)
    linhas_conv.append((imdb_g, ml_g, int(tem_ml.sum()), 100 * tem_imdb[tem_ml].mean(), int(tem_imdb.sum()), 100 * tem_ml[tem_imdb].mean()))
musica = ambos["genres_imdb"].str.contains("Music") & ~ambos["genres_imdb"].str.contains("Musical")
linhas_conv.append(("Music (não convertido)", "Musical", np.nan, np.nan, int(musica.sum()),
                    100 * ambos.loc[musica, "generos_ml"].map(lambda s: "Musical" in s).mean()))
conversao = pd.DataFrame(linhas_conv, columns=["IMDb", "MovieLens", "filmes com o gênero no MovieLens",
                                               "% que o IMDb também marca", "filmes com o gênero no IMDb",
                                               "% que o MovieLens também marca"])
comum = np.array([len(set(a) & set(b)) for a, b in zip(ambos["generos_ml"], ambos["generos_imdb"])])
uniao = np.array([len(set(a) | set(b)) for a, b in zip(ambos["generos_ml"], ambos["generos_imdb"])])
print(f"\nConversão de gêneros, medida em {len(ambos):,} filmes com gênero nas duas fontes "
      f"(pelo menos um gênero em comum: {100 * (comum > 0).mean():.1f}%; mesmos gêneros: {100 * (comum == uniao).mean():.1f}%; "
      f"Jaccard médio: {(comum / uniao).mean():.3f}):")
print(conversao.sort_values("% que o MovieLens também marca", ascending=False).round(1).to_string(index=False))

# Ano: startYear do IMDb × ano do título, onde existem os dois (indicador da qualidade da chave; nada é corrigido).
filmes_pd["ano_titulo"] = filmes_pd["title"].str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)
filmes_pd["ano_imdb"] = pd.to_numeric(filmes_pd["startYear"], errors="coerce")
com_os_dois = filmes_pd[filmes_pd["ano_titulo"].notna() & filmes_pd["ano_imdb"].notna()]
diferenca = (com_os_dois["ano_imdb"] - com_os_dois["ano_titulo"]).abs()
print(f"\nAno: {len(com_os_dois):,} filmes com ano nas duas fontes; igual em {100 * (diferenca == 0).mean():.1f}%; "
      f"até 1 ano de diferença em {100 * (diferenca <= 1).mean():.1f}%; mais de 1 ano: {(diferenca > 1).sum():,}")

# Preenchimento: só onde o MovieLens não tem o valor.
sem_genero = filmes_pd["genres"] == "Unknown"
preenche_genero = sem_genero & (filmes_pd["generos_imdb"].str.len() > 0)
filmes_pd["genero_final"] = filmes_pd["genres"].where(~preenche_genero, filmes_pd["generos_imdb"].str.join("|"))
filmes_pd["origem_genero"] = np.select([~sem_genero, preenche_genero], ["MovieLens", "IMDb"], "sem gênero")
sem_ano = filmes_pd["ano_titulo"].isna()
preenche_ano = sem_ano & filmes_pd["ano_imdb"].notna()
filmes_pd["ano_final"] = filmes_pd["ano_titulo"].where(~sem_ano, filmes_pd["ano_imdb"])
filmes_pd["origem_ano"] = np.select([~sem_ano, preenche_ano], ["título", "IMDb"], "sem ano")


def resumo_origem(coluna):
    r = filmes_pd.groupby(coluna).agg(filmes=("movieId", "size"), avaliacoes=("avaliacoes", "sum"))
    r["%_filmes"] = 100 * r["filmes"] / r["filmes"].sum()
    r["%_avaliacoes"] = 100 * r["avaliacoes"] / r["avaliacoes"].sum()
    return r


print(f"\nOrigem do gênero (denominadores: {len(filmes_pd):,} filmes e {int(filmes_pd['avaliacoes'].sum()):,} avaliações):")
print(resumo_origem("origem_genero").round(3).to_string())
print(f"Filmes com gênero do IMDb: {preenche_genero.sum():,}; continuam sem gênero: {(sem_genero & ~preenche_genero).sum():,} "
      f"(com 5 ou mais avaliações: {(sem_genero & ~preenche_genero & (filmes_pd['avaliacoes'] >= 5)).sum():,})")
print("\nOrigem do ano:")
print(resumo_origem("origem_ano").round(3).to_string())
print("\nTipo do título no IMDb dos filmes sem gênero no MovieLens:",
      filmes_pd.loc[sem_genero, "titleType"].fillna("não encontrado").value_counts().to_dict())
tv = filmes_pd[preenche_genero & filmes_pd["titleType"].isin(["tvEpisode", "tvMiniSeries", "tvSeries"])]
print(f"Amostra dos títulos de TV preenchidos ({len(tv):,}), para conferir título e ano à mão:")
print(tv.sort_values("avaliacoes", ascending=False)[["movieId", "title", "primaryTitle", "startYear", "titleType", "genero_final"]]
      .head(8).to_string(index=False))

# Por origem: o IMDb lista no máximo 3 gêneros, então os filmes preenchidos tendem a ter menos gêneros.
por_origem = []
for origem in ["MovieLens", "IMDb"]:
    gs = filmes_pd.loc[filmes_pd["origem_genero"] == origem, "genero_final"].str.split("|")
    por_origem.append((origem, len(gs), gs.str.len().mean(), 100 * gs.map(lambda s: "Thriller" in s).mean(),
                       100 * gs.map(lambda s: "Musical" in s).mean(), 100 * gs.map(lambda s: "Drama" in s).mean()))
print("\nGêneros por filme, por origem:")
print(pd.DataFrame(por_origem, columns=["origem", "filmes", "gêneros por filme", "% Thriller", "% Musical", "% Drama"])
      .round(2).to_string(index=False))

# Atualiza movies_clean: `genres` passa a ser o gênero final; as colunas originais ficam ao lado.
esquema = StructType([StructField("movieId", LongType()), StructField("genero_final", StringType()),
                      StructField("origem_genero", StringType()), StructField("ano_titulo", IntegerType()),
                      StructField("ano", IntegerType()), StructField("origem_ano", StringType())])
inteiro = lambda v: None if pd.isna(v) else int(v)
registros = [(int(r.movieId), r.genero_final, r.origem_genero, inteiro(r.ano_titulo), inteiro(r.ano_final), r.origem_ano)
             for r in filmes_pd.itertuples()]
preenchimento = spark.createDataFrame(registros, esquema)
movies_clean = (movies_clean.withColumnRenamed("genres", "genres_movielens")
    .join(F.broadcast(preenchimento), "movieId", "left")
    .withColumnRenamed("genero_final", "genres")
    .cache())
movies_clean.createOrReplaceTempView("movies_clean")
assert movies_clean.count() == len(filmes_pd)

fig, axes = plt.subplots(1, 2, figsize=(20, 6.5))
conv = conversao.dropna(subset=["% que o IMDb também marca"]).sort_values("% que o MovieLens também marca")
pos = np.arange(len(conv))
axes[0].barh(pos - 0.2, conv["% que o IMDb também marca"], height=0.4, color="#4682a9", label="MovieLens → IMDb")
axes[0].barh(pos + 0.2, conv["% que o MovieLens também marca"], height=0.4, color="#d18b24", label="IMDb → MovieLens")
axes[0].set_yticks(pos, conv["MovieLens"])
axes[0].set(title="Concordância do gênero nos dois sentidos (filmes com gênero nas duas fontes)", xlabel="%", xlim=(0, 100))
axes[0].legend(loc="lower right")
axes[0].grid(axis="x", alpha=.2)
preenchidos_pct = [100 * preenche_genero.sum() / sem_genero.sum(),
                   100 * filmes_pd.loc[preenche_genero, "avaliacoes"].sum() / filmes_pd.loc[sem_genero, "avaliacoes"].sum(),
                   100 * preenche_ano.sum() / sem_ano.sum(),
                   100 * filmes_pd.loc[preenche_ano, "avaliacoes"].sum() / filmes_pd.loc[sem_ano, "avaliacoes"].sum()]
rotulos = [f"filmes sem gênero ({sem_genero.sum():,})", "avaliações dos filmes sem gênero",
           f"filmes sem ano ({sem_ano.sum():,})", "avaliações dos filmes sem ano"]  # rótulos distintos: o eixo é categórico
axes[1].barh(rotulos[::-1], preenchidos_pct[::-1], color=["#d18b24", "#d18b24", "#7a9e7e", "#7a9e7e"])
for i, v in enumerate(preenchidos_pct[::-1]):
    axes[1].text(v + 1, i, f"{v:.1f}%", va="center")
axes[1].set(title="Quanto o IMDb preenche", xlabel="% preenchido", xlim=(0, 110))
axes[1].grid(axis="x", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Junção:** a ligação pela tabela `links` é 1:1. O IMDb encontra 62.294 dos 62.423 filmes; os 129 não encontrados (17.085 avaliações) têm códigos ausentes do arquivo do IMDb, e só 15 deles estão entre os filmes sem gênero.
# - **Conversão de gêneros:** nos 57.100 filmes com gênero nas duas fontes, 97,7% têm pelo menos um gênero em comum (conjuntos idênticos: 46,9%; Jaccard médio 0,719).
#   - Concordância alta nos dois sentidos: Animation e Documentary (acima de 96%), Horror, Comedy e Drama (de 79% a 94%).
#   - Concordância baixa: Musical (48% e 43%), Mystery (67% e 48%) e Film-Noir (73% e 45%). As duas fontes não usam esses rótulos do mesmo jeito.
#   - `Music` não vira Musical: só 15,4% desses filmes são Musical no MovieLens.
# - **Ano:** nos 61.880 filmes com ano nas duas fontes, o ano é igual em 92,8% e difere em até 1 ano em 99,0%. Os 644 com diferença maior ficam como indicador da qualidade da chave; nada é corrigido.
# - **Gênero preenchido:** 4.822 dos 5.062 filmes sem gênero recebem o gênero do IMDb (25.221 avaliações, 0,10% do total). Continuam sem gênero 240 filmes (1.406 avaliações, 0,006%), dos quais só 55 têm 5 ou mais avaliações.
# - **Ano preenchido:** o IMDb dá o ano de 409 dos 412 filmes sem ano no título (11.747 avaliações); 3 filmes, com 2 avaliações, continuam sem ano.
# - **Tipos de título:** dos filmes sem gênero no MovieLens, 3.684 são longas no IMDb, 628 são curtas e quase todo o resto são títulos de TV. Na amostra dos 149 títulos de TV preenchidos, título e ano batem com o catálogo (por exemplo, Cosmos, The OA, 13 Reasons Why e Death Note são séries).
# - **Por origem:** os filmes preenchidos têm 1,78 gênero por filme, contra 1,87 dos que já tinham gênero. Têm menos Thriller (7,4% contra 15,1%), mais Drama (52,5% contra 44,6%) e mais Musical (3,1% contra 1,8%). O IMDb usa menos Thriller e lista no máximo três gêneros, como a concordância já indicava.
#
# **Limitação:** a concordância mede o quanto as duas fontes rotulam do mesmo jeito, não qual delas está certa. O IMDb atualiza seus gêneros com o tempo; por isso o recorte usado fica identificado pelo MD5.
#
# **Consequência:** `Unknown` passa de 26.627 para 1.406 avaliações. As colunas `origem_genero` e `origem_ano` permitem executar P1 com e sem os valores preenchidos. Num perfil, um filme preenchido entra como qualquer outro: no vetor base (3.1), cada avaliação conta uma vez em cada gênero do filme; no vetor revisado (3.8), o peso de cada avaliação é dividido entre os gêneros do filme.

# %% [markdown]
# #### O gênero previsto pelo público: kNN
#
# A integração resolve quase todos os filmes sem gênero, mas não usa o comportamento dos usuários. Esta célula mede se o próprio MovieLens contém a informação de gênero: o gênero de um filme é previsto a partir de **quem o avaliou**, e a previsão é validada de duas formas.
#
# - **Atributos do filme:** a média do perfil de gêneros dos seus avaliadores, calculada sem o próprio filme (para o gênero dele não vazar), dividida pela média geral (lift) e padronizada. Só os gêneros originais do MovieLens são usados como resposta.
# - **kNN (k = 15):** os 15 filmes com o público mais parecido votam; cada vizinho vota em todos os seus gêneros; em empate, vence o gênero com mais filmes no catálogo de treino.
# - **Comparações:** chutar Drama (a moda), a contagem dos gêneros vistos pelo público sem peso e com peso 1/√N (N = filmes do gênero no catálogo de treino) e o Naive Bayes. O erro do Naive Bayes é a **taxa de ruído de referência**: mostra quanto um classificador simples não consegue explicar com a mesma informação; um método mais elaborado só compensa se errar menos que isso.
# - **Validação interna, por faixa de avaliadores:** validação cruzada de 5 partes nos filmes com gênero e pelo menos 10 avaliadores; cada uma das 5 partes também testa todos os filmes com menos de 10 avaliadores, que nunca entram no treino. Assim todas as faixas têm média e desvio-padrão.
# - **Critério, fixado antes da execução com os dados reais:** o kNN só é considerado útil numa faixa se acertar pelo menos 10 pontos percentuais a mais que chutar Drama. O valor de 10 pontos vem da regra da versão anterior; aplicá-lo por faixa foi decidido depois de uma medição exploratória (`analises/`). O código calcula o mínimo de avaliadores: a menor faixa a partir da qual todas as faixas superiores passam.
# - **Validação externa:** nos próprios filmes sem gênero no MovieLens, a previsão é comparada com o gênero que veio do IMDb. Isso mede concordância com o IMDb, não a verdade.
#
# Como o IMDb já preenche quase todos os filmes sem gênero, o kNN não é usado para preencher: ele é o método de mineração que confirma, com os dados de comportamento, a informação que veio da fonte externa.

# %%
# [2.2-c] kNN: gênero previsto pelo público do filme, validado por faixa de avaliadores e contra o IMDb
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

K_VIZINHOS = 15
GANHO_MINIMO = 10  # p.p. acima de chutar Drama
FAIXAS_AVALIADORES = [(1, 2), (3, 4), (5, 9), (10, 49), (50, None)]
rotulo_faixa = lambda a, b: f"{a}+" if b is None else f"{a}–{b}"
na_faixa = lambda n, a, b: (n >= a) & (n <= (np.inf if b is None else b))
generos_reais = sorted(r["genre"] for r in movies_clean.select(F.explode(F.split("genres_movielens", r"\|")).alias("genre"))
                       .distinct().collect() if r["genre"] != "Unknown")
cols_g = ["g_" + g.lower().replace("-", "_") for g in generos_reais]

# Perfil do público de cada filme: média dos perfis dos avaliadores (sem o próprio filme) e soma das avaliações por gênero.
filmes_vetor = (movies_clean.select("movieId", F.split("genres_movielens", r"\|").alias("gs"))
    .withColumn("conhecido", F.when(F.array_contains("gs", "Unknown"), 0).otherwise(1))
    .withColumn("k", F.size("gs"))
    .select("movieId", "conhecido",
            *[F.when(F.array_contains("gs", g), 1.0 / F.col("k")).otherwise(0.0).alias(c) for g, c in zip(generos_reais, cols_g)]))
avaliacoes_vetor = ratings_clean.select("userId", "movieId").join(F.broadcast(filmes_vetor), "movieId")
totais_usuario = avaliacoes_vetor.groupBy("userId").agg(F.sum("conhecido").alias("n_conhecidos"),
                                                        *[F.sum(c).alias("u_" + c) for c in cols_g])
denominador = F.col("n_conhecidos") - F.col("conhecido")
publico_pd = (avaliacoes_vetor.join(totais_usuario, "userId")
    .where(denominador > 0)
    .select("movieId",
            *[F.when(denominador > 0, (F.col("u_" + c) - F.col(c)) / denominador).alias(c) for c in cols_g],
            *[(F.col("u_" + c) - F.col(c)).alias("s_" + c) for c in cols_g])
    .groupBy("movieId").agg(F.count("*").alias("n_avaliadores"), *[F.avg(c).alias(c) for c in cols_g],
                            *[F.sum("s_" + c).alias("s_" + c) for c in cols_g])
    .join(movies_clean.select("movieId", "title", "genres_movielens", "genres", "origem_genero"), "movieId")
    .orderBy("movieId")  # ordem fixa: os sorteios dão o mesmo resultado em qualquer máquina
    .toPandas())
conhecido = (publico_pd["genres_movielens"] != "Unknown").to_numpy()
n_aval = publico_pd["n_avaliadores"].to_numpy()
binarizador = MultiLabelBinarizer(classes=generos_reais)
Y = binarizador.fit_transform([s.split("|") if k else [] for s, k in zip(publico_pd["genres_movielens"], conhecido)])
P = publico_pd[cols_g].to_numpy()
S = publico_pd[["s_" + c for c in cols_g]].to_numpy()
pos_drama = generos_reais.index("Drama")
base = np.where(conhecido & (n_aval >= 10))[0]
pequenos = np.where(conhecido & (n_aval < 10))[0]
print(f"Filmes com gênero no MovieLens e público conhecido: {conhecido.sum():,} ({len(base):,} com 10 ou mais avaliadores)")


def atributos(treino):
    """Lift do perfil do público em relação à média dos filmes de treino, padronizado no treino."""
    lift = P / P[treino].mean(axis=0)
    return StandardScaler().fit(lift[treino]).transform(lift)


def votar(X, treino, alvo):
    """Os K_VIZINHOS filmes de treino mais parecidos votam em todos os seus gêneros."""
    _, vizinhos = NearestNeighbors(n_neighbors=K_VIZINHOS).fit(X[treino]).kneighbors(X[alvo])
    votos = Y[treino][vizinhos].sum(axis=1).astype(float)
    desempate = Y[treino].sum(axis=0) / (10 * len(treino))  # empate: vence o gênero com mais filmes no treino
    return (votos + desempate).argmax(axis=1), vizinhos, votos


metodos = ["chutar Drama", "contagem sem peso", "contagem 1/√N", "Naive Bayes", f"kNN (k={K_VIZINHOS})"]
TODOS_10 = (10, None)  # todos os filmes com 10 ou mais avaliadores (só a parte de teste de cada rodada)
acertos = {(f, m): [] for f in FAIXAS_AVALIADORES + [TODOS_10] for m in metodos}
for parte_treino, parte_teste in KFold(n_splits=5, shuffle=True, random_state=42).split(base):
    treino, alvo = base[parte_treino], np.concatenate([base[parte_teste], pequenos])
    X = atributos(treino)
    N_treino = Y[treino].sum(axis=0)
    previsto = {"chutar Drama": np.full(len(alvo), pos_drama),
                "contagem sem peso": S[alvo].argmax(axis=1),
                "contagem 1/√N": (S[alvo] / np.sqrt(N_treino)).argmax(axis=1),
                "Naive Bayes": OneVsRestClassifier(GaussianNB()).fit(X[treino], Y[treino]).predict_proba(X[alvo]).argmax(axis=1),
                f"kNN (k={K_VIZINHOS})": votar(X, treino, alvo)[0]}
    for a, b in FAIXAS_AVALIADORES + [TODOS_10]:
        sel = na_faixa(n_aval[alvo], a, b)
        for m in metodos:
            acertos[((a, b), m)].append(100 * Y[alvo[sel], previsto[m][sel]].mean() if sel.any() else np.nan)

linhas_faixa = []
for a, b in FAIXAS_AVALIADORES:
    linha = {"avaliadores": rotulo_faixa(a, b), "filmes": int((conhecido & na_faixa(n_aval, a, b)).sum())}
    for m in metodos:
        v = np.array(acertos[((a, b), m)])
        linha[m] = "—" if np.isnan(v).all() else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
    linha["kNN − chute (p.p.)"] = np.mean(acertos[((a, b), metodos[-1])]) - np.mean(acertos[((a, b), metodos[0])])
    linhas_faixa.append(linha)
validacao_faixas = pd.DataFrame(linhas_faixa)
print("\nValidação interna: acerto (%) por faixa de avaliadores, média ± desvio-padrão entre as 5 partes")
print(validacao_faixas.round(1).to_string(index=False))
acerto_10 = {m: np.mean(acertos[(TODOS_10, m)]) for m in metodos}
print(f"Filmes com 10 ou mais avaliadores: kNN {acerto_10[metodos[-1]]:.2f}%; Naive Bayes {acerto_10['Naive Bayes']:.2f}%; "
      f"chutar Drama {acerto_10['chutar Drama']:.2f}%")
print(f"Erro do Naive Bayes (taxa de ruído de referência): {100 - acerto_10['Naive Bayes']:.2f}%; "
      f"erro do kNN: {100 - acerto_10[metodos[-1]]:.2f}%")
ganhos = dict(zip(FAIXAS_AVALIADORES, validacao_faixas["kNN − chute (p.p.)"]))
avaliadas = [f for f in FAIXAS_AVALIADORES if not np.isnan(ganhos[f])]  # faixas sem filmes não entram no critério
minimo_avaliadores = next((a for i, (a, b) in enumerate(avaliadas)
                           if all(ganhos[f] >= GANHO_MINIMO for f in avaliadas[i:])), None)
print(f"Mínimo de avaliadores pelo critério (kNN ≥ chute + {GANHO_MINIMO} p.p. em todas as faixas a partir dele): "
      f"{minimo_avaliadores}")

# Validação externa: nos filmes sem gênero no MovieLens que receberam gênero do IMDb.
X = atributos(base)
externos = np.where((~conhecido) & (publico_pd["origem_genero"] == "IMDb").to_numpy())[0]
previsto_ext, vizinhos_ext, votos_ext = votar(X, base, externos)
gabarito = publico_pd.loc[externos, "genres"].str.split("|").to_numpy()
N_base = Y[base].sum(axis=0)
raiz_ext = (S[externos] / np.sqrt(N_base)).argmax(axis=1)
linhas_ext = []
for a, b in FAIXAS_AVALIADORES:
    sel = na_faixa(n_aval[externos], a, b)
    if sel.any():
        ok = lambda pred: 100 * np.mean([generos_reais[p] in g for p, g in zip(pred[sel], gabarito[sel])])
        linhas_ext.append((rotulo_faixa(a, b), int(sel.sum()), ok(np.full(len(externos), pos_drama)), ok(raiz_ext),
                           ok(previsto_ext)))
validacao_externa = pd.DataFrame(linhas_ext, columns=["avaliadores", "filmes sem gênero", "chutar Drama %", "contagem 1/√N %",
                                                      f"kNN (k={K_VIZINHOS}) %"])
print("\nValidação externa: concordância com o gênero do IMDb nos filmes sem gênero no MovieLens")
print(validacao_externa.round(1).to_string(index=False))

# Exemplo passo a passo: o filme sem gênero mais avaliado da faixa 5–9.
faixa_exemplo = na_faixa(n_aval[externos], 5, 9)
i_ex = np.where(faixa_exemplo)[0][np.argmax(n_aval[externos][faixa_exemplo])]
filme_ex = publico_pd.loc[externos[i_ex]]
viz = base[vizinhos_ext[i_ex]]
print(f"\nExemplo: {filme_ex['title']} — {filme_ex['n_avaliadores']} avaliadores; gênero no IMDb: {filme_ex['genres']}")
print("Os 15 filmes com o público mais parecido (cada um vota em todos os seus gêneros):")
print(publico_pd.loc[viz, ["title", "genres_movielens", "n_avaliadores"]].to_string(index=False))
contagem_votos = pd.Series(votos_ext[i_ex], index=generos_reais)
print("Votos:", contagem_votos[contagem_votos > 0].sort_values(ascending=False).astype(int).to_dict())
print(f"Previsto pelo kNN: {generos_reais[previsto_ext[i_ex]]}; está entre os gêneros do IMDb: "
      f"{generos_reais[previsto_ext[i_ex]] in filme_ex['genres'].split('|')}")

fig, axes = plt.subplots(1, 2, figsize=(20, 5.5))
rotulos = [rotulo_faixa(a, b) for a, b in FAIXAS_AVALIADORES]
for m, cor in [("chutar Drama", "#999999"), ("contagem 1/√N", "#7656a5"), (f"kNN (k={K_VIZINHOS})", "#d18b24")]:
    medias = [np.mean(acertos[(f, m)]) for f in FAIXAS_AVALIADORES]
    dps = [np.std(acertos[(f, m)], ddof=1) for f in FAIXAS_AVALIADORES]
    axes[0].errorbar(rotulos, medias, yerr=dps, marker="o", capsize=4, color=cor, label=m)
chute = [np.mean(acertos[(f, "chutar Drama")]) + GANHO_MINIMO for f in FAIXAS_AVALIADORES]
axes[0].plot(rotulos, chute, ls="--", color="crimson", label=f"chute + {GANHO_MINIMO} p.p. (critério)")
axes[0].set(title="Validação interna por faixa de avaliadores", xlabel="avaliadores do filme", ylabel="% de acerto",
            ylim=(0, 100))
axes[0].legend()
axes[0].grid(alpha=.2)
largura = 0.27
pos = np.arange(len(validacao_externa))
for k, (col, cor) in enumerate([("chutar Drama %", "#999999"), ("contagem 1/√N %", "#7656a5"),
                                (f"kNN (k={K_VIZINHOS}) %", "#d18b24")]):
    axes[1].bar(pos + (k - 1) * largura, validacao_externa[col], width=largura, color=cor, label=col.replace(" %", ""))
axes[1].set_xticks(pos, validacao_externa["avaliadores"])
axes[1].set(title="Validação externa: concordância com o IMDb nos filmes sem gênero", xlabel="avaliadores do filme",
            ylabel="%", ylim=(0, 100))
axes[1].legend()
axes[1].grid(axis="y", alpha=.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Leitura dos resultados.**
#
# - **Validação interna:** nos 24.010 filmes com gênero e pelo menos 10 avaliadores, o kNN acerta 83,4%, contra 47,3% de chutar Drama. O Naive Bayes acerta 69,6%; seu erro, de 30,4%, é a taxa de ruído de referência, e o kNN fica bem abaixo dela (erro de 16,7%). Contar os gêneros vistos pelo público quase não ajuda: de 47% a 54% sem peso e de 49% a 62% com peso 1/√N.
# - **Por faixa de avaliadores:** o acerto do kNN cresce com o público do filme: 38,3% com 1–2 avaliadores, 55,3% com 3–4, 67,7% com 5–9, 77,9% com 10–49 e 87,9% com 50 ou mais. O desvio-padrão entre as 5 partes fica abaixo de 1 ponto.
#   - Pelo critério de 10 pontos acima do chute, o mínimo calculado é **3 avaliadores** (3–4: 12,3 pontos acima).
#   - Com 1–2 avaliadores, o kNN fica 4,4 pontos abaixo do chute: o público de um ou dois usuários não diz nada sobre o filme.
# - **Validação externa:** nos filmes sem gênero, a concordância do kNN com o IMDb segue a mesma ordem: 66,6% com 5–9 avaliadores, 70,9% com 10–49 e 80,6% com 50 ou mais, contra 48,7% a 58,3% do chute. Com 3–4 avaliadores, o kNN fica só 3,3 pontos acima do chute e empata com a contagem 1/√N (55,8%); com 1–2, fica abaixo do chute.
# - **Interna × externa:** a vantagem do kNN sobre o chute é menor no teste externo (na faixa 5–9, 14,8 pontos contra 24,0 na validação interna). Os filmes sem gênero não são iguais aos que têm gênero, e o gabarito externo é outra fonte, que rotula de outro jeito.
# - **Exemplo:** St. Giuseppe Moscati: Doctor to the Poor (2007) tem 9 avaliadores e nenhum gênero no MovieLens. Entre os 15 filmes de público mais parecido, 13 têm Drama, e o kNN prevê Drama, que é o gênero do IMDb.
#
# **Limitação:** o critério interno aceita a faixa 3–4, mas o teste externo mostra que ali o ganho é pequeno. A faixa 5–9 é a primeira que passa nos dois testes.
#
# **Consequência:** o comportamento dos usuários contém a informação de gênero: o kNN recupera o gênero de 83% dos filmes com 10 ou mais avaliadores e concorda com o IMDb nos filmes sem gênero a partir de 5 avaliadores. Como o IMDb já preenche 4.822 dos 5.062 filmes sem gênero e só 55 dos 240 restantes têm 5 ou mais avaliações, o kNN não é usado para preencher.

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
          "nos gêneros, os filmes sem gênero recebem o gênero do IMDb ou continuam `Unknown` (2.2-b); 2.6 mostra o efeito.")
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
print(f"Amostra de notas: n={len(amostra_pd):,} (aleatória simples por hash de userId e movieId, semente 42)")
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
# - **Amostra de notas:** as 100.065 notas diferem da população em no máximo 0,18 ponto percentual (nota 4,5). A média é 3,5332, contra 3,5339, e o desvio-padrão é 1,0614, contra 1,0607.
# - **Amostra de usuários:** nos 3.000 usuários, médias e desvios ficam próximos dos da população (por exemplo, `nota_media` 3,6786 contra 3,6792).
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
# - **(b)** filmes, avaliações, média e desvio-padrão por gênero, antes e depois do preenchimento;
# - **(c)** as notas dos filmes que receberam gênero do IMDb, comparadas às do gênero inteiro;
# - **(d)** o qui-quadrado gênero × grupo de nota, com o V de Cramér antes e depois.
#
# Volume, tempo e correlações por filme não dependem do gênero. A década de 1.8-c usa só os filmes com ano no título (dados-alvo); o ano preenchido entra no perfil revisado (3.8). As correlações por usuário mudam só nas avaliações dos filmes preenchidos, cuja quantidade aparece em (b).

# %%
# [2.6] Distribuições depois do tratamento: o que muda com o preenchimento do gênero
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

# (c) Avaliações dos filmes que receberam gênero do IMDb, por gênero, comparadas às do gênero inteiro.
preenchidos_notas = (ratings_clean.join(movies_clean.filter(F.col("origem_genero") == "IMDb")
                                        .select("movieId", F.explode(F.split("genres", r"\|")).alias("genre")), "movieId")
    .groupBy("genre").agg(F.countDistinct("movieId").alias("filmes"), F.count("*").alias("avaliacoes"),
                          F.avg("rating").alias("media_preenchidos"), F.stddev("rating").alias("dp_preenchidos")).toPandas())
if len(preenchidos_notas):
    preenchidos_notas = preenchidos_notas.merge(depois[["media", "avaliacoes"]].rename(
        columns={"media": "media_do_genero", "avaliacoes": "avaliacoes_do_genero"}), left_on="genre", right_index=True)
    preenchidos_notas["%_das_avaliacoes_do_genero"] = 100 * preenchidos_notas["avaliacoes"] / preenchidos_notas["avaliacoes_do_genero"]
    print("\n(c) Notas dos filmes que receberam gênero do IMDb, comparadas à média do gênero:")
    print(preenchidos_notas.drop(columns="avaliacoes_do_genero").sort_values("avaliacoes", ascending=False)
          .round(3).to_string(index=False))

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
axes[0].set(title="Avaliações que ganharam ou perderam gênero com o preenchimento", xlabel="Δ avaliações (depois − antes)")
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
# - **(b) Gêneros:** `Unknown` perde 4.822 filmes e 25.221 avaliações (de 26.627 para 1.406). As avaliações vão sobretudo para Drama (+12.902), Comedy (+7.269), Adventure (+4.127), Action (+3.708) e Fantasy (+3.675). Fora de `Unknown`, a média de cada gênero muda no máximo 0,0015 e o desvio-padrão no máximo 0,0013 (os dois em Documentary, que ganha 707 filmes). A média de `Unknown` sobe de 3,33 para 3,45, agora sobre só 1.406 avaliações.
# - **(c) Filmes preenchidos:** em geral, têm notas menores que a média do gênero: Drama 3,43 contra 3,68, Comedy 3,21 contra 3,42 e Action 3,02 contra 3,47. Horror é a exceção (3,39 contra 3,29). São filmes pouco vistos, e os filmes pouco vistos têm notas mais baixas (consulta "nicho × populares" de 1.7-c). Como pesam no máximo 0,87% das avaliações de um gênero (Documentary), as médias dos gêneros quase não mudam.
# - **(d) Associação gênero × nota:** fora de `Unknown`, o V de Cramér muda no máximo 0,0001 (Comedy e Action). O de `Unknown` cai de 0,0051 para 0,0024. Drama continua o gênero mais associado à nota (V = 0,119), e todas as associações seguem fracas.
#
# **Consequência:** o preenchimento não altera as distribuições estudadas na Seção 1. `Unknown` passa a 0,006% das avaliações, e o efeito chega ao perfil por usuário em 3.1.
