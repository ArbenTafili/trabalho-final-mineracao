# Análises exploratórias

Medições feitas **fora do notebook** para decidir, com números, o que entra nele. Cada script lê os dados reais de `dados/` (sem Spark) e o resultado fica gravado ao lado, em `resultados_*.txt`.

| Script | Decisão | O que mede |
|---|---|---|
| `imputacao_por_faixa.py` | D2 | acerto da imputação de gênero por número de avaliadores: chute (Drama), contagem com peso 1/N^α e kNN; média e desvio-padrão entre as 5 partes; mínimo de avaliadores pelo critério de 10 pontos |
| `imdb.py` | D2 e D4 | cobertura do IMDb, concordância de gêneros e de ano com o MovieLens, teste externo do kNN nos filmes sem gênero |
| `chave_imdb.py` | D2 (proposta 2-C) | conferência automática da chave MovieLens → IMDb em todos os filmes: título, ano e tipo, validados pela concordância de gêneros contra uma chave sorteada; margem de erro dos gêneros e dos anos preenchidos; efeito por usuário; lista dos filmes sem gênero mais avaliados que a regra não confirma (substitui a conferência à mão) |
| `casos_sem_tratamento.py` | D6 (proposta 3) | o que continua sem tratamento depois do IMDb (filmes sem gênero, sem ano e títulos repetidos), o peso de cada caso em avaliações e por usuário, e o teste do kNN nos curtas |
| `pca_criterios.py` | D7 (proposta 4) | PCA como no método da disciplina (min-max, centralizado, covariância) e o d′ de cada critério: % acumulada, broken-stick, Kaiser em escala, D2 e MLE; distribuição das distâncias no min-max e no PCA de cada d′ (`--com-dp-genero` acrescenta o desvio por gênero) |
| `atributos_por_genero.py` | D7 (proposta 4) | cobertura e redundância da média e do desvio-padrão da nota por gênero em cada usuário, com e sem encolhimento |
| `dimensao_por_genero.py` | D7 (proposta 4, item 10) | dimensionalidade intrínseca e d′ do PCA no público de cada gênero (10% mais altos), com n igual, pareamento por atividade, usuários sorteados e placebo; prévia, refazer com o vetor revisado da proposta 4 |
| `preencher_genero_distribuicao.py` | D8 (proposta 5) | se preencher o gênero dos filmes sem gênero (1/√N, versões fracionárias, gênero mais comum, distribuição do catálogo) mantém a distribuição: em curtas com gênero conhecido, escondido, mede a distância da distribuição verdadeira e o erro no perfil de cada usuário |

Para rodar (da raiz do repositório):

```bash
.venv/bin/python scripts/baixar_dados.py --imdb
```

```bash
cd analises && ../.venv/bin/python imputacao_por_faixa.py && ../.venv/bin/python imdb.py && ../.venv/bin/python chave_imdb.py && ../.venv/bin/python casos_sem_tratamento.py
```

Os da revisão de 26/11/2026 (D7 e D8); só o último usa o recorte do IMDb:

```bash
cd analises && ../.venv/bin/python pca_criterios.py && ../.venv/bin/python atributos_por_genero.py && ../.venv/bin/python dimensao_por_genero.py && ../.venv/bin/python preencher_genero_distribuicao.py
```

Arquivo do IMDb usado nos resultados gravados: `title.basics.tsv.gz` baixado em 18/11/2026, MD5 `694fa9cd5bad4d6f3f94384b7ed61c74`. O IMDb atualiza o arquivo todo dia, então um download novo pode mudar os números na última casa.
