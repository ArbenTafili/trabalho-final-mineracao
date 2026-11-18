# Análises exploratórias

Medições feitas **fora do notebook** para decidir, com números, o que entra nele. Cada script lê os dados reais de `dados/` (sem Spark) e o resultado fica gravado ao lado, em `resultados_*.txt`.

| Script | Decisão | O que mede |
|---|---|---|
| `imputacao_por_faixa.py` | D2 | acerto da imputação de gênero por número de avaliadores: chute (Drama), contagem com peso 1/N^α e kNN; média e desvio-padrão entre as 5 partes; mínimo de avaliadores pelo critério de 10 pontos |
| `imdb.py` | D2 e D4 | cobertura do IMDb, concordância de gêneros e de ano com o MovieLens, teste externo do kNN nos filmes sem gênero |

Para rodar (da raiz do repositório):

```bash
.venv/bin/python scripts/baixar_dados.py --imdb
```

```bash
cd analises && ../.venv/bin/python imputacao_por_faixa.py && ../.venv/bin/python imdb.py
```

Arquivo do IMDb usado nos resultados gravados: `title.basics.tsv.gz` baixado em 18/11/2026, MD5 `694fa9cd5bad4d6f3f94384b7ed61c74`. O IMDb atualiza o arquivo todo dia, então um download novo pode mudar os números na última casa.
