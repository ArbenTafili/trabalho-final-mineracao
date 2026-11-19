# Conferência manual

A tabela desta pasta é preenchida à mão, abrindo os links. As anotações entram no repositório e voltam para a mesa; as decisões aprovadas passam a ser lidas pelo notebook. Para regenerar a tabela em branco: `python scripts/gerar_conferencia_manual.py` (ele não sobrescreve uma tabela que já tenha anotações).

> A tabela dos 38 filmes sem gênero mais avaliados (decisão D2) foi retirada em 19/11/2026: a chave MovieLens → IMDb passou a ser conferida por código em todos os filmes (proposta 2-C, `analises/chave_imdb.py`). A conferência dos títulos repetidos (D3) ainda é manual e vai ter uma rodada própria na mesa.

## `titulos_repetidos_candidatos.csv` (decisão D3)

São os 8 pares com o mesmo título, o mesmo ano e o mesmo código TMDb. Pelo código IMDb, todos seriam filmes diferentes; pelo TMDb, seriam o mesmo. A conferência decide.

Para cada linha:

1. abra `imdb_1`, `imdb_2` e o `tmdb`;
2. em `mesmo_filme`, escreva `sim` ou `não`;
3. em `evidencia`, anote o que decidiu (ex.: "mesmo diretor e mesma duração"; "um é o curta, outro o longa");
4. se for o mesmo filme, em `generos_decididos` escreva os gêneros finais, separados por `|`. Regra da mesa: a interseção dos gêneros das duas cópias; se ela parecer apagar um gênero correto da cópia mais avaliada, use os gêneros da cópia mais avaliada e explique em `observacao`.

Interseção vazia (nenhum gênero em comum) é sinal de que não é o mesmo filme. Nesse caso, só marque `sim` com uma evidência forte.
