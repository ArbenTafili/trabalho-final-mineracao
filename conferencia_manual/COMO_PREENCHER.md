# Conferência manual

As tabelas desta pasta são preenchidas à mão, abrindo os links. O notebook lê as anotações, então elas fazem parte do trabalho e entram no repositório. Para regenerar as tabelas em branco: `python scripts/gerar_conferencia_manual.py` (ele não sobrescreve uma tabela que já tenha anotações).

## `filmes_sem_genero_mais_avaliados.csv` (decisão D2)

São os 38 filmes sem gênero com pelo menos 50 avaliações. Juntos, eles têm 44% das avaliações de filmes sem gênero.

Para cada linha:

1. abra o `link_imdb` e confira se é o filme certo (título e ano batem? é longa, curta ou série?);
2. em `generos_no_imdb`, copie os gêneros que o IMDb mostra, separados por `|` (ex.: `Comedy|Music`);
3. em `generos_movielens`, converta para a lista do MovieLens pela tabela abaixo, separados por `|`;
4. em `observacao`, anote o que chamar atenção (ex.: "é o curta de 2013, não o longa de 2014").

| IMDb | MovieLens |
|---|---|
| Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western | igual |
| Family | Children |
| Music | Musical, só se for musical de ficção; documentário musical fica Documentary |
| Biography, History, Sport, News, Reality-TV, Talk-Show, Game-Show, Short | não existe no MovieLens: ignorar |

Se nenhum gênero sobrar depois da conversão, deixe `generos_movielens` vazio e explique em `observacao`.

## `titulos_repetidos_candidatos.csv` (decisão D3)

São os 8 pares com o mesmo título, o mesmo ano e o mesmo código TMDb. Pelo código IMDb, todos seriam filmes diferentes; pelo TMDb, seriam o mesmo. A conferência decide.

Para cada linha:

1. abra `imdb_1`, `imdb_2` e o `tmdb`;
2. em `mesmo_filme`, escreva `sim` ou `não`;
3. em `evidencia`, anote o que decidiu (ex.: "mesmo diretor e mesma duração"; "um é o curta, outro o longa");
4. se for o mesmo filme, em `generos_decididos` escreva os gêneros finais, separados por `|`. Regra da mesa: a interseção dos gêneros das duas cópias; se ela parecer apagar um gênero correto da cópia mais avaliada, use os gêneros da cópia mais avaliada e explique em `observacao`.

Interseção vazia (nenhum gênero em comum) é sinal de que não é o mesmo filme. Nesse caso, só marque `sim` com uma evidência forte.
