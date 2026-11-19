# Conferência manual

As tabelas desta pasta são preenchidas à mão, abrindo os links. As anotações entram no repositório e voltam para a mesa; as decisões aprovadas passam a ser lidas pelo notebook. Para regenerar as tabelas em branco: `python scripts/gerar_conferencia_manual.py` (ele não sobrescreve uma tabela que já tenha anotações).

## `filmes_sem_genero_mais_avaliados.csv` (decisão D2: verificação da chave)

São os 38 filmes sem gênero com pelo menos 50 avaliações (44% das avaliações de filmes sem gênero). O gênero deles vem do IMDb, pelo código da tabela `links` (seção 2.2-b). A conferência manual verifica se **o código aponta para o filme certo**: se apontar para outro título, o gênero que veio do IMDb também está errado.

As colunas `titulo_no_imdb`, `ano_no_imdb`, `tipo_no_imdb` e `generos_no_imdb` já vêm preenchidas a partir do recorte do IMDb. Para cada linha:

1. compare `titulo` (MovieLens) com `titulo_no_imdb`, `ano_no_imdb` e `tipo_no_imdb`; na dúvida, abra o `link_imdb`;
2. em `chave_confere`, escreva `sim` (é o mesmo filme) ou `não` (o código aponta para outro título);
3. em `observacao`, anote o que chamar atenção. Exemplos: "Black Mirror: o código aponta para um curta de 2011, não para a série"; "Whiplash (2013): é o curta que deu origem ao filme de 2014, então o código está certo".

Os casos `não` voltam para a mesa: o gênero desses filmes pode ser corrigido à mão ou o filme pode continuar `Unknown`.

Tabela de conversão de gêneros usada pelo notebook (para conferir `generos_no_imdb`):

| IMDb | MovieLens |
|---|---|
| Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western | igual |
| Family | Children |
| Music, Biography, History, Sport, News, Reality-TV, Talk-Show, Game-Show, Short | não existe no MovieLens: ignorado |

## `titulos_repetidos_candidatos.csv` (decisão D3)

São os 8 pares com o mesmo título, o mesmo ano e o mesmo código TMDb. Pelo código IMDb, todos seriam filmes diferentes; pelo TMDb, seriam o mesmo. A conferência decide.

Para cada linha:

1. abra `imdb_1`, `imdb_2` e o `tmdb`;
2. em `mesmo_filme`, escreva `sim` ou `não`;
3. em `evidencia`, anote o que decidiu (ex.: "mesmo diretor e mesma duração"; "um é o curta, outro o longa");
4. se for o mesmo filme, em `generos_decididos` escreva os gêneros finais, separados por `|`. Regra da mesa: a interseção dos gêneros das duas cópias; se ela parecer apagar um gênero correto da cópia mais avaliada, use os gêneros da cópia mais avaliada e explique em `observacao`.

Interseção vazia (nenhum gênero em comum) é sinal de que não é o mesmo filme. Nesse caso, só marque `sim` com uma evidência forte.
