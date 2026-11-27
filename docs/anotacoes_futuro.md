# Anotações para as próximas etapas (P1 e P2)

Anotações do aluno registradas em 25/11/2026, com uma nota sobre como cada uma se encaixa no trabalho. Nada disto está implementado: o notebook atual termina na seção 3, e P1 (agrupamento de usuários) e P2 (padrões frequentes e regras de associação) vêm depois. Quando cada ideia virar decisão, ela passa pelo aluno e pela mesa, e a justificativa entra em `docs/justificativas.md`.

## P2 — padrões frequentes e regras de associação

1. **Cesta de compras (*market basket*).** É o exemplo clássico de regra de associação: cada transação é uma cesta, e a regra diz que quem leva A costuma levar B. Aqui, a cesta é o conjunto de filmes de um usuário. O plano já registrado usa transações = usuários e itens = filmes com nota ≥ 4,0.

2. **Filmes vistos juntos e gêneros vistos juntos.** São dois níveis de item:
   - no nível de **filme**, há dezenas de milhares de itens, e as regras são específicas ("quem gostou de X gostou de Y");
   - no nível de **gênero**, são 19 itens, e as regras são gerais ("quem vê Animation vê Children").

   Os dois podem ser minerados e comparados.

3. **A priori multidimensional.** Os itens podem vir de mais de uma dimensão ao mesmo tempo, por exemplo filme, gênero e faixa de nota. "Os 3 filmes vistos juntos" é um conjunto frequente de 3 itens. A ordenação anotada (alfabética, ou por identificador) é o que o Apriori faz por dentro: os itens de cada conjunto ficam numa ordem fixa, para gerar os candidatos maiores sem repetir o mesmo conjunto em outra ordem.

4. **Suporte mínimo: baixo, mas não baixo demais.** Com 162.541 usuários:
   - **1%** são cerca de 1.625 usuários: um bom ponto de partida, e vale ser analisado;
   - **10⁻⁴** são cerca de 16 usuários: gera regras demais, muitas por acaso;
   - **"0,1"** precisa ser esclarecido. Se for **0,1%**, são cerca de 163 usuários. Se for **10%**, são cerca de 16.254, o que só deixa os filmes mais populares.

   A escolha deve ser fixada antes de rodar e comparada em mais de um valor.

5. **Sequências: não sugerir um filme antes de o anterior ter sido visto** (por exemplo, a continuação antes do primeiro filme). Isso é mineração de **padrões sequenciais** (algoritmos como GSP e PrefixSpan), em que a ordem das transações importa.

   Cuidado com os dados: o `timestamp` é a data da **avaliação**, não a de quando o filme foi visto. Na seção 1.6-b, mais da metade dos usuários fez todas as avaliações no mesmo dia, então a ordem das avaliações diz pouco sobre a ordem em que os filmes foram vistos. Uma alternativa é usar a ordem de lançamento dos filmes, por exemplo, para não sugerir a continuação antes do original.

6. **Apriori por vizinhos top-k.** As regras são mineradas só entre os k usuários mais parecidos com o usuário-alvo, ou dentro de cada grupo encontrado em P1. Isso combina com a orientação de que P2 depende de P1: minerar em cada perfil e na população geral, para comparar. As buscas de vizinhos podem usar os índices medidos na 3.9 (kd-tree, pivôs ou HNSW no espaço PCA).

7. **Função de distância entre filmes.** Mede se dois filmes estão próximos, por exemplo pelo Jaccard ou pelo cosseno entre os públicos (quem avaliou cada um), como na 2.2-c. Ela serve para agrupar filmes parecidos antes da mineração, ou para filtrar regras entre filmes que já são quase o mesmo público.

8. **Cauda longa (sugestão da mesa na D8, 27/11/2026).** Com suporte de 0,1% dos usuários (163), só 5.530 dos 62.423 filmes do catálogo podem entrar numa regra; com 1%, só 1.562. É isso, e não a falta de gênero, que limita os filmes que podem ser sugeridos. Alavancas: suporte calculado dentro de cada grupo da P1 (um filme frequente num grupo gera regras ali), mais de um suporte comparado e regras em dois níveis (filme e gênero). Os 240 filmes excluídos na 2.2-d ficam fora também da P2 (decisão do aluno: um dado tem um tratamento só); o custo é Black Mirror (382 avaliações positivas) e, com suportes menores, Beastie Boys: Sabotage (50). Os 3.376 filmes do catálogo sem nenhuma avaliação não podem ser sugeridos por métodos baseados em avaliações (começo a frio, fora do escopo).

## Decisões da seção 3 que ainda estão abertas e afetam P1

- **Desvios relativos por grupo na entrada principal (D7):** com eles, o PCA do vetor revisado guarda menos variância (9 componentes, 67,32%) e concentra mais as distâncias que o PCA do vetor base (7, 85,73%) e que o revisado sem os desvios (7, 80,34%). A entrada principal registrada continua o PCA do vetor revisado; P1 compara as três.
- *Resolvidas desde a anotação:* D5 (IMAX fora, Animation e Children juntos; v8 e v10) e os gêneros raros com o teste do corte de 2% (D4; v8).
