# Registro de decisões

Cada mudança no trabalho segue o mesmo caminho:

1. a ideia é proposta (pelo aluno, pela dupla ou pelo Claude) com os números reais da base;
2. o aluno filtra o que vai adiante;
3. uma **mesa de avaliação** simulada (agente de IA independente, que não participou das propostas) vota cada ideia. Até 18/11/2026 ela teve cinco membros (professor de Mineração de Dados, estatístico, engenheiro de dados, especialista em sistemas de recomendação e colega de turma). A partir da proposta 2-B, tem sete:
   - professor de Mineração de Dados (rigor do KDD);
   - professor de IA / Aprendizado de Máquina (métodos, validação, vazamento);
   - professor de Banco de Dados (chaves, integridade, junções, duplicatas);
   - cientista de dados (se a decisão compensa na prática);
   - engenheiro de dados (reprodutibilidade);
   - estatístico (viés, amostragem, se os números sustentam a conclusão);
   - colega de turma (dá para explicar e conferir à mão?);
4. os ajustes pedidos pela mesa voltam ao aluno, e só então a ideia é implementada.

Os critérios de cada decisão são fixados **antes** de ver o resultado.

---

## D1 — Repositório e testes locais

**Status:** aprovada com ajustes em 18/11/2026; ajustes aceitos pelo aluno; em implementação.

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- edita-se só em `fonte/*.py`; o `.ipynb` é sempre gerado por `scripts/montar_notebook.py` e nunca editado à mão;
- a execução que gera as saídas entregues é feita no Colab; o Mac serve de pré-teste;
- sementes fixas e uma lista de números de controle comparada entre Mac e Colab;
- pico de memória medido na execução local; versões das bibliotecas iguais às do Colab;
- dados sintéticos com casos de borda (filme sem gênero, sem ano, título repetido, filmes com 9 e 10 avaliações);
- dados do MovieLens e do IMDb fora do repositório.

## D2 — Preencher o gênero com contagem ponderada por 1/√N

**Status:** aprovada com ajustes em 18/11/2026; aguardando o limite de perda de acerto.

**Proposta:** para um filme sem gênero, contar as avaliações que os avaliadores dele deram a filmes de cada gênero g (sem o próprio filme), dividir por √N_g e escolher o maior. N_g = número de filmes do gênero g no catálogo; um filme com vários gêneros conta em cada um.

**Votos:** 3 aprovam com ajuste, 2 rejeitam (estatístico e especialista em recomendação, porque a regra provavelmente acerta menos que o kNN de 83,4%).

**Ajustes:**
- fixar antes de rodar a perda máxima de acerto aceitável em relação ao kNN; se a regra passar do limite, o kNN continua e a regra entra como comparação;
- medir α = 0; 0,5 e 1 (peso N^−α) nas mesmas 5 partes da validação, com N_g calculado só no treino; informar média e desvio-padrão do acerto;
- os 38 filmes sem gênero mais avaliados são conferidos à mão no IMDb e ficam com o gênero conferido; eles servem de teste externo (substituído em 19/11/2026 pela proposta 2-C: a chave é conferida por código em todos os filmes);
- mostrar no notebook um filme calculado passo a passo.

**Medição exploratória (18/11/2026, fora do notebook).** Acerto (gênero previsto entre os verdadeiros) em filmes com gênero conhecido, por número de avaliadores. Faixas abaixo de 10: modelo treinado nos filmes com 10 ou mais; faixas de 10 em diante: validação cruzada de 5 partes. O kNN aqui usa a proporção de gêneros do público, não a média dos perfis da v5, por isso fica um pouco abaixo dos 83,4% do notebook.

| Avaliadores | Filmes | Chutar Drama | Regra 1/√N | kNN (k = 15) |
|---|---:|---:|---:|---:|
| 1–2 | 15.026 | 42,7% | 40,5% | 42,5% |
| 3–4 | 7.514 | 43,0% | 48,1% | 52,1% |
| 5–9 | 7.929 | 43,7% | 47,6% | 59,0% |
| 10–49 | 10.872 | 46,6% | 50,5% | 71,1% |
| 50+ | 13.138 | 47,9% | 56,7% | 84,2% |

A regra 1/√N não passa 10 pontos acima do chute em nenhuma faixa; o kNN passa a partir de 5 avaliadores. Uma regra de "perfil típico do gênero" (centróide, similaridade do cosseno) ficou abaixo do chute e foi descartada. Peso 1/N (α = 1) acerta menos de 2%.

**Rodada 2 (proposta revisada), 18/11/2026: aprovada com ajustes por unanimidade (5 de 5).** O kNN continua como método principal; o mínimo de avaliadores deixa de ser fixo (10) e passa a ser calculado pelo código: a menor faixa a partir da qual todas as faixas superiores superam o chute por pelo menos 10 pontos. O valor de 10 pontos vem da regra da v5, fixada antes da execução real; aplicá-lo por faixa foi decidido depois da medição exploratória. Ajustes: média e DP também nas faixas abaixo de 10; amostra da faixa 5–9 conferida no IMDb; mínimo e total preenchido nos números de controle; explicar o peso 1/N; coluna que marca o gênero imputado; exemplo passo a passo com um filme da faixa 5–9; papel definido para o erro do Naive Bayes. Medição com 5 partes em `analises/resultados_imputacao_por_faixa.txt`: kNN 58,7% ± 0,2 na faixa 5–9 (chute 43,7%); mínimo calculado = 5. O peso 1/N escolhe IMAX em 56% dos casos (é o menor gênero, 195 filmes), por isso acerta tão pouco.

**Implementação (v6, 19/11/2026).** Seções 2.2-b (integração com o IMDb) e 2.2-c (kNN validado). Resultados na execução com os dados reais: 4.822 dos 5.062 filmes sem gênero e 409 dos 412 sem ano preenchidos; 240 continuam sem gênero (1.406 avaliações, 0,006%); kNN com 83,4% de acerto nos filmes com 10 ou mais avaliadores; mínimo de avaliadores calculado pelo critério = 3 (a faixa 3–4 passa na validação interna, +12,3 pontos, mas no teste externo fica só 3,3 pontos acima do chute; a faixa 5–9 é a primeira que passa nos dois). A conferência manual dos 38 filmes passou a ser verificação da chave e já mostrou um caso: o código IMDb de "Black Mirror" aponta para um curta de 2011, não para a série.

Correções encontradas durante a implementação: o Spark lia 11 títulos com aspas errado (opção `escape` do CSV) e a amostra de 3.000 usuários da 1.8-a dependia da ordem das linhas.

**Proposta 2-B (IMDb + kNN): aprovada com ajustes pela mesa de 7 em 18/11/2026 (6 aprovam com ajuste, 1 aprova).** Ajustes pedidos:
- tabela inversa da conversão: para cada gênero do IMDb, % de filmes em que o MovieLens também marca o gênero correspondente, justificando cada linha da conversão;
- integridade da junção: 1:1, códigos com 8 dígitos, amostra dos tvEpisode e tvMiniSeries conferida por título e ano, os 644 anos com diferença maior que 1 como indicador de qualidade da chave (sem correção automática);
- colunas originais intactas; colunas novas de gênero final, ano final e origem de cada um;
- dizer quantos dos 240 filmes sem gênero convertível têm 5 ou mais avaliadores; se forem poucos, o kNN fica como método validado, sem etapa de reserva;
- declarar que o acerto externo é concordância com o IMDb, comentar a queda da vantagem do kNN entre a validação interna e a externa e explicar a mudança nos números da regra 1/√N (N medido no catálogo inteiro na primeira medição e só nos filmes de treino na segunda);
- comparar por origem a média de gêneros por filme e a frequência de Thriller e Musical; dizer como um filme de um só gênero entra no perfil da P1;
- guardar o recorte do IMDb usado num local privado da dupla e medir o tempo extra no Colab;
- apresentar a conferência manual como verificação da chave.

Verificações feitas logo depois da votação: a junção é 1:1 (nenhum código IMDb repetido em nenhum dos lados); os 129 filmes não encontrados têm códigos com 3 a 8 dígitos e somam 17.085 avaliações, quase todos de filmes que já têm gênero; dos 240 filmes sem gênero convertível, só 55 têm 5 ou mais avaliadores (1.065 avaliações), e a maioria é curta-metragem (gênero "Short" no IMDb, 145 filmes).

Texto da proposta, como foi à mesa: O aluno perguntou se os ~4 mil filmes que continuariam `Unknown` não poderiam ser preenchidos. Com 1–2 avaliadores nenhum método baseado no público supera o chute; é falta de informação. A proposta usa o gênero oficial do IMDb (`title.basics`, pelo código da tabela `links`) como fonte principal e mantém o kNN como método de mineração, validado contra o IMDb e usado onde o IMDb não tem gênero. Medições em `analises/resultados_imdb.txt`:
- o IMDb preenche 4.822 dos 5.062 filmes sem gênero (95%; 25.221 das 26.627 avaliações);
- nos 57.100 filmes com gênero nas duas fontes, 97,7% têm pelo menos um gênero em comum (mesmos gêneros: 46,9%; Jaccard médio 0,719; menor concordância em Musical, 48%);
- "Music" do IMDb não é convertido para Musical: só 15% desses filmes são Musical no MovieLens;
- teste externo do kNN nos próprios filmes sem gênero: 63,8% na faixa 5–9 (chute 51,8%), 66,4% em 10–49, 77,8% em 50+; com 1–4 avaliadores, igual ou pior que o chute.

**Proposta 2-C (conferência automática da chave), 19/11/2026: aprovada pela mesa de 7 (6 aprovam com ajuste, 1 aprova); ajustes aceitos e implementados em 19/11/2026.** O aluno aceitou os ajustes 1–4 e 6–9 e delegou à mesa e ao Claude a decisão sobre o ajuste 5.

Texto da proposta, como foi à mesa: o aluno pediu que a conferência da chave não fosse feita filme a filme, e sim por uma regra executada por código sobre todos os filmes. `analises/chave_imdb.py` compara três sinais (título padronizado, ano e tipo no IMDb) e valida cada um nos 57.100 filmes com gênero nas duas fontes: com a chave verdadeira, 97,67% têm pelo menos um gênero em comum; com a chave sorteada, 40,02%. A concordância fica perto de 97% em todas as categorias de título, ano e tipo; a exceção é "sem ano no MovieLens" (147 filmes, 91,8%). Títulos diferentes são, em geral, traduções (Hachi-ko = The Story of Hachikō). Decisão proposta: nenhuma regra de exclusão; manter os 4.822 gêneros do IMDb e declarar a margem de erro (3,88% das chaves; até 187 filmes e 979 avaliações, 0,0039% do total); a conferência dos 38 filmes deixa de ser feita. Alternativas não escolhidas pelo aluno: (B) devolver a "sem gênero" 28 filmes com título diferente e ano divergente ou ausente (71 avaliações); (C) excluir curtas e episódios (539 filmes, 3.421 avaliações), que concordam 97,8% e 96,7%; (D) o kNN do público como segunda opinião: com ~5% de chaves erradas e ~17% de erro do kNN, só 10% a 16% dos marcados teriam chave errada, e o kNN cobre só 301 filmes preenchidos (10+ avaliadores).

Votos: o cientista de dados aprova (mesmo que os 4.822 estivessem todos errados, seriam 0,10% das avaliações); os outros seis aprovam com ajuste. Ninguém rejeita a decisão central. A mesa conferiu as contas e apontou as premissas frágeis:
- o grupo de controle não representa os filmes preenchidos: 90% deles entraram depois do ML-20M (movieId > 131.262), têm 20 vezes mais filmes sem ano (5,27% × 0,26%) e mais curtas (9,85% × 3,83%). No controle recente, a concordância é 97,01% e a chave sorteada, 35,27%;
- a conta de 3,88% supõe que uma chave errada concorda como uma chave sorteada; uma chave que aponta para um título parecido (episódio, homônimo) concorda mais, então a margem não é "pior caso";
- entre os filmes sem ano, título igual é justamente a marca de um homônimo (Black Mirror), e não um argumento a favor da chave;
- nenhum sinal pega o Black Mirror (título igual, sem ano, curta): é a limitação da regra.

Ajustes pedidos:
1. margem recalculada no controle comparável (movieId > 131.262), separada pelo sinal de ano e ponderada por avaliações: cerca de 5,0% dos filmes (240) e 5,4% das avaliações (1.357; 0,0054% do total);
2. chamar a margem de "estimativa sob a hipótese de que chave errada concorda como chave sorteada", e declarar junto que 3,0% dos filmes do controle comparável não têm nenhum gênero em comum;
3. estender a margem ao ano: 409 anos preenchidos (11.747 avaliações) estão no grupo sem ano, em que o controle concorda 135 de 147 (IC 95% de 86,3% a 95,3%);
4. medir o efeito por usuário: 7.793 usuários avaliaram algum filme preenchido; no mais exposto, eles são 10,7% das avaliações; 102 usuários passam de 5%;
5. conferência à mão reduzida, escolhida pelo código: entre os 38 mais avaliados, só os 14 que não têm título igual, ano igual ou com 1 ano de diferença e tipo filme (13 preenchidos, 4.023 avaliações; a mesa tinha escrito 4.033);
6. registrar o Black Mirror como controle positivo e limitação (homônimo sem ano), e o Sherlock Holmes (a série aponta para um episódio: gênero certo, chave errada);
7. corrigir o texto: a faixa de 96,7% a 97,9% não inclui "especial" (99,6%, tipo em que a chave sorteada já concorda 96,9%);
8. gravar o MD5 do `title.basics` no topo de `resultados_chave_imdb.txt` e registrar que o sinal de título usa `originalTitle`, que o recorte do notebook não tem;
9. registrar o motivo de cada alternativa rejeitada (feito acima).

Onde entra: só como texto na leitura da 2.2-b (4 a 6 linhas), sem célula de código nova, para não exigir outra execução no Colab (`montar_notebook.py --saidas` aproveita as saídas das células que não mudaram). A tabela completa fica neste registro e em `analises/`. O professor de Mineração preferiria a tabela no notebook na próxima vez que tudo for executado de novo; nesse caso, o recorte do IMDb precisa ganhar `originalTitle`.

**Rodada de consenso sobre o ajuste 5, 19/11/2026: contraproposta aceita por unanimidade (5 aceitam, 2 aceitam com ajuste).** Ninguém preenche tabela: o código lista os 14 filmes (dos 38 sem gênero com 50 ou mais avaliações, os que não passam em "título igual, ano igual ou com 1 ano de diferença, tipo filme"), com título, ano e tipo do IMDb lado a lado e, por pedido do professor de Banco de Dados, os outros filmes do MovieLens com o mesmo título (igual depois de padronizar, ou um é o começo do outro). A tabela `conferencia_manual/filmes_sem_genero_mais_avaliados.csv` e a parte D2 do gerador e das instruções foram retiradas. Nenhum gênero muda. O que a lista mostra:
- 11 se resolvem pela própria linha: séries sem ano no MovieLens (Cosmos, The OA, Death Note, Cosmos: A Spacetime Odyssey, 13 Reasons Why), traduções e títulos alternativos (Fantastic Beasts 2 = The Crimes of Grindelwald; 120 BPM; Somnio = Infinity Chamber), Paterson (sem ano no MovieLens), um episódio e um especial com título igual;
- Whiplash (2013) é o curta; o longa de 2014 é outro filme do MovieLens (movieId 112552), com o mesmo gênero (Drama);
- o Sherlock Holmes sem subtítulo aponta para "Mortal Fight", uma das três partes do filme de 1980; as outras duas (King of Blackmailers e The Hunt for the Tiger) estão separadas no MovieLens, então a chave provavelmente está certa;
- Black Mirror é o controle positivo: a chave está errada (homônimo sem ano) e nenhum sinal da regra a pega. Esse filme não recebeu gênero, porque o único gênero do curta no IMDb é "Short". A limitação fica declarada na 2.2-b.

Fontes externas consultadas em 19/11/2026 (pedido do engenheiro de dados, porque não se refazem por código): IMDb tt2492564 e TMDb 452830 (Black Mirror: os dois apontam para o curta de 4 minutos de Doug Aitken, de 2011); TMDb 127605 (Sherlock Holmes: página removida, endereço com o nome da série); TMDb 428948, títulos alternativos (Somnio consta como título de Infinity Chamber).

**Implementação (19/11/2026).** `analises/chave_imdb.py` grava em `analises/resultados_chave_imdb.txt`, com o MD5 do `title.basics` lido (`694fa9cd5bad4d6f3f94384b7ed61c74`): a margem no controle comparável (29.412 filmes com movieId > 131.262; concordância 97,01%, chave sorteada 35,37%, 2,99% sem nenhum gênero em comum), separada pelo sinal de ano e ponderada por avaliações: 5,0% dos filmes (240) e 5,4% das avaliações (1.359; 0,0054% do total); anos: 135 de 147 concordam no controle sem ano (IC 95% de 86,3% a 95,3%), cerca de 56 dos 409 anos preenchidos possivelmente errados (até 94); por usuário: 7.793 usuários afetados, no máximo 10,7% das avaliações de um usuário, 102 acima de 5%, efeito esperado de 0,58 ponto percentual no mais exposto. No notebook, só texto na leitura da 2.2-b (conferência da chave, margem de erro e a limitação do homônimo sem ano); as células de código e as saídas da execução no Colab não mudaram.

## D3 — Títulos repetidos

**Status:** aprovada com ajustes em 18/11/2026; aguardando a conferência manual dos 8 pares.

**Fatos medidos:** 98 títulos idênticos (nome + ano), 196 filmes; nenhum par com o mesmo código IMDb; 8 pares com o mesmo código TMDb. Os outros 90 são filmes diferentes com o mesmo nome (ex.: War of the Worlds (2005) de Spielberg e a produção homônima). 2.747 nomes aparecem com anos diferentes (refilmagens).

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- TMDb igual é só candidato; a fusão depende da conferência manual registrada em `conferencia_manual/titulos_repetidos.csv`;
- interseção de gêneros vazia não leva à fusão (Blackout (2007) fica separado, salvo prova em contrário);
- nos pares confirmados, comparar interseção × gêneros da cópia mais avaliada e justificar;
- fundir antes de qualquer contagem por filme, com o mesmo mapeamento em ratings, tags e links; registrar a tabela movieId antigo → novo e as avaliações removidas (usuário que avaliou as duas cópias fica com a mais recente);
- mostrar no notebook os contraexemplos da regra só por título + ano.

## D4 — Filmes sem ano no título e gêneros raros

**Status:** aprovada com ajustes em 18/11/2026; arquivo do IMDb baixado em 18/11/2026 (MD5 `694fa9cd5bad4d6f3f94384b7ed61c74`). O IMDb tem ano para 409 dos 412 filmes sem ano. Nos 61.880 filmes com ano nas duas fontes, o ano é igual em 92,8% e difere em até 1 ano em 99,0% (`analises/resultados_imdb.txt`).

**Fatos medidos:** 412 filmes sem ano no título (11.749 avaliações; 264 deles também sem gênero), incluindo Ready Player One e Moonlight.

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- buscar o ano no `title.basics` do IMDb pelo código `tt` + 7 dígitos; validar contra os anos que já estão nos títulos e informar a concordância; registrar preenchidos e não preenchidos (filmes e avaliações); registrar data e MD5 do arquivo;
- deixar escrito o denominador de cada participação: a contagem simples (1.6-c, um filme conta em cada gênero) e a fracionária (3.8, peso 1/k);
- testar a sensibilidade do corte de 2% da 3.8 com War (1,80% na contagem fracionária), o gênero mais perto do corte;
- decidir o tratamento do IMAX no vetor do perfil (ver D5).

> Correção ao parecer da mesa: ela comparou o corte de 2% com as participações da contagem simples (Musical 3,86%, IMAX 4,25%). O corte da 3.8 usa a contagem fracionária, em que Musical tem 1,14% e IMAX 1,05%.

## D5 — Gêneros parecidos (sobreposição)

**Status:** proposta, aguardando aprovação do aluno para ir à mesa.

**Fatos medidos:** dos 195 filmes IMAX, só 1 não tem outro gênero; nas avaliações, 63% dos IMAX também são Action, 60% Adventure e 45% Sci-Fi. Animation × Children é o par mais sobreposto: Jaccard 0,21 nos filmes, 78% das avaliações de Animation também são Children, correlação 0,82 entre usuários (3.2).

**Proposta:** IMAX sai do vetor do perfil (é formato de exibição, não gênero); Animation e Children viram um grupo só no vetor; os rótulos continuam nos dados.

## Pendências do aluno

- Rodada própria sobre a D3: a conferência à mão dos 8 pares de títulos repetidos choca com o princípio do aluno de não conferir filme a filme.
- Aprovação da D5 para ir à mesa.
- Decisão sobre Musical e gêneros raros (agrupar só no vetor ou já no pré-processamento).
