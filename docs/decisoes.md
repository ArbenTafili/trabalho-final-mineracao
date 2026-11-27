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

**Status:** aprovada com ajustes em 18/11/2026; ajustes aceitos pelo aluno; implementada na v6 (19/11/2026) e ampliada na v7 (testes da fusão e da exclusão, 21 números de controle).

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- edita-se só em `fonte/*.py`; o `.ipynb` é sempre gerado por `scripts/montar_notebook.py` e nunca editado à mão;
- a execução que gera as saídas entregues é feita no Colab; o Mac serve de pré-teste;
- sementes fixas e uma lista de números de controle comparada entre Mac e Colab;
- pico de memória medido na execução local; versões das bibliotecas iguais às do Colab;
- dados sintéticos com casos de borda (filme sem gênero, sem ano, título repetido, filmes com 9 e 10 avaliações);
- dados do MovieLens e do IMDb fora do repositório.

## D2 — Preencher o gênero com contagem ponderada por 1/√N

**Status:** superada. A regra 1/√N ficou só como comparação (rodada 2); o gênero que falta vem do IMDb (proposta 2-B, v6), a chave é conferida por código (proposta 2-C) e os filmes que continuam sem gênero são excluídos (D6, v7).

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

**Status:** substituída pela D6 em 19/11/2026: os 8 pares são decididos por regra automática, sem conferência à mão (7 fundidos, Blackout separado).

**Fatos medidos:** 98 títulos idênticos (nome + ano), 196 filmes; nenhum par com o mesmo código IMDb; 8 pares com o mesmo código TMDb. Os outros 90 são filmes diferentes com o mesmo nome (ex.: War of the Worlds (2005) de Spielberg e a produção homônima). 2.747 nomes aparecem com anos diferentes (refilmagens).

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- TMDb igual é só candidato; a fusão depende da conferência manual registrada em `conferencia_manual/titulos_repetidos.csv`;
- interseção de gêneros vazia não leva à fusão (Blackout (2007) fica separado, salvo prova em contrário);
- nos pares confirmados, comparar interseção × gêneros da cópia mais avaliada e justificar;
- fundir antes de qualquer contagem por filme, com o mesmo mapeamento em ratings, tags e links; registrar a tabela movieId antigo → novo e as avaliações removidas (usuário que avaliou as duas cópias fica com a mais recente);
- mostrar no notebook os contraexemplos da regra só por título + ano.

## D4 — Filmes sem ano no título e gêneros raros

**Status:** aprovada com ajustes em 18/11/2026; ano preenchido pelo IMDb na v6; teste de sensibilidade do corte de 2% feito na v8 (3.8-a): com 1,5%, só War muda de lado; com 2,5%, só Mystery; o limite fica em 2%. Musical (1,25%) fica em `raros`, e o agrupamento é feito só no vetor revisado, não no pré-processamento. Arquivo do IMDb baixado em 18/11/2026 (MD5 `694fa9cd5bad4d6f3f94384b7ed61c74`). O IMDb tem ano para 409 dos 412 filmes sem ano. Nos 61.880 filmes com ano nas duas fontes, o ano é igual em 92,8% e difere em até 1 ano em 99,0% (`analises/resultados_imdb.txt`).

**Fatos medidos:** 412 filmes sem ano no título (11.749 avaliações; 264 deles também sem gênero), incluindo Ready Player One e Moonlight.

**Votos:** 4 aprovam com ajuste, 1 aprova.

**Ajustes:**
- buscar o ano no `title.basics` do IMDb pelo código `tt` + 7 dígitos; validar contra os anos que já estão nos títulos e informar a concordância; registrar preenchidos e não preenchidos (filmes e avaliações); registrar data e MD5 do arquivo;
- deixar escrito o denominador de cada participação: a contagem simples (1.6-c, um filme conta em cada gênero) e a fracionária (3.8, peso 1/k);
- testar a sensibilidade do corte de 2% da 3.8 com War (1,80% na contagem fracionária), o gênero mais perto do corte;
- decidir o tratamento do IMAX no vetor do perfil (ver D5).

> Correção ao parecer da mesa: ela comparou o corte de 2% com as participações da contagem simples (Musical 3,86%, IMAX 4,25%). O corte da 3.8 usa a contagem fracionária, em que Musical tem 1,14% e IMAX 1,05%.

## D5 — Gêneros parecidos (sobreposição)

**Status:** aprovada pelo aluno em 25/11/2026 ("faz sentido isso do perfil ter o Animation e Children"; IMAX fora: "as duas") e pela mesa de 7 (2 aprovam, 5 aprovam com ajuste); implementada na v8 (25/11/2026).

**Fatos medidos:** dos 195 filmes IMAX, só 1 não tem outro gênero; nas avaliações, 63% dos IMAX também são Action, 60% Adventure e 45% Sci-Fi. Animation × Children é o par mais sobreposto: Jaccard 0,21 nos filmes, 78% das avaliações de Animation também são Children, correlação 0,82 entre usuários (3.2).

**Proposta:** IMAX sai do vetor do perfil (é formato de exibição, não gênero); Animation e Children viram um grupo só no vetor; os rótulos continuam nos dados.

**Votação (25/11/2026).** A proposta foi à mesa só para o vetor revisado (3.8): IMAX fora; Animation e Children num grupo `Animation/Children`; k = número de grupos distintos do filme; limite de 2% aplicado aos 17 grupos. Ficaram de fora: juntar também no vetor base (o base fica como representação fiel e comparação) e renomear Children para Family (Family é o rótulo do IMDb e concorda só de 53% a 62% com Children). Votos: 2 aprovam (prof. de Banco de Dados e cientista de dados) e 5 aprovam com ajuste. Ajustes:
1. registrar os números esperados antes da execução (abaixo);
2. declarar que a decisão é de taxonomia, não de redundância (correlação de 0,82, abaixo do limiar de 0,9 da 3.2), e que vale qualquer que seja o resultado da 3.8-b; mostrar a v7 e a v8 lado a lado, só para transparência;
3. informar a sobreposição nos dois sentidos (77,7% das avaliações de Animation são de filmes também Children; 59,6% no sentido contrário) e que o grupo fica com cerca de 3,33%, não 4,03%: contado uma vez só por filme, ele passa 0,70 ponto aos gêneros que aparecem junto;
4. sensibilidade do limite (fecha a pendência da D4): imprimir o destino de cada grupo com 1,5%, 2% e 2,5%;
5. uma constante única de agrupamento, k por grupos distintos, coluna `frac_animation_children`, `assert` de 17 grupos e de soma 1;
6. testes dos quatro casos de borda ("Action|IMAX", só IMAX, "Animation|Children|Comedy", só "Animation");
7. exemplo calculado na 3.8-a (Toy Story e Avatar);
8. fechar a D5, a decisão 24 e as pendências da D4 (agrupamento só no vetor; Musical em `raros`; teste com War feito).

**Números esperados, registrados antes da execução v8** (conta da mesa sobre os dados reais, sem as fusões da D3, cujo efeito é desprezível): 17 grupos; Animation/Children perto de 3,33%; 12 grupos mantidos (Drama 20,80%, Comedy 17,06%, Action 9,60%, Thriller 9,19%, Adventure 7,33%, Romance 6,77%, Crime 5,78%, Sci-Fi 5,74%, Fantasy 3,46%, Animation/Children 3,33%, Horror 3,17%, Mystery 2,49%); `raros` com War 1,81%, Musical 1,25%, Documentary 1,11%, Western 0,80% e Film-Noir 0,30% (cerca de 5,27%); vetor revisado com 20 atributos; 1 filme só com IMAX (movieId 4460, Encounter in the Third Dimension, 27 avaliações); com o limite em 1,5%, só War muda; em 2,5%, só Mystery. A seção 1 até a 3.7 (vetor base) não muda.

**Implementação (v8, 25/11/2026).** Só a célula 3.8-a mudou: constante única `GRUPO_DO_GENERO`, k = número de grupos distintos do filme, coluna `frac_animation_children`, `assert` de 17 grupos e de soma 1, os quatro casos de borda testados em `pesos_do_filme` dentro do notebook, a sobreposição nos dois sentidos, o exemplo calculado (Toy Story e Avatar, conferido contra o cálculo do Spark) e a tabela de sensibilidade do limite. Novo número de controle: grupos do perfil. Na execução com os dados reais no Mac, os números esperados bateram: 17 grupos; Animation/Children com 3,34%; 12 grupos mantidos; `raros` com War, Musical, Documentary, Western e Film-Noir (5,27%); 1 filme só com IMAX (27 avaliações); com 1,5% só War muda e com 2,5% só Mystery. Sobreposição: 77,6% e 59,6% (a mesa tinha 77,7%, sem as fusões da D3); Jaccard 0,213. Efeito no vetor revisado: o PCA passa a ter d′ = 7 (73,94% da variância; na v7, 6 e 69,13%); atípicos 1.171 (v7: 1.078; vetor base: 1.293); Hopkins robusto 0,68; diversidade de gêneros normalizada por log(17). A seção 1 até a 3.7 não mudou. No Colab (25/11/2026, das 11:18 às 11:50, em segundo plano com `jupyter nbconvert`), os 22 números de controle saíram idênticos aos do Mac.

## D6 — Excluir o que não tem tratamento automático (proposta 3)

**Status:** aprovada com ajustes pela mesa de 7 em 19/11/2026 (7 aprovam com ajuste); aceita pelo aluno; implementada na v7 (19/11/2026).

**Princípio do aluno:** um caso que nenhuma regra automática consegue tratar, e que não pesa de forma expressiva, é excluído dos dados-alvo, com a justificativa registrada.

**Fatos medidos** (`analises/casos_sem_tratamento.py` → `analises/resultados_casos_sem_tratamento.txt`):
- **A. Filmes que continuam sem gênero depois do IMDb:** 240 filmes, 1.406 avaliações (0,0056%). Motivos: 209 só têm gêneros que não existem no MovieLens (145 só "Short"), 16 têm IMDb sem gênero e 15 têm código ausente do IMDb. Por avaliadores: 7 com 0, 131 com 1–2, 47 com 3–4, 36 com 5–9 e 19 com 10 ou mais. Excluir os 233 com avaliação afeta 823 usuários, no máximo 5,0% das avaliações de um usuário; ninguém perde todas, e o menor resto é de 19 avaliações.
- **B. Filmes que continuam sem ano:** 3 filmes, 2 avaliações; os 2 avaliados também estão em A, e o terceiro não tem avaliação.
- **C. Títulos repetidos com o mesmo TMDb (D3):** em 6 dos 8 pares, um dos dois códigos IMDb não existe mais (é sempre o da cópia mais avaliada); Enron tem as duas entradas no IMDb com o mesmo ano e 109 × 110 minutos; Blackout são filmes diferentes (2008, 85 min, Horror × 2007, 95 min, Drama). 171 usuários avaliaram as duas cópias de algum dos 7 pares.

**Proposta, como foi à mesa:** em A, gênero pelo kNN da 2.2-c para os 55 filmes com 5 ou mais avaliadores e exclusão dos 178 com 1 a 4 e dos 7 sem avaliação; alternativa simples: excluir os 240 sem kNN. Em C, fusão automática pelo IMDb (código ausente, ou duração até 5 minutos de diferença e gênero em comum); alternativa: excluir a cópia menos avaliada (363 avaliações).

**Votos:** 7 aprovam com ajuste. A mesa prefere a alternativa simples de A e a fusão em C.

**Por que sem kNN em A:** dos 55 filmes com 5 ou mais avaliadores, 46 são curtas no IMDb. Nos 1.100 curtas com gênero no MovieLens e 5 ou mais avaliadores, o kNN acerta 51,0%, e chutar o gênero mais comum entre os curtas (Animation) acerta 52,7%; na faixa 5–9, 41,2% contra 53,1%. A validação da 2.2-c (filmes de longa-metragem, linha de base Drama) não vale para essa população; só 14,8% dos curtas são Drama. Num grupo de 117 filmes parecidos, mas formado sobretudo por códigos ausentes do IMDb e longas, o kNN vai bem (62,4% contra 27,4%; medição feita pela mesa, que não está nos arquivos de `analises/`), mas ele representa pouco os 55. Na faixa 10–49, o kNN supera o chute nos curtas em 7,8 pontos (56,2% contra 48,4%), abaixo dos 10 pontos do critério. O ajuste da 2-B já previa não ter etapa de reserva se os restantes fossem poucos. Pelo princípio do aluno, esses filmes não têm tratamento automático seguro.

**Ajustes pedidos:**
1. A: excluir os 240 filmes sem gênero (1.406 avaliações), sem kNN; a 2.2-c continua como validação; registrar que o Black Mirror sai com chave errada;
2. os 7 sem avaliação saem por necessidade do código (para `Unknown` deixar de existir e `prop_unknown` não virar coluna constante), não pelo peso;
3. C: fundir pelo critério já registrado na D3 (mesmo título, mesmo ano, mesmo TMDb e pelo menos um gênero do MovieLens em comum: funde 7, separa Blackout); o que o IMDb mostra entra como evidência; sem o limite de 5 minutos;
4. integridade da fusão: a cópia que fica herda o código IMDb que existe; a tabela movieId antigo → novo vale para ratings, tags e links; a fusão vem antes da 2.2-b e da 2.2-c; a seção 1 descreve os dados antes da fusão;
5. escrever o critério de "não pesa" em números (fração do total de avaliações, máximo por usuário, ninguém perdendo todas), declarando que foi fixado depois de ver A e que vale daqui em diante;
6. textos: o roteiro deixa de dizer que todos os usuários têm pelo menos 20 avaliações depois da limpeza (1 fica com 19); contar as avaliações excluídas por motivo;
7. execução: registrar antes os números de controle esperados (avaliações 25.000.095 → 24.998.518; filmes 62.423 → 62.176; usuários 162.541); casos sintéticos e testes para a fusão e a exclusão; retirar `conferencia_manual/` e o gerador; uma só execução no Colab.

**Aceite do aluno (19/11/2026):** "sim", com o pedido de guardar todas as justificativas (desta e das decisões anteriores) num lugar só: `docs/justificativas.md`.

**Números de controle esperados, registrados antes da execução com os dados reais** (contas a partir das medições acima):
- avaliações nos dados tratados: 25.000.095 − 171 repetidas − 1.406 excluídas = 24.998.518;
- filmes no catálogo: 62.423 − 7 fundidos − 240 excluídos = 62.176;
- filmes avaliados: 59.047 − 7 fundidos − 233 excluídos com avaliação = 58.807;
- usuários: 162.541 (ninguém perde todas as avaliações);
- cópias fundidas: 7; filmes excluídos: 240; avaliações excluídas: 1.406;
- seção 1, gênero do IMDb (4.822) e filmes que continuam sem gênero na 2.2-b (240): sem mudança;
- kNN, PCA, atípicos e Hopkins: mudam, porque `prop_unknown` sai do vetor base e a fusão mexe em 7 filmes; sem valor esperado.

**Implementação (v7, 19/11/2026).** Duas células novas na seção 2: 2.2-a (fusão dos títulos repetidos, antes da integração com o IMDb) e 2.2-d (exclusão, depois da validação do kNN). Na execução com os dados reais no Mac, todos os números esperados bateram: 24.998.518 avaliações, 62.176 filmes no catálogo, 58.807 filmes avaliados, 162.541 usuários, 7 cópias fundidas, 171 avaliações repetidas, 240 filmes e 1.406 avaliações excluídos. Detalhes:
- **gêneros do filme fundido:** os da cópia mais avaliada (regra sem julgamento caso a caso; a interseção da D3 apagaria, por exemplo, o Musical de Lagaan, que o IMDb também marca);
- **código IMDb:** 6 das 7 cópias que ficaram herdaram o código da cópia fundida, porque o delas não existe mais no IMDb;
- **critério de "não pesa"** (fixado depois de ver os casos, vale daqui em diante): no máximo 0,1% das avaliações, no máximo 10% das avaliações de um usuário e nenhum usuário perdendo todas. Resultado: 0,0056%, 5,0% e 0; 823 usuários afetados; 1 usuário fica com 19 avaliações;
- **seção 3:** sem `prop_unknown`, o vetor base tem 21 atributos; o PCA passa a ter d′ = 6 (66,84%, antes 7 e 68,41%); atípicos 1.293 (antes 1.365); Hopkins robusto 0,705 no PCA. O vetor revisado quase não muda (69,13%; 1.078 atípicos). O kNN da 2.2-c acerta 83,38% (antes 83,35%), porque a fusão muda as partes da validação cruzada;
- **correções encontradas:** o gerador de dados sintéticos sorteava "Children" como gênero do IMDb para filmes sem gênero (o IMDb usa "Family"), o que deixava filmes populares sem gênero; a 3.8-a imprimia "log(20)" fixo na normalização da diversidade, que agora usa o número real de gêneros (19).

## D7 — Revisão pedida pelo professor (proposta 4, 26/11/2026)

**Status:** aprovada pela mesa de 7 (os 10 itens; a maioria com ajuste); dúvidas de escopo resolvidas com o aluno (esclarecimentos abaixo); implementada na v10 (26/11/2026), exceto o agrupamento (item 7, P1) e os itens 8 e 9 (só registro).

**Origem:** anotações do aluno da conversa com o professor. Algumas são ambíguas e foram interpretadas pelo Claude com base nos slides da disciplina (PCA e escolha de d′; seleção por CFS; agrupamento; validação). Medições em `analises/pca_criterios.py`, `analises/atributos_por_genero.py`, `analises/dimensao_por_genero.py` e nos resultados correspondentes.

**Itens e ajustes da mesa:**
1. **PCA como no método da disciplina:** min-max por dimensão, centralizar na média e autovalores da covariância. O z-score sai da 3.4 à 3.10 e fica só como comparação na 3.3 (amplia gêneros raros, z até 47) e no critério univariado |z| da 3.7-a. Critérios de d′, todos impressos e marcados num só scree plot: % acumulada (80% e 90%), broken-stick pela regra do slide (último autovalor acima; hoje igual à regra antiga), Kaiser com λ / média(λ) ≥ 1, dimensão de correlação D2 com d′ = ⌈D2⌉ + 1 (gráfico log C(r) × log r e sensibilidade à faixa e à semente), MLE como comparação. Medição no vetor base (20 atributos, sem IMAX): 3, 5, 6, 9, 8 e 13 componentes. Regra fixada antes de executar: entrada principal com d′ = ⌈D2⌉ + 1; broken-stick como alternativa compacta. O "fim estranho" do gráfico se explica: com o z-score, autovalores do fim ficam acima do broken-stick; com min-max e covariância, nenhum depois do 3º. Gráfico de distâncias da 3.5 com min-max e o PCA de cada critério, sem a curva do z-score.
2. **Dimensionalidade intrínseca global e local:** LID por usuário com o mesmo estimador do MLE (k de 10 a 20), no espaço PCA escolhido, resumida por gênero dominante (mediana, IQR, n). A parte "depois do agrupamento" vai com P1.
3. **IMAX fora no pré-processamento, antes da 2.2-b:** o único filme só com IMAX (movieId 4460) passa a ser preenchido pelo IMDb (Adventure, Comedy, Drama) e não é excluído. Esperado: 4.823 filmes com gênero do IMDb; 240 excluídos e 24.998.518 avaliações sem mudança; vetor base com 20 atributos.
4. **Desvio-padrão da nota por gênero:** na forma relativa, n/(n + 5) · (desvio no grupo − desvio geral), 0 com menos de 2 avaliações no grupo; pelos 13 grupos do vetor revisado (cobertura de 71% a 99,9%); só no vetor revisado. As formas "encolhida para o desvio geral" e "bruta com o geral no lugar" foram descartadas: repetem o desvio geral (13 pares acima de 0,8 com `nota_dp`) e dominariam o PCA. Afinidade (média no grupo − média geral): sem redundância; a maioria recomenda deixar fora por enquanto (4 × 3), com a escolha devolvida ao aluno.
5. **Limiar de redundância de 0,8** (|Pearson| e |Spearman|), registrado como escolha do projeto ao aplicar o CFS (o slide usa ranking, sem limiar) e como valor sugerido pelo professor depois de ver os pares. Na 3.2, sai o atributo de maior |Pearson| médio com os demais. Efeito esperado: sai Animation ou Children no base e `nota_media` ou o viés do usuário no revisado.
6. **Coordenadas paralelas com 100 usuários**, a mesma amostra fixa, em escala min-max.
7. **Agrupamento (P1), se o aluno confirmar o escopo:** k-means em todos os usuários (inicialização aleatória, várias repetições); hierárquico com ligação média e completa sobre os centróides de cerca de 1.000 microgrupos do k-means ou numa amostra fixa, corte no maior salto do dendrograma conferido pela silhueta; comparação das duas partições por Rand e Jaccard; silhueta por objeto, grupo e partição numa amostra; WSS e BSS; linha de base com dados uniformes; com e sem atípicos; DBSCAN opcional. Parâmetros pré-registrados na mesa; execução própria, depois da transformação.
8. **"Aquelas avaliações":** registrar; as leituras possíveis (atípicos, filmes pouco avaliados, notas baixas, chaves possivelmente erradas do IMDb) já têm tratamento previsto. Perguntar ao professor.
9. **Períodos:** registrar e não fazer (coerente com a decisão de deixar o tempo fora do perfil).
10. **Dimensionalidade intrínseca por gênero (pedido do aluno, com atenção especial; fora do tema original):** estudo em `analises/dimensao_por_genero.py`, prévia num vetor revisado aproximado. Desenho: o público de um gênero são os 10% de usuários com maior proporção naquele grupo (13 públicos de cerca de 16 mil; o gênero dominante não serve, porque Drama tem 54,2% dos usuários e Mystery 0,01%); min-max global; 10 amostras de 2.000 usuários por público, pareadas por atividade; usuários sorteados e placebos (10% mais altos em nota média e em atividade) como linha de base. Regra fixada antes: um gênero só precisa de outro número de componentes se a faixa do seu d′ não se sobrepõe à dos sorteados e a diferença passa da do placebo. Resultado (pareado): todo gênero precisa de 8 componentes por ⌈D2⌉ + 1, como os sorteados (Thriller de 7 a 8); D2 de 6,05 (Thriller) a 6,86 (Animation/Children), contra 6,54 nos sorteados, diferença do tamanho do placebo da nota média (6,28); 80% dá 7 em 12 dos 13 (Horror 8). A razão entre a variação de um gênero para outro e a variação entre amostras é 11,1 no D2, mas 0,4 no ⌈D2⌉ + 1. O que muda de verdade é a forma: o 1º componente explica de 18,1% (Horror) a 26,0% (Fantasy) da variância. Voto: 7 × 0 pelo estudo em `analises/`, 0 por seção nova no notebook, 6 × 1 por um resumo curto na leitura da 3.5-b (o cientista de dados prefere uma frase). A versão "depois do agrupamento" (dimensionalidade dentro de cada grupo de P1, contra subconjuntos sorteados e dados uniformes) vai na validação de P1. Refazer quando o vetor revisado da proposta 4 estiver pronto.

**Esclarecimentos do aluno (26/11/2026), que mudam os itens 1 e 4:**
- **"Até mais 1" é um quarto critério, não um "+1" na dimensão intrínseca.** O pedido é um PCA para cada critério: broken-stick, Kaiser-Guttman, dimensão intrínseca e mais um. O slide "Estimadores para caso PCA" separa os critérios em três famílias: gráfico (scree plot com limiar de %), estatístico (broken-stick e Kaiser-Guttman) e dimensão intrínseca (ρ-score e dimensão fractal). Pela estrutura do slide, o quarto é o gráfico, ou seja, a % acumulada da variância. A dimensão intrínseca passa a dar d′ = ⌈D⌉, sem o +1. O ρ-score do slide, ρ = μ² / (2σ²) da distribuição dos pares de distância, entra ao lado da dimensão fractal. No vetor base (min-max, 20 atributos), ρ fica entre 3,9 e 4,03 conforme a amostra de 2.000 usuários, o que dá 4 ou 5 componentes. Como o arredondamento para cima é sensível nesse ponto, ρ deve ser a mediana de várias amostras. Com ⌈D2⌉ = ⌈6,21⌉ são 7 componentes (85,4%). O PCA com 9 componentes de que o aluno gostou vinha do z-score; no método da disciplina, 80% dá 6 componentes e 90% dá 9.
- **O desvio-padrão por gênero vem da figura "Atributos do perfil por usuário"** (a 3.1-b, com 24 atributos antes da 3.2). O professor viu o histograma de `nota_dp` e pediu o mesmo para cada gênero: o desvio-padrão das notas do usuário dentro de cada gênero. Logo, o desvio por gênero entra no perfil da 3.1, e não só no vetor revisado.
  - Medição (script de conferência, dados sem a limpeza da seção 2): o desvio bruto só existe para quem tem 2 ou mais avaliações no gênero, de 99,9% dos usuários em Drama a 21,7% em Documentary. A média vai de 0,63 (Film-Noir) a 0,93 (Comedy).
  - Na forma relativa, os 18 desvios no vetor base (38 atributos) não criam pares acima de 0,8, mas sobem a dimensionalidade: D2 de 6,21 para 9,11, Kaiser de 5 para 9, 80% de 6 para 15 componentes, ρ de 4,0 para 6,35.
  - Eles também trazem de volta o "fim estranho" do broken-stick: os autovalores 32 a 38 ficam acima da curva, então a regra do último acima dá 38 e a dos primeiros acima dá 3. O motivo são 18 colunas fracas e quase independentes (variância média 0,0036 no min-max, contra 0,0089 das proporções).

**Plano:** dúvidas de escopo com o aluno; medir em `analises/` e registrar os números esperados; seção 2 (IMAX); seção 3 na ordem 3.1 → 3.10; uma execução no Mac e uma no Colab; documentos. P1 depois, com proposta pré-registrada.

**Escolha do aluno sobre o desvio por gênero (26/11/2026):** "Figura + vetor revisado": o desvio bruto por gênero entra no perfil e na figura da 3.1-b; no PCA, só no vetor revisado, na forma relativa e pelos 13 grupos. A 3.4 não ganhou a linha "e se os 18 entrassem no vetor base": a medição está aqui e em `analises/` (38 atributos; D2 de 6,21 para 9,11; broken-stick com o fim cruzando a curva).

**Implementação (v10, 26/11/2026).** Números esperados conferidos na execução com os dados reais no Mac (`execucoes/real_v10.ipynb`):
- *IMAX (item 3):* 195 filmes; o único só com IMAX (4460) recebeu Adventure, Comedy e Drama do IMDb; 4.823 filmes com gênero do IMDb, 240 excluídos e 24.998.518 avaliações, como esperado. O kNN da 2.2-c, sem o IMAX entre as classes, acerta 83,37% (antes 83,38%).
- *Limiar de 0,8 (item 5):* sai `prop_children` no vetor base (19 atributos) e `nota_media` no revisado, como esperado.
- *PCA (item 1):* vetor base em min-max, 19 atributos: broken-stick 3 (64,91%), Kaiser 4 (72,46%), ρ-score 3,95 → 4, 80% → 6 (82,07%), D2 = 6,17 → 7 (85,73%). O fim não cruza a curva broken-stick. D2 vai de 5,5 a 6,9 conforme a faixa de raios.
- *Dimensão local (itens 2 e 10):* na 3.5-c, todos os 18 públicos precisam de 7 componentes, como os sorteados (Western de 6 a 7); nenhum gênero difere pela regra fixada. Confirma a prévia em `analises/`.
- *Desvio por gênero (item 4):* no perfil, de 21,7% (Documentary) a 99,9% (Drama) dos usuários; no revisado, `dprel` nos 13 grupos. O vetor revisado fica com 32 atributos e PCA com 9 componentes (D2 = 8,25; 67,32%); o broken-stick dá 4, com os autovalores 24 a 31 acima da curva no fim (a regra usa os iniciais). Transparência: sem os desvios, o revisado teria 7 componentes (80,34%), variância relativa das distâncias 0,344 (com eles, 0,310), Hopkins 0,681 (0,656) e 1.070 atípicos (1.348).
- *Coordenadas paralelas (item 6):* 100 usuários em min-max.
- *Atípicos:* 1.156 no vetor base (antes 1.293), agora puxados por Horror, Animation e pela nota, e não pelos gêneros raros.

**Ponto em aberto para o aluno:** com os desvios relativos, o PCA do vetor revisado guarda menos variância e concentra mais as distâncias que o PCA do vetor base e que o revisado sem os desvios. A entrada principal registrada continua a mesma, e o notebook registra o custo; a escolha final entre as três entradas fica para P1 ou para o aluno.

## Medição complementar: PCA com 80% da variância (25/11/2026)

O aluno pediu para ver o PCA com 9 componentes (critério de 80% da variância, registrado desde a 3.4 como alternativa) ao lado do PCA principal. Não muda nenhuma decisão: só acrescenta a comparação, na v9, da 3.5 à 3.9 (concentração de distâncias, dimensionalidade intrínseca, t-SNE, Hopkins, tabela da 3.8-b e índices da 3.9), nos dois vetores. Resultado na execução com os dados reais: com 9 componentes, a variância retida sobe (base 80,07%; revisado 82,37%), a variância relativa das distâncias cai um pouco (0,417 e 0,354, contra 0,426 e 0,369), o Hopkins robusto fica em 0,718 e 0,685 (contra 0,705 e 0,684), os atípicos sobem (1.422 e 1.351, contra 1.293 e 1.171) e os pivôs evitam 99,2% e 99,0% das distâncias (contra 99,9% e 99,8%). A escolha entre 6/7 e 9 componentes fica para P1.

## Pendências do aluno

- Validar o limiar de nota positiva (4,0). Ele só define `prop_positivas`, que já está fora da entrada de P1 (redundante com `nota_media`, 3.2); só volta a importar se esse atributo voltar.
- Próximas etapas: P1 e P2 (anotações do aluno em `docs/anotacoes_futuro.md`).
