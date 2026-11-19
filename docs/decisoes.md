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
- os 38 filmes sem gênero mais avaliados são conferidos à mão no IMDb e ficam com o gênero conferido; eles servem de teste externo;
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

- Conferência manual dos 38 filmes (D2) e dos 8 pares (D3).
- Aprovação da D5 para ir à mesa.
- Decisão sobre Musical e gêneros raros (agrupar só no vetor ou já no pré-processamento).
