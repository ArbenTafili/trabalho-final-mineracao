# Justificativas das decisões

Este documento reúne num lugar só as justificativas de todas as decisões do trabalho final de Mineração de Dados sobre o MovieLens 25M. O trabalho segue o processo KDD (*Knowledge Discovery in Databases*: seleção → pré-processamento → transformação → mineração → interpretação), e o notebook vai até a seção 3 (Transformação). É a versão consolidada e legível de [`docs/decisoes.md`](decisoes.md), que guarda o registro detalhado: votos da mesa de avaliação, ajustes pedidos e contas. O aluno pediu este documento ao aceitar a D6, em 19/11/2026.

**Como ler.** As decisões aparecem na ordem do processo KDD: seleção (seção 1 do notebook), pré-processamento (seção 2) e transformação (seção 3). Depois vêm as decisões de infraestrutura. Cada decisão segue o mesmo roteiro: o que foi decidido, por quê (com os números reais), alternativas descartadas, limitações, onde está a evidência e status. Quando a decisão mudou com o tempo, um item "Como chegou aqui" conta a sequência e diz o que vale hoje. Os códigos D1 a D6, 2-B e 2-C remetem ao registro em `docs/decisoes.md`; números como 2.2-b remetem às seções do notebook.

Os números são os da v7, executada com os dados reais em 19/11/2026 (`execucoes/real_v7.ipynb`), salvo indicação. A seção 1 descreve os dados-alvo antes da fusão e da exclusão da v7, e por isso seus números não mudaram. Os tempos da 3.9 dependem da máquina e não são citados aqui. Os termos técnicos são explicados na primeira vez em que aparecem e, de novo, no [glossário](#glossario).

## Índice

- [Princípio geral: como as decisões foram tomadas](#principio)

| # | Decisão | Onde | Status |
|---:|---|---|---|
| | **Seleção (seção 1)** | | |
| 1 | [Problema e tarefa: P1 agrupamento, P2 associação, sem classe](#d1) | seção 0 | implementada (v5) |
| 2 | [Dados-alvo: avaliações e filmes; tags fora; códigos externos só como chave](#d2) | 1.3, 1.9, 2.2 | implementada (v5; `tmdbId` na fusão desde a v7) |
| 3 | [Cinco grupos de nota](#d3) | 1.5, 1.7 | implementada (v5) |
| 4 | [Nota positiva: 4,0 ou mais](#d4) | 1.5, 3.1 | implementada (v5) |
| 5 | [O tempo fica fora do perfil](#d5) | 1.6-b | implementada (v5) |
| 6 | [Atributos em [0, 1] antes das matrizes de covariância](#d6) | 1.8 | implementada (v5) |
| 7 | [Testes lidos pelo tamanho de efeito](#d7) | 1.7-c, 1.8-d | implementada (v5) |
| | **Pré-processamento (seção 2)** | | |
| 8 | [Limpeza: nenhuma avaliação removida por qualidade; notas baixas mantidas](#d8) | 2.1, 2.2, 2.6 | implementada (v5) |
| 9 | [Títulos repetidos: fusão pela regra automática](#d9) | D3, D6; 2.2-a | implementada (v7) |
| 10 | [Gênero que falta: IMDb como fonte, kNN só como validação](#d10) | D2, 2-B; 2.2-b, 2.2-c | implementada (v6) |
| 11 | [Conversão dos gêneros do IMDb](#d11) | 2-B; 2.2-b | implementada (v6) |
| 12 | [Conferência da chave MovieLens → IMDb por código](#d12) | 2-C; 2.2-b | implementada (19/11/2026, texto da 2.2-b) |
| 13 | [Ano que falta: IMDb](#d13) | D4; 2.2-b | implementada (v6); um pedido sem registro |
| 14 | [Excluir os filmes que continuam sem gênero](#d14) | D6; 2.2-d | implementada (v7) |
| 15 | [Amostras só para gráficos, com representatividade conferida](#d15) | 1.5-b, 2.4 | implementada (v5; sorteio pelo hash na v6) |
| 16 | [Não imputar a matriz usuário × filme](#d16) | 2.5 | implementada (v5) |
| | **Transformação (seção 3)** | | |
| 17 | [Perfil por usuário em proporções](#d17) | 3.1 | implementada (v5) |
| 18 | [Seleção de características por filtro, com limiar de 0,9](#d18) | 3.2 | implementada (v5) |
| 19 | [Escala: z-score como base, min-max como alternativa](#d19) | 3.3 | implementada (v5) |
| 20 | [PCA do vetor base com 6 componentes](#d20) | 3.4 a 3.6 | implementada (v5); hoje é comparação |
| 21 | [Atípicos mantidos e marcados](#d21) | 3.7-a | implementada (v5) |
| 22 | [Seguir para o agrupamento, esperando separação fraca](#d22) | 3.7-b | implementada (v5) |
| 23 | [Vetor revisado e seu PCA como entrada principal de P1](#d23) | 3.8, 3.10 | implementada (v5) |
| 24 | [Gêneros que se sobrepõem e gêneros raros no vetor](#d24) | D4, D5 | proposta (pendente) |
| 25 | [Solução indexada para buscas por similaridade](#d25) | 3.9 | implementada (v5) |
| | **Infraestrutura** | | |
| 26 | [Repositório: fontes em `fonte/`, testes locais e execução final no Colab](#d26) | D1; README | implementada (v6; ampliada na v7) |
| 27 | [Reprodutibilidade: o mesmo resultado em qualquer máquina](#d27) | README | implementada (v6) |
| 28 | [Recorte fixo do IMDb](#d28) | README | implementada (v6) |
| 29 | [Medir fora do notebook antes de mudar o notebook](#d29) | `analises/` | em uso desde 18/11/2026 |

- [Glossário](#glossario)

---

<a id="principio"></a>

## Princípio geral: como as decisões foram tomadas

### Decidir junto

Nenhuma decisão foi tomada pronta. Cada mudança segue o mesmo caminho:

1. a ideia é proposta (pelo aluno, pela dupla ou pelo Claude) com os números reais da base;
2. o aluno filtra o que vai adiante;
3. uma **mesa de avaliação simulada** vota a ideia. A mesa é um agente de IA independente, que não participou das propostas e examina a ideia de vários pontos de vista. Até 18/11/2026 ela teve cinco membros (professor de Mineração de Dados, estatístico, engenheiro de dados, especialista em sistemas de recomendação e colega de turma). A partir da proposta 2-B, passou a ter sete: professor de Mineração de Dados, professor de IA e Aprendizado de Máquina, professor de Banco de Dados, cientista de dados, engenheiro de dados, estatístico e colega de turma;
4. os ajustes pedidos pela mesa voltam ao aluno, e só então a ideia é implementada.

Os critérios de cada decisão são fixados **antes** de ver o resultado. Quando isso não aconteceu, fica declarado: aplicar o critério de 10 pontos por faixa de avaliadores (decisão 10) e o critério de "não pesa" da D6 (decisão 14) foram definidos depois de ver os números.

### Regra que se confere à mão num caso, mas é executada por código em todos

O aluno quer métodos que ele conseguiria aplicar à mão num caso, mas que o código executa sobre todos os casos. Não há conferência filme a filme, nem tabela para preencher à mão.

- *Gênero que falta (decisão 10):* consultar o gênero de um filme no IMDb é algo que dá para fazer à mão. O notebook faz isso por código para todos os filmes sem gênero.
- *Conferência da chave (decisão 12):* a chave é o código que liga um filme do MovieLens ao mesmo título no IMDb. A ideia inicial era abrir no IMDb, um a um, os 38 filmes sem gênero mais avaliados. Ela foi trocada por uma regra (título, ano e tipo do título) executada em todos os filmes.
- *Títulos repetidos (decisão 9):* a conferência à mão dos 8 pares candidatos foi trocada por uma regra automática (mesmo título, ano e código TMDb, e pelo menos um gênero em comum). A pasta da conferência manual e o script que gerava a tabela foram retirados do repositório.
- *Mostrar a regra num caso:* o notebook mostra um exemplo calculado. Na 2.2-c, o filme St. Giuseppe Moscati: Doctor to the Poor (2007) tem 9 avaliadores e nenhum gênero no MovieLens. Entre os 15 filmes com o público mais parecido, 13 têm Drama, e o kNN prevê Drama, que é o gênero do IMDb. (O kNN, *k nearest neighbors*, prevê o rótulo de um objeto pelos rótulos dos k objetos mais parecidos com ele.)

### Excluir, com justificativa, o que não tem tratamento automático e não pesa

Um caso que nenhuma regra automática consegue tratar, e que não pesa de forma expressiva, sai dos **dados-alvo** (os dados escolhidos para a mineração), com a justificativa registrada. "Não pesa" quer dizer, em números:

- as avaliações excluídas somam no máximo 0,1% do total;
- nenhum usuário perde mais de 10% das suas avaliações;
- nenhum usuário perde todas as avaliações.

No caso que originou o critério (os 240 filmes que continuam sem gênero, decisão 14), os números são 0,0056% das avaliações, no máximo 5,0% das avaliações de um usuário e nenhum usuário perdendo todas. O critério foi escrito depois de ver esse caso. Isso fica declarado, e o critério vale como regra para os próximos casos.

### Duas práticas que acompanham o princípio

- *O que é preenchido fica identificável.* As colunas originais ficam intactas (`genres_movielens`, `ano_titulo`), e as colunas `origem_genero` e `origem_ano` dizem de onde veio cada valor. Assim, qualquer análise pode ser repetida sem os valores preenchidos (decisão 10).
- *Marcar em vez de apagar.* Os usuários atípicos são marcados, não removidos, e P1 pode ser executado com e sem eles (decisão 21). Os 644 filmes cujo ano difere em mais de 1 ano entre as fontes ficam como indicador, sem correção automática (decisão 13).

---

## Seleção (seção 1)

<a id="d1"></a>

### 1. Problema e tarefa: P1 agrupamento, P2 associação, sem classe

*Notebook: seção 0.*

**O que foi decidido.** P1 é uma tarefa de **agrupamento** (aprendizado não supervisionado: o algoritmo forma grupos sem receber respostas certas), em que cada usuário é uma instância. P2 é uma tarefa de **regras de associação**: cada usuário é uma transação, e os itens são os filmes que ele avaliou positivamente. P2 é construída a partir do resultado de P1.

**Por quê.**
- O MovieLens não traz atributo-alvo (classe ou meta): não há um valor certo a prever.
- A população é de 162.541 usuários anônimos, com avaliações de 1995 a 2019 e sem dados demográficos. O que existe é o comportamento de avaliação, e é ele que os perfis descrevem.

**Alternativas descartadas.**
- *Classificação:* exige uma classe, que a base não tem.
- *Seleção de características que depende de classe (CFS completo, Fisher, ReliefF):* não se aplica antes de P1. Depois de P1, os grupos podem servir de classe, mas só para descrever o que separa os grupos (decisão 18).

**Limitações.** Os perfis descrevem comportamento observado de avaliação, não características pessoais. Todos os usuários têm pelo menos 20 avaliações nos dados originais; depois da limpeza da v7, um fica com 19 (decisão 14).

**Onde está a evidência.** Seção 0 (`fonte/00_roteiro_e_problema.py`).

**Status.** Implementada desde a v5; o texto da população foi atualizado na v7.

<a id="d2"></a>

### 2. Dados-alvo: avaliações e filmes; tags fora; códigos externos só como chave

*Notebook: 1.3, 1.4, 1.9 e tabela da 2.2.*

**O que foi decidido.**
- Os dados-alvo são as tabelas `ratings` (avaliações) e `movies` (filmes).
- As tags ficam fora nesta fase: não entram em P1 nem em P2.
- Da tabela `links`, o `imdbId` serve de chave para buscar no IMDb o gênero e o ano que faltam (decisões 10 e 13), e o `tmdbId` só entra na regra dos títulos repetidos (decisão 9). Nenhum dos dois entra como atributo.
- Filmes sem nenhuma avaliação ficam fora de P1 e P2.
- A exploração da seção 1 é feita sobre as tabelas como foram carregadas, antes de qualquer tratamento.

**Por quê.**
- As tags são texto livre, com alta cardinalidade (muitos valores diferentes). Descrevem poucos usuários e não respondem a P1 nem a P2 (1.9).
- `imdbId` e `tmdbId` são identificadores: não fazem sentido em médias, distâncias ou correlações. Os 107 links sem `tmdbId` não pedem tratamento.
- 3.376 filmes do catálogo não têm nenhuma avaliação: não há comportamento a descrever.
- A exploração feita antes do tratamento é a que orienta a escolha do pré-processamento (1.4 e a tabela de fechamento da 1.9).

**Alternativas descartadas.** *Usar as tags no perfil:* adiado. Elas podem entrar no fim do trabalho, se uma pergunta justificar (1.3).

**Limitações.** A informação das tags fica sem uso nesta fase. Um filme sem `tmdbId` fica fora da regra de fusão (2.2).

**Onde está a evidência.** 1.3 (tabela de tipos e domínios), 1.9 (tabela de fechamento da seleção), 2.2 (tabela de ausentes).

**Status.** Implementada desde a v5. Desde a v6, o `imdbId` é a chave para o IMDb; desde a v7, o `tmdbId` entra na regra de fusão.

<a id="d3"></a>

### 3. Cinco grupos de nota

*Notebook: roteiro (seção 0), 1.5-c a 1.5-f, 1.6-b, 1.7.*

**O que foi decidido.** Os histogramas da nota usam cinco grupos, cada um com duas notas vizinhas: 0,5 e 1,0 → grupo 1; 1,5 e 2,0 → 2; 2,5 e 3,0 → 3; 3,5 e 4,0 → 4; 4,5 e 5,0 → 5. É o histograma comum a todas as consultas (por gênero e com filtro), sempre com média, desvio-padrão e o denominador de cada proporção.

**Por quê.**
- A nota tem só dez valores possíveis e muitos empates; faixas com a mesma quantidade de avaliações não são possíveis (ver equi-depth abaixo).
- Até 2002 não havia notas de meio ponto (0% até 2002, 29,15% em 2003, 39% a 44% depois). Com os cinco grupos, cada grupo das avaliações anteriores a 2003 contém só a nota inteira correspondente (1.6-b).
- O resultado é legível: o grupo 4 concentra 39,27% das avaliações, e o grupo 1, 4,68%.

**Alternativas** (apresentadas em 1.5 como comparação, não como histograma principal).
- *Equi-width (três faixas de mesma largura):* põe 62,52% das avaliações na faixa alta [3,5; 5,0].
- *Equi-depth (faixas com quantidades parecidas):* não consegue faixas iguais (17,89%, 32,30% e 49,81%), pelos empates.
- *V-Optimal (faixas que minimizam a soma dos quadrados dentro de cada faixa):* 0,5–2,0, 2,5–3,5 e 4,0–5,0 (12,84%, 37,35% e 49,81%). A última fronteira coincide com o limiar de nota positiva (decisão 4).

**Limitações.** A nota é ordinal (os valores têm ordem, mas a distância entre eles não é garantida). Média e desvio-padrão a tratam como intervalar, prática usual que fica registrada (1.5-a).

**Onde está a evidência.** Roteiro na seção 0; 1.5-c a 1.5-f; 1.6-b; 1.7.

**Status.** Implementada desde a v5.

<a id="d4"></a>

### 4. Nota positiva: 4,0 ou mais

*Notebook: 1.5-a, 1.5-f, 3.1, 3.2.*

**O que foi decidido.** Uma avaliação é positiva quando a nota é 4,0 ou mais. O limiar define `prop_positivas` no perfil (3.1) e os itens de P2 (filmes avaliados positivamente).

**Por quê.** As notas de 4,0 para cima somam 49,81% das avaliações, e a última fronteira do V-Optimal cai exatamente em 4,0 (1.5-f).

**Alternativas descartadas.** Nenhuma registrada.

**Limitações.** A proporção de notas positivas de um usuário deve ser lida contra essa base de quase 50%. O código da 3.2 registra que o limiar ainda precisa ser validado; por isso, e pela redundância com a nota média, `prop_positivas` sai da distância (decisão 18).

**Onde está a evidência.** 1.5-a, 1.5-f; constante `LIMIAR_POSITIVA` em 3.1.

**Status.** Implementada desde a v5. A validação do limiar está pendente.

<a id="d5"></a>

### 5. O tempo fica fora do perfil

*Notebook: 1.6-b.*

**O que foi decidido.** O momento da avaliação (`timestamp`) não entra no perfil de P1. A mudança de escala de 2003 (início das notas de meio ponto) fica registrada para a interpretação. Observação: o "ano médio dos filmes vistos" do vetor revisado (decisão 23) é o ano de lançamento dos filmes, não o momento da avaliação.

**Por quê.**
- O ano da avaliação explica só 0,26% da variação das notas (η = 0,051; η² = 0,0026). A razão de correlação η mede quanto uma variável categórica (aqui, o ano) explica da variação de uma numérica (a nota).
- A média anual varia pouco: de 3,43 (2004) a 3,65 (2013), sem contar 1995, que tem só 3 avaliações.
- A mediana do período de atividade (dias entre a primeira e a última avaliação do usuário) é 0 dia, com Q3 = 38,3 dias: mais da metade dos usuários fez todas as avaliações no mesmo dia.

**Alternativas descartadas.** *Usar o tempo como atributo do perfil:* pelos números acima, ele quase não se associa à nota, e a maioria dos usuários não tem histórico ao longo do tempo.

**Limitações.** Para a maioria dos usuários, o perfil reflete uma única sessão de avaliação, não preferências acompanhadas ao longo do tempo.

**Onde está a evidência.** 1.6-b.

**Status.** Implementada desde a v5.

<a id="d6"></a>

### 6. Atributos em [0, 1] antes das matrizes de covariância

*Notebook: 1.8-a, 1.8-b.*

**O que foi decidido.** Antes de cada matriz da exploração multivariada, todos os atributos vão para o intervalo [0, 1]: contagens divididas pelo máximo, nota pela posição na escala de 0,5 a 5,0, desvio-padrão pelo maior valor observado e ano entre o primeiro e o último ano. A covariância na escala original aparece só como consulta.

**Por quê.**
- A covariância mede quanto dois atributos variam juntos, na unidade de cada um. Na escala original, as variâncias por usuário iam de 1,1×10⁻¹⁰ a 0,23; por filme, a variância do ano (599) escondia todas as outras.
- Em [0, 1], a covariância fica comparável entre atributos. Pearson (correlação linear) e Spearman (correlação entre as ordens dos valores) não mudam com essa transformação.

**Alternativas descartadas.** *Covariância só na escala original:* não permite comparar atributos.

**Limitações.** A atividade fica quase toda perto de zero (média 0,0048, mediana 0,0022), porque o usuário mais ativo tem 32.202 avaliações.

**Onde está a evidência.** 1.8-a, 1.8-b, tabela da 1.9.

**Status.** Implementada desde a v5.

<a id="d7"></a>

### 7. Testes lidos pelo tamanho de efeito

*Notebook: 1.7-c, 1.8-d.*

**O que foi decidido.** Os testes estatísticos são lidos pelo **tamanho de efeito** (o tamanho da diferença ou da associação), não só pelo valor-p. As medidas usadas são r, η², V de Cramér (força da associação entre duas variáveis categóricas, de 0 a 1) e d de Cohen (diferença de médias dividida pelo desvio-padrão combinado).

**Por quê.** Com milhões de avaliações, qualquer diferença é estatisticamente significativa.
- Com 162.541 usuários, até r = −0,17 é significativo.
- A ANOVA (F = 104,9) e o Kruskal-Wallis (H = 923,4) rejeitam a igualdade das médias por década com p ≈ 10⁻¹⁹⁰, mas o efeito é η² = 0,067.
- Os 20 testes qui-quadrado gênero × grupo de nota rejeitam a independência, mas o maior V de Cramér é 0,119 (Drama).
- Nas consultas com filtro, o maior efeito é o da popularidade (d = 0,73 entre filmes populares e de nicho); os outros filtros têm efeito pequeno.

No qui-quadrado, cada avaliação conta uma vez por teste (avaliações do gênero contra as demais), para respeitar a independência mesmo com filmes de vários gêneros.

**Alternativas descartadas.** *Decidir só pelo valor-p:* tudo pareceria relevante.

**Limitações.** As associações são descritivas, não causais. Popularidade e nota se influenciam mutuamente: filmes bons atraem avaliações (1.7-c).

**Onde está a evidência.** 1.7-c, 1.8-d.

**Status.** Implementada desde a v5.

---

## Pré-processamento (seção 2)

<a id="d8"></a>

### 8. Limpeza: nenhuma avaliação removida por qualidade; notas baixas mantidas

*Notebook: 1.5-b, 2.1, 2.2, 2.3-a, 2.6.*

**O que foi decidido.** Nenhuma avaliação é removida por problema de qualidade. As notas 0,5 e 1,0 ficam, mesmo aparecendo como pontos fora do boxplot. A regra de deduplicação (fica a avaliação mais recente pelo `timestamp`) continua no código, para reprodutibilidade.

**Por quê.**
- A auditoria da 2.1 encontrou 0 nulos, 0 duplicatas e 0 notas inválidas em `ratings`.
- As notas 0,5 e 1,0 (4,68% das avaliações) ficam abaixo do limite inferior do boxplot (1,5), mas pertencem à escala válida: são atípicos legítimos, não erros.
- Na v7, saem 1.577 das 25.000.095 avaliações (0,0063% dos dados-alvo), por outros motivos: 171 repetidas na fusão dos títulos (decisão 9) e 1.406 de filmes sem gênero (decisão 14). A proporção de cada nota muda no máximo 0,0006 ponto percentual (nota 4,0), e os histogramas e resumos da seção 1 continuam válidos (2.6).

**Alternativas descartadas.** *Remover as notas fora do boxplot:* são valores válidos da escala.

**Limitações.** Nenhuma declarada.

**Onde está a evidência.** 1.5-b, 2.1, 2.2, 2.3-a (avaliações removidas por motivo), 2.6.

**Status.** Implementada desde a v5. Desde a v7, a 2.3-a conta as avaliações removidas por motivo.

<a id="d9"></a>

### 9. Títulos repetidos: fusão pela regra automática

*Registro: D3; D6 (proposta 3), parte C. Notebook: 2.2-a.*

**O que foi decidido.** Duas cópias de um título são o mesmo filme quando têm o mesmo título (com o ano), o mesmo código TMDb e pelo menos um gênero do MovieLens em comum. Nesse caso, viram uma só:
- fica a cópia mais avaliada, com os gêneros dela;
- as avaliações da outra cópia passam para ela, e quem avaliou as duas fica com a avaliação mais recente;
- se o código IMDb da cópia que fica não existe mais no IMDb, ela herda o código da cópia fundida.

Sem gênero em comum, as cópias ficam separadas, mesmo com o código TMDb igual. A regra funde 7 pares e deixa Blackout separado.

**Como chegou aqui.**
1. *D3, 18/11/2026, mesa de 5 (4 aprovam com ajuste, 1 aprova).* A medição encontrou 98 títulos idênticos (nome + ano), somando 196 filmes; nenhum par com o mesmo código IMDb; 8 pares com o mesmo código TMDb. A mesa aprovou fundir só depois de uma conferência à mão dos 8 pares, numa tabela em `conferencia_manual/`.
2. *A conferência à mão contrariava o princípio do aluno* (nada de conferir filme a filme) e ficou pendente, à espera de uma rodada própria.
3. *D6, parte C, 19/11/2026, mesa de 7 (7 aprovam com ajuste).* A proposta era fundir pelo IMDb (código ausente, ou duração com até 5 minutos de diferença e gênero em comum). A mesa preferiu fundir pelo critério já escrito na D3, sem o limite de 5 minutos; o que o IMDb mostra entra como evidência. A D3 está registrada como substituída pela D6.
4. *v7, 19/11/2026:* implementada na célula 2.2-a, antes da integração com o IMDb. A pasta `conferencia_manual/` e o script que gerava a tabela foram retirados.

**Por quê.**
- *TMDb igual é o candidato, e o IMDb confirma:* em 6 dos 7 pares fundidos, o código IMDb da cópia mais avaliada não existe mais no IMDb. O IMDb apaga um dos códigos quando descobre um cadastro em dobro, então um código que sumiu é mais uma evidência de duplicata. Em Enron, as duas entradas do IMDb têm o mesmo ano e 109 × 110 minutos.
- *O gênero em comum separa o caso diferente:* Blackout são dois filmes (no IMDb, 2008, 85 min, Horror × 2007, 95 min, Drama), sem gênero em comum no MovieLens.
- *Só título + ano não basta:* em 90 dos 98 títulos repetidos, as cópias têm códigos TMDb diferentes ou ausentes; são filmes diferentes com o mesmo nome (por exemplo, War of the Worlds (2005), de Spielberg, e a produção homônima). Além disso, 2.747 nomes aparecem com anos diferentes (refilmagens).
- *Gêneros da cópia mais avaliada:* é uma regra sem julgamento caso a caso. A outra cópia tinha gêneros parecidos: em Lagaan, fica Comedy|Drama|Musical|Romance e sai Adventure|Drama|Romance.

**Alternativas descartadas.**
- *Conferir os 8 pares à mão (D3):* contraria o princípio de regra executada por código.
- *Fundir pelo IMDb com o limite de 5 minutos (proposta da D6):* a mesa preferiu a regra já registrada na D3, com o IMDb só como evidência.
- *Dar ao filme fundido a interseção dos gêneros das duas cópias (pedido da D3):* apagaria gêneros corretos, como o Musical de Lagaan, que o IMDb também marca.
- *Excluir a cópia menos avaliada:* sairiam 8 filmes e 363 avaliações, com 332 usuários afetados (no máximo 2,6% das avaliações de um usuário). A mesa preferiu a fusão; pelo princípio do aluno, a exclusão fica para o que não tem regra automática, e aqui existe uma.
- *Fundir por título + ano:* juntaria filmes diferentes (os 90 homônimos).

**Limitações.** A regra depende do código TMDb e dos gêneros. Duas cópias do mesmo filme sem gênero em comum ficariam separadas; em Blackout, o único caso assim, o IMDb confirma que são filmes diferentes. As tags estão fora dos dados-alvo; a tabela de correspondência `mapa_fusao` (movieId antigo → novo) fica pronta para elas.

**Onde está a evidência.** 2.2-a; `analises/casos_sem_tratamento.py` → `analises/resultados_casos_sem_tratamento.txt` (parte C, com os 8 pares e os dados do IMDb lado a lado); `docs/decisoes.md` (D3 e D6).

**Status.** Implementada na v7 (19/11/2026).

**Números finais (v7).** Dos 98 títulos repetidos, 8 têm o mesmo código TMDb. 7 pares foram fundidos: Berlin Calling, Clockstoppers, Confessions of a Dangerous Mind, Enron, Lagaan, Seven Years Bad Luck e The Unauthorized Bash Brothers Experience. Blackout ficou separado. Saíram 171 avaliações repetidas (ficou a mais recente), e as avaliações passaram de 25.000.095 para 24.999.924. Em 6 das 7 cópias que ficaram, o código IMDb não existia mais e foi herdado da cópia fundida. Os números de controle bateram com os esperados, registrados antes da execução.

<a id="d10"></a>

### 10. Gênero que falta: IMDb como fonte, kNN só como validação

*Registro: D2 (com a rodada 2) e proposta 2-B. Notebook: 2.2-b, 2.2-c, 2.6.*

**O que foi decidido (vale hoje).** O gênero dos filmes sem gênero no MovieLens vem do IMDb: do arquivo público `title.basics`, ligado pelo código IMDb da tabela `links`. O kNN não preenche nada. Ele prevê o gênero de um filme a partir de **quem o avaliou** e serve para validar, com o comportamento dos usuários, a informação que veio de fora. O que o IMDb não resolve é excluído (decisão 14).

**Como chegou aqui.**
1. *v5 (18/11/2026):* o gênero era imputado (estimado) pelo kNN, com k = 15, a partir de quem avaliou o filme. Só filmes com pelo menos 10 avaliações podiam receber gênero, e só se o kNN superasse o chute (prever sempre Drama, o gênero mais comum) por pelo menos 10 pontos percentuais. O kNN acertava 83,4%. Isso alcançava os 320 filmes sem gênero com 10 ou mais avaliações, que somam 63% das avaliações sem gênero (1.6-c); os demais continuavam `Unknown`.
2. *D2 (18/11/2026), mesa de 5:* proposta de preencher com uma regra de contagem no lugar do kNN. A regra soma as avaliações que o público do filme deu a cada gênero e divide pela raiz do número de filmes do gênero (peso 1/√N). Votos: 3 aprovam com ajuste, 2 rejeitam (estatístico e especialista em recomendação, porque a regra provavelmente acertaria menos que o kNN). Na primeira medição por faixa de avaliadores, a regra não passou 10 pontos acima do chute em nenhuma faixa, e o kNN passou a partir de 5 avaliadores.
3. *D2, rodada 2 (18/11/2026), 5 de 5 aprovam com ajuste:* o kNN continua. O mínimo de avaliadores deixa de ser fixo (10) e passa a ser calculado pelo código: é a menor faixa a partir da qual todas as faixas superiores superam o chute por pelo menos 10 pontos. Na medição exploratória, o mínimo calculado foi 5. Mesmo assim, cerca de 4 mil filmes continuariam `Unknown`.
4. *Proposta 2-B (18/11/2026), mesa de 7 (6 aprovam com ajuste, 1 aprova):* o aluno perguntou se esses filmes não poderiam ser preenchidos. Com 1–2 avaliadores, nenhum método baseado no público supera o chute: é falta de informação. A proposta foi usar o IMDb como fonte principal e o kNN como método de mineração, validado contra o IMDb.
5. *v6 (19/11/2026):* implementado na 2.2-b (IMDb) e na 2.2-c (kNN validado). Como o IMDb preenche 4.822 dos 5.062 filmes sem gênero e só 55 dos 240 restantes têm 5 ou mais avaliações, o kNN não foi usado para preencher.
6. *v7 (19/11/2026):* os 240 restantes são excluídos na 2.2-d, sem kNN (decisão 14). A D2 está registrada como superada.

**Por quê.**
- *Cobertura:* o IMDb preenche 4.822 dos 5.062 filmes sem gênero (95%), com 25.221 das 26.627 avaliações desses filmes. O kNN só funciona com público suficiente.
- *As fontes concordam:* nos 57.099 filmes com gênero nas duas (depois da fusão da 2.2-a), 97,7% têm pelo menos um gênero em comum; os conjuntos são idênticos em 46,9%, e o Jaccard médio é 0,719. (O índice de Jaccard é o número de gêneros em comum dividido pelo número de gêneros distintos dos dois lados.)
- *Validação interna do kNN:* na validação cruzada de 5 partes (treina em 4/5 dos filmes, testa no 1/5 restante e repete cinco vezes), nos 24.006 filmes com gênero e 10 ou mais avaliadores, o kNN acerta 83,38%, contra 47,3% de chutar Drama. O Naive Bayes (classificador probabilístico simples) acerta 69,6%. Seu erro, de 30,4%, é a taxa de ruído de referência: o kNN só compensa se errar menos que isso, e erra 16,6%.
- *Por faixa:* o acerto do kNN cresce com o público do filme: 38,2% com 1–2 avaliadores, 55,4% com 3–4, 67,8% com 5–9, 78,0% com 10–49 e 87,8% com 50 ou mais. Com 1–2, fica 4,5 pontos abaixo do chute.
- *Validação externa:* nos próprios filmes sem gênero, o kNN concorda com o IMDb em 66,6% (5–9 avaliadores), 70,9% (10–49) e 80,6% (50 ou mais), contra 48,7% a 58,3% do chute. O comportamento dos usuários contém a informação de gênero e confirma o IMDb.
- *Efeito nas distribuições (2.6):* `Unknown` deixa de existir: 4.822 filmes (25.221 avaliações) recebem o gênero do IMDb e 240 (1.406 avaliações) são excluídos. A média de cada gênero muda no máximo 0,0015 e o desvio-padrão no máximo 0,0013 (os dois em Documentary, que ganha 706 filmes); o V de Cramér muda no máximo 0,0001.

**Alternativas descartadas.**
- *Dar Drama a todos:* acerta 47,3% nos filmes com 10 ou mais avaliadores.
- *kNN como preenchimento (v5):* só alcança filmes com público suficiente e deixava milhares de filmes sem gênero.
- *Contagem com peso 1/√N (D2):* fica sempre bem abaixo do kNN. Na validação interna da v7, acerta de 49% a 62% conforme a faixa; na faixa 5–9, 53,8% contra 67,8% do kNN.
- *Contagem com peso 1/N:* escolhe IMAX em 56% dos casos (é o menor gênero, com 195 filmes) e acerta de 2,8% a 10,9%, conforme a faixa (medição exploratória).
- *Perfil típico do gênero (centróide, com similaridade do cosseno):* ficou abaixo do chute.
- *Converter o gênero Music do IMDb em Musical:* ver decisão 11.

**Limitações.**
- A concordância mede o quanto as duas fontes rotulam do mesmo jeito, não qual delas está certa. O acerto externo do kNN é concordância com o IMDb.
- A vantagem do kNN sobre o chute é menor no teste externo (na faixa 5–9, 14,8 pontos, contra 24,1 na validação interna): os filmes sem gênero não são iguais aos que têm gênero, e o gabarito externo é outra fonte, que rotula de outro jeito.
- O critério interno aceita a faixa 3–4 (12,4 pontos acima do chute), e por isso o mínimo calculado no notebook é 3. No teste externo, porém, essa faixa fica só 3,3 pontos acima; a faixa 5–9 é a primeira que passa nos dois testes.
- O valor de 10 pontos vem da regra da v5, fixada antes da execução real; aplicá-lo por faixa foi decidido depois da medição exploratória.
- O IMDb lista no máximo três gêneros e usa menos Thriller. Os filmes preenchidos têm 1,78 gênero por filme (contra 1,87), Thriller em 7,4% (contra 15,1%), Drama em 52,5% (contra 44,6%) e Musical em 3,1% (contra 1,8%).
- Os filmes preenchidos têm notas menores que a média do gênero (Drama 3,43 contra 3,68; Comedy 3,21 contra 3,42), porque são pouco vistos. Pesam no máximo 0,87% das avaliações de um gênero (Documentary).
- Uma chave errada traz o gênero de outro título (decisão 12).
- O acerto de 83,38% da v7 difere do 83,35% da v6 porque a fusão muda as partes da validação cruzada.
- Os números de `analises/` e os do notebook diferem um pouco. O kNN exploratório usa a proporção de gêneros do público, e não a média dos perfis usada no notebook. Na regra 1/√N, o N foi medido no catálogo inteiro na primeira medição e só nos filmes de treino na segunda. Os números do notebook são os da versão implementada.

**Onde está a evidência.** 2.2-b, 2.2-c, 2.6; `analises/imputacao_por_faixa.py` → `analises/resultados_imputacao_por_faixa.txt`; `analises/imdb.py` → `analises/resultados_imdb.txt`; `docs/decisoes.md` (D2, rodada 2 e 2-B).

**Status.** IMDb como fonte e kNN como validação: implementados na v6 (19/11/2026). Exclusão dos 240 restantes: implementada na v7 (decisão 14).

<a id="d11"></a>

### 11. Conversão dos gêneros do IMDb

*Registro: ajuste da 2-B. Notebook: 2.2-b.*

**O que foi decidido.** Os 17 gêneros com o mesmo nome nas duas fontes passam direto; `Family` vira `Children`; os gêneros que não existem no MovieLens (Biography, History, Music, Sport, Short etc.) são ignorados.

**Por quê.** Cada linha da conversão foi justificada nos dois sentidos, nos filmes que têm gênero nas duas fontes (tabela inversa pedida pela mesa na 2-B).
- Concordância alta nos dois sentidos: Animation e Documentary (acima de 96%); Horror, Comedy e Drama (de 79% a 94%).
- `Children`: 63% dos filmes Children do MovieLens recebem `Family` no IMDb.

**Alternativas descartadas.** *Converter Music em Musical:* dos 1.879 filmes com Music no IMDb (e sem Musical), o MovieLens marca Musical em só 15% e Documentary em 33%.

**Limitações.**
- Concordância baixa em Musical (48% e 43%), Mystery (67% e 48%) e Film-Noir (73% e 45%): as duas fontes não usam esses rótulos do mesmo jeito.
- Os gêneros sem correspondente se perdem. 209 dos 240 filmes que continuam sem gênero só têm gêneros que não existem no MovieLens (145 só Short e 26 só Music), e por isso são excluídos (decisão 14).

**Onde está a evidência.** 2.2-b; `analises/resultados_imdb.txt` (parte 2).

**Status.** Implementada na v6.

<a id="d12"></a>

### 12. Conferência da chave MovieLens → IMDb por código

*Registro: D2, proposta 2-C. Notebook: leitura da 2.2-b. Análise: `analises/chave_imdb.py`.*

**O que foi decidido.** A chave (o código IMDb que a tabela `links` dá para cada filme) é conferida por código em todos os filmes, e não filme a filme. Nenhum filme é excluído por essa conferência: os 4.822 gêneros vindos do IMDb ficam, e a margem de erro é declarada.

**Como chegou aqui.**
1. A D2 previa conferir à mão, no IMDb, os 38 filmes sem gênero mais avaliados (50 ou mais avaliações) e usá-los como teste externo.
2. O aluno pediu que a conferência da chave não fosse feita filme a filme, e sim por uma regra executada por código sobre todos os filmes. A proposta 2-C foi aprovada pela mesa de 7 em 19/11/2026 (6 aprovam com ajuste, 1 aprova).
3. Num dos ajustes, a mesa pediu ainda uma conferência à mão reduzida (14 filmes). A rodada de consenso (5 aceitam, 2 aceitam com ajuste) trocou isso por uma lista gerada pelo código, que ninguém preenche.

**A regra.** Três sinais: título padronizado (igual, quase igual, contido, diferente), ano (igual, 1 ano de diferença, mais de 1 ano, sem ano) e tipo do título no IMDb (filme, curta, série, episódio, especial). Cada sinal é validado pela concordância de gêneros nos 57.100 filmes com gênero nas duas fontes e comparado com a concordância de uma chave sorteada. A medição foi feita antes da fusão da 2.2-a.

**Por quê.**
- *Uma chave errada aparece como queda de concordância:* com a chave verdadeira, 97,7% dos filmes têm pelo menos um gênero em comum; com a chave sorteada, 40,0%.
- *Nenhum sinal separa um grupo suspeito:* a concordância fica perto de 97% em todas as categorias de título, ano e tipo. Título diferente costuma ser tradução (Hachi-ko = The Story of Hachikō) e concorda 97,3%. A exceção é "sem ano no MovieLens" (147 filmes, 91,8%).
- *Margem de erro dos gêneros:* o controle comparável são os 29.412 filmes que entraram no catálogo na mesma época dos preenchidos (movieId acima de 131.262), com concordância de 97,01% e chave sorteada de 35,37%. Separando pelo sinal de ano e pesando pelas avaliações, cerca de 5,0% dos gêneros preenchidos (240 filmes) e 5,4% das avaliações deles (1.359; 0,0054% do total) podem vir de chave errada. Esses 240 não são os 240 filmes da decisão 14: aqueles nem receberam gênero.
- *Margem dos anos:* no controle sem ano no MovieLens, 135 de 147 concordam (intervalo de confiança de 95%, ou IC 95%, de 86,3% a 95,3%). Cerca de 56 dos 409 anos preenchidos podem estar errados (até 94).
- *Por usuário:* 7.793 usuários avaliaram algum filme preenchido. No mais exposto, esses filmes são 10,7% das avaliações, o que dá cerca de 0,58 ponto percentual com gênero possivelmente errado; 102 usuários passam de 5%.
- *Peso total:* mesmo que os 4.822 gêneros estivessem todos errados, seriam 0,10% das avaliações (voto do cientista de dados).

**A lista dos 14.** Dos 38 filmes sem gênero com 50 ou mais avaliações, o código separa os 14 que não passam em "título igual, ano igual ou com 1 ano de diferença, tipo filme". Para cada um, mostra título, ano e tipo do IMDb lado a lado e os outros filmes do MovieLens com o mesmo título. Nenhum gênero muda.
- 11 se resolvem pela própria linha: séries sem ano no MovieLens (Cosmos, The OA, Death Note, Cosmos: A Spacetime Odyssey, 13 Reasons Why), traduções e títulos alternativos (Fantastic Beasts 2 = The Crimes of Grindelwald; 120 BPM; Somnio = Infinity Chamber), Paterson (sem ano no MovieLens), um episódio e um especial com título igual.
- Whiplash (2013) é o curta; o longa de 2014 é outro filme do MovieLens (movieId 112552), com o mesmo gênero (Drama).
- O Sherlock Holmes sem subtítulo aponta para "Mortal Fight", uma das três partes do filme de 1980; as outras duas estão separadas no MovieLens, então a chave provavelmente está certa.
- Black Mirror é o controle positivo (um caso sabidamente errado, que mostra o que a regra não pega): o código aponta para um curta de 2011 com o mesmo nome, e não para a série, e nenhum sinal da regra detecta isso. O filme não recebeu gênero, porque o único gênero do curta no IMDb é Short, e sai na exclusão da 2.2-d (decisão 14).

**Alternativas descartadas** (motivos registrados na 2-C, salvo indicação).
- *Conferir os 38 filmes à mão:* contraria o princípio; substituída pela regra.
- *Conferência à mão reduzida a 14 filmes (ajuste da mesa):* trocada pela lista gerada por código.
- *Devolver a "sem gênero" 28 filmes com título diferente e ano divergente ou ausente (71 avaliações):* não foi escolhida pelo aluno. Pelos números do controle, o grupo parecido (título diferente e ano com mais de 1 ano de diferença) tem só 5% de chaves erradas estimadas.
- *Excluir curtas e episódios (539 filmes, 3.421 avaliações):* eles concordam 97,8% e 96,7%, como os demais.
- *Usar o kNN do público como segunda opinião:* com cerca de 5% de chaves erradas e cerca de 17% de erro do kNN, só 10% a 16% dos filmes marcados teriam chave errada; além disso, o kNN cobre só 301 filmes preenchidos (os com 10 ou mais avaliadores).

**Limitações.**
- A margem é uma estimativa sob a hipótese de que uma chave errada concorda como uma chave sorteada. Uma chave que aponta para um título parecido (episódio, homônimo) concorda mais, então a margem não é o pior caso. No controle comparável, 2,99% dos filmes não têm nenhum gênero em comum.
- O controle não representa bem os preenchidos: 90% deles entraram depois do ML-20M (movieId acima de 131.262), têm 20 vezes mais filmes sem ano (5,27% × 0,26%) e mais curtas (9,85% × 3,83%).
- Entre os filmes sem ano, título igual é justamente a marca de um homônimo (Black Mirror), e não um argumento a favor da chave. Nenhum sinal pega um homônimo sem ano.
- O sinal de título usa `originalTitle`, que o recorte do IMDb usado no notebook não tem; por isso a conferência roda fora do notebook.

**Onde está a evidência.** `analises/chave_imdb.py` → `analises/resultados_chave_imdb.txt` (com o MD5 do `title.basics` lido, `694fa9cd5bad4d6f3f94384b7ed61c74`); leitura da 2.2-b; `docs/decisoes.md` (2-C, com as fontes externas consultadas em 19/11/2026 para Black Mirror, Sherlock Holmes e Somnio).

**Status.** Aprovada e implementada em 19/11/2026, só como texto na leitura da 2.2-b, sem célula de código nova; na v7, continua assim. O professor de Mineração da mesa preferiria a tabela no notebook numa execução completa; para isso, o recorte do IMDb precisa ganhar `originalTitle`.

<a id="d13"></a>

### 13. Ano que falta: IMDb

*Registro: D4. Notebook: 2.2-b.*

**O que foi decidido.** O ano dos filmes sem ano no título vem do IMDb, pela mesma chave. Só se preenche o que falta; o ano do título nunca é trocado. A análise por década da seção 1 continua só com o ano do título; o ano preenchido entra no vetor revisado (3.8, ano médio dos filmes vistos).

**Por quê.**
- 412 filmes não têm ano no título (11.749 avaliações; 264 deles também sem gênero), entre eles Ready Player One e Moonlight.
- O IMDb tem o ano de 409 deles (11.747 avaliações).
- Nos 61.879 filmes com ano nas duas fontes (depois da fusão), o ano é igual em 92,8% e difere em até 1 ano em 99,0%.

**Alternativas descartadas.** *Corrigir automaticamente os 644 filmes com diferença maior que 1 ano:* nada é corrigido; eles ficam como indicador da qualidade da chave (ajuste da 2-B).

**Limitações.**
- Cerca de 56 dos 409 anos preenchidos podem vir de chave errada (até 94) (decisão 12).
- 3 filmes continuam sem ano (2 avaliações). Os 2 avaliados estão entre os 240 excluídos na decisão 14; o terceiro não tem avaliação.

**Outros pedidos da D4.**
- Escrever o denominador de cada participação de gênero: contagem simples na 1.6-c (um filme conta em cada um dos seus gêneros) e fracionária na 3.8 (peso 1/k). Os dois estão no texto do notebook.
- Testar a sensibilidade do corte de 2% da 3.8 com War (1,80% na contagem fracionária, o gênero mais perto do corte): não há registro de que o teste foi feito.
- Decidir o IMAX no vetor: ver decisão 24.

**Onde está a evidência.** 2.2-b; `analises/resultados_imdb.txt` (parte 3); `analises/resultados_chave_imdb.txt` (parte 9).

**Status.** Preenchimento do ano implementado na v6. Teste de sensibilidade com War: pedido, sem registro de execução.

<a id="d14"></a>

### 14. Excluir os filmes que continuam sem gênero

*Registro: D6 (proposta 3), partes A, B e D. Notebook: 2.2-d. Análise: `analises/casos_sem_tratamento.py`.*

**O que foi decidido.** Os 240 filmes que continuam sem gênero depois do IMDb saem dos dados-alvo, com 1.406 avaliações (0,0056% do total), sem kNN. A 2.2-c continua como validação. Com isso, o rótulo `Unknown` deixa de existir, e o perfil da 3.1 passa a ter 19 gêneros.

**Como chegou aqui.** Na v6, esses filmes ficavam como `Unknown`. A proposta 3 levou à mesa o princípio do aluno: excluir o que não tem tratamento automático e não pesa. A proposta dava gênero pelo kNN aos 55 filmes com 5 ou mais avaliadores e excluía os 178 com 1 a 4 e os 7 sem avaliação; a alternativa simples era excluir os 240, sem kNN. A mesa de 7 (19/11/2026, 7 aprovam com ajuste) preferiu a alternativa simples, e o aluno aceitou os ajustes no mesmo dia. A exclusão foi implementada na v7, na célula 2.2-d, depois da validação do kNN.

**Por quê.**
- *Não há tratamento automático:* no IMDb, 209 filmes só têm gêneros que não existem no MovieLens (145 só Short e 26 só Music), 16 não têm gênero e 15 têm código ausente.
- *O kNN não serve para esses filmes:* dos 55 com 5 ou mais avaliadores, 46 são curtas no IMDb (os outros: 6 com código ausente, 1 especial de TV, 1 longa e 1 vídeo). Nos 1.100 curtas com gênero no MovieLens e 5 ou mais avaliadores, o kNN acerta 51,0%, e chutar o gênero mais comum entre os curtas (Animation) acerta 52,7%; na faixa 5–9, 41,2% contra 53,1%. A validação da 2.2-c foi feita em filmes com gênero, quase todos longas, com Drama como chute, e não vale para essa população: só 14,8% dos curtas são Drama.
- *Já estava previsto:* o ajuste da 2-B dizia que, se os filmes restantes fossem poucos, não haveria etapa de reserva.
- *Não pesa:* por número de avaliações, são 7 filmes com 0, 131 com 1–2, 47 com 3–4, 36 com 5–9 e 19 com 10 ou mais. As avaliações excluídas são 0,0056% do total (limite: 0,1%); 823 usuários perdem alguma avaliação, no máximo 5,0% das suas (limite: 10%), e nenhum perde todas.
- *`Unknown` não é preferência:* registra falta de informação no catálogo. Na v6, `prop_unknown` era um dos atributos que geravam os valores z mais extremos e dominava parte dos usuários atípicos (3.7 da v6).
- *Os 7 sem avaliação* saem por necessidade do código (para `Unknown` deixar de existir, em vez de virar um atributo sempre igual a zero), não pelo peso.
- *Sem ano (parte B):* 3 filmes, 2 avaliações. Os 2 avaliados estão entre os 240; o terceiro não tem avaliação. Não pedem regra própria.

**Alternativas descartadas.**
- *Manter como `Unknown` (v6):* mantém um atributo que mede falta de informação e gera atípicos.
- *kNN para os 55 com 5 ou mais avaliadores, excluindo só os 178 com 1 a 4 e os 7 sem avaliação:* afetaria menos usuários (124, no máximo 2,6% das avaliações de um usuário, menor resto de 38), mas, no conjunto dos curtas com 5 ou mais avaliadores, o kNN não supera o chute.
- *kNN para todos:* com 1–2 avaliadores, o kNN fica abaixo do chute (2.2-c).

**Limitações.**
- O critério de "não pesa" foi fixado depois de ver estes números. Isso fica declarado, e o critério vale daqui em diante.
- A exclusão tem direção conhecida: saem sobretudo curtas, vídeos musicais e títulos sem cadastro no IMDb. O efeito no perfil é pequeno, porque esses filmes somam no máximo 5% das avaliações de um usuário.
- Na faixa de 10 a 49 avaliadores, o kNN supera o chute nos curtas (56,2% contra 48,4%); a decisão olhou o conjunto dos curtas com 5 ou mais avaliadores.
- Black Mirror (456 avaliações) sai junto, com a chave errada registrada (decisão 12).
- Um usuário fica com 19 avaliações; os outros continuam com pelo menos 20.

**Onde está a evidência.** 2.2-d; `analises/casos_sem_tratamento.py` → `analises/resultados_casos_sem_tratamento.txt` (partes A e B e, na parte D, o teste do kNN nos curtas); `docs/decisoes.md` (D6). Um grupo de 117 filmes parecidos com os 55, mas formado sobretudo por códigos ausentes do IMDb e longas, em que o kNN vai bem (62,4% contra 27,4%), está só no registro da D6; ele representa pouco os 55.

**Status.** Implementada na v7 (19/11/2026).

**Números finais (v7).** Saíram 240 filmes e 1.406 avaliações (0,0056% de 24.999.924, depois da fusão; limite 0,1%). 823 de 162.541 usuários foram afetados, com no máximo 5,0% das avaliações de um usuário (limite 10%), e nenhum perdeu todas: o critério de "não pesa" foi atendido. Depois da limpeza (fusão e exclusão), ficaram 24.998.518 avaliações, 62.176 filmes no catálogo, 58.807 filmes avaliados, 162.541 usuários e 19 gêneros. Todos os números de controle esperados, registrados antes da execução, bateram.

<a id="d15"></a>

### 15. Amostras só para gráficos, com representatividade conferida

*Notebook: 1.5-b, 1.8-a, 2.4.*

**O que foi decidido.** Todas as medidas usam a base completa. As amostras servem só aos gráficos que ficariam pesados com 25 milhões de pontos: boxplot e violin plot com cerca de 100 mil avaliações, e matriz de dispersão com 3.000 usuários. A amostra de notas é sorteada pelo hash de (`userId`, `movieId`, semente 42): um número calculado a partir do par usuário–filme, que dá a cada avaliação a mesma chance de entrar.

**Por quê.**
- A amostra precisa representar a população, e isso é conferido na 2.4. As 100.065 notas diferem da população em no máximo 0,18 ponto percentual (nota 4,5). A média é 3,5332, contra 3,5339; o desvio-padrão, 1,0614, contra 1,0607. Nos 3.000 usuários, `nota_media` é 3,6786, contra 3,6792.
- O sorteio pelo hash dá a mesma amostra em qualquer máquina (decisão 27).

**Alternativas descartadas.**
- *Truncar com `limit`:* pega as primeiras linhas, não uma amostra aleatória.
- *Sorteio interno do Spark (v5):* depende de como o Spark divide os dados, e por isso muda com o número de núcleos.

**Limitações.** Nenhuma declarada.

**Onde está a evidência.** 1.5-b, 2.4.

**Status.** Implementada desde a v5; sorteio pelo hash desde a v6.

<a id="d16"></a>

### 16. Não imputar a matriz usuário × filme

*Notebook: tabela da 2.2; 2.5.*

**O que foi decidido.** Os pares usuário–filme sem avaliação não são preenchidos. O perfil de cada usuário usa só os filmes que ele avaliou: é a alternativa de ignorar os valores ausentes.

**Por quê.**
- A matriz usuário × filme tem 9.558.548.587 células (162.541 usuários × 58.807 filmes avaliados), das quais só 24.998.518 (0,26%) têm nota. O usuário mediano avaliou 0,12% dos filmes; o mais ativo, 54,6%.
- Um par não avaliado não é um valor que falta: preencher criaria avaliações artificiais.
- O teste foi feito num recorte de 500 usuários sorteados e dos 300 filmes mais avaliados, em que 17,2% das células têm nota. Com a imputação, o desvio-padrão das notas cai de 1,02 para 0,45 (média da categoria) ou 0,52 (média do filme): a variação real cai à metade. A correlação média entre usuários sobe de 0,12 para 0,19 ou 0,54: os usuários passam a parecer semelhantes porque compartilham os valores preenchidos, não porque avaliaram igual.

**Alternativas descartadas.**
- *Preencher pela média da categoria (primeiro gênero do filme):* reduz a variação real à metade.
- *Preencher pela média do filme:* além disso, infla a correlação entre usuários.
- *Preencher pela média do perfil:* só faz sentido depois de P1, quando os segmentos existirem.

**Limitações.** O teste usa um recorte de 500 usuários e dos filmes mais populares, onde a matriz é muito menos vazia que no total.

**Onde está a evidência.** 2.5.

**Status.** Implementada desde a v5.

---

## Transformação (seção 3)

<a id="d17"></a>

### 17. Perfil por usuário em proporções

*Notebook: 1.6, 3.1, 3.1-b.*

**O que foi decidido.** Cada usuário vira uma linha (instância), com atributos em proporção e denominador explícito: proporção das avaliações em cada um dos 19 gêneros (denominador: avaliações do usuário), nota média, desvio-padrão da nota, proporção de notas positivas e participação no total de avaliações.

**Por quê.**
- O volume é muito assimétrico: o número de avaliações por usuário tem assimetria de 16,3 (média 153,8, mediana 71), e o mais ativo tem 32.202 avaliações. Contagens brutas mediriam atividade, não gosto; proporções comparam usuários com volumes diferentes (1.6).
- Em média, 44,3% das avaliações de um usuário são de Drama, 34,8% de Comedy e 31,4% de Action.

**Alternativas descartadas.** *Contagens brutas por gênero:* seriam dominadas pela atividade do usuário.

**Limitações.** As proporções de gênero de um usuário somam mais que 1, porque um filme pode ter vários gêneros. O vetor revisado corrige isso com a contagem fracionária (decisão 23).

**Onde está a evidência.** 1.6, 3.1, 3.1-b.

**Status.** Implementada desde a v5.

<a id="d18"></a>

### 18. Seleção de características por filtro, com limiar de 0,9

*Notebook: 3.2, 3.10.*

**O que foi decidido.** A seleção de características (escolher quais atributos entram na entrada de P1) usa a abordagem de **filtro**: mede a redundância entre pares de atributos, sem depender de um algoritmo. Um par é redundante quando |Pearson| e |Spearman| são ambos ≥ 0,9. Resultado: sai `prop_positivas` (redundante com `nota_media`), e `prop_avaliacoes_total` fica fora da distância. A entrada candidata tem 21 atributos (nota média, desvio-padrão e 19 proporções de gênero).

**Por quê.**
- O filtro independe do método de aprendizado. *Wrappers* e *embedded* (que escolhem atributos testando um algoritmo) acrescentariam um segundo viés.
- Sem classe, do CFS (que busca atributos ligados à classe e pouco ligados entre si) só a segunda parte se aplica.
- Só o par `nota_media` × `prop_positivas` passou do limiar (Pearson 0,922; Spearman 0,946). Fica o atributo que usa a escala completa da nota; `prop_positivas` também depende de um limiar ainda a validar (decisão 4).
- `prop_avaliacoes_total` mede atividade, é muito assimétrico e dominaria a distância; serve para descrever os grupos.
- O segundo par mais associado, `prop_animation` × `prop_children` (0,823 / 0,804), ficou abaixo do limiar, e os dois foram mantidos.

**Alternativas descartadas.**
- *Wrappers e embedded:* dependem do algoritmo.
- *Fisher e ReliefF:* exigem classe. Depois de P1, podem usar os grupos como classe para descrever o que os separa, não para escolher a entrada.

**Limitações.** O limiar de 0,9 é uma escolha explícita e pode ser revisto.

**Onde está a evidência.** 3.2, 3.10.

**Status.** Implementada desde a v5. Na v7, sem `prop_unknown`, a entrada candidata passou de 22 para 21 atributos.

<a id="d19"></a>

### 19. Escala: z-score como base, min-max como alternativa

*Notebook: 3.3, 3.4, 3.5, 3.10.*

**O que foi decidido.** A matriz padronizada por **z-score** (subtrair a média e dividir pelo desvio-padrão de cada atributo) é a base do PCA e a entrada candidata interpretável. A re-escala **min-max** (levar cada atributo para [0, 1]) fica como alternativa (`X_p1_minmax`).

**Por quê.**
- A escala muda o peso de cada atributo na distância entre usuários e no PCA.
- No z-score, cada atributo tem o mesmo peso na variância total (4,76%).
- No min-max, os gêneros frequentes dominam a distância: Action pesa 11,54% e Drama 10,12%, contra 0,23% de Documentary.
- Na amostra de 2.000 usuários, o min-max é o espaço que mais concentra as distâncias (variância relativa 0,3590, contra 0,3740 no z-score e 0,4259 no PCA). Distâncias concentradas (a menor e a maior ficam proporcionalmente próximas) prejudicam os métodos baseados em distância.

**Alternativas descartadas.** *Min-max como escala principal:* deixa os gêneros frequentes decidirem a distância e concentra mais as distâncias.

**Limitações.** O z-score amplia os gêneros raros: máximos de 47,1 em Documentary, 38,9 em Western e 38,2 em Film-Noir. O vetor revisado trata isso agrupando os gêneros raros (decisão 23).

**Onde está a evidência.** 3.3, 3.5, 3.10.

**Status.** Implementada desde a v5.

<a id="d20"></a>

### 20. PCA do vetor base com 6 componentes

*Notebook: 3.4, 3.5, 3.5-b, 3.6.*

**O que foi decidido.** O **PCA** (análise de componentes principais: troca os atributos por combinações deles, ordenadas pela variância que explicam) é aplicado à matriz padronizada. O número de componentes, d′ = 6, vem do critério de Kaiser-Guttman (ficar com os componentes de autovalor ≥ 1). A matriz PCA é uma entrada a comparar em P1, não uma substituição obrigatória dos 21 atributos. O critério de 80% da variância (9 componentes) fica como alternativa a testar em P1.

**Por quê.**
- Os critérios discordam: Kaiser-Guttman dá 6 componentes (66,84% da variância); o broken-stick (ficar com os componentes que superam a variância esperada ao acaso) dá 2; e 80% da variância acumulada dá 9. Cada um tem seu viés, e os três ficam registrados.
- A dimensionalidade intrínseca (quantas dimensões efetivas descrevem os dados, estimada pelo método de Levina–Bickel) é de cerca de 13 no espaço z de 21 atributos (12,7 a 14,1, conforme o número de vizinhos usado) e de 6,4 no espaço PCA de 6 componentes. Há redundância, mas a estrutura não cabe em 2 ou 3 dimensões.
- O espaço PCA é o que menos concentra as distâncias (contraste relativo 110,66; variância relativa 0,4259).
- Os dois primeiros componentes explicam 39,68%. As projeções 2D (PCA, MDS e t-SNE) mostram uma nuvem contínua, sem grupos isolados.

**Alternativas.**
- *Broken-stick (2 componentes):* muito abaixo da dimensionalidade intrínseca.
- *80% da variância (9 componentes):* mais perto da dimensionalidade intrínseca; fica como alternativa em P1.
- *Os 21 atributos sem PCA:* continuam como entrada candidata (`X_p1_padronizado`).

**Limitações.** O PCA com 6 componentes guarda 66,84% da variância e descarta parte da estrutura.

**Onde está a evidência.** 3.4, 3.5, 3.5-b, 3.6, 3.10.

**Status.** Implementada desde a v5. Até a v6, com `prop_unknown`, eram 7 componentes e 68,41%. Como entrada principal de P1, foi substituída pelo PCA do vetor revisado (decisão 23); o PCA do vetor base fica como comparação.

<a id="d21"></a>

### 21. Atípicos mantidos e marcados

*Notebook: 3.7-a, 3.8-b.*

**O que foi decidido.** Um usuário é marcado como atípico quando está, ao mesmo tempo, longe do centro e isolado dos vizinhos:
- *longe do centro:* a distância de Mahalanobis ao centro (uma distância que leva em conta a correlação entre os atributos), no espaço PCA, passa do quantil 0,999 da χ² com 6 graus de liberdade;
- *isolado:* a distância ao 10º vizinho mais próximo passa do percentil 99.

Nenhum usuário é removido: a marca `atipico` permite executar P1 com e sem esses usuários.

**Por quê.**
- O k-means e outros métodos baseados em médias são sensíveis a objetos muito afastados, que deslocam os centróides. Por isso é preciso identificá-los.
- Atípicos podem ser objetos legítimos: aqui, usuários com gosto concentrado em poucos gêneros. Em quase todos, o atributo mais extremo é um gênero raro: Documentary (222 usuários), Film-Noir (215), Western (183) e War (118). Eles avaliam menos filmes (mediana de 35 contra 71) e dão notas um pouco mais altas (mediana da média 3,88 contra 3,70).
- Exigir os dois critérios evita marcar nichos: 2.270 usuários estão longe do centro, mas têm vizinhos próximos.
- Resultado: 1.293 usuários (0,80%) marcados.

**Alternativas descartadas.**
- *|z| > 3 em algum atributo (critério univariado):* marca 31.263 usuários (19,23%). Com caudas tão longas, marca a cauda de cada atributo, não casos isolados.
- *Só Mahalanobis:* marca 3.563 usuários (2,19%), cerca de 22 vezes os 163 esperados sob normalidade, incluindo possíveis grupos de nicho.
- *Remover os atípicos:* perderia usuários legítimos; a marca permite comparar.

**Limitações.**
- Os limiares (quantil 0,999 e percentil 99) são escolhas registradas, e a referência χ² supõe normalidade, que os dados não seguem.
- No vetor revisado, os atípicos são outros: 1.078, dos quais só 379 também são atípicos no vetor base (Jaccard 0,19).

**Onde está a evidência.** 3.7-a, 3.8-b, 3.10.

**Status.** Implementada desde a v5. Na v6, com `Unknown`, eram 1.365 atípicos; sem ele, os valores extremos vêm dos gêneros raros de verdade, que o vetor revisado agrupa.

<a id="d22"></a>

### 22. Seguir para o agrupamento, esperando separação fraca

*Notebook: 3.6, 3.7-b.*

**O que foi decidido.** Vale agrupar, mas deve-se esperar separação fraca entre os grupos. A validação de P1 precisa comparar seus índices com os de partições de dados uniformes no mesmo domínio.

**Por quê.**
- A estatística de Hopkins compara distâncias de pontos uniformes e de usuários reais: H ≈ 0,5 indica dados indistinguíveis de pontos uniformes; H perto de 1 indica dados concentrados em regiões. O controle com dados uniformes dá H = 0,500 nos dois espaços, o que confirma o cálculo.
- No domínio robusto (percentis 1 a 99), H = 0,705 no PCA e 0,767 no z-score; sem os atípicos, 0,696 e 0,762. O desvio-padrão entre as 10 repetições é no máximo 0,0040. Os usuários são mais concentrados que pontos uniformes, e isso não é efeito dos atípicos.
- A tendência é moderada: uma referência usual é H > 0,75 para tendência clara, e o espaço PCA fica abaixo dela. As projeções 2D mostram um núcleo denso com caudas, não grupos separados.

**Alternativas descartadas.** *Domínio mín–máx:* dá H = 0,960 (PCA) e 0,953 (z-score), inflado pelos extremos, que alargam o domínio dos pontos uniformes. Sem os atípicos, cai para 0,930 e 0,932.

**Limitações.** H mede concentração, não separação: um único aglomerado denso também dá H alto. A separação entre grupos só pode ser avaliada em P1.

**Onde está a evidência.** 3.6, 3.7-b.

**Status.** Implementada desde a v5.

<a id="d23"></a>

### 23. Vetor revisado e seu PCA como entrada principal de P1

*Notebook: 3.8, 3.8-b, 3.10.*

**O que foi decidido.** A entrada principal de P1 é `X_rev_pca`: o PCA do vetor revisado, com 6 componentes (Kaiser-Guttman, 69,13% da variância). O vetor revisado:
- usa a **proporção fracionária**: cada avaliação distribui peso 1/k entre os k gêneros do filme, e o vetor de gêneros soma 1;
- junta no grupo `raros` os gêneros com menos de 2% das avaliações na contagem fracionária (War, Animation, Musical, Documentary, IMAX, Western e Film-Noir, que somam 7,85%); os rótulos continuam nos dados;
- acrescenta atributos derivados: viés do usuário (nota menos a média do filme), atividade em escala log, popularidade média e ano médio dos filmes vistos, e diversidade de gêneros (entropia normalizada);
- tem um gênero dominante, usado só para descrever.

O vetor base e as matrizes z-score ficam como comparação. Até a v6, o vetor revisado também tirava `Unknown`; na v7, `Unknown` já não existe (decisão 14).

**Por quê.**
- O vetor base tem três problemas: proporções que somam mais que 1 (contagem dupla de gêneros); gêneros raros que geram os valores extremos e a maior parte dos atípicos; e a atividade fora da distância.
- A popularidade dos filmes vistos é o filtro com maior efeito sobre a nota (d de Cohen 0,73 entre filmes populares e de nicho, 1.7-c).
- Os gêneros com pouca participação nas avaliações são informação real do catálogo, por isso continuam nos dados; no vetor, são agrupados para não distorcer as distâncias (1.6-c).
- Com o mesmo número de componentes, o vetor revisado guarda mais variância (69,13% contra 66,84% no vetor base) e gera 17% menos atípicos (1.078 contra 1.293). O broken-stick retém 3 componentes (contra 2), sinal de mais estrutura nos primeiros eixos. Os eixos são interpretáveis: CP3 é nota média e viés; CP4, atividade contra popularidade; CP6, época dos filmes. A estatística de Hopkins fica próxima (0,68 contra 0,70 no PCA).
- Redundância: o par mais associado (nota média × viés, Pearson 0,903, Spearman 0,881) só passa em um dos dois critérios; nenhum atributo sai, e o vetor tem 20 atributos.

**Alternativas.** *Vetor base (21 atributos ou PCA com 6 componentes):* fica como comparação, inclusive para medir o efeito da concentração de distâncias um pouco maior do vetor revisado.

**Limitações.**
- As distâncias ficam um pouco mais concentradas no vetor revisado: no PCA, o contraste relativo cai de 110,66 para 59,81, e a variância relativa, de 0,426 para 0,377.
- A dimensionalidade intrínseca é a mesma nos dois PCA (6,4).
- O limite de 2% é uma escolha registrada. Animation (1,66%) ficou em `raros`, mas o público infantil continua representado por Children (2,4%).
- Há pontos em aberto sobre os gêneros raros e sobrepostos (decisão 24).

**Onde está a evidência.** 3.8, 3.8-b, 3.10; 1.6-c e 1.7-c para a origem dos atributos.

**Status.** Implementada desde a v5. Desde a v6, o ano médio usa também o ano preenchido pelo IMDb.

<a id="d24"></a>

### 24. Gêneros que se sobrepõem e gêneros raros no vetor

*Registro: D4 (ajustes) e D5.*

**O que está proposto.** IMAX sai do vetor do perfil, por ser formato de exibição, não gênero. Animation e Children viram um grupo só no vetor. Os rótulos continuam nos dados.

**Por quê.**
- Dos 195 filmes IMAX, só 1 não tem outro gênero. Nas avaliações, 63% dos IMAX também são Action, 60% Adventure e 45% Sci-Fi.
- Animation × Children é o par mais sobreposto: Jaccard de 0,21 nos filmes; 78% das avaliações de Animation também são Children; correlação de 0,82 entre usuários (3.2).

**Alternativa em vigor.** O vetor revisado atual: IMAX e Animation no grupo `raros`, Children separado.

**Pontos em aberto.**
- O corte de 2% usa a contagem fracionária, em que Musical tem 1,14% e IMAX 1,05%. (A mesa tinha comparado o corte com a contagem simples, em que Musical tem 3,86% e IMAX 4,25%; o registro corrigiu isso.)
- A D4 pediu testar a sensibilidade do corte com War (1,80%); não há registro do teste.
- Falta decidir se Musical e os gêneros raros são agrupados só no vetor ou já no pré-processamento.

**Onde está a evidência.** `docs/decisoes.md` (D4 e D5); 3.2 e 3.8.

**Status.** Proposta; aguardando aprovação do aluno para ir à mesa.

<a id="d25"></a>

### 25. Solução indexada para buscas por similaridade

*Notebook: 3.9, 3.10.*

**O que foi decidido.** P1 trabalha no espaço PCA, onde os índices funcionam. As buscas exatas (DBSCAN e distância ao k-ésimo vizinho) usam kd-tree ou pivôs com seleção incremental; o HNSW atende buscas aproximadas.

Os métodos comparados:
- *varredura linear:* compara a consulta com todos os objetos; é a referência exata;
- *kd-tree e ball-tree:* índices exatos que dividem o espaço em regiões e descartam regiões distantes;
- *pivôs:* objetos de referência. Pela desigualdade triangular, |d(q,p) − d(o,p)| ≤ d(q,o); se esse limite inferior passa do raio, o objeto é descartado sem calcular a distância;
- *HNSW:* índice aproximado baseado em grafo, avaliado pelo recall@10 (fração dos 10 vizinhos verdadeiros que ele recupera).

**Por quê.**
- P1 depende de buscas por similaridade: o DBSCAN busca todos os vizinhos dentro de um raio, a escolha desse raio usa a distância ao k-ésimo vizinho, e o kNN busca os k mais próximos. Com 162.541 usuários, a varredura linear compara cada consulta com todos.
- Com 16 pivôs de seleção incremental, a busca por abrangência descarta 99,9% das distâncias nos dois espaços PCA, contra 92,6% (z revisado) e 89,2% (z base). Essa fração não depende da máquina.
- O HNSW chega a recall@10 de 0,98 a 1,00 já com efSearch = 16.

**Alternativas descartadas.**
- *Ball-tree:* não ganha da varredura linear no PCA.
- *Árvores nos espaços z (20 ou 21 dimensões):* muito mais lentas que a varredura linear. Em muitas dimensões, as partições deixam de descartar regiões.
- *Pivôs por variância máxima:* a seleção incremental ganha na maioria dos casos, sobretudo nos espaços z (com 2 pivôs no z revisado, cerca de 60% contra 46%).

**Limitações.** Os tempos dependem da máquina; a fração de distâncias evitadas não. Na v6 (Colab), a kd-tree foi o índice exato mais rápido no PCA revisado. Na execução da v7 no Mac, ela ganhou da varredura linear no PCA base, mas perdeu no PCA revisado; na execução da v7 no Colab, que é a entregue, ela foi o índice exato mais rápido nos dois espaços PCA. É um exemplo de por que os tempos dependem da máquina.

**Onde está a evidência.** 3.9, 3.10.

**Status.** Implementada desde a v5.

---

## Infraestrutura

<a id="d26"></a>

### 26. Repositório: fontes em `fonte/`, testes locais e execução final no Colab

*Registro: D1. README: "Regra de ouro", "Fluxo de trabalho" e "Execução final no Colab".*

**O que foi decidido.**
- Edita-se só em `fonte/*.py` (um arquivo por seção, no formato *percent* do jupytext). O `.ipynb` é sempre gerado por `scripts/montar_notebook.py` e nunca é editado à mão.
- A execução que gera as saídas entregues é feita no Colab, em segundo plano, com `jupyter nbconvert`; o Mac serve de pré-teste.
- Testes locais: teste de estrutura (o notebook monta, toda célula de código tem cabeçalho e o notebook termina na seção 3) e teste com dados sintéticos com os casos de borda (filmes sem gênero, com 9 e 10 avaliadores, sem ano, títulos repetidos, IMAX). Na v7, entraram testes para a fusão e a exclusão.
- Sementes fixas e números de controle comparados entre Mac e Colab; pico de memória medido na execução local; versões das bibliotecas iguais às do Colab (Python 3.13, Spark 4.0.5, pandas 2.2.3).
- Dados do MovieLens e do IMDb fora do repositório.

**Por quê.**
- Uma edição feita direto no `.ipynb` (inclusive no Colab) se perde na próxima montagem.
- Executando em segundo plano na máquina do Colab, as saídas não se perdem se a aba do navegador ficar em segundo plano.
- O pré-teste no Mac é rápido: o teste com dados sintéticos leva cerca de 2 minutos no M1 Pro (pico de 1,8 GB de memória) e a execução completa local, cerca de 4 minutos (pico de 4,1 GB); no Colab gratuito, a execução leva uns 30 minutos.
- A licença do MovieLens não permite redistribuir os dados; os arquivos do IMDb são para uso não comercial.

**Alternativas descartadas.** *Editar o notebook direto, inclusive no Colab:* a edição se perde na montagem seguinte.

**Limitações.** Cada mudança em célula de código exige nova execução no Colab. Por isso a proposta 2-C entrou só como texto: `montar_notebook.py --saidas` aproveita as saídas das células que não mudaram.

**Correção encontrada no caminho (v7).** O gerador de dados sintéticos sorteava "Children" como gênero do IMDb para filmes sem gênero (o IMDb usa "Family"), o que deixava filmes populares sem gênero.

**Onde está a evidência.** `README.md`; `scripts/`; `tests/test_notebook.py`; `docs/decisoes.md` (D1).

**Status.** Aprovada com ajustes em 18/11/2026 (4 aprovam com ajuste, 1 aprova); implementada na v6 (19/11/2026) e ampliada na v7, com os testes da fusão e da exclusão e 21 números de controle.

<a id="d27"></a>

### 27. Reprodutibilidade: o mesmo resultado em qualquer máquina

*README: "Reprodutibilidade" e "Números de controle".*

**O que foi decidido.**
- Antes de qualquer sorteio com semente (amostras da 3.5 à 3.9 e validação cruzada da 2.2-c), os dados são ordenados por `userId` ou `movieId`.
- A amostra de notas (1.5-b, 2.4) é sorteada pelo hash de `userId` e `movieId`, e não pela divisão interna do Spark.
- Os quantis da seção 1 são exatos, não aproximados.
- `scripts/numeros_controle.py` extrai números que resumem a execução (avaliações, usuários, preenchimento pelo IMDb, kNN, PCA, atípicos, Hopkins etc.) e compara duas execuções célula a célula. Eram 16 na v6; na v7 passaram a 21, com os da fusão e da exclusão (D1).

**Por quê.** O Spark devolve as linhas numa ordem que muda com o número de núcleos; sem essas regras, as amostras mudariam de máquina para máquina. Conferido em 18/11/2026 (v5): com os dados reais, as execuções com 8 e com 2 núcleos deram os mesmos números de controle. Na v6 (19/11/2026), o Mac (8 núcleos) e o Colab (2 núcleos) deram os 16 números idênticos. Só mudam os tempos da 3.9, que dependem da máquina, e a ordem de dois gêneros empatados na lista de votos do exemplo da 2.2-c.

**Correções encontradas no caminho.**
- v6: o Spark lia errado 11 títulos com aspas (sem a opção `escape` do CSV, eles ficavam com aspas sobrando e sem o ano no fim); a amostra de 3.000 usuários da 1.8-a dependia da ordem das linhas.
- v7: a 3.8-a imprimia "log(20)" fixo na normalização da diversidade de gêneros; agora usa o número real de gêneros (19).

**Alternativas descartadas.** *Sorteio interno do Spark e quantis aproximados (como na v5):* dependem de como o Spark divide os dados.

**Limitações.** Os tempos dependem da máquina. O IMDb muda todo dia (decisão 28).

**Onde está a evidência.** `README.md`; `scripts/numeros_controle.py`; `docs/decisoes.md` (D1 e implementação da v6).

**Status.** Implementada na v6.

<a id="d28"></a>

### 28. Recorte fixo do IMDb

*README: "Recorte do IMDb". Script: `scripts/recortar_imdb.py`.*

**O que foi decidido.** O notebook usa um recorte fixo do `title.basics` com as 62.294 linhas dos filmes do MovieLens (1 MB). O recorte é identificado pelo MD5, uma "impressão digital" do arquivo que muda se qualquer byte mudar. O recorte usado na v6 tem MD5 `f041a014c94b30678e0d4e9a18bc9c6d`; o arquivo de origem, baixado em 18/11/2026, tem MD5 `694fa9cd5bad4d6f3f94384b7ed61c74`. O recorte fica fora do git; a dupla guarda uma cópia num local privado (por exemplo, no Drive) e a envia ao Colab junto com o notebook.

**Por quê.** O IMDb publica só a versão do dia e atualiza o arquivo todo dia. Sem o recorte, o notebook baixa a versão do dia, e os números podem mudar na última casa. Com o recorte, todas as execuções usam os mesmos dados.

**Alternativas descartadas.**
- *Baixar o IMDb a cada execução:* os números podem mudar.
- *Pôr o recorte no git:* os arquivos do IMDb são para uso não comercial.

**Limitações.** O recorte não tem `originalTitle`; por isso a conferência da chave por título roda fora do notebook, sobre o arquivo completo (decisão 12).

**Onde está a evidência.** `README.md`; `analises/LEIAME.md`; `docs/decisoes.md` (D4 e 2-B).

**Status.** Implementada na v6.

<a id="d29"></a>

### 29. Medir fora do notebook antes de mudar o notebook

*Pasta: `analises/` (ver `analises/LEIAME.md`).*

**O que foi decidido.** As medições que embasam uma decisão são feitas fora do notebook, em scripts de `analises/` que leem os dados reais sem Spark e gravam o resultado ao lado (`resultados_*.txt`). Só o que for aprovado entra no notebook.

| Script | Decisão | Resultado gravado |
|---|---|---|
| `imputacao_por_faixa.py` | D2 (decisão 10) | `resultados_imputacao_por_faixa.txt` |
| `imdb.py` | D2 e D4 (decisões 10, 11 e 13) | `resultados_imdb.txt` |
| `chave_imdb.py` | 2-C (decisão 12) | `resultados_chave_imdb.txt` |
| `casos_sem_tratamento.py` | D6 (decisões 9 e 14), incluindo o teste do kNN nos curtas | `resultados_casos_sem_tratamento.txt` |

**Por quê.** Para decidir com números o que entra no notebook, antes de mexer nele. Os resultados gravados ficam como evidência conferível, com o MD5 do arquivo do IMDb usado.

**Limitações.**
- O kNN exploratório não é idêntico ao do notebook, e os números diferem um pouco (decisão 10).
- As medições usam os dados antes da fusão da v7 (por exemplo, 57.100 filmes com gênero nas duas fontes, contra 57.099 no notebook).
- Um download novo do IMDb pode mudar os números na última casa.

**Onde está a evidência.** `analises/LEIAME.md` e os arquivos da tabela acima.

**Status.** Em uso desde 18/11/2026.

---

<a id="glossario"></a>

## Glossário

| Termo | Significado |
|---|---|
| KDD | *Knowledge Discovery in Databases*: processo em etapas (seleção, pré-processamento, transformação, mineração, interpretação) para extrair conhecimento de dados. |
| Dados-alvo | Os dados escolhidos na seleção para seguir pelo processo. |
| Chave | Código que liga um registro de uma tabela a outro; aqui, o código IMDb da tabela `links`. |
| Junção 1:1 | Cada filme de um lado corresponde a no máximo um registro do outro. |
| Chute (linha de base) | Prever sempre o valor mais comum (aqui, Drama); um método só vale se o superar. |
| kNN | *k nearest neighbors*: prevê o rótulo de um objeto pelos rótulos dos k objetos mais parecidos. |
| Naive Bayes | Classificador probabilístico simples; seu erro serve de taxa de ruído de referência. |
| Validação cruzada de 5 partes | Divide os dados em 5 partes; treina em 4, testa na restante e repete 5 vezes. |
| Jaccard | Elementos em comum divididos pelo total de elementos distintos dos dois conjuntos (de 0 a 1). |
| IC 95% | Intervalo de confiança de 95%: faixa de valores plausíveis para a medida. |
| MD5 | "Impressão digital" de um arquivo; muda se qualquer byte mudar. |
| z-score | Subtrai a média e divide pelo desvio-padrão de cada atributo. |
| Min-max | Leva cada atributo para o intervalo [0, 1]. |
| PCA, d′ | Análise de componentes principais; d′ é o número de componentes mantidos. |
| Kaiser-Guttman, broken-stick | Critérios para escolher d′: autovalor ≥ 1; variância acima da esperada ao acaso. |
| Dimensionalidade intrínseca | Número de dimensões efetivas que descrevem os dados. |
| Concentração de distâncias | Em muitas dimensões, a menor e a maior distância ficam proporcionalmente próximas. Medida pelo contraste relativo, (máx − mín) / mín, e pela variância relativa, desvio-padrão / média das distâncias. |
| Mahalanobis | Distância ao centro que leva em conta a correlação entre atributos. |
| Hopkins (H) | Compara os dados com pontos uniformes: 0,5 = sem tendência de agrupamento; perto de 1 = dados concentrados. |
| η (razão de correlação) | Quanto uma variável categórica explica da variação de uma numérica. |
| d de Cohen | Diferença de médias dividida pelo desvio-padrão combinado. |
| V de Cramér | Força da associação entre duas variáveis categóricas (de 0 a 1). |
| kd-tree, ball-tree | Índices exatos que dividem o espaço em regiões para evitar cálculos de distância. |
| Pivôs | Objetos de referência que, pela desigualdade triangular, descartam objetos sem calcular a distância. |
| HNSW, recall@10 | Índice aproximado baseado em grafo; recall@10 é a fração dos 10 vizinhos verdadeiros recuperados. |
