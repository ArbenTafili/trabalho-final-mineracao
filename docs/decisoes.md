# Registro de decisões

Cada mudança no trabalho segue o mesmo caminho:

1. a ideia é proposta (pelo aluno, pela dupla ou pelo Claude) com os números reais da base;
2. o aluno filtra o que vai adiante;
3. uma **mesa de avaliação** com cinco pontos de vista vota cada ideia:
   - professor de Mineração de Dados (rigor do KDD);
   - estatístico (viés e validação);
   - engenheiro de dados (reprodutibilidade);
   - especialista em sistemas de recomendação (sentido para P1/P2);
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

**Status:** aprovada com ajustes em 18/11/2026; aguardando permissão para baixar o arquivo do IMDb.

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

- Perda máxima de acerto aceita para trocar o kNN (D2).
- Conferência manual dos 38 filmes (D2) e dos 8 pares (D3).
- Permissão para baixar o `title.basics` do IMDb (D4).
- Aprovação da D5 para ir à mesa.
- Decisão sobre Musical e gêneros raros (agrupar só no vetor ou já no pré-processamento).
