# %% [markdown]
# # Trabalho Final — MovieLens 25M
#
# ## Roteiro segundo o processo KDD
#
# O notebook segue o processo KDD (Fayyad et al., 1996):
#
# **Dados → Seleção → dados-alvo → Pré-processamento → dados pré-processados → Transformação → dados transformados → Mineração → padrões → Interpretação/Avaliação → conhecimento.**
#
# O processo é iterativo: um resultado pode exigir retorno a uma etapa anterior. A qualidade dos dados de entrada limita a qualidade dos padrões obtidos.
#
# | Etapa | O que é feito aqui |
# |---|---|
# | 0. Problema e tarefa | P1 = agrupamento de usuários; P2 = associação; P2 depende de P1; não há classe/meta |
# | 1. Seleção | fontes, grão, instância × atributos, tipos e domínios; exploração univariada com histogramas de todos os atributos (média e DP marcados), consultas por gênero, por objeto e com filtro, e exploração multivariada com atributos em escala [0, 1] **sobre os dados-alvo, antes de qualquer tratamento** |
# | 2. Pré-processamento | qualidade (completude, unicidade, validade, integridade), limpeza, fusão dos títulos cadastrados duas vezes, gênero e ano que faltam preenchidos pelo IMDb (com o gênero também previsto pelo público, por kNN, como validação), exclusão dos filmes sem gênero que nada resolve, integração das tabelas, amostragem, valores ausentes e distribuições depois do tratamento |
# | 3. Transformação | perfil por usuário em proporções, seleção de características, normalização, PCA, concentração de distâncias, dimensionalidade intrínseca, visualização, valores atípicos, tendência de agrupamento, vetor revisado com atributos derivados e solução indexada |
#
# Esta versão vai até a etapa 3 (Transformação). A mineração (P1 e P2) e a interpretação ficam para a versão seguinte.
#
# > **Cinco grupos de nota:** 0,5 e 1,0 → grupo 1; 1,5 e 2,0 → 2; 2,5 e 3,0 → 3; 3,5 e 4,0 → 4; 4,5 e 5,0 → 5. Histogramas e resumos mostram média e desvio-padrão, e cada proporção declara seu denominador.
#
# **Execução:** notebook executado integralmente no Google Colab em 19/11/2026, em segundo plano com `jupyter nbconvert` na máquina do Colab, com as 46 células de código concluídas sem erro. Os dados são o MovieLens 25M, obtido de uma cópia hospedada no Hugging Face; as contagens conferem com a documentação oficial (25.000.095 avaliações, 162.541 usuários, 62.423 filmes).

# %% [markdown]
# ## 0. Problema e tarefa de mineração
#
# - **P1 — perfis de usuários:** tarefa de agrupamento (aprendizado não supervisionado). A instância analítica é o usuário.
# - **P2 — padrões frequentes:** tarefa de associação. A transação é o usuário e os itens são filmes avaliados positivamente. P2 é construída a partir do resultado de P1.
# - **Sem classe/meta:** o MovieLens não traz um atributo-alvo. Por isso, a classificação e os filtros de seleção de características que dependem de classe (CFS, Fisher e ReliefF em sua forma completa) não se aplicam diretamente antes de P1.
# - **População:** 162.541 usuários anônimos, todos com pelo menos 20 avaliações nos dados originais (depois da limpeza da 2.2, um fica com 19), entre 1995 e 2019, sem dados demográficos. Os perfis descrevem comportamento observado de avaliação, não características pessoais.
