# %% [markdown]
# ## Estado da análise
#
# | Etapa | Situação |
# |---|---|
# | 0. Problema e tarefa | definido: P1 agrupamento → P2 associação; sem classe |
# | 1. Seleção | concluída: estrutura, tipos, exploração univariada com histogramas de todos os atributos (incluindo tempo e distribuição dos gêneros), consultas por gênero, por objeto e com filtro, exploração multivariada em escala [0, 1] e testes estatísticos |
# | 2. Pré-processamento | concluído: auditoria, limpeza (nenhuma avaliação removida), integração, imputação do gênero dos filmes sem gênero, verificação de amostragem, valores ausentes da matriz usuário × filme (sem imputação) e distribuições depois do tratamento |
# | 3. Transformação | concluída: perfil por usuário, seleção de características, z-score × min-max, PCA (d′ = 7), concentração de distâncias, dimensionalidade intrínseca, projeções 2D, valores atípicos, tendência de agrupamento, vetor revisado (PCA com d′ = 6, entrada principal de P1) e solução indexada |
