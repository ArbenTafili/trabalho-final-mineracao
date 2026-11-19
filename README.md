# MovieLens 25M — Trabalho Final de Mineração de Dados

Trabalho final da disciplina de Mineração de Dados, organizado pelo processo KDD (seleção → pré-processamento → transformação → mineração → interpretação) sobre o [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/).

- **P1:** agrupamento de usuários por perfil (proporções de gêneros, nota média etc.).
- **P2:** regras de associação entre filmes, depois de P1.

O notebook atual vai até a **seção 3 (Transformação)**. P1 e P2 são as próximas etapas.

## Estrutura

```
fonte/                fontes do notebook, um arquivo .py por seção  ← só aqui se edita
notebook/             notebook montado, com as saídas da última execução no Colab
scripts/              montar, executar, baixar dados, dados sintéticos, números de controle, conferência manual
tests/                testes (estrutura e execução com dados sintéticos)
conferencia_manual/   tabelas conferidas à mão no IMDb/TMDb (lidas pelo notebook)
analises/             medições exploratórias que embasam as decisões (fora do notebook)
docs/decisoes.md      registro das decisões e dos votos da mesa de avaliação
dados/                MovieLens 25M e arquivo do IMDb (fora do git)
execucoes/            execuções locais (fora do git)
```

## Regra de ouro: edita-se só em `fonte/`

O `.ipynb` é sempre **gerado** a partir de `fonte/*.py` por `scripts/montar_notebook.py`. Uma edição feita direto no `.ipynb` (inclusive no Colab) se perde na próxima montagem.

Os arquivos de `fonte/` estão no formato *percent* do jupytext: cada célula começa com `# %%` (código) ou `# %% [markdown]` (texto, com as linhas comentadas). O VS Code mostra um botão **Run Cell** em cada `# %%`, e o Jupyter abre esses arquivos como notebook com a extensão jupytext.

## Preparar o ambiente (macOS)

O ambiente local reproduz o Colab: Python 3.13, Spark 4.0.5 (exige Java 17 ou mais) e pandas 2.2.3.

```bash
brew install python@3.13 openjdk@21
```

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python scripts/baixar_dados.py
```

O último comando baixa o arquivo oficial do GroupLens (262 MB), confere o MD5 e descompacta em `dados/ml-25m`. Com `--imdb`, baixa também o `title.basics` do IMDb (231 MB), usado para os gêneros e anos que faltam no MovieLens, e grava a data e o MD5 do download. Em Linux ou Windows: Python 3.13, Java 17 ou 21 e a variável `JAVA_HOME` apontando para o Java.

## Fluxo de trabalho

1. **Editar** as fontes em `fonte/`.
2. **Teste de estrutura** (segundos): o notebook monta, toda célula de código tem cabeçalho, o material não cita slides nem aulas e termina na seção 3.
   ```bash
   .venv/bin/pytest -m "not lento"
   ```
3. **Teste com dados sintéticos** (cerca de 2 minutos no M1 Pro, pico de 1,8 GB de memória): gera um MovieLens pequeno com os casos de borda (filmes sem gênero, com 9 e 10 avaliadores, sem ano, títulos repetidos, IMAX) e executa o notebook inteiro.
   ```bash
   .venv/bin/pytest
   ```
4. **Execução completa local, opcional** (cerca de 9 minutos no M1 Pro, pico de 3,6 GB; no Colab gratuito leva uns 22): os dados reais, no Mac. Mostra o tempo de cada célula e o pico de memória. Com `--nucleos 2`, o Spark usa 2 núcleos, como no Colab.
   ```bash
   .venv/bin/python scripts/executar_notebook.py --dados dados/ml-25m
   ```
5. **Execução final no Colab**, a que vai para a entrega (passo a passo abaixo).
6. **Conferir** que Mac e Colab deram os mesmos números:
   ```bash
   .venv/bin/python scripts/numeros_controle.py execucoes/<execução local>.ipynb notebook/MovieLens25M_KDD.ipynb
   ```
7. **Commit** das fontes, do notebook executado e das tabelas de conferência.

## Execução final no Colab

A execução roda em segundo plano na própria máquina do Colab, com `jupyter nbconvert`. Assim as saídas não se perdem se a aba do navegador ficar em segundo plano.

1. Monte o notebook sem saídas:
   ```bash
   .venv/bin/python scripts/montar_notebook.py --destino execucoes/para_colab.ipynb
   ```
2. No Colab, abra um notebook qualquer, conecte o ambiente de execução e envie pelo painel **Arquivos** (eles vão para `/content`):
   - `para_colab.ipynb`;
   - `imdb_recorte_movielens.tsv.gz`, o recorte fixo do IMDb (ver abaixo). Sem ele, o notebook baixa a versão do dia do IMDb, e os números podem mudar na última casa.
3. Numa célula, inicie a execução:
   ```
   !nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output executado /content/para_colab.ipynb > /content/execucao.log 2>&1 &
   ```
4. Acompanhe, de vez em quando, numa outra célula. A execução terminou quando o log mostrar `Writing ... executado.ipynb`:
   ```
   !tail -c 300 /content/execucao.log; ls -la /content/executado.ipynb
   ```
5. Baixe o resultado:
   ```
   from google.colab import files; files.download('/content/executado.ipynb')
   ```
6. No Mac, grave as saídas no notebook do repositório. O script avisa se alguma célula ficou sem saída:
   ```bash
   .venv/bin/python scripts/montar_notebook.py --saidas ~/Downloads/executado.ipynb
   ```

## Recorte do IMDb

O gênero e o ano que faltam no MovieLens vêm do arquivo público `title.basics` do IMDb (seção 2.2-b). O IMDb publica só a versão do dia, então o notebook usa um **recorte fixo** com as 62.294 linhas dos filmes do MovieLens (1 MB):

```bash
.venv/bin/python scripts/baixar_dados.py --imdb
```

```bash
.venv/bin/python scripts/recortar_imdb.py
```

O recorte fica em `dados/imdb/` (fora do git: o IMDb é para uso não comercial), com a data e o MD5 do arquivo de origem em `imdb_recorte_movielens.info`. A dupla guarda uma cópia num local privado (por exemplo, no Drive) para que as duas execuções usem o mesmo recorte. O recorte usado na v6 tem MD5 `f041a014c94b30678e0d4e9a18bc9c6d`.

## Decisões e conferência manual

- [`docs/decisoes.md`](docs/decisoes.md): cada mudança passa pelo aluno e por uma mesa de avaliação com cinco pontos de vista; os votos, os ajustes pedidos e as pendências ficam registrados.
- [`conferencia_manual/`](conferencia_manual/COMO_PREENCHER.md): filmes sem gênero e títulos repetidos conferidos à mão no IMDb e no TMDb.

## Reprodutibilidade

Os resultados não dependem da máquina nem do número de núcleos. O Spark devolve as linhas numa ordem que muda com os núcleos, então:

- antes de qualquer sorteio com semente (amostras da 3.5 à 3.9 e validação cruzada da 2.2-b), os dados são ordenados por `userId` ou `movieId`;
- a amostra de notas (1.5-b, 2.4) é sorteada pelo hash de `userId` e `movieId`, e não pela divisão interna do Spark;
- os quantis da seção 1 são exatos, não aproximados.

Conferido em 18/11/2026: com os dados reais, as execuções com 8 e com 2 núcleos deram os 15 números de controle idênticos. Só os tempos da 3.9 mudam, porque dependem da máquina.

## Números de controle

`scripts/numeros_controle.py` extrai 15 números que resumem a execução (avaliações, usuários, imputação, PCA, atípicos, Hopkins etc.) e compara duas execuções célula a célula. Na v5 executada no Colab em 18/11/2026: 25.000.095 avaliações, 162.541 usuários, 320 filmes com gênero imputado, 1.409 usuários atípicos.

## Dados e licença

O MovieLens 25M é do [GroupLens](https://grouplens.org/datasets/movielens/) e não pode ser redistribuído; por isso fica fora do repositório e é baixado por `scripts/baixar_dados.py`. Os arquivos públicos do IMDb (quando usados) também ficam fora.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4: 19:1–19:19.
