"""Testes do notebook.

- Estrutura (segundos): as fontes montam um notebook válido e seguem as regras do material para o professor.
- Execução com dados sintéticos (poucos minutos, marcado como `lento`): o notebook roda inteiro, sem erro, sobre
  um MovieLens sintético com os casos de borda.

    pytest                 # tudo
    pytest -m "not lento"  # só a estrutura
"""
import re
import sys
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from executar_notebook import executar  # noqa: E402
from gerar_sinteticos import gerar  # noqa: E402
from montar_notebook import montar  # noqa: E402
from numeros_controle import controles, textos_por_celula  # noqa: E402


@pytest.fixture(scope="module")
def notebook():
    nb, _, _ = montar()
    return nb


def test_fontes_montam_notebook_valido(notebook):
    nbformat.validate(notebook)
    assert sum(c.cell_type == "code" for c in notebook.cells) > 0


def test_cada_celula_de_codigo_tem_cabecalho(notebook):
    sem_cabecalho = [c.source[:60] for c in notebook.cells
                     if c.cell_type == "code" and not re.match(r"# \[\d+\.\d+(-[a-z])?\] ", c.source)]
    assert not sem_cabecalho, sem_cabecalho


def test_material_nao_cita_slides_nem_aulas(notebook):
    texto = "\n".join(c.source for c in notebook.cells).lower()
    assert "slide" not in texto
    assert not re.search(r"\baulas?\b", texto)


def test_nenhuma_leitura_pendente(notebook):
    pendentes = [c.source[:60] for c in notebook.cells if "LEITURA_PENDENTE" in c.source]
    assert not pendentes, pendentes


def test_notebook_termina_na_secao_3(notebook):
    titulos = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert not any(re.match(r"## [4-9]\.", t) for t in titulos), titulos


@pytest.mark.lento
def test_executa_inteiro_com_dados_sinteticos(tmp_path_factory):
    dados = tmp_path_factory.mktemp("ml-25m-sintetico")
    gerar(dados, n_usuarios=12000, n_filmes=3000)
    saida = tmp_path_factory.mktemp("execucao") / "sintetico.ipynb"
    executar(dados, saida)  # termina com SystemExit se alguma célula falhar
    valores = controles(textos_por_celula(saida))
    assert valores["cópias fundidas (títulos repetidos)"] == "1"  # só a "Cópia Duplicada"; as outras ficam separadas
    assert valores["filmes excluídos (sem gênero)"] not in ("—", "0")
    assert float(valores["diferença máxima da nota após tratamento (p.p.)"]) < 0.5
    assert valores["grupos do perfil (vetor revisado)"] == "17"  # 18 gêneros (IMAX saiu na 2.2), Animation + Children juntos
    assert valores["filmes com IMAX retirado do gênero"] not in ("—", "0")
    assert valores["filmes só com IMAX preenchidos pelo IMDb"] == "1 de 1"  # fica sem gênero e o IMDb preenche
    for nome in ("d′ do vetor base (⌈D2⌉)", "d′ do vetor base (broken-stick)", "d′ do vetor revisado (⌈D2⌉)"):
        assert valores[nome] != "—", nome
    assert valores["filmes com gênero do IMDb"] != "—"
    assert valores["mínimo de avaliadores (kNN)"] != "—"
