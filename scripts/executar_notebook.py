"""Monta o notebook a partir das fontes e executa todas as células localmente, como no Colab.

Uso:
    python scripts/executar_notebook.py --dados dados/ml-25m            # dados reais
    python scripts/executar_notebook.py --dados /tmp/sinteticos --saida execucoes/teste.ipynb

Mede o tempo de cada célula e o pico de memória (kernel + Spark). Se uma célula falhar, o notebook parcial é
salvo mesmo assim e o script termina com erro. A execução final, a que vai para a entrega, é feita no Colab
(ver README); esta serve de pré-teste.
"""
import argparse
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import nbformat
import psutil
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from montar_notebook import montar  # noqa: E402

JAVA_BREW = Path("/opt/homebrew/opt/openjdk@21")


class MedidorMemoria(threading.Thread):
    """Soma a memória residente deste processo e de todos os descendentes (kernel e JVM do Spark)."""

    def __init__(self, intervalo=1.0):
        super().__init__(daemon=True)
        self.intervalo, self.pico, self._parar = intervalo, 0, threading.Event()

    def run(self):
        eu = psutil.Process()
        while not self._parar.is_set():
            total = 0
            for p in [eu, *eu.children(recursive=True)]:
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.pico = max(self.pico, total)
            self._parar.wait(self.intervalo)

    def parar(self):
        self._parar.set()


def executar(dados, saida, timeout=None):
    dados = Path(dados).resolve()
    if not (dados / "ratings.csv").is_file():
        sys.exit(f"ratings.csv não encontrado em {dados}")
    nb, _, _ = montar()
    os.environ["MOVIELENS_DIR"] = str(dados)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    if "JAVA_HOME" not in os.environ and JAVA_BREW.exists():
        os.environ["JAVA_HOME"] = str(JAVA_BREW)
    trabalho = Path(tempfile.mkdtemp(prefix="execucao_"))
    inicio_cel = {}

    def antes(cell, cell_index, **_):
        inicio_cel[cell_index] = time.time()

    def depois(cell, cell_index, **_):
        cabecalho = cell.source.splitlines()[0][:70] if cell.source else ""
        print(f"  [{cell_index:3d}] {time.time() - inicio_cel[cell_index]:7.1f}s  {cabecalho}", flush=True)

    cliente = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False,
                             resources={"metadata": {"path": str(trabalho)}},
                             on_cell_execute=antes, on_cell_executed=depois)
    medidor = MedidorMemoria()
    medidor.start()
    t0, erro = time.time(), None
    print(f"Executando com dados de {dados} (pasta de trabalho {trabalho})", flush=True)
    try:
        cliente.execute()
    except CellExecutionError as e:
        erro = e
    finally:
        medidor.parar()
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, saida)
    resumo = {"duracao_min": (time.time() - t0) / 60, "pico_memoria_gb": medidor.pico / 2**30, "saida": str(saida)}
    print(f"\nDuração: {resumo['duracao_min']:.1f} min; pico de memória: {resumo['pico_memoria_gb']:.1f} GB; "
          f"notebook executado em {saida}")
    if erro is not None:
        print(f"\nERRO na execução:\n{str(erro)[-3000:]}")
        raise SystemExit(1)
    return resumo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", required=True, help="pasta com movies.csv, ratings.csv, tags.csv e links.csv")
    p.add_argument("--saida", default=str(RAIZ / "execucoes" / f"execucao_{datetime.now():%Y%m%d_%H%M}.ipynb"))
    p.add_argument("--timeout", type=int, default=None, help="limite por célula, em segundos (padrão: sem limite)")
    a = p.parse_args()
    executar(a.dados, a.saida, a.timeout)


if __name__ == "__main__":
    main()
