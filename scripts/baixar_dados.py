"""Baixa o MovieLens 25M oficial do GroupLens para `dados/`, confere o MD5 e descompacta.

Uso:
    python scripts/baixar_dados.py          # MovieLens 25M
    python scripts/baixar_dados.py --imdb   # também o title.basics do IMDb (231 MB), com data e MD5 registrados
Os dados ficam fora do repositório (a licença do MovieLens não permite redistribuição; o IMDb é para uso não
comercial). O IMDb atualiza o arquivo todo dia, então a data e o MD5 do download ficam gravados ao lado dele.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
URL_IMDB = "https://datasets.imdbws.com/title.basics.tsv.gz"
ARQUIVOS = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def md5(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_imdb():
    pasta = DADOS / "imdb"
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / "title.basics.tsv.gz"
    if arquivo.is_file():
        print(f"IMDb já presente em {arquivo} (MD5 {(pasta / 'title.basics.md5').read_text().strip()})")
        return
    print(f"Baixando {URL_IMDB} (cerca de 231 MB)...")
    urllib.request.urlretrieve(URL_IMDB, arquivo)
    (pasta / "title.basics.md5").write_text(md5(arquivo) + "\n")
    (pasta / "title.basics.data_download").write_text(f"{datetime.now():%Y-%m-%d %H:%M}\n")
    print(f"IMDb em {arquivo}; MD5 {md5(arquivo)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imdb", action="store_true", help="baixar também o title.basics do IMDb")
    if p.parse_args().imdb:
        baixar_imdb()
    DADOS.mkdir(exist_ok=True)
    destino = DADOS / "ml-25m"
    if all((destino / a).is_file() for a in ARQUIVOS):
        print(f"Dados já presentes em {destino}")
        return
    zip_local = DADOS / "ml-25m.zip"
    if not zip_local.is_file():
        print(f"Baixando {URL} (262 MB)...")
        urllib.request.urlretrieve(URL, zip_local)
    oficial = urllib.request.urlopen(URL + ".md5").read().decode().split()[0]
    calculado = md5(zip_local)
    if calculado != oficial:
        sys.exit(f"MD5 diferente do oficial: {calculado} != {oficial}. Apague {zip_local} e baixe de novo.")
    print(f"MD5 conferido: {calculado}")
    with zipfile.ZipFile(zip_local) as z:
        z.extractall(DADOS)
    print(f"Dados em {destino}")


if __name__ == "__main__":
    main()
