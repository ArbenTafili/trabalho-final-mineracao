"""Baixa o MovieLens 25M oficial do GroupLens para `dados/`, confere o MD5 e descompacta.

Uso: python scripts/baixar_dados.py
Os dados ficam fora do repositório (a licença do MovieLens não permite redistribuição).
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ARQUIVOS = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def md5(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
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
