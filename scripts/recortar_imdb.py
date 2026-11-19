"""Cria o recorte fixo do IMDb usado pelo notebook: só as linhas dos filmes do MovieLens.

Uso: python scripts/recortar_imdb.py   (depois de `python scripts/baixar_dados.py --imdb`)

O IMDb publica só a versão do dia do `title.basics`. Para que todas as execuções usem os mesmos dados, o notebook
lê este recorte (poucos MB), e não o arquivo completo. O recorte fica fora do repositório (uso não comercial do
IMDb): a dupla guarda uma cópia num local privado e, no Colab, envia-o para /content junto com o notebook.

Gera em dados/imdb/:
- imdb_recorte_movielens.tsv.gz: tconst, titleType, primaryTitle, startYear, genres;
- imdb_recorte_movielens.info: origem, data e MD5 do arquivo completo e MD5 do recorte.
"""
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "dados" / "imdb"
COMPLETO = PASTA / "title.basics.tsv.gz"
RECORTE = PASTA / "imdb_recorte_movielens.tsv.gz"
LINKS = RAIZ / "dados" / "ml-25m" / "links.csv"


def md5(caminho):
    return hashlib.md5(Path(caminho).read_bytes()).hexdigest()


def main():
    if not COMPLETO.is_file() or not LINKS.is_file():
        sys.exit("Faltam os dados: rode `python scripts/baixar_dados.py --imdb`")
    codigos = set(pd.read_csv(LINKS)["imdbId"].map(lambda i: f"tt{int(i):07d}"))
    imdb = pd.read_csv(COMPLETO, sep="\t", usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
                       dtype=str, na_values="\\N", keep_default_na=False, quoting=3)
    recorte = imdb[imdb["tconst"].isin(codigos)].sort_values("tconst")
    recorte.to_csv(RECORTE, sep="\t", index=False, na_rep="\\N", compression={"method": "gzip", "mtime": 0})
    data_download = (PASTA / "title.basics.data_download").read_text().strip() if (PASTA / "title.basics.data_download").exists() else "?"
    info = (f"origem: https://datasets.imdbws.com/title.basics.tsv.gz\n"
            f"download do arquivo completo: {data_download}\n"
            f"MD5 do arquivo completo: {md5(COMPLETO)}\n"
            f"recorte criado em: {datetime.now():%Y-%m-%d %H:%M}\n"
            f"linhas do recorte: {len(recorte)} (de {len(codigos)} códigos IMDb do MovieLens)\n"
            f"MD5 do recorte: {md5(RECORTE)}\n")
    (PASTA / "imdb_recorte_movielens.info").write_text(info)
    print(info, end="")
    print(f"Recorte em {RECORTE.relative_to(RAIZ)} ({RECORTE.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
