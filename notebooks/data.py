"""Carregamento e validação dos dados de treinamento da competição."""

from pathlib import Path
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt


def load_training_data(path: str | Path,
                    *,target: str = "Will_Buy_EV",id_col: str = "id",) -> pd.DataFrame:

    
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")

    df = pd.read_csv(path)
    
    missing_cols = {id_col, target} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_cols)}")

    if df[id_col].isna().any() or not df[id_col].is_unique:
        raise ValueError(f"A coluna '{id_col}' contém IDs ausentes ou repetidos.")

    if df[target].isna().any():
        raise ValueError(f"O alvo '{target}' contém valores ausentes.")

    labels = set(df[target].unique())
    if labels == {"Yes", "No"}:
        df[target] = df[target].map({"Yes": 1, "No": 0})
    elif labels != {0, 1}:
        raise ValueError(
            f"O alvo '{target}' deve conter as duas classes Yes/No ou 1/0. "
            f"Valores encontrados: {labels}"
        )

    df[target] = df[target].astype("int8")
    return df



def show_cat_unique(df, colunas):
    for col in colunas:
        print(f'valores únicos da coluna: {col}')
        print(list(df[col].unique()))
        print(50*'=')


def data_stats(df, target="Will_Buy_EV", id_col="id"):

    df.info()
    print("\n" + "="*50 + "\n") # Linha separadora para organizar
    
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    if id_col in numerical_cols:
        numerical_cols.remove(id_col)
    
    print("--- Estatísticas Numéricas ---")
    display(df[numerical_cols].describe().T)
    print("\n" + "="*50 + "\n")


    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    print("--- Valores Únicos Categóricos ---")
    show_cat_unique(df, categorical_cols)


    # gráfico de distribuição de targets

    total = len(df[target])
    targ_1pct = (len(df[df[target] == 1])/total)*100
    targ_0pct = (len(df[df[target] == 0])/total)*100

    print("--- Valores Únicos Categóricos ---")
    print(f'Não Comprou (0): {targ_0pct:.2f}%')
    print(f'Comprou (1): {targ_1pct:.2f}%')

    categorias = [f'Não Comprou (0): {targ_0pct:.2f}%', f'Comprou (1): {targ_1pct:.2f}%']
    pct = [targ_0pct, targ_1pct]
    

    plt.figure(figsize=(12,6))
    plt.bar(categorias, pct, c='green')


    plt.title('Distribuição de Targets')
    plt.show()