import os
import json
import pandas as pd

def carregar_configuracao():
    """
    Carrega as configurações de um arquivo JSON chamado 'config.json' no diretório atual.
    """
    caminho_config = "config.json"
    if not os.path.exists(caminho_config):
        print(f"Erro: Arquivo de configuração '{caminho_config}' não encontrado no diretório atual.")
        return None
    try:
        with open(caminho_config, 'r') as f:
            return json.load(f)  # O arquivo será fechado automaticamente após a leitura
    except Exception as e:
        print(f"Erro ao carregar o arquivo de configuração: {e}")
        return None

def carregar_todos_csv(diretorio):
    """
    Carrega todos os arquivos CSV, XLS ou XLSX de um diretório e retorna uma lista de DataFrames.
    """
    if not os.path.exists(diretorio):
        raise FileNotFoundError(f"O diretório '{diretorio}' não foi encontrado.")

    dataframes = []
    arquivos_suportados = [
        arquivo for arquivo in os.listdir(diretorio)
        if arquivo.endswith(('.csv', '.xls', '.xlsx')) and not arquivo.startswith('~$')
    ]

    if not arquivos_suportados:
        print(f"Nenhum arquivo CSV, XLS ou XLSX encontrado no diretório '{diretorio}'.")
        return dataframes

    for arquivo in arquivos_suportados:
        caminho_arquivo = os.path.join(diretorio, arquivo)
        try:
            if arquivo.endswith('.csv'):
                df = pd.read_csv(caminho_arquivo)
            elif arquivo.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(caminho_arquivo, engine='openpyxl')  # Especifica o engine para arquivos Excel
            dataframes.append(df)
            print(f"Arquivo carregado com sucesso: {arquivo}")
        except Exception as e:
            print(f"Erro ao carregar o arquivo {arquivo}: {e}")
    return dataframes