import argparse
import pandas as pd
import os  # Para manipulação de arquivos e diretórios
import json  # Para ler o arquivo de configuração
from openai import OpenAI, AsyncOpenAI # Biblioteca da OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Biblioteca da Hugging Face
from rouge_score import rouge_scorer  # Biblioteca para calcular o ROUGE Score
import csv
from huggingface_hub import login, HfApi
import re
import json
import psutil  # Biblioteca para monitorar o uso de memória
import time
# Variáveis globais
api_key = None  # Será definida pelo arquivo de configuração
modelo = None  # Modelo a ser usado
uri = None  # URI opcional
use_local_llm = False  # Define se usará LLM local ou API
local_model = None  # Modelo local da Hugging Face
usar_prompt_local = False  # Define se o prompt será processado localmente
config_model = None  # Configuração carregada do arquivo JSON
import tqdm
import time
import json
import os
from datetime import datetime
from pathlib import Path
import tiktoken

def contar_tokens_tiktoken(texto, modelo_nome):
    try:
        encoding = tiktoken.encoding_for_model(modelo_nome)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback padrão
    return len(encoding.encode(texto))

def salvar_log(modelo, modo_prompt, input_tokens, output_tokens, prompt, resposta):
    data_str = datetime.now().strftime("%Y-%m-%d")
    nome_arquivo = f"{data_str}_{modelo.replace('/', '-')}_{modo_prompt}.json"
    caminho = Path("logs") / nome_arquivo
    caminho.parent.mkdir(parents=True, exist_ok=True)

    entrada = {
        "timestamp": datetime.now().isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt": prompt,
        "resposta": resposta
    }

    if caminho.exists():
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
        except Exception:
            # Se o arquivo estiver corrompido, sobrescreve com uma nova lista
            print(f"Aviso: arquivo de log corrompido, será recriado: {caminho}")
            dados = []
    else:
        dados = []

    dados.append(entrada)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def enviar_prompt_para_llm(prompt):
    """
    Envia o prompt para a API ou modelo local, calcula tokens com tiktoken
    e salva estatísticas num JSON baseado em data, modelo e modo_prompt.
    """
    import os
    global modelo, use_local_llm
    time.sleep(1.5)

    modo_prompt = os.getenv("MODO_PROMPT", "default")

    if use_local_llm:
        resposta = enviar_prompt_para_local(prompt)
    else:
        resposta = enviar_prompt_para_api(prompt)

    input_tokens = contar_tokens_tiktoken(prompt, modelo)
    output_tokens = contar_tokens_tiktoken(resposta, modelo)

    salvar_log(modelo, modo_prompt, input_tokens, output_tokens, prompt, resposta)
    return resposta

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

def enviar_prompt_para_api(prompt):
    """
    Envia um prompt para a API e retorna a resposta.
    """
    global api_key, uri, modelo
    try:
        # Verifica se o prompt está vazio
        if not prompt.strip():
            return "Erro: O prompt está vazio e não pode ser enviado para a API."

        # Envia o prompt para a API
        client = OpenAI(api_key=api_key, base_url=uri)
        resposta = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,  # Certifique-se de que o prompt está sendo enviado aqui
                }
            ],
            model=modelo,
            stream=False
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"Erro ao chamar a API: {e}"
    
    
def carregar_modelo_local():
    """
    Carrega um modelo de linguagem da Hugging Face.
    """
    global local_model
    try:
        # Usa o modelo especificado no arquivo de configuração
        local_model = pipeline("text-generation", model=modelo, truncation=True)  # Explicitly set truncation=True
        print(f"Modelo local '{modelo}' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o modelo local: {e}")

def enviar_prompt_para_local(prompt):
    """
    Envia um prompt para o modelo local DeepSeek-R1 e retorna a resposta.
    """
    global local_model, tokenizer, config_model
    try:
        
        if local_model is None or tokenizer is None:
            # Processa config_model para extrair argumentos
            config_args = {}
            if config_model:
                for item in config_model.split(","):
                    key, value = item.split("=")
                    config_args[key.strip()] = eval(value.strip())  # Converte strings como "True" para booleanos

            # Carrega o modelo e o tokenizer do DeepSeek-R1 com os argumentos processados
            tokenizer = AutoTokenizer.from_pretrained(modelo, **config_args)
            local_model = AutoModelForCausalLM.from_pretrained(modelo, **config_args)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        # Usa o pipeline para geração de texto
        pipe = pipeline("text-generation", model=local_model, tokenizer=tokenizer, max_new_tokens=250)
        response = pipe(formatted_prompt)[0]["generated_text"]

        return response
    except Exception as e:
        return f"Erro ao processar o prompt localmente: {e}"

def calcular_rouge_score(resposta_anterior, nova_resposta):
    """
    Calcula o ROUGE Score entre a resposta anterior e a nova resposta.
    Retorna o F1 score da métrica ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(resposta_anterior, nova_resposta)
    return scores['rougeL'].fmeasure

def gerar_dica(prompt):
    """
    Gera uma dica progressiva com base na resposta anterior.
    """
    prompt_dica = f"""Hint: The category is near: {prompt} """
    return prompt_dica


def is_valid_json(texto):
    """
    Verifica se o texto fornecido é um JSON válido.
    
    Args:
        texto (str): O texto a ser verificado.
    
    Returns:
        bool: True se for um JSON válido, False caso contrário.
    """
    try:
        json.loads(texto)
        return True
    except json.JSONDecodeError:
        return False

def extrair_cat(texto):
    """
    Procura e retorna o primeiro código CAT (CAT1 a CAT12) encontrado no texto.
    Se não encontrar, retorna None.
    """
    import re
    match = re.search(r'\bCAT([1-9]|1[0-2])\b', texto.upper())
    if match:
        return f"CAT{match.group(1)}"
    return texto

def extrair_security_incidents(texto):
    """
    Extrai 'Category' e 'Explanation' do texto fornecido.
    Verifica se o texto é um JSON válido. Caso seja, extrai as informações diretamente do JSON.
    Caso contrário, utiliza regex para extrair as informações.

    Args:
        texto (str): O texto contendo as informações de 'Category' e 'Explanation'.

    Returns:
        dict: Um dicionário contendo os valores de 'Category' e 'Explanation'.
    """
    # Verifica se o texto é um JSON válido
    if is_valid_json(texto):
        dados = json.loads(texto)
        # Verifica se o JSON contém as chaves 'Category' e 'Explanation'
        if "Category" in dados and "Explanation" in dados:
            return {
                "Category": dados["Category"].strip(),
                "Explanation": dados["Explanation"].strip()
            }

    # Caso o texto não seja um JSON válido, continua com regex
    padrao = r"(?:\*\*Category:\*\*|Category:)\s*(.*?)\s*(?:\*\*Explanation:\*\*|Explanation:|Explanation:)\s*(.*?)(?=\n|$)"
    matches = re.findall(padrao, texto, re.DOTALL)

    # Retorna apenas a última ocorrência válida, se existir
    if matches:
        ultima_ocorrencia = matches[-1]
        return {
            "Category": extrair_cat(ultima_ocorrencia[0].replace("*", "").replace("\n", "").strip()),
            "Explanation": ultima_ocorrencia[1].replace("*", "").replace("\n", "").strip()
        }
    else:
        return {"Category": "unknown", "Explanation": "unknown"}

    
def progressive_hint_prompting(prompt, row, colunas, max_hints=4, limite_rouge=0.9):
    """
    Implementa a funcionalidade de progressive hints.
    Gera dicas progressivas usando o próprio LLM com base na resposta anterior.
    Interrompe a execução se o ROUGE Score entre as respostas for maior que o limite.
    """
    prompt = f""" {prompt} {output}"""
    resposta = enviar_prompt_para_llm(prompt)
    
    resposta_anterior = resposta
    
    resultados = []  # Lista para armazenar os resultados
    informacoes_das_colunas = " / ".join(
        [f"{coluna}: {row[coluna]}" if coluna in row and pd.notnull(row[coluna]) else f"{coluna}: [valor ausente]" for coluna in colunas]
    )

    # Inicializa as variáveis para evitar erros
    nova_resposta = resposta
    rouge_score = 0.0

    if max_hints == 0:
        
        resultados.append({
            
            "informacoes_das_colunas": informacoes_das_colunas,
            "categoria": extrair_security_incidents(nova_resposta)["Category"],
            "rouge": rouge_score
        })
        #print(extrair_security_incidents(nova_resposta)['Category'])
        return resultados

    for i in range(max_hints):
        # Gera uma dica com base na resposta anterior
        dica = gerar_dica(extrair_security_incidents(resposta_anterior)["Category"])
        prompt = f""" {dica} {prompt}"""
        nova_resposta = enviar_prompt_para_llm(prompt)
        
        #print(f"Dica {i + 1}: {nova_resposta}")
        # Calcula o ROUGE Score entre a resposta anterior e a nova resposta
        rouge_score = calcular_rouge_score(extrair_security_incidents(resposta_anterior)["Category"],extrair_security_incidents(nova_resposta)["Category"])

        # Interrompe se atingir o limite de ROUGE ou o número máximo de dicas
        
             # Salva os resultados na lista
            
        if (i + 1) == max_hints or rouge_score >= limite_rouge:
             # Salva os resultados na lista
            resultados.append({
                "informacoes_das_colunas": informacoes_das_colunas,
                "categoria": extrair_security_incidents(nova_resposta)["Category"],
                "rouge": rouge_score
            })
            #print(extrair_security_incidents(nova_resposta)['Category'])
            break
        

        # Atualiza a resposta anterior para a nova resposta
        resposta_anterior = nova_resposta
    
    return resultados

def hypothesis_testing_prompting(prompt, row, colunas, max_iter=12, limite_qualidade=0.9):
    """
    Para cada categoria CATx, testa a hipótese usando as keywords correspondentes.
    Retorna uma lista de resultados para cada categoria testada.
    """
    results = []
    incident = " / ".join(
        [f"{coluna}: {row[coluna]}" if coluna in row and pd.notnull(row[coluna]) else f"{coluna}: [valor ausente]" for coluna in colunas]
    )

    categories = [f"CAT{i}" for i in range(1, 13)]
    i = 0
    results = ({
            "informacoes_das_colunas": incident,
            "categoria_testada": 'Unknown',
            "categoria": 'Unknown',
            "explicacao": 'Unknown'
            })
    while i < len(categories):
        category = categories[i]
        #print(f"categoria para hipotese: {category}")
        keywords = subcategories(category)
        keywords_str = ', '.join(keywords)
        prompt_llm = f"""
        Security Incident Analysis System - HTP

        Incident Description:
        "{prompt}"

        Instructions:
        For NIST category Considered tha Category: {category} and keyword: {keywords_str} perform the following steps:

        1. True Hypothesis:
           - Assume the incident belongs to this category. Justify based on the description and keywords.
           - Indicate whether the hypothesis is SUPPORTED or NOT SUPPORTED.

        2. False Hypothesis:
           - Assume the incident does NOT belong to this category. Justify based on the lack of evidence.
           - Indicate whether the hypothesis is SUPPORTED or NOT SUPPORTED.
        Return the hypotheses in the following format:
            True Hypothesis:[SUPPORTED/NOT SUPPORTED] 
            False Hypothesis:[SUPPORTED/NOT SUPPORTED]    

        """
        hipoteses = enviar_prompt_para_llm(prompt_llm)
        verificar_hipoteses = f"""
        {hipoteses}
        Final Classification Decision:
           - If the True Hypothesis is SUPPORTED and the False Hypothesis is NOT SUPPORTED, return:
             Category: same {category}
             Explanation: [Justification for the chosen same considered {category}]
           - Otherwise, return "UNKNOWN" as the final classification.
             Category: UNKNOWN
             Explanation: UNKNOWN
        """
        response_2 = enviar_prompt_para_api(verificar_hipoteses)
        incident_info = extrair_security_incidents(response_2)
        result_cat = incident_info.get('Category', 'UNKNOWN')
        result_exp = incident_info.get('Explanation', 'UNKNOWN')

        #print(incident_info)
        results = ({
            "informacoes_das_colunas": incident,
            "categoria_testada": category,
            "categoria": result_cat,
            "explicacao": result_exp
            })
        if(calcular_rouge_score(result_cat, category) >= limite_qualidade):
    
            break
        i += 1

    return results
    

def subcategories(categoria):
        """
    Dado o valor da chave (ex: 'CAT1'), retorna a lista de palavras-chave correspondente.
    Se não encontrar, retorna uma lista vazia.
    """
        palavras_chave = {
        "CAT1": ["phishing", "brute force", "unauthorized access", "compromised password", "credential theft", "account compromise", "token", "oauth", "ssh", "suspicious login"],
        "CAT2": ["malware", "ransomware", "trojan", "virus", "spyware", "rootkit", "infection", "malicious code"],
        "CAT3": ["ddos", "dos", "denial of service", "flood", "syn flood", "udp flood", "botnet", "api outage", "site down"],
        "CAT4": ["data leak", "exposed data", "leaked credentials", "sensitive information", "data exfiltration", "unauthorized disclosure"],
        "CAT5": ["exploit", "vulnerability", "cve", "remote execution", "sql injection", "injection", "rce", "security flaw"],
        "CAT6": ["insider", "internal abuse", "employee", "internal leak", "sabotage", "intentional action", "staff"],
        "CAT7": ["social engineering", "phishing", "vishing", "fraud", "deception", "spoofing", "manipulation", "scam", "ceo fraud"],
        "CAT8": ["physical access", "equipment theft", "burglary", "unauthorized entry", "broken door", "physical breach"],
        "CAT9": ["modification", "defacement", "unauthorized change", "erased", "altered record", "tampering"],
        "CAT10": ["misuse", "resource abuse", "crypto mining", "compromised server", "malware hosting", "unauthorized use"],
        "CAT11": ["third party", "supplier", "partner", "vendor", "supply chain", "external breach", "saas issue"],
        "CAT12": ["intrusion attempt", "scan", "reconnaissance", "probing", "port scan", "blocked exploit", "failed attempt"],
        "UNKNOWN": ["unknown", "unspecified", "not categorized", "no category", "undefined", "other"]
    }
        return palavras_chave.get(categoria.strip().upper(), [])

def progressive_rectification_prompting(prompt_inicial, row, colunas, max_iter=4, limite_qualidade=0.9):

    informacoes_das_colunas = " / ".join(
        [f"{coluna}: {row[coluna]}" if coluna in row and pd.notnull(row[coluna]) else f"{coluna}: [valor ausente]" for coluna in colunas]
    )

    
    def build_rectification_prompt(prompt, categoria_rejectada=""):
        return (
            f"""
            Incident: {prompt}
            The answer is probably not: {categoria_rejectada}
            Let's think step by step to reclassify.
            {output}
            """
        )

    def validate_response(prompt, category, subcategory):
        #print(f"Validando resposta: {category} e mascarado {subcategory}")
        prompt_verification = f"""{prompt}
        Incident (masked): {subcategory}
        Hypothesis: {category}
        Question: What is the value of X in the Incident? (If not applicable, respond 'Unknown')
        {output}
        """
        return extrair_security_incidents(enviar_prompt_para_llm(prompt_verification))['Category']
    def mascarated_prompt(prompt, subcat):
        mascarar = f"""
        {prompt}
        Replace all occurrences of the word '{subcat}' with 'X' in a given text. Ensure that the substitution preserves the original context and sentence structure.
        """
        return enviar_prompt_para_llm(mascarar)
    attempt = 0
    resultados = []
    r_0 = enviar_prompt_para_llm(prompt_inicial)
    #print(f"Resposta inicial r_0: {r_0}")
    a_0 = extrair_security_incidents(r_0)
    categoria_atual = a_0["Category"]
    categoria_rejectada = ''
    while attempt < max_iter:
        # Gera resposta inicial
        #print(f"Categoria atual a_0: {categoria_atual}")
        
        #print(f"Iteração {attempt + 1}:")
       
        subcategory = subcategories(categoria_atual)
        for subcat in subcategory:
            #print(f"Lista de subcategorias rejeitadas: {categoria_rejectada}")
            categoria_anterior = categoria_atual
            incidente_mascarado = mascarated_prompt(prompt_inicial, subcat)
            #print(f"Incidente mascarado: {incidente_mascarado}")
            categoria_atual = validate_response(incidente_mascarado, categoria_atual, subcat)
            #print(f"Categoria da validação: {categoria_atual}")
            qualidade = calcular_rouge_score(str(categoria_anterior), str(categoria_atual))
            if qualidade >= limite_qualidade:
                #print('Resultado final:', categoria_atual)
                resultados.append({
                    "informacoes_das_colunas": informacoes_das_colunas,
                    "categoria": categoria_atual,
                })
                return resultados
            else:
                categoria_rejectada = ',' + categoria_atual
                rectification_prompt = build_rectification_prompt(prompt_inicial, categoria_rejectada)
                response = enviar_prompt_para_llm(rectification_prompt)
                incident = extrair_security_incidents(response)
                categoria_atual = incident["Category"]
                #print(f"Categoria atual retification: {categoria_atual}")
        attempt += 1


def self_hint_prompting(prompt, row, colunas, max_iter=4, limite_qualidade=0.9):
    plano = f""" Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."""
    prompt_inicial = f"""{prompt} {plano}"""
    plano_intermediario = enviar_prompt_para_llm(prompt_inicial)
    response = enviar_prompt_para_llm(prompt_inicial + output)
   
    categoria_anterior = extrair_security_incidents(response)["Category"]
    resultados = []
    informacoes_das_colunas = " / ".join(
        [f"{coluna}: {row[coluna]}" if coluna in row and pd.notnull(row[coluna]) else f"{coluna}: [valor ausente]" for coluna in colunas]
    )
    
    for i in range(max_iter):
        prompt_reflexao = f"""{prompt} {plano_intermediario} The category is: {categoria_anterior} {output}"""
        response = enviar_prompt_para_llm(prompt_reflexao)
        incident = extrair_security_incidents(response)
        categoria_atual = incident["Category"]
        if (i+1 == max_iter) or calcular_rouge_score(categoria_anterior, categoria_atual) >= limite_qualidade:
            #print('Resultado final:', response)
            resultados.append({
                "informacoes_das_colunas": informacoes_das_colunas,
                "categoria": categoria_atual,
            })
            return resultados
        else: 
            categoria_anterior = categoria_atual
            prompt_inicial = f"""{prompt} {plano} The category is: {categoria_anterior}"""
            plano_intermediario = enviar_prompt_para_llm(prompt_inicial)
            
def salvar_resultados_csv(resultados, nome_arquivo):
    """
    Salva os resultados em um arquivo CSV usando pandas.
    """
    try:
        # Cria o diretório se não existir
        diretorio = os.path.dirname(nome_arquivo)
        if diretorio:  # Verifica se há um diretório no caminho
            os.makedirs(diretorio, exist_ok=True)

        # Verifica se há resultados para salvar
        if not resultados:
            print("Nenhum resultado para salvar no arquivo CSV.")
            return

        # Converte a lista de dicionários em um DataFrame
        df = pd.DataFrame(resultados)

        # Salva o DataFrame em um arquivo CSV
        df.to_csv(nome_arquivo, index=False, encoding='utf-8')
        print(f"Resultados salvos com sucesso no arquivo '{nome_arquivo}'.")
    except Exception as e:
        print(f"Erro ao salvar os resultados no arquivo CSV: {e}")


def salvar_resultados_json(resultados, nome_arquivo):
    """
    Salva os resultados em um arquivo JSON.
    """
    try:
        # Cria o diretório se não existir
        diretorio = os.path.dirname(nome_arquivo)
        if diretorio:  # Verifica se há um diretório no caminho
            os.makedirs(diretorio, exist_ok=True)

        # Verifica se há resultados para salvar
        if not resultados:
            print("Nenhum resultado para salvar no arquivo JSON.")
            return

        # Abre o arquivo JSON para escrita
        with open(nome_arquivo, mode='w', encoding='utf-8') as arquivo_json:
            json.dump(resultados, arquivo_json, ensure_ascii=False, indent=4)
        print(f"Resultados salvos com sucesso no arquivo '{nome_arquivo}'.")
    except Exception as e:
        print(f"Erro ao salvar os resultados no arquivo JSON: {e}")
        
def salvar_resultados_xlsx(resultados, nome_arquivo):
    """
    Salva os resultados em um arquivo XLSX usando pandas.
    """
    try:
        # Cria o diretório se não existir
        diretorio = os.path.dirname(nome_arquivo)
        if diretorio:  # Verifica se há um diretório no caminho
            os.makedirs(diretorio, exist_ok=True)

        # Verifica se há resultados para salvar
        if not resultados:
            print("Nenhum resultado para salvar no arquivo XLSX.")
            return

        # Converte a lista de dicionários em um DataFrame
        df = pd.DataFrame(resultados)

        # Salva o DataFrame em um arquivo XLSX
        df.to_excel(nome_arquivo, index=False, engine='openpyxl')
        print(f"Resultados salvos com sucesso no arquivo '{nome_arquivo}'.")
    except Exception as e:
        print(f"Erro ao salvar os resultados no arquivo XLSX: {e}")
                

def main():
    global api_key, modelo, uri, use_local_llm, usar_prompt_local, config_model  # Acessa as variáveis globais

    # Monitoramento de memória no início
    processo = psutil.Process()
    memoria_inicial = processo.memory_info().rss / (1024 * 1024)  # Em MB
    print(f"Uso de memória no início: {memoria_inicial:.2f} MB")

    # Monitoramento de tempo no início
    inicio_tempo = time.time()

    # Carrega as configurações do arquivo JSON
    config = carregar_configuracao()
    if not config:
        print("Erro: Configuração não encontrada.")
        return

    token = config.get('hugginface_api_key')

    # Configuração dos argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Processa arquivos CSV e executa comandos.")
    parser.add_argument('diretorio', type=str, help="Caminho do diretório contendo arquivos CSV")
    parser.add_argument('--colunas', nargs='+', required=True, help="Nomes das colunas a serem usadas como prompt")
    parser.add_argument('--limite_rouge', type=float, default=0.9, help="Limite do ROUGE Score para interromper o loop de dicas (padrão: 0.9)")
    parser.add_argument('--formato', type=str, choices=['csv', 'json', 'xlsx'], default='csv', help="Formato de saída dos resultados (csv ou json). Padrão: csv")
    parser.add_argument('--provider', nargs='+', required=True, help="Definir o provider a ser utilizado (local ou api)")
    parser.add_argument('--limite_hint', type=int, default=0, help="Número máximo de dicas progressivas (padrão: 0)")
    parser.add_argument('--nist', dest='nist', action='store_true', help="Ativar categorizador conforme padrão NIST (padrão: True)")
    parser.add_argument('--no-nist', dest='nist', action='store_false', help="Desativar categorizador conforme padrão NIST")
    parser.add_argument('--modo', type=str, choices=['php', 'prp', 'shp','htp'], default='php',
                        help="Escolha o modo de execução: 'php' para progressive hints ou 'prp' para prompt refinement process (padrão: php)")
    
    parser.set_defaults(nist=True)

    args = parser.parse_args()

    # Verifica se o ativo está na lista de ativos disponíveis
    ativos_disponiveis = config.get("ativos", [])
    config_ativo = args.provider[0]  # Pega o primeiro ativo da lista de argumentos
    if not config_ativo:
        print("Erro: Nenhum ativo foi fornecido.")
        return
    if config_ativo not in ativos_disponiveis:
        print(f"Erro: A configuração ativa '{config_ativo}' não está na lista de ativos disponíveis: {ativos_disponiveis}.")
        return

    # Carrega as configurações correspondentes
    config_provider = config.get(config_ativo)
    if not config_provider:
        print(f"Erro: Configuração para '{config_ativo}' não encontrada no arquivo de configuração.")
        return

    # Define as variáveis globais com base na configuração ativa
    provider = config_provider.get("provider")
    use_local_llm = config_provider.get("usar_prompt_local")

    if use_local_llm:
        modelo = config_provider.get("modelo")  # Modelo padrão para local
        config_model = config_provider.get("config_model")  # Configuração do modelo local
    else:
        modelo = config_provider.get("modelo")  # Modelo padrão para API
        api_key = config_provider.get("api_key")
        uri = config_provider.get("uri")  # URI opcional
        if not api_key or not uri:
            print("Erro: A chave da API e a URI são obrigatórias quando o provider é 'api'.")
            return

    # Login no Hugging Face, se necessário
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as {user['name']}")
    except Exception as e:
        print("Você não está logado no Hugging Face. Realizando login...")
        login(token=token)

    # Carregar arquivos CSV
    dataframes = []
    try:
        dataframes = carregar_todos_csv(args.diretorio)
        if not dataframes:
            print("Nenhum arquivo CSV encontrado no diretório.")
            return
    except Exception as e:
        print(f"Erro ao carregar arquivos do diretório: {e}")
        return

    # Processar todas as linhas de todos os arquivos
    resultados = []
    total_linhas = sum(len(df) for df in dataframes)  # Calcula o total de linhas em todos os DataFrames
    linhas_processadas = 0  # Contador de linhas processadas
    with tqdm.tqdm(total=total_linhas, desc="Processando linhas") as pbar:
        for df in dataframes:
            for index, row in df.iterrows():
                # Atualiza o contador de linhas processadas
                linhas_processadas += 1
                #porcentagem = (linhas_processadas / total_linhas) * 100
                #print(f"Progresso: {linhas_processadas}/{total_linhas} linhas processadas ({porcentagem:.2f}%)")
                global output
                # Construção do prompt
                prompt = f"""
                You are a security expert.
                Categorize the following incident description into a Category and an Explanation. 

                Description:
                    ```
                """
                for coluna in args.colunas:
                    # Verifica se a coluna existe na linha e se o valor não é nulo
                    if coluna in df.columns and pd.notnull(row[coluna]):
                        prompt += f" [{coluna}]: [{row[coluna]}]"

                if args.nist:
                    prompt += f"""
                                    NIST Categories Available for Classification:
                    - CAT1: Account Compromise – unauthorized access to user or administrator accounts.
                        Examples: credential phishing, SSH brute force, OAuth token theft.
                    - CAT2: Malware – infection by malicious code.
                        Examples: ransomware, Trojan horse, macro virus.
                    - CAT3: Denial of Service Attack – making systems unavailable.
                        Examples: volumetric DoS or DDoS (UDP flood, SYN flood, HTTP, HTTPS), attack on publicly available APIs or websites, botnet Mirai attacking an institution's server.
                    - CAT4: Data Leak – unauthorized disclosure of sensitive data.
                        Examples: database theft, leaked credentials.
                    - CAT5: Vulnerability Exploitation – using technical flaws for attacks.
                        Examples: exploitation of critical CVE, remote code execution (RCE), SQL injection in web applications. Includes known vulnerabilities or insecure patterns that allow remote exploitation without authentication, traffic amplification, or information leaks, such as NTP servers with monlist/readvar commands enabled, DNS responding to ANY queries, or Memcached servers open to the internet.
                    - CAT6: Insider Abuse – malicious actions by internal users.
                        Examples: copying confidential data, sabotage.
                    - CAT7: Social Engineering – deception to gain access or data.
                        Examples: phishing, vishing, CEO fraud.
                    - CAT8: Physical Incident – impact due to unauthorized physical access.
                        Examples: laptop theft, data center break-in.
                    - CAT9: Unauthorized Modification – improper changes to systems or data.
                        Examples: defacement, record manipulation.
                    - CAT10: Misuse of Resources – unauthorized use for other purposes.
                        Examples: cryptocurrency mining, malware distribution.
                    - CAT11: Third-Party Issues – security failures by suppliers.
                        Examples: SaaS breach, supply chain attack.
                    - CAT12: Intrusion Attempt – unconfirmed attacks.
                        Examples: network scans, brute force, blocked exploits.

                    Your task:
                    - Classify the incident below using the most appropriate category code (CAT1 to CAT12).
                    - Justify based on the explanation of the selected category.
                    """
                if(args.nist):
                        output = f"""
                    If classification is not possible, return:
                    Category: Unknown
                    Explanation: Unknown
                    
                    OUTPUT:
                    Category: [NIST code]
                    Explanation: [Justification for the chosen category]
                    """
                else:
                    output += f"""
                    Rules for returning the NIST Category and Explanation:
                    - If no clear Category is found, return "Unknown"
                    - If no clear Explanation is found, return "Unknown"

                    Category: [Identified Category Title]
                    Explanation: [Detailed Description of the Category]
                    """
            
                if args.modo == 'shp':
                    resultado_analisado = self_hint_prompting(prompt, row, args.colunas, max_iter=args.limite_hint, limite_qualidade=args.limite_rouge)
                    resultados.extend(resultado_analisado)
                elif args.modo == 'prp':
                    resultado_analisado = progressive_rectification_prompting(prompt, row, args.colunas, max_iter=args.limite_hint, limite_qualidade=args.limite_rouge)
                    resultados.extend(resultado_analisado)
                elif args.modo == 'htp':
                    resultado_analisado = hypothesis_testing_prompting(prompt, row, args.colunas, max_iter=args.limite_hint, limite_qualidade=args.limite_rouge)
                    resultados.extend(resultado_analisado)
                else:
                    resultado_analisado = progressive_hint_prompting(prompt, row, args.colunas, max_hints=args.limite_hint, limite_rouge=args.limite_rouge)
                    resultados.extend(resultado_analisado)
                pbar.update(1)
    nome_arquivo = f"resultados_{provider}_{args.modo}.{args.formato}"
    if args.formato == 'csv':
        salvar_resultados_csv(resultados, nome_arquivo)
    elif args.formato == 'json':
        salvar_resultados_json(resultados, nome_arquivo)
    elif args.formato == 'xlsx':
        salvar_resultados_xlsx(resultados, nome_arquivo)

    # Monitoramento de memória no final
    memoria_final = processo.memory_info().rss / (1024 * 1024)  # Em MB
    print(f"Uso de memória no final: {memoria_final:.2f} MB")
    print(f"Memória utilizada durante a execução: {memoria_final - memoria_inicial:.2f} MB")

    # Monitoramento de tempo no final
    fim_tempo = time.time()
    tempo_total = fim_tempo - inicio_tempo
    print(f"Tempo total de execução: {tempo_total:.2f} segundos")


if __name__ == "__main__":
    main()