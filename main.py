import argparse
import pandas as pd
import os
import json
import re
import psutil
import time
import tqdm

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge_score import rouge_scorer
from huggingface_hub import login, HfApi
from utils.configs import carregar_configuracao, carregar_todos_csv

# Variáveis globais
api_key = None
modelo = None
uri = None
use_local_llm = False
local_model = None
usar_prompt_local = False
config_model = None
output = "" # Definindo output globalmente para ser acessível

def enviar_prompt_para_llm(prompt):
    """
    Envia o prompt para a API ou modelo local e retorna a resposta.
    """
    global modelo, use_local_llm
    time.sleep(1.5)  # Previne sobrecarga da API

    if use_local_llm:
        return enviar_prompt_para_local(prompt)
    else:
        return enviar_prompt_para_api(prompt)

def enviar_prompt_para_api(prompt):
    """
    Envia um prompt para a API OpenAI-compatível e retorna a resposta.
    """
    global api_key, uri, modelo
    if not prompt or not prompt.strip():
        print("Erro: O prompt está vazio.")
        return '{"Category": "Unknown", "Explanation": "Prompt vazio recebido."}'
    try:
        client = OpenAI(api_key=api_key, base_url=uri)
        resposta = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo,
            stream=False,
            temperature=0.0 # Baixa temperatura para respostas mais determinísticas e factuais
        )
        return resposta.choices[0].message.content
    except Exception as e:
        print(f"Erro ao chamar a API: {e}")
        return f'{{"Category": "Unknown", "Explanation": "Erro na API: {e}"}}'

def carregar_modelo_local():
    """
    Carrega um modelo de linguagem da Hugging Face.
    """
    global local_model, modelo
    try:
        local_model = pipeline("text-generation", model=modelo, truncation=True)
        print(f"Modelo local '{modelo}' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o modelo local: {e}")

def enviar_prompt_para_local(prompt):
    """
    Envia um prompt para o modelo local e retorna a resposta.
    """
    global local_model, tokenizer, config_model, modelo
    try:
        if local_model is None:
            config_args = {key.strip(): eval(value.strip()) for key, value in (item.split("=") for item in config_model.split(","))} if config_model else {}
            tokenizer = AutoTokenizer.from_pretrained(modelo, **config_args)
            local_model = AutoModelForCausalLM.from_pretrained(modelo, **config_args)
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        pipe = pipeline("text-generation", model=local_model, tokenizer=tokenizer, max_new_tokens=250)
        response = pipe(formatted_prompt)[0]["generated_text"]
        return response
    except Exception as e:
        return f"Erro ao processar o prompt localmente: {e}"

def calcular_rouge_score(texto_a, texto_b):
    """
    Calcula o ROUGE-L F1 score entre dois textos.
    """
    if not texto_a or not texto_b:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(texto_a, texto_b)
    return scores['rougeL'].fmeasure

def extrair_json_do_texto(texto):
    """
    Extrai o primeiro objeto JSON válido de uma string.
    """
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        try:
            json_str = match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def extrair_security_incidents(texto):
    """
    Extrai 'Category' e 'Explanation' de uma string, priorizando formato JSON.
    """
    dados_json = extrair_json_do_texto(texto)
    if dados_json and "Category" in dados_json and "Explanation" in dados_json:
        return {
            "Category": str(dados_json["Category"]).strip(),
            "Explanation": str(dados_json["Explanation"]).strip()
        }
    
    # Fallback para regex se JSON falhar
    padrao = r"(?:\*\*?Category:\*\*?|Category:)\s*(.*?)\s*(?:\*\*?Explanation:\*\*?|Explanation:)\s*(.*)"
    match = re.search(padrao, texto, re.DOTALL)
    if match:
        categoria = match.group(1).replace("*", "").strip()
        explicacao = match.group(2).replace("*", "").strip()
        # Garante que a categoria tem o formato CATx
        cat_match = re.search(r'\bCAT([1-9]|1[0-2])\b', categoria.upper())
        return {
            "Category": f"CAT{cat_match.group(1)}" if cat_match else "Unknown",
            "Explanation": explicacao
        }
    return {"Category": "Unknown", "Explanation": "Formato de saída inválido"}


def progressive_hint_prompting(prompt, row, colunas, max_hints=4, limite_rouge=0.9):
    """
    Implementa a funcionalidade de progressive hints (PHP).
    """
    global output
    prompt_completo = f"{prompt} {output}"
    resposta_str = enviar_prompt_para_llm(prompt_completo)
    resposta_anterior = extrair_security_incidents(resposta_str)

    informacoes_das_colunas = " / ".join(
        [f"{coluna}: {row[coluna]}" for coluna in colunas if coluna in row and pd.notna(row[coluna])]
    )
    
    if max_hints == 0:
        return [{
            "informacoes_das_colunas": informacoes_das_colunas,
            "categoria": resposta_anterior["Category"],
            "explicacao": resposta_anterior["Explanation"],
            "rouge": 0.0
        }]

    for i in range(max_hints):
        dica = f"Hint: The previous classification was '{resposta_anterior['Category']}'. Refine your analysis."
        prompt_com_dica = f"{dica}\n{prompt_completo}"
        nova_resposta_str = enviar_prompt_para_llm(prompt_com_dica)
        nova_resposta = extrair_security_incidents(nova_resposta_str)
        
        rouge_score = calcular_rouge_score(resposta_anterior["Category"], nova_resposta["Category"])

        if (i + 1) == max_hints or rouge_score >= limite_rouge:
            return [{
                "informacoes_das_colunas": informacoes_das_colunas,
                "categoria": nova_resposta["Category"],
                "explicacao": nova_resposta["Explanation"],
                "rouge": rouge_score
            }]
        
        resposta_anterior = nova_resposta

    return [{
        "informacoes_das_colunas": informacoes_das_colunas,
        "categoria": resposta_anterior["Category"],
        "explicacao": resposta_anterior["Explanation"],
        "rouge": calcular_rouge_score(resposta_anterior["Category"], nova_resposta["Category"])
    }]

def rices_progressive_rectification(prompt_base, row, colunas, max_iter=3, limite_qualidade=0.9):
    """
    Implementa o RICES + Progressive Rectification Prompting (PRP) para classificação robusta.
    """
    texto_incidente = " / ".join(
        [f"{coluna}: {row[coluna]}" for coluna in colunas if coluna in row and pd.notna(row[coluna])]
    )

    # --- 1. PROMPT INICIAL (RICES) ---
    prompt_inicial = f"""
    **Role**: You are an expert cybersecurity analyst specializing in incident classification according to NIST SP 800-61r3 standards.

    **Instructions**:
    1. Analyze the incident details provided below.
    2. Classify the incident into ONE of the provided NIST categories.
    3. Your output MUST be a single, valid JSON object with two keys: "Category" (e.g., "CAT5") and "Explanation" (a brief justification).
    4. Do not include any text before or after the JSON object.

    **Context**: This classification is critical for a Security Operations Center (SOC) to trigger the correct response playbook. Accuracy is paramount. Here are the categories:
    - CAT1: Account Compromise (e.g., phishing, brute force)
    - CAT2: Malware (e.g., ransomware, virus)
    - CAT3: Denial of Service (e.g., DDoS, SYN flood)
    - CAT4: Data Leak (e.g., database theft, exposed credentials)
    - CAT5: Vulnerability Exploitation (e.g., CVE, RCE, SQL injection)
    - CAT6: Insider Abuse (e.g., employee sabotage)
    - CAT7: Social Engineering (e.g., phishing, CEO fraud)
    - CAT8: Physical Incident (e.g., laptop theft)
    - CAT9: Unauthorized Modification (e.g., defacement)
    - CAT10: Misuse of Resources (e.g., crypto mining)
    - CAT11: Third-Party Issues (e.g., SaaS breach)
    - CAT12: Intrusion Attempt (e.g., network scans)

    **Examples**:
    - Example 1 Input: "[vulnerability_name]: CVE-2023-1234 in web server detected. [action]: Blocked remote code execution attempt."
    - Example 1 Output: {{"Category": "CAT5", "Explanation": "The incident involves an attempt to exploit a known vulnerability (CVE), which directly corresponds to the Vulnerability Exploitation category."}}
    
    - Example 2 Input: "[log_entry]: Multiple failed login attempts from IP 1.2.3.4 followed by a successful login. [user_report]: User reports unauthorized access."
    - Example 2 Output: {{"Category": "CAT1", "Explanation": "The combination of multiple failed logins followed by a success indicates a brute force attack, leading to an Account Compromise."}}

    **Specificity**: Now, classify the following incident.

    **Incident Details**:
    "{texto_incidente}"
    """
    
    resposta_inicial_str = enviar_prompt_para_llm(prompt_inicial)
    incidente_atual = extrair_security_incidents(resposta_inicial_str)
    categoria_atual = incidente_atual.get("Category", "Unknown")
    
    categorias_rejeitadas = []
    
    for tentativa in range(max_iter):
        if categoria_atual == "Unknown":
             break # Não há o que validar ou retificar

        # --- 2. VALIDAÇÃO (Simplificada para eficiência) ---
        prompt_validacao = f"""
        **Role**: You are a logical reasoning expert.
        **Instruction**: Your task is to validate a cybersecurity incident classification. I will provide an incident description and a hypothesized category. Re-classify the incident based only on the provided text and determine if the hypothesis is sound.
        **Context**: This is a verification step. We are testing the robustness of an initial classification.
        **Hypothesis**: The category is '{categoria_atual}'.
        **Incident**: "{texto_incidente}"
        **Specificity**: Provide your re-classification in a strict JSON format: {{"Category": "...", "Explanation": "..."}}.
        """
        resposta_validacao_str = enviar_prompt_para_llm(prompt_validacao)
        resultado_validacao = extrair_security_incidents(resposta_validacao_str)
        categoria_validada = resultado_validacao.get("Category", "Unknown")

        qualidade = calcular_rouge_score(categoria_atual, categoria_validada)

        if qualidade >= limite_qualidade:
            # Confiança alta, a classificação é consistente.
            incidente_atual["Confidence"] = "High"
            break
        
        # --- 3. RETIFICAÇÃO (Se a validação falhar) ---
        if categoria_atual not in categorias_rejeitadas:
            categorias_rejeitadas.append(categoria_atual)
        
        rejeitadas_str = ", ".join(categorias_rejeitadas)
        
        prompt_retificacao = f"""
        **Role**: You are a senior cybersecurity analyst performing a differential diagnosis.
        **Instruction**: Your previous analysis of an incident was challenged and found to be potentially incorrect. Re-analyze the original incident from scratch.
        **Context**: Based on further review, you MUST AVOID the following categories as they were deemed incorrect: **{rejeitadas_str}**. Think step-by-step to find the more accurate classification from the remaining NIST categories.
        **Incident**: "{texto_incidente}"
        **Specificity**: Return your final, corrected classification in a strict JSON format: {{"Category": "...", "Explanation": "..."}}.
        """
        
        resposta_retificada_str = enviar_prompt_para_llm(prompt_retificacao)
        incidente_atual = extrair_security_incidents(resposta_retificada_str)
        categoria_atual = incidente_atual.get("Category", "Unknown")

        if tentativa == max_iter - 1:
            incidente_atual["Confidence"] = "Low (max iterations reached)"

    return [{
        "informacoes_das_colunas": texto_incidente,
        "categoria": incidente_atual.get("Category"),
        "explicacao": incidente_atual.get("Explanation"),
        "confianca": incidente_atual.get("Confidence", "Low")
    }]


def salvar_resultados(resultados, nome_base, formato):
    """
    Salva os resultados no formato especificado (CSV, JSON, XLSX).
    """
    if not resultados:
        print(f"Nenhum resultado para salvar no formato {formato}.")
        return

    nome_arquivo = f"{nome_base}.{formato}"
    diretorio = os.path.dirname(nome_arquivo)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)

    try:
        df = pd.DataFrame(resultados)
        if formato == 'csv':
            df.to_csv(nome_arquivo, index=False, encoding='utf-8')
        elif formato == 'json':
            df.to_json(nome_arquivo, orient='records', indent=4, force_ascii=False)
        elif formato == 'xlsx':
            df.to_excel(nome_arquivo, index=False, engine='openpyxl')
        print(f"Resultados salvos com sucesso em '{nome_arquivo}'.")
    except Exception as e:
        print(f"Erro ao salvar os resultados em {formato}: {e}")


def main():
    global api_key, modelo, uri, use_local_llm, config_model, output

    processo = psutil.Process()
    memoria_inicial = processo.memory_info().rss / (1024 * 1024)
    inicio_tempo = time.time()
    print(f"Uso de memória no início: {memoria_inicial:.2f} MB")

    config = carregar_configuracao()
    if not config:
        print("Erro: Arquivo de configuração 'config.json' não encontrado ou inválido.")
        return

    parser = argparse.ArgumentParser(description="Processa e classifica incidentes de segurança usando LLMs.")
    parser.add_argument('diretorio', type=str, help="Caminho do diretório com arquivos CSV de incidentes.")
    parser.add_argument('--colunas', nargs='+', required=True, help="Nomes das colunas a serem usadas como prompt.")
    parser.add_argument('--provider', type=str, required=True, help="Provedor a ser utilizado (ex: 'openai', 'local_model').")
    parser.add_argument('--modo', type=str, choices=['php', 'prp'], default='prp', help="Modo de prompting: 'php' (progressive-hint) ou 'prp' (rices-progressive-rectification).")
    parser.add_argument('--formato', type=str, choices=['csv', 'json', 'xlsx'], default='csv', help="Formato do arquivo de saída.")
    parser.add_argument('--limite_rouge', type=float, default=0.9, help="Limite de ROUGE Score para convergência.")
    parser.add_argument('--limite_hint', type=int, default=3, help="Número máximo de iterações/dicas.")
    args = parser.parse_args()

    config_provider = config.get(args.provider)
    if not config_provider:
        print(f"Erro: Provider '{args.provider}' não encontrado no config.json.")
        return

    use_local_llm = config_provider.get("usar_prompt_local", False)
    modelo = config_provider.get("modelo")
    if use_local_llm:
        config_model = config_provider.get("config_model")
    else:
        api_key = config_provider.get("api_key")
        uri = config_provider.get("uri")
        if not api_key or not modelo:
            print("Erro: 'api_key' e 'modelo' são obrigatórios para provedores de API.")
            return

    # Construção do prompt base
    prompt_base = f"""
    You are a security expert. Categorize the following incident description into a Category and an Explanation.
    Description:
    ```
    """

    output = f"""
    If classification is not possible, return:
    Category: Unknown
    Explanation: Unknown
    
    OUTPUT:
    Category: [NIST code]
    Explanation: [Justification for the chosen category]
    """

    dataframes = carregar_todos_csv(args.diretorio)
    if not dataframes:
        print("Nenhum arquivo CSV encontrado no diretório.")
        return

    resultados = []
    total_linhas = sum(len(df) for df in dataframes)
    
    with tqdm.tqdm(total=total_linhas, desc=f"Classificando incidentes via {args.provider.upper()}/{args.modo.upper()}") as pbar:
        for df in dataframes:
            for index, row in df.iterrows():
                if args.modo == 'prp':
                    resultado_analisado = rices_progressive_rectification(
                        prompt_base, row, args.colunas, 
                        max_iter=args.limite_hint, 
                        limite_qualidade=args.limite_rouge
                    )
                else: # php
                    resultado_analisado = progressive_hint_prompting(
                        prompt_base, row, args.colunas, 
                        max_hints=args.limite_hint, 
                        limite_rouge=args.limite_rouge
                    )
                
                if resultado_analisado:
                    resultados.extend(resultado_analisado)
                pbar.update(1)

    nome_base_arquivo = f"resultados_{args.provider}_{args.modo}"
    salvar_resultados(resultados, nome_base_arquivo, args.formato)

    memoria_final = processo.memory_info().rss / (1024 * 1024)
    fim_tempo = time.time()
    tempo_total = fim_tempo - inicio_tempo
    
    print(f"\nUso de memória no final: {memoria_final:.2f} MB")
    print(f"Memória utilizada durante a execução: {memoria_final - memoria_inicial:.2f} MB")
    print(f"Tempo total de execução: {tempo_total:.2f} segundos")

if __name__ == "__main__":
    main()