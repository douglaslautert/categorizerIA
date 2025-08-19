
import json
from typing import List, Dict, Any

def extrair_security_incidents(resposta_llm: str) -> Dict[str, Any]:
    #Função para parsear a resposta do LM (ex: JSON) para um dicionário.
    import json
    try:
        return json.loads(resposta_llm)
    except json.JSONDecodeError:
        print(f"AVISO: Não foi possível parsear o JSON: {resposta_llm}")
        return {"Category": "Unknown"}

def subcategories(categoria: str) -> List[str]:
    # Retorna subcategorias conhecidas para uma dada categoria para o mascaramento.
    categorias_map = {
        "Malware": ["ransomware", "virus", "trojan"],
        "Phishing": ["email", "link malicioso", "credenciais"],
        "Data Leak": ["banco de dados", "informações de clientes", "documentos confidenciais"]
    }
    return categorias_map.get(categoria, [])

def calcular_rouge_score(texto1: str, texto2: str) -> float:
    # Calcula uma pontuação de similaridade. Aqui, um simples acerto/erro.
    return 1.0 if texto1.lower() == texto2.lower() else 0.0