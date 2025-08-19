import json
from pathlib import Path
from datetime import datetime
import tiktoken

class OutPutModels:
    def __init__(self, model_name: str) -> None:
        self.encoding_model = "cl100k_base"  # fallback padrão
        self.model_name = model_name

    def count_tokens(self, text_for_count: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding(self.encoding_model)
        return len(encoding.encode(text_for_count))

    def save_logs(self, modo_prompt: str, resposta: str, prompt: str) -> None:
        output_tokens = self.count_tokens(resposta)
        input_tokens = self.count_tokens(prompt)

        self.salvar_log_models(
            modelo=self.model_name,
            modo_prompt=modo_prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt,
            resposta=resposta
        )

    def salvar_log_models(
        self,
        modelo: str,
        modo_prompt: str,
        input_tokens: int,
        output_tokens: int,
        prompt: str,
        resposta: str
    ) -> None:
        data_str = datetime.now().strftime("%Y-%m-%d")
        nome_arquivo = f"{data_str}_{modelo.replace('/', '-')}_{modo_prompt}.json"
        caminho = Path("output_models") / nome_arquivo
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
                print(f"Aviso: arquivo de log corrompido, será recriado: {caminho}")
                dados = []
        else:
            dados = []

        dados.append(entrada)

        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)