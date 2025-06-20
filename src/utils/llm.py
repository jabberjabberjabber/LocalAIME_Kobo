import openai
from openai.types.chat import ChatCompletion
import requests, json
from utils.logger import Logger

TEMPERATURE = 0.6
TOP_P = 0.95
REP_PEN = 1
MIN_P = 0
TOP_K = 100

class LLM:
    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
    
    def get_answer(self, question: str, max_tokens: int, timeout: float) -> tuple[str | None, int | None]:
        messages=[
            {"role": "user", "content": question}
        ]

        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_p": MIN_P,
            "rep_pen": REP_PEN
        }
    
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json["choices"][0]["message"]["content"].strip()
            response_tokens = response_json["usage"]["completion_tokens"]

        except Exception as e:
            Logger.error('get_answer', f'The response from the model was invalid (no content): {e}')
            return None, None
        
        return response_text, response_tokens
