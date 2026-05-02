from app.services.openrouter_client import OpenRouterClient

llm = OpenRouterClient()
result = llm.chat(
    model='google/gemma-3-27b-it',
    messages=[{'role': 'user', 'content': 'Return JSON only: {"python_code": "result = 1+1"}'}],
)
print(result)
