import os
from dotenv import load_dotenv
from utils.llm_manager import analyze_news_context

load_dotenv()
headlines = [
    "Stocks slide as rate fears grow",
    "Tech companies report strong earnings despite inflation",
    "Fed hints at possible rate hike in next meeting"
]
macro = "Inflation remains sticky. Yield curve is inverted."

res = analyze_news_context(headlines, macro)
print("Final result bias vector:", res['bias_vector'])
print("Final result narrative:", res['narrative'])
