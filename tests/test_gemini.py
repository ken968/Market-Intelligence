import os
from dotenv import load_dotenv
from utils.llm_manager import _call_gemini

if __name__ == "__main__":
    load_dotenv()
    res = _call_gemini("Test prompt. Return JSON with 'confidence' key = 0.5")
    print(res)

