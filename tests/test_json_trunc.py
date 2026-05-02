import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY_1'))
model = genai.GenerativeModel('gemini-2.5-flash')

prompt = """
OUTPUT FORMAT — respond ONLY with valid JSON, no extra text:
{
  "supply_shock_severity": 0.0,
  "geopolitical_stress": 0.0,
  "monetary_policy_hawkishness": 0.0,
  "risk_appetite": 0.0,
  "market_sentiment": 0.0,
  "confidence": 0.0,
  "dominant_regime": "describe in 3 words",
  "time_horizon_bias": "short_term|medium_term|long_term",
  "narrative": "2-3 sentence explanation of your assessment"
}
"""

print("Test 1: Normal")
resp1 = model.generate_content(prompt)
print(repr(resp1.text))

print("\nTest 2: With config")
resp2 = model.generate_content(prompt, generation_config={"temperature": 0.1, "max_output_tokens": 512, "response_mime_type": "application/json"})
print(repr(resp2.text))
