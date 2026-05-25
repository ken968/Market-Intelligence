import base64
import requests
import re

code = """
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '24px'}}}%%
mindmap
  root((Market Intelligence<br/>System Terminal))
    ("System Ontology<br/>& Taxonomy")
"""

clean = re.sub(r'^%%\{.*?\}%%\s*', '', code.strip(), flags=re.DOTALL)
print("Cleaned code:")
print(repr(clean))

# Test urlsafe_b64encode
b64_urlsafe = base64.urlsafe_b64encode(clean.encode("utf-8")).decode("utf-8")
url_urlsafe = f"https://mermaid.ink/img/{b64_urlsafe}?type=png"
print("\nURL Safe URL:", url_urlsafe)
try:
    r = requests.get(url_urlsafe, timeout=10)
    print("URL Safe Status:", r.status_code)
    print("URL Safe Content Length:", len(r.content))
except Exception as e:
    print("URL Safe Error:", e)

# Test standard b64encode
b64_std = base64.b64encode(clean.encode("utf-8")).decode("utf-8")
url_std = f"https://mermaid.ink/img/{b64_std}"
print("\nStandard B64 URL:", url_std)
try:
    r = requests.get(url_std, timeout=10)
    print("Standard Status:", r.status_code)
    print("Standard Content Length:", len(r.content))
except Exception as e:
    print("Standard Error:", e)

# Test standard b64encode with ?type=png
url_std_png = f"https://mermaid.ink/img/{b64_std}?type=png"
print("\nStandard B64 URL with ?type=png:", url_std_png)
try:
    r = requests.get(url_std_png, timeout=10)
    print("Standard+png Status:", r.status_code)
    print("Standard+png Content Length:", len(r.content))
except Exception as e:
    print("Standard+png Error:", e)
