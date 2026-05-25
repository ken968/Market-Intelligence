import re
import base64
import requests

path = r"D:\Market-Intelligence\README.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Extract mermaid blocks
mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
print(f"Found {len(mermaid_blocks)} mermaid blocks.")

for i, block in enumerate(mermaid_blocks):
    print(f"\nBlock {i+1} preview (first 150 chars):")
    print(repr(block[:150]))
    
    clean = re.sub(r'^%%\{.*?\}%%\s*', '', block.strip(), flags=re.DOTALL)
    print("Cleaned preview (first 150 chars):")
    print(repr(clean[:150]))
    
    # Let's test different encodings
    # URL safe base64
    b64_urlsafe = base64.urlsafe_b64encode(clean.encode("utf-8")).decode("utf-8")
    url_urlsafe = f"https://mermaid.ink/img/{b64_urlsafe}?type=png"
    print("URL Safe URL Length:", len(url_urlsafe))
    try:
        r = requests.get(url_urlsafe, timeout=20)
        print("URL Safe Status:", r.status_code)
        print("URL Safe Content Length:", len(r.content))
        if r.status_code != 200:
            print("Response:", r.text[:200])
    except Exception as e:
        print("URL Safe Error:", e)

    # Standard B64
    b64_std = base64.b64encode(clean.encode("utf-8")).decode("utf-8")
    url_std = f"https://mermaid.ink/img/{b64_std}"
    print("Standard B64 URL Length:", len(url_std))
    try:
        r = requests.get(url_std, timeout=20)
        print("Standard Status:", r.status_code)
        print("Standard Content Length:", len(r.content))
        if r.status_code != 200:
            print("Response:", r.text[:200])
    except Exception as e:
        print("Standard Error:", e)
