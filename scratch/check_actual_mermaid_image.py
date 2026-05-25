import re
import base64
import requests

path = r"D:\Market-Intelligence\README.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
block = mermaid_blocks[0]
clean = re.sub(r'^%%\{.*?\}%%\s*', '', block.strip(), flags=re.DOTALL)

# Test urlsafe
b64_urlsafe = base64.urlsafe_b64encode(clean.encode("utf-8")).decode("utf-8")
url_urlsafe = f"https://mermaid.ink/img/{b64_urlsafe}?type=png"
r_urlsafe = requests.get(url_urlsafe)
print("URL Safe Status:", r_urlsafe.status_code)
print("URL Safe Content-Type:", r_urlsafe.headers.get("Content-Type"))
print("URL Safe starts with PNG:", r_urlsafe.content[:4] == b'\x89PNG')

with open(r"d:\Market-Intelligence\scratch\actual_urlsafe.png", "wb") as f:
    f.write(r_urlsafe.content)

# Test standard
b64_std = base64.b64encode(clean.encode("utf-8")).decode("utf-8")
url_std = f"https://mermaid.ink/img/{b64_std}"
r_std = requests.get(url_std)
print("Standard Status:", r_std.status_code)
print("Standard Content-Type:", r_std.headers.get("Content-Type"))
print("Standard starts with JPEG/PNG:", r_std.content[:4])

with open(r"d:\Market-Intelligence\scratch\actual_std.png", "wb") as f:
    f.write(r_std.content)
