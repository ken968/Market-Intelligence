import requests

url_urlsafe = "https://mermaid.ink/img/bWluZG1hcAogIHJvb3QoKE1hcmtldCBJbnRlbGxpZ2VuY2U8YnIvPlN5c3RlbSBUZXJtaW5hbCkpCiAgICAoIlN5c3RlbSBPbnRvbG9neTxici8-JiBUYXhvbm9teSIp?type=png"
url_std = "https://mermaid.ink/img/bWluZG1hcAogIHJvb3QoKE1hcmtldCBJbnRlbGxpZ2VuY2U8YnIvPlN5c3RlbSBUZXJtaW5hbCkpCiAgICAoIlN5c3RlbSBPbnRvbG9neTxici8+JiBUYXhvbm9teSIp"

try:
    r = requests.get(url_urlsafe)
    print("URL Safe content type:", r.headers.get("Content-Type"))
    print("URL Safe starts with PNG header:", r.content[:4] == b'\x89PNG')
    with open(r"d:\Market-Intelligence\scratch\mermaid_urlsafe.png", "wb") as f:
        f.write(r.content)
        
    r2 = requests.get(url_std)
    print("Std content type:", r2.headers.get("Content-Type"))
    print("Std starts with PNG header:", r2.content[:4] == b'\x89PNG')
    with open(r"d:\Market-Intelligence\scratch\mermaid_std.png", "wb") as f:
        f.write(r2.content)
except Exception as e:
    print("Error:", e)
