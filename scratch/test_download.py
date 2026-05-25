import requests

urls = [
    "https://raw.githubusercontent.com/MartinPaulEve/meTypeset/master/docx/utils/maths/mml2omml.xsl",
]

for url in urls:
    try:
        print(f"Fetching {url}...")
        r = requests.get(url, timeout=10)
        print("Status code:", r.status_code)
        if r.status_code == 200:
            print("Content length:", len(r.content))
            print("Preview:", r.text[:200])
    except Exception as e:
        print("Error:", e)
