import requests

url = "https://raw.githubusercontent.com/MartinPaulEve/meTypeset/master/docx/utils/maths/mml2omml.xsl"
dest = r"d:\Market-Intelligence\scripts\mml2omml.xsl"

try:
    print(f"Downloading XSL from {url}...")
    r = requests.get(url, timeout=20)
    print("Status code:", r.status_code)
    if r.status_code == 200:
        with open(dest, "wb") as f:
            f.write(r.content)
        print("XSL saved successfully to", dest)
    else:
        print("Failed to download. Status:", r.status_code)
except Exception as e:
    print("Error:", e)
