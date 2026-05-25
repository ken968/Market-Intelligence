import lxml.etree as ET

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
tree = ET.parse(XSL_PATH)
root = tree.getroot()

templates = root.findall(".//{http://www.w3.org/1999/XSL/Transform}template")
print(f"Total templates: {len(templates)}")

for t in templates[:20]:
    match = t.get("match")
    name = t.get("name")
    mode = t.get("mode")
    print(f"Match: {match}, Name: {name}, Mode: {mode}")
