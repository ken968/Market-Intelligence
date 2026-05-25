import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import lxml.etree as ET
from docx.oxml import parse_xml as docx_parse_xml
import latex2mathml.converter
import re

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
xslt_doc = ET.parse(XSL_PATH)
transform = ET.XSLT(xslt_doc)

latex = r"y_t = \frac{\text{Price}_{t + H} - \text{Price}_t}{\text{Price}_t}"
mathml_str = latex2mathml.converter.convert(latex)
mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))

omml_tree = transform(mathml_xml)
omml_bytes = ET.tostring(omml_tree, encoding="utf-8")
omml_str = omml_bytes.decode("utf-8")
omml_str = re.sub(r'<\?xml[^>]*\?>', '', omml_str).strip()

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Let's see what happens with the replace in export_to_docx.py
omml_str_replaced = omml_str
for tag in ['oMath', 'm:oMath']:
    if f'<{tag}' in omml_str_replaced and 'xmlns:m=' not in omml_str_replaced:
        omml_str_replaced = omml_str_replaced.replace(
            f'<{tag}',
            f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}"',
            1
        )

print("omml_str_replaced starts with:")
print(omml_str_replaced[:200])
print("omml_str_replaced ends with:")
print(omml_str_replaced[-100:])

try:
    docx_parse_xml(omml_str_replaced)
    print("Parsing succeeded!")
except Exception as e:
    print("Parsing failed:", e)
