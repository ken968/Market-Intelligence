import lxml.etree as ET
import latex2mathml.converter
import re

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
xslt_doc = ET.parse(XSL_PATH)
transform = ET.XSLT(xslt_doc)

for latex in [r"\rho", r"y_t = \frac{\text{Price}_{t + H} - \text{Price}_t}{\text{Price}_t}"]:
    mathml_str = latex2mathml.converter.convert(latex)
    mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))
    
    omml_tree = transform(mathml_xml)
    omml_bytes = ET.tostring(omml_tree, encoding="utf-8")
    omml_str = omml_bytes.decode("utf-8")
    omml_str = re.sub(r'<\?xml[^>]*\?>', '', omml_str).strip()
    
    print(f"\nLaTeX: {latex}")
    print("Contains 'xmlns:m=':", 'xmlns:m=' in omml_str)
    print("Raw OMML starts with:")
    print(omml_str[:250])
