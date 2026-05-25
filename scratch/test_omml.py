import os
import sys

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import lxml.etree as ET
from docx.oxml import parse_xml as docx_parse_xml
import latex2mathml.converter

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
XSL_PATH = os.path.join(SCRIPT_DIR, "mml2omml.xsl")

print("XSL_PATH exists:", os.path.exists(XSL_PATH))
if os.path.exists(XSL_PATH):
    print("XSL_PATH size:", os.path.getsize(XSL_PATH))
    try:
        xslt_doc = ET.parse(XSL_PATH)
        transform = ET.XSLT(xslt_doc)
        print("XSLT parsed successfully.")
    except Exception as e:
        print("Failed to parse XSLT:", e)

latex = r"y_t = \frac{\text{Price}_{t + H} - \text{Price}_t}{\text{Price}_t}"
print("\nConverting LaTeX:", latex)

try:
    mathml_str = latex2mathml.converter.convert(latex)
    print("MathML:", mathml_str)
    mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))
    
    omml_tree = transform(mathml_xml)
    omml_bytes = ET.tostring(omml_tree, encoding="utf-8")
    omml_str = omml_bytes.decode("utf-8")
    print("Raw OMML:", omml_str[:300])
    
    # Strip xml decl
    import re
    omml_str = re.sub(r'<\?xml[^>]*\?>', '', omml_str).strip()
    
    # Check namespace
    M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    for tag in ['oMath', 'm:oMath']:
        if f'<{tag}' in omml_str and 'xmlns:m=' not in omml_str:
            omml_str = omml_str.replace(
                f'<{tag}',
                f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}"',
                1
            )
            
    print("Modified OMML:", omml_str[:300])
    
    element = docx_parse_xml(omml_str)
    print("docx_parse_xml succeeded! Element tag:", element.tag)
except Exception as e:
    import traceback
    print("Failed:")
    traceback.print_exc()
