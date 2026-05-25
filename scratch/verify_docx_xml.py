import docx

doc_path = r"d:\Market-Intelligence\scratch\test_math_out.docx"
doc = docx.Document(doc_path)
p = doc.paragraphs[0]

print("Paragraph text:", p.text)
print("Paragraph XML:")
xml_str = p._p.xml
print(xml_str[:1000])

# Check for oMath element inside paragraph XML
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
omath_elements = p._p.findall(f".//{{{M_NS}}}oMath")
print(f"Found {len(omath_elements)} oMath elements.")
