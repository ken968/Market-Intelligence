import lxml.etree as ET

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
mathml_str = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"><mrow><msub><mi>y</mi><mi>t</mi></msub><mo>&#x0003D;</mo><mfrac><mrow><msub><mtext>Price</mtext><mrow><mi>t</mi><mo>&#x0002B;</mo><mi>H</mi></mrow></msub><mo>&#x02212;</mo><msub><mtext>Price</mtext><mi>t</mi></msub></mrow><mrow><msub><mtext>Price</mtext><mi>t</mi></msub></mrow></mfrac></mrow></math>'

xslt_doc = ET.parse(XSL_PATH)
transform = ET.XSLT(xslt_doc)

mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))
omml_tree = transform(mathml_xml)

print("Type of omml_tree:", type(omml_tree))
print("Errors during transformation:", transform.error_log)

# Try serializing using str(omml_tree)
try:
    s = str(omml_tree)
    print("str(omml_tree) worked, length:", len(s))
except Exception as e:
    print("str(omml_tree) failed:", e)

# Try serializing using bytes(omml_tree)
try:
    b = bytes(omml_tree)
    print("bytes(omml_tree) worked, length:", len(b))
except Exception as e:
    print("bytes(omml_tree) failed:", e)

# Try using ET.tostring(omml_tree)
try:
    bts = ET.tostring(omml_tree, encoding="utf-8")
    if bts is None:
        print("ET.tostring(omml_tree) returned None")
    else:
        print("ET.tostring(omml_tree) worked, length:", len(bts))
except Exception as e:
    print("ET.tostring(omml_tree) failed:", e)

# Try using ET.tostring(omml_tree.getroot())
try:
    root = omml_tree.getroot()
    print("root type:", type(root))
    bts_root = ET.tostring(root, encoding="utf-8")
    print("ET.tostring(root) worked, length:", len(bts_root))
    print("Preview root:", bts_root[:300].decode("utf-8", errors="replace"))
except Exception as e:
    print("ET.tostring(root) failed:", e)

