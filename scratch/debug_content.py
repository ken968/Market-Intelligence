import lxml.etree as ET

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
mathml_str = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"><mrow><msub><mi>y</mi><mi>t</mi></msub><mo>&#x0003D;</mo><mfrac><mrow><msub><mtext>Price</mtext><mrow><mi>t</mi><mo>&#x0002B;</mo><mi>H</mi></mrow></msub><mo>&#x02212;</mo><msub><mtext>Price</mtext><mi>t</mi></msub></mrow><mrow><msub><mtext>Price</mtext><mi>t</mi></msub></mrow></mfrac></mrow></math>'

xslt_doc = ET.parse(XSL_PATH)
transform = ET.XSLT(xslt_doc)

mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))
omml_tree = transform(mathml_xml)

print("bytes(omml_tree) repr:")
print(repr(bytes(omml_tree)))

print("\nstr(omml_tree) repr:")
print(repr(str(omml_tree)))
