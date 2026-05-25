import lxml.etree as ET
import re

XSL_PATH = r"d:\Market-Intelligence\scripts\mml2omml.xsl"
with open(XSL_PATH, "r", encoding="utf-8") as f:
    xsl_content = f.read()

# Let's uncomment the match="/" template
uncommented = xsl_content.replace(
    """<!--
  <xsl:template match="/">
    <oMath>
      <xsl:apply-templates mode="mml"  />
    </oMath>
  </xsl:template>
-->""",
    """<xsl:template match="/">
    <oMath>
      <xsl:apply-templates mode="mml"  />
    </oMath>
  </xsl:template>"""
)

# If the exact string replace didn't find it, let's use regex or a generic replace
if uncommented == xsl_content:
    print("Exact replacement failed, trying regex...")
    # Find <!-- <xsl:template match="/"> ... </xsl:template> -->
    uncommented = re.sub(
        r'<!--\s*(<xsl:template\s+match="/".*?</xsl:template>)\s*-->',
        r'\1',
        xsl_content,
        flags=re.DOTALL
    )

# Let's write the modified XSL to a temporary file
TEMP_XSL_PATH = r"d:\Market-Intelligence\scratch\temp_mml2omml.xsl"
with open(TEMP_XSL_PATH, "w", encoding="utf-8") as f:
    f.write(uncommented)

# Now test transformation with the new stylesheet
try:
    xslt_doc = ET.parse(TEMP_XSL_PATH)
    transform = ET.XSLT(xslt_doc)
    
    mathml_str = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline"><mrow><msub><mi>y</mi><mi>t</mi></msub><mo>&#x0003D;</mo><mfrac><mrow><msub><mtext>Price</mtext><mrow><mi>t</mi><mo>&#x0002B;</mo><mi>H</mi></mrow></msub><mo>&#x02212;</mo><msub><mtext>Price</mtext><mi>t</mi></msub></mrow><mrow><msub><mtext>Price</mtext><mi>t</mi></msub></mrow></mfrac></mrow></math>'
    mathml_xml = ET.fromstring(mathml_str.encode("utf-8"))
    
    omml_tree = transform(mathml_xml)
    omml_bytes = ET.tostring(omml_tree, encoding="utf-8")
    print("Success! OMML length:", len(omml_bytes))
    print("OMML preview:")
    # Replace non-cp1252 chars for printing
    print(omml_bytes[:1000].decode("utf-8", errors="replace"))
except Exception as e:
    import traceback
    traceback.print_exc()
