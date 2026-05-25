import shutil

src = r"d:\Market-Intelligence\scratch\temp_mml2omml.xsl"
dest = r"d:\Market-Intelligence\scripts\mml2omml.xsl"

try:
    shutil.copy(src, dest)
    print("Copied temp XSL to scripts/mml2omml.xsl successfully!")
except Exception as e:
    print("Error copying:", e)
