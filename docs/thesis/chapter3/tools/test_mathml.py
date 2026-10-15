from lxml import etree

xsl = etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
transform = etree.XSLT(xsl)
mathml = """<math xmlns="http://www.w3.org/1998/Math/MathML">
<mrow><msub><mi>y</mi><mi>i</mi></msub><mo>:</mo><msub><mi>Ω</mi><mi>o</mi></msub><mo>→</mo><mi>ℝ</mi></mrow>
</math>"""
res = transform(etree.fromstring(mathml.encode("utf-8")))
print("root", res.getroot().tag if res.getroot() is not None else None)
print(etree.tostring(res.getroot(), encoding="unicode")[:1000])
