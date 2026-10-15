from __future__ import annotations

import html
import re
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DOCX = ROOT / "submission.docx"
OUT = ROOT / "第3章 基于多尺度调制隐式神经表示的流场重建方法.docx"
MEDIA = REPO_ROOT / "outputs" / "chapter3" / "work_inspect" / "media"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
MATH_NS = "http://www.w3.org/1998/Math/MathML"


def e(text: str) -> str:
    return html.escape(text, quote=False)


def tag(name: str, *parts: str, **attrs: str) -> str:
    attr = "".join(f' {k}="{e(v)}"' for k, v in attrs.items())
    return f"<{name}{attr}>{''.join(parts)}</{name}>"


def mi(text: str) -> str:
    return tag("mi", e(text))


def mn(text: str | int) -> str:
    return tag("mn", e(str(text)))


def mo(text: str) -> str:
    return tag("mo", e(text))


def mt(text: str) -> str:
    return tag("mtext", e(text))


def row(*parts: str) -> str:
    return tag("mrow", *parts)


def sub(base: str, lower: str) -> str:
    return tag("msub", base, lower)


def sup(base: str, upper: str) -> str:
    return tag("msup", base, upper)


def subsup(base: str, lower: str, upper: str) -> str:
    return tag("msubsup", base, lower, upper)


def frac(num: str, den: str) -> str:
    return tag("mfrac", num, den)


def sqrt(expr: str) -> str:
    return tag("msqrt", expr)


def over(expr: str, accent: str) -> str:
    return tag("mover", expr, mo(accent), accent="true")


def under(expr: str, lower: str) -> str:
    return tag("munder", expr, lower)


def underover(expr: str, lower: str, upper: str) -> str:
    return tag("munderover", expr, lower, upper)


def parens(expr: str) -> str:
    return row(mo("("), expr, mo(")"))


def brackets(expr: str) -> str:
    return row(mo("["), expr, mo("]"))


def braces(expr: str) -> str:
    return row(mo("{"), expr, mo("}"))


def norm(expr: str) -> str:
    return row(mo("‖"), expr, mo("‖"))


def mathml(*parts: str) -> str:
    return f'<math xmlns="{MATH_NS}">{row(*parts)}</math>'


def fsub(name: str, lower: str) -> str:
    return sub(mi(name), row(*[mi(p) if p.isalpha() else mo(p) for p in re.split(r"([,])", lower) if p]))


class MathConverter:
    def __init__(self) -> None:
        self._transform = etree.XSLT(etree.parse(str(MML2OMML)))

    def omml(self, mml: str):
        root = etree.fromstring(mml.encode("utf-8"))
        result = self._transform(root)
        return parse_xml(etree.tostring(result.getroot(), encoding="unicode"))


CONVERTER = MathConverter()


INLINE: dict[str, str] = {}


def register_inline(expr: str, mml: str) -> None:
    INLINE[expr] = mml


def setup_inline_math() -> None:
    simple = {
        "Ω": mathml(mi("Ω")),
        "Ω_o": mathml(sub(mi("Ω"), mi("o"))),
        "S_i": mathml(sub(mi("S"), mi("i"))),
        "D_i": mathml(sub(mi("D"), mi("i"))),
        "R^2": mathml(sup(mi("ℝ"), mn(2))),
        "R^d": mathml(sup(mi("ℝ"), mi("d"))),
        "R^{d_z}": mathml(sup(mi("ℝ"), sub(mi("d"), mi("z")))),
        "R^{d_h}": mathml(sup(mi("ℝ"), sub(mi("d"), mi("h")))),
        "R^{d_out}": mathml(sup(mi("ℝ"), sub(mi("d"), mt("out")))),
        "R^{T×P}": mathml(sup(mi("ℝ"), row(mi("T"), mo("×"), mi("P")))),
        "x": mathml(mi("x")),
        "i": mathml(mi("i")),
        "j": mathml(mi("j")),
        "k": mathml(mi("k")),
        "ℓ": mathml(mi("ℓ")),
        "L": mathml(mi("L")),
        "H": mathml(mi("H")),
        "N": mathml(mi("N")),
        "T": mathml(mi("T")),
        "P": mathml(mi("P")),
        "z_i": mathml(sub(mi("z"), mi("i"))),
        "ẑ_i": mathml(sub(mi("ẑ"), mi("i"))),
        "tilde(z)_i": mathml(sub(over(mi("z"), "~"), mi("i"))),
        "y_i(x)": mathml(row(sub(mi("y"), mi("i")), parens(mi("x")))),
        "ŷ_i(x)": mathml(row(sub(mi("ŷ"), mi("i")), parens(mi("x")))),
        "u_o(x)": mathml(row(sub(mi("u"), mi("o")), parens(mi("x")))),
        "v_o(x)": mathml(row(sub(mi("v"), mi("o")), parens(mi("x")))),
        "F_Θ(·)": mathml(row(sub(mi("ℱ"), mi("Θ")), parens(mo("·")))),
        "Θ": mathml(mi("Θ")),
        "Z": mathml(mi("Z")),
        "d_z": mathml(sub(mi("d"), mi("z"))),
        "d_p": mathml(sub(mi("d"), mi("p"))),
        "d_h": mathml(sub(mi("d"), mi("h"))),
        "d_out": mathml(sub(mi("d"), mt("out"))),
        "w_k^(g)": mathml(subsup(mi("w"), mi("k"), parens(mi("g")))),
        "w_k^(f)": mathml(subsup(mi("w"), mi("k"), parens(mi("f")))),
        "w_k^(l)": mathml(subsup(mi("w"), mi("k"), parens(mi("l")))),
        "b_k^(g)": mathml(subsup(mi("b"), mi("k"), parens(mi("g")))),
        "b_k^(l)": mathml(subsup(mi("b"), mi("k"), parens(mi("l")))),
        "μ_k": mathml(sub(mi("μ"), mi("k"))),
        "γ_k": mathml(sub(mi("γ"), mi("k"))),
        "α_f": mathml(sub(mi("α"), mi("f"))),
        "σ_ℓ": mathml(sub(mi("σ"), mi("ℓ"))),
        "σ_0": mathml(sub(mi("σ"), mn(0))),
        "f_mix(x,z_i)": mathml(row(sub(mi("f"), mt("mix")), parens(row(mi("x"), mo(","), sub(mi("z"), mi("i")))))),
        "f_ℓ(x,ẑ_i)": mathml(row(sub(mi("f"), mi("ℓ")), parens(row(mi("x"), mo(","), sub(mi("ẑ"), mi("i")))))),
        "s̃_ℓ,k": mathml(sub(over(mi("s"), "~"), row(mi("ℓ"), mo(","), mi("k")))),
        "M_ℓ(·)": mathml(row(sub(mi("M"), mi("ℓ")), parens(mo("·")))),
        "h_ℓ": mathml(sub(mi("h"), mi("ℓ"))),
        "v_ℓ": mathml(sub(mi("v"), mi("ℓ"))),
        "W_ℓ^(h)": mathml(subsup(mi("W"), mi("ℓ"), parens(mi("h")))),
        "W_ℓ^(z)": mathml(subsup(mi("W"), mi("ℓ"), parens(mi("z")))),
        "W_out": mathml(sub(mi("W"), mt("out"))),
        "b_out": mathml(sub(mi("b"), mt("out"))),
        "C_z": mathml(sub(mi("C"), mi("z"))),
        "C_y": mathml(sub(mi("C"), mi("y"))),
        "λ_z": mathml(sub(mi("λ"), mi("z"))),
        "ℓ_2": mathml(sub(mi("ℓ"), mn(2))),
        "p_lat": mathml(sub(mi("p"), mt("lat"))),
        "10^-2": mathml(sup(mn(10), row(mo("-"), mn(2)))),
        "s∈{5%,25%,50%}": mathml(row(mi("s"), mo("∈"), braces(row(mn(5), mo("%"), mo(","), mn(25), mo("%"), mo(","), mn(50), mo("%"))))),
        "MSE(C_z,C_y)": mathml(row(mt("MSE"), parens(row(sub(mi("C"), mi("z")), mo(","), sub(mi("C"), mi("y")))))),
        "z_{t_k}^{(d_z)}": mathml(subsup(mi("z"), sub(mi("t"), mi("k")), parens(sub(mi("d"), mi("z"))))),
        "k=1,…,T": mathml(row(mi("k"), mo("="), mn(1), mo(","), mo("…"), mo(","), mi("T"))),
        "Z∈R^{T×d_z}": mathml(row(mi("Z"), mo("∈"), sup(mi("ℝ"), row(mi("T"), mo("×"), sub(mi("d"), mi("z")))))),
        "Y∈R^{T×P}": mathml(row(mi("Y"), mo("∈"), sup(mi("ℝ"), row(mi("T"), mo("×"), mi("P"))))),
        "NMSE": mathml(mt("NMSE")),
        "MSE": mathml(mt("MSE")),
    }
    INLINE.update(simple)


def inline_math(expr: str) -> str:
    return INLINE.get(expr, mathml(mt(expr)))


def set_cell_text(cell, text: str, bold: bool = False, underline: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run)
    run.bold = bold
    run.underline = underline


def set_run_font(run, size: float | None = 10.5, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def set_paragraph_font(paragraph, size: float = 10.5) -> None:
    for run in paragraph.runs:
        set_run_font(run, size=size)


def add_math_to_paragraph(paragraph, mml: str) -> None:
    paragraph._p.append(CONVERTER.omml(mml))


def add_text_with_math(paragraph, text: str, size: float = 10.5) -> None:
    parts = re.split(r"(\$[^$]+\$)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            add_math_to_paragraph(paragraph, inline_math(part[1:-1]))
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size)


def add_paragraph(doc: Document, text: str = "", style: str | None = None, align=None):
    paragraph = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if align is not None:
        paragraph.alignment = align
    add_text_with_math(paragraph, text)
    return paragraph


def add_heading(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_heading(level=level)
    run = paragraph.add_run(text)
    set_run_font(run, size=16 if level == 1 else 14 if level == 2 else 12, bold=True)


def add_caption(doc: Document, text: str) -> None:
    paragraph = add_paragraph(doc, text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in paragraph.runs:
        run.italic = False


def add_picture(doc: Document, image_name: str, width: float = 6.3) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(MEDIA / image_name), width=Inches(width))


def add_formula(doc: Document, lines: str | list[str], number: int) -> None:
    if isinstance(lines, str):
        lines = [lines]
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.allow_autofit = False
    table.cell(0, 0).width = Inches(5.7)
    table.cell(0, 1).width = Inches(0.7)
    formula_cell = table.cell(0, 0)
    number_cell = table.cell(0, 1)
    formula_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    number_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    formula_cell.text = ""
    for idx, line in enumerate(lines):
        p = formula_cell.paragraphs[0] if idx == 0 else formula_cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_math_to_paragraph(p, line)
    number_cell.text = ""
    pnum = number_cell.paragraphs[0]
    pnum.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = pnum.add_run(f"({number})")
    set_run_font(run)
    for row_ in table.rows:
        for cell in row_.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
    # Remove visible borders so the layout behaves like a displayed equation.
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        elem = OxmlElement(f"w:{edge}")
        elem.set(qn("w:val"), "nil")
        borders.append(elem)
    tbl_pr.append(borders)


def create_formulas() -> dict[int, str | list[str]]:
    x = mi("x")
    zi = sub(mi("z"), mi("i"))
    zhat_i = sub(mi("ẑ"), mi("i"))
    yi = sub(mi("y"), mi("i"))
    yhat_i = sub(mi("ŷ"), mi("i"))
    theta_f = sub(mi("ℱ"), mi("Θ"))
    fj = sub(mi("x"), mi("j"))
    yixj = row(yi, parens(fj))
    in_si = row(fj, mo("∈"), sub(mi("S"), mi("i")))
    di_set = row(sub(mi("D"), mi("i")), mo("="), braces(row(parens(row(fj, mo(","), yixj)), mo("|"), in_si)))
    return {
        1: mathml(row(yi, mo(":"), sub(mi("Ω"), mi("o")), mo("→"), mi("ℝ"))),
        2: mathml(di_set),
        3: mathml(row(mt("current"), parens(x), mo("="), sqrt(row(sup(row(sub(mi("u"), mi("o")), parens(x)), mn(2)), mo("+"), sup(row(sub(mi("v"), mi("o")), parens(x)), mn(2)))))),
        4: mathml(row(mt("current"), mo("∈"), sup(mi("ℝ"), row(mn(731), mo("×"), mn(69), mo("×"), mn(117))))),
        5: mathml(row(mi("s"), mo("∈"), braces(row(mn(5), mo("%"), mo(","), mn(25), mo("%"), mo(","), mn(50), mo("%"))))),
        6: mathml(row(yhat_i, parens(x), mo("="), theta_f, parens(row(x, mo(","), zhat_i)))),
        7: mathml(row(mi("Z"), mo("="), braces(row(sub(mi("z"), mn(1)), mo(","), sub(mi("z"), mn(2)), mo(","), mo("…"), mo(","), sub(mi("z"), mi("N")))), mo(","), zi, mo("∈"), sup(mi("ℝ"), sub(mi("d"), mi("z"))))),
        8: mathml(row(sub(mi("D"), mi("i")), mo("="), braces(row(parens(row(fj, mo(","), yixj)), mo(":"), in_si)))),
        9: mathml(row(under(mt("min"), row(mi("Θ"), mo(","), mi("Z"))), mo(" "), mi("ℒ"), mo("="), underover(mo("∑"), row(mi("i"), mo("="), mn(1)), mi("N")), under(mo("∑"), row(fj, mo("∈"), sub(mi("S"), mi("i")))), sup(norm(row(theta_f, parens(row(fj, mo(","), zhat_i)), mo("-"), yixj)), mn(2)))),
        10: [
            mathml(row(sub(over(mi("z"), "~"), mi("i")), mo("="), mi("m"), mo("⊙"), zi, mo(","))),
            mathml(row(mi("m"), mo("∼"), mt("Bernoulli"), parens(mi("p")), mo("."))),
        ],
        11: [
            mathml(row(zhat_i, mo("="), sup(mi("P"), mo("⊤")), sub(over(mi("z"), "~"), mi("i")), mo(","))),
            mathml(row(mi("P"), mo("∈"), sup(mi("ℝ"), row(sub(mi("d"), mi("z")), mo("×"), sub(mi("d"), mi("p")))), mo("."))),
        ],
        12: mathml(row(sub(mi("D"), mi("k")), parens(x), mo("="), sup(norm(x), mn(2)), mo("+"), sup(norm(sub(mi("μ"), mi("k"))), mn(2)), mo("-"), mn(2), sup(x, mo("⊤")), sub(mi("μ"), mi("k")))),
        13: mathml(row(sub(mi("ϕ"), mi("k")), parens(x), mo("="), sub(mi("α"), mi("f")), sup(parens(subsup(mi("w"), mi("k"), parens(mi("f")))), mo("⊤")), x)),
        14: [
            mathml(row(sub(mt("f"), row(mt("Gabor"), mo(","), mi("k"))), parens(x), mo("="), mt("sin"), parens(row(sup(parens(subsup(mi("w"), mi("k"), parens(mi("g")))), mo("⊤")), x, mo("+"), subsup(mi("b"), mi("k"), parens(mi("g"))))))),
            mathml(row(mo("·"), mt("exp"), parens(row(mo("-"), frac(mn(1), mn(2)), sub(mi("γ"), mi("k")), sub(mi("D"), mi("k")), parens(x))))),
        ],
        15: mathml(row(sub(mt("f"), row(mt("Fourier"), mo(","), mi("k"))), parens(x), mo("="), brackets(row(mt("sin"), parens(row(sub(mi("ϕ"), mi("k")), parens(x))), mo(","), mt("cos"), parens(row(sub(mi("ϕ"), mi("k")), parens(x))))))),
        16: mathml(row(sub(mt("f"), row(mt("Linear"), mo(","), mi("k"))), parens(x), mo("="), sup(parens(subsup(mi("w"), mi("k"), parens(mi("l")))), mo("⊤")), x, mo("+"), subsup(mi("b"), mi("k"), parens(mi("l"))))),
        17: [
            mathml(row(sub(mt("f"), mt("Gabor")), parens(x), mo("="), sup(brackets(row(sub(mi("f"), row(mi("G"), mo(","), mn(1))), parens(x), mo(","), mo("…"), mo(","), sub(mi("f"), row(mi("G"), mo(","), mi("H"))), parens(x))), mo("⊤")))),
            mathml(row(sub(mt("f"), mt("Fourier")), parens(x), mo("="), sup(brackets(row(sub(mi("f"), row(mi("F"), mo(","), mn(1))), parens(x), mo(","), mo("…"), mo(","), sub(mi("f"), row(mi("F"), mo(","), mi("H"))), parens(x))), mo("⊤")))),
            mathml(row(sub(mt("f"), mt("Linear")), parens(x), mo("="), sup(brackets(row(sub(mi("f"), row(mi("L"), mo(","), mn(1))), parens(x), mo(","), mo("…"), mo(","), sub(mi("f"), row(mi("L"), mo(","), mi("H"))), parens(x))), mo("⊤")))),
        ],
        18: [
            mathml(row(sub(mi("w"), mi("i")), mo("="), brackets(row(sub(mi("w"), mi("g")), mo(","), sub(mi("w"), mi("f")), mo(","), sub(mi("w"), mi("l")))), mo("="), mt("Softmax"), parens(row(mi("𝒢"), parens(zhat_i))), mo(","))),
            mathml(row(sub(mi("w"), mi("g")), mo("+"), sub(mi("w"), mi("f")), mo("+"), sub(mi("w"), mi("l")), mo("="), mn(1), mo("."))),
        ],
        19: [
            mathml(row(sub(mt("f"), mt("mix")), parens(row(x, mo(","), zhat_i)), mo("="), sub(mi("w"), mi("g")), parens(zhat_i), sub(mt("f"), mt("Gabor")), parens(x))),
            mathml(row(mo("+"), sub(mi("w"), mi("f")), parens(zhat_i), sub(mt("f"), mt("Fourier")), parens(x), mo("+"), sub(mi("w"), mi("l")), parens(zhat_i), sub(mt("f"), mt("Linear")), parens(x))),
        ],
        20: [
            mathml(row(sub(mt("f"), mt("mix")), parens(row(x, mo(","), zhat_i)), mo("="), sub(mi("w"), mi("g")), parens(zhat_i), sub(mt("f"), row(mt("Gabor"), mo(","), mi("k"))), parens(x))),
            mathml(row(mo("+"), sub(mi("w"), mi("f")), parens(zhat_i), sub(mt("f"), row(mt("Fourier"), mo(","), mi("k"))), parens(x), mo("+"), sub(mi("w"), mi("l")), parens(zhat_i), sub(mt("f"), row(mt("Linear"), mo(","), mi("k"))), parens(x))),
        ],
        21: mathml(row(sub(mi("σ"), mi("ℓ")), mo("="), frac(sub(mi("σ"), mn(0)), sqrt(row(mi("ℓ"), mo("+"), mn(1)))))),
        22: mathml(row(sub(mi("h"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("∈"), sup(mi("ℝ"), sub(mi("d"), mi("h"))), mo(","), mi("ℓ"), mo("="), mn(0), mo(","), mo("…"), mo(","), mi("L"), mo("-"), mn(1), mo(","))),
        23: mathml(row(sub(mi("h"), mn(0)), parens(row(x, mo(","), zhat_i)), mo("="), mn(0), mo("."))),
        24: [
            mathml(row(sub(mi("s"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), subsup(mi("W"), mi("ℓ"), parens(mi("h"))), sub(mi("h"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("+"), subsup(mi("W"), mi("ℓ"), parens(mi("z"))), zhat_i, mo("+"), sub(mi("b"), mi("ℓ")), mo(","))),
            mathml(row(sub(mi("s"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("∈"), sup(mi("ℝ"), sub(mi("d"), mi("h"))), mo("."))),
        ],
        25: mathml(row(sub(over(mi("s"), "~"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), mi("ϕ"), parens(row(sub(mi("s"), mi("ℓ")), parens(row(x, mo(","), zhat_i)))))),
        26: mathml(row(sub(mi("u"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), sub(over(mi("s"), "~"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("⊙"), sub(mi("f"), mi("ℓ")), parens(row(x, mo(","), zhat_i)))),
        27: mathml(row(brackets(row(sub(mi("γ"), mi("ℓ")), mo(","), sub(mi("β"), mi("ℓ")))), mo("="), sub(mi("𝓜"), mi("ℓ")), parens(zhat_i), mo(","), sub(mi("γ"), mi("ℓ")), mo(","), sub(mi("β"), mi("ℓ")), mo("∈"), sup(mi("ℝ"), sub(mi("d"), mi("h"))))),
        28: [
            mathml(row(sub(mi("v"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), parens(row(mn(1), mo("+"), sub(mi("γ"), mi("ℓ")))), mo("⊙"), sub(mi("u"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("+"), sub(mi("β"), mi("ℓ")), mo(","))),
            mathml(row(sub(mi("v"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("∈"), sup(mi("ℝ"), sub(mi("d"), mi("h"))), mo("."))),
        ],
        29: mathml(row(sub(over(mi("v"), "~"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), sub(mi("α"), mi("ℓ")), mo("⊙"), sub(mi("v"), mi("ℓ")), parens(row(x, mo(","), zhat_i)))),
        30: mathml(row(sub(mi("m"), mi("ℓ")), mo("∼"), mt("Bernoulli"), parens(sub(mi("p"), mi("ℓ"))))),
        31: mathml(row(sub(over(mi("v"), "¯"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("="), sub(mi("m"), mi("ℓ")), sub(over(mi("v"), "~"), mi("ℓ")), parens(row(x, mo(","), zhat_i)))),
        32: mathml(row(sub(over(mi("h"), "~"), row(mi("ℓ"), mo("+"), mn(1))), parens(row(x, mo(","), zhat_i)), mo("="), sub(mi("h"), mi("ℓ")), parens(row(x, mo(","), zhat_i)), mo("+"), sub(over(mi("v"), "¯"), mi("ℓ")), parens(row(x, mo(","), zhat_i)))),
        33: mathml(row(sub(mi("h"), row(mi("ℓ"), mo("+"), mn(1))), parens(row(x, mo(","), zhat_i)), mo("="), mt("LayerNorm"), parens(row(sub(over(mi("h"), "~"), row(mi("ℓ"), mo("+"), mn(1))), parens(row(x, mo(","), zhat_i)))))),
        34: mathml(row(mi("ŷ"), parens(row(x, mo(","), mi("i"))), mo("="), sub(mi("W"), mt("out")), sub(mi("h"), mi("L")), parens(row(x, mo(","), zhat_i)), mo("+"), sub(mi("b"), mt("out")))),
        35: mathml(row(yhat_i, parens(x), mo("="), theta_f, parens(row(x, mo(","), zhat_i)))),
        36: mathml(row(sub(mi("D"), mi("i")), mo("="), braces(row(parens(row(fj, mo(","), yixj)), mo("|"), in_si)), mo(","))),
        37: mathml(row(under(mt("min"), row(mi("Θ"), mo(","), mi("Z"))), mo(" "), sub(mi("ℒ"), mt("rec")), mo("="), underover(mo("∑"), row(mi("i"), mo("="), mn(1)), mi("N")), under(mo("∑"), row(fj, mo("∈"), sub(mi("S"), mi("i")))), sup(norm(row(theta_f, parens(row(fj, mo(","), zhat_i)), mo("-"), yixj)), mn(2)))),
        38: mathml(row(mi("ℒ"), mo("="), sub(mi("ℒ"), mt("rec")), mo("+"), sub(mi("λ"), mi("z")), underover(mo("∑"), row(mi("i"), mo("="), mn(1)), mi("N")), sup(zi, mo("⊤")), zi)),
        39: mathml(row(mt("MSE"), mo("="), frac(mn(1), row(mo("|"), sub(mi("Ω"), mi("o")), mo("|"))), under(mo("∑"), row(x, mo("∈"), sub(mi("Ω"), mi("o")))), sup(parens(row(yhat_i, parens(x), mo("-"), yi, parens(x))), mn(2)), mo("."))),
        40: mathml(row(mt("Improvement"), mo("="), frac(row(sub(mt("MSE"), mt("2nd-best")), mo("-"), sub(mt("MSE"), mt("M2GNet"))), sub(mt("MSE"), mt("2nd-best"))))),
        41: mathml(row(sub(mi("d"), mi("z")), mo("∈"), braces(row(mn(1), mo(","), mn(2), mo(","), mn(4), mo(","), mn(8), mo(","), mn(16), mo(","), mn(32), mo(","), mn(64), mo(","), mn(128), mo(","), mn(256), mo(","), mn(512))))),
        42: mathml(row(mt("MSE"), parens(row(sub(mi("C"), mi("z")), mo(","), sub(mi("C"), mi("y")))), mo(","))),
    }


def add_table_1(doc: Document) -> None:
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["模型", "任务1", "任务2", "任务3", "任务4"]
    for idx, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[idx], h, bold=True)

    blocks = [
        (
            "s=5%",
            [
                ("ResMLP", "2.81×10^-2", "2.87×10^-2", "2.56×10^-2", "2.56×10^-2", False, False),
                ("FFN-P", "3.11×10^-2", "3.23×10^-2", "2.31×10^-2", "2.30×10^-2", False, False),
                ("FFN-G", "4.32×10^-2", "4.47×10^-2", "3.49×10^-2", "3.45×10^-2", False, False),
                ("SIREN", "3.06×10^-2", "3.17×10^-2", "2.70×10^-2", "3.95×10^-2", False, False),
                ("MMGN", "1.53×10^-2", "1.90×10^-2", "1.84×10^-2", "2.01×10^-2", False, True),
                ("M2GNet", "1.47×10^-2", "1.74×10^-2", "1.14×10^-2", "1.51×10^-2", True, False),
                ("提升幅度", "3.90%", "8.42%", "38.04%", "24.50%", False, False),
            ],
        ),
        (
            "s=25%",
            [
                ("ResMLP", "2.26×10^-2", "2.26×10^-2", "2.20×10^-2", "2.10×10^-2", False, False),
                ("FFN-P", "1.28×10^-2", "1.38×10^-2", "1.02×10^-2", "1.04×10^-2", False, False),
                ("FFN-G", "2.57×10^-2", "2.88×10^-2", "2.14×10^-2", "2.17×10^-2", False, False),
                ("SIREN", "2.35×10^-2", "2.67×10^-2", "2.01×10^-2", "2.73×10^-2", False, False),
                ("MMGN", "7.28×10^-3", "8.19×10^-3", "6.14×10^-3", "7.20×10^-3", False, True),
                ("M2GNet", "5.63×10^-3", "6.78×10^-3", "4.42×10^-3", "5.50×10^-3", True, False),
                ("提升幅度", "22.66%", "17.20%", "28.01%", "23.61%", False, False),
            ],
        ),
        (
            "s=50%",
            [
                ("ResMLP", "2.02×10^-2", "2.03×10^-2", "1.93×10^-2", "1.85×10^-2", False, False),
                ("FFN-P", "8.59×10^-3", "9.18×10^-3", "8.58×10^-3", "8.58×10^-3", False, False),
                ("FFN-G", "1.88×10^-2", "1.93×10^-2", "1.86×10^-2", "1.80×10^-2", False, False),
                ("SIREN", "2.67×10^-2", "3.44×10^-2", "2.17×10^-2", "2.16×10^-2", False, False),
                ("MMGN", "5.15×10^-3", "5.66×10^-3", "5.26×10^-3", "5.64×10^-3", False, True),
                ("M2GNet", "3.92×10^-3", "4.36×10^-3", "3.07×10^-3", "3.69×10^-3", True, False),
                ("提升幅度", "23.88%", "22.97%", "41.63%", "34.57%", False, False),
            ],
        ),
    ]
    for label, rows in blocks:
        row_cells = table.add_row().cells
        merged = row_cells[0].merge(row_cells[4])
        set_cell_text(merged, label, bold=True)
        for row_data in rows:
            cells = table.add_row().cells
            name, *values, bold, underline = row_data
            set_cell_text(cells[0], name, bold=name in {"M2GNet", "提升幅度"})
            for idx, value in enumerate(values, start=1):
                set_cell_text(cells[idx], value, bold=bold and name == "M2GNet", underline=underline)
    for row_ in table.rows:
        for cell in row_.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def extract_media() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_DOCX) as zf:
        for name in zf.namelist():
            if name.startswith("word/media/"):
                (MEDIA / Path(name).name).write_bytes(zf.read(name))


def build_doc() -> None:
    setup_inline_math()
    extract_media()
    formulas = create_formulas()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("第3章 基于多尺度调制隐式神经表示的流场重建方法")
    set_run_font(run, size=18, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text_with_math(p, "原题：基于多尺度隐式神经表示的连续海洋流场重建")

    add_paragraph(doc, "作者：Yingjie Mei，Shuo Jin，Jing Hao，Zuotao Ni，Jingbo Jiang，Tao Jin，Hongjuan Zhou，Zhiquan Zhou，Chenxu Wang，Yunzhou Li，Juncheng Wang", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "单位：中国科学院海洋研究所（青岛 266071）；哈尔滨工业大学（威海 264200）；青岛市生态环境科学研究院（青岛 266003）；山东省科学院海洋仪器仪表研究所（青岛 266001）", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph(doc, "通信作者邮箱：jintao@hit.edu.cn；myj@hit.edu.cn；jiangjingbo@qdio.ac.cn", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_heading(doc, "亮点", 2)
    highlights = [
        "M2GNet 可由稀疏观测重建连续海洋流速场。",
        "潜在门控混合频率编码能够适应多尺度流场结构。",
        "上下文条件化 FiLM 模块提升了模型对时变采样的鲁棒性。",
        "渤海实验表明，该方法在四类采样场景下均取得稳定的 MSE 改进。",
        "模型支持在有效海洋网格点上的任意坐标查询。",
    ]
    for item in highlights:
        p = doc.add_paragraph(style="List Bullet")
        add_text_with_math(p, item)

    add_heading(doc, "摘要", 2)
    add_paragraph(doc, "在近岸海域，从稀疏、异构且时变的观测中重建高分辨率海洋流场仍然十分困难，因为多尺度动力过程、不规则采样以及复杂海陆边界会相互耦合。现有统计方法、低秩方法和固定网格深度学习方法往往会过度平滑小尺度结构，依赖预定义网格，或在观测布局变化时性能下降。本文提出 M2GNet，一种用于连续海洋表层流速重建的上下文条件化隐式神经场。该模型结合样本级上下文嵌入、潜在门控混合频率编码、乘性交互与层级 FiLM 调制，以及稳定化残差堆叠。通过上下文依赖门控融合 Gabor、Fourier 和线性谱分支，M2GNet 能够自适应表示大尺度环流和局地高梯度结构。实验使用由 Copernicus Marine 产品构建的渤海日平均表层流速数据，时间跨度为 731 天，空间分辨率约为 1/12°。在四种观测随机性场景和三种稀疏度水平下，M2GNet 在代表性隐式神经表示基线中取得最低 MSE。相较最强基线，该模型在固定布局、5% 采样情形下将误差降低 3.9%，并在观测位置和数量随机变化时获得更大提升，说明其适用于稀疏且不规则的海洋观测网络。")
    add_heading(doc, "关键词", 2)
    add_paragraph(doc, "隐式神经表示；海洋流场重建；稀疏不规则采样；多尺度谱表示；基于坐标的深度学习")

    add_heading(doc, "3.1 引言", 1)
    add_paragraph(doc, "从稀疏观测中重建连续、高分辨率海洋流场，是物理海洋学、动力过程分析、数值预报评估和海洋工程规划中的基础问题。实际观测网络受传感器布设数量、恶劣海况、平台漂移、质量控制缺口和卫星覆盖不完整等因素限制，通常呈现稀疏、异构和时变特征。在具有复杂海陆边界、锋面和多尺度涡旋活动的近岸海域，这些限制会造成明显的时空盲区，使动力一致的流场重建更具挑战性。")
    add_paragraph(doc, "现有方法大体可分为物理模型与资料同化，以及统计插值或低维分解两类。物理方法能够给出具有动力一致性的估计，但对模型配置、参数化方案和边界条件较为敏感，计算代价也通常较高。统计插值和低秩方法依赖平稳协方差或低秩假设，在强非平稳、强非线性动力过程中这些假设往往失效，从而导致过度平滑和细尺度结构损失。近年来，数据驱动的海洋重建方法在次表层热场、四维温度临近预报以及稀疏观测融合等方面发展迅速，但固定网格假设仍然限制了其对采样变化和连续坐标查询的鲁棒性。")
    add_paragraph(doc, "隐式神经表示（INR）通过学习从时空坐标到物理量的连续映射，提供了一种替代范式，天然支持任意点评估与插值。然而，将 INR 直接用于真实海洋流场重建仍存在困难。首先，许多 INR 以逐样本拟合方式训练，缺少对样本级上下文的显式建模，限制了跨时间和跨观测场景的泛化能力。其次，单一形式的频率编码难以同时表示大尺度背景流和小尺度涡旋结构。第三，有限的条件调制削弱了模型对不同流动状态和观测稀疏度的适应性；而当高频表示与深层残差传播结合时，深层 INR 架构还可能面临训练不稳定问题。")
    add_paragraph(doc, "为解决上述问题，本文提出 M2GNet，这是一种面向稀疏和不规则观测下海洋流场重建的条件隐式神经场。M2GNet 集成四个关键组件：用于捕获大尺度状态信息的样本级上下文嵌入、用于增强多尺度表达能力的混合频率谱编码、用于实现上下文依赖频率选择的乘性交互与层级条件调制，以及用于提升训练鲁棒性的稳定化深度残差堆叠。这些组件使 M2GNet 能够在不同空间尺度和采样条件下自适应表示异构流场结构。实验基于 Copernicus Marine 产品构建的渤海表层流速数据集，覆盖 2022 至 2024 年共 731 天，空间分辨率约为 1/12°。结果表明，M2GNet 在所有测试场景中均优于代表性 INR 基线，尤其在高随机性和高稀疏性采样下提升更为明显。")

    add_heading(doc, "3.2 问题形式化与数据集", 1)
    add_heading(doc, "3.2.1 问题形式化", 2)
    add_paragraph(doc, "令 $Ω⊂R^2$ 表示在规则经纬度网格上离散化的目标空间域，令 $Ω_o$ 表示由海陆掩膜确定的有效海洋网格点集合。对于第 $i$ 个时空样本（本文中为第 $i$ 天，$i=1,…,N$），目标标量场记为")
    add_formula(doc, formulas[1], 1)
    add_paragraph(doc, "其中 $y_i(x)$ 表示位置 $x∈Ω_o$ 处的海表流速，具体定义见第 3.2.2 节。")
    add_paragraph(doc, "在稀疏观测条件下，仅有一部分位置被观测到。令 $S_i⊂Ω_o$ 为样本 $i$ 的观测位置集合，并将稀疏观测集定义为")
    add_formula(doc, formulas[2], 2)
    add_paragraph(doc, "目标是由 $D_i$ 重建连续函数 $ŷ_i(x)$，使其能够在任意 $x∈Ω_o$ 处查询，并在完整海洋网格上以参考场 $y_i(x)$ 作为基准进行评估。需要注意的是，观测集合 $S_i$ 的点数和空间位置都可能随时间变化，因此可用于系统评估不同采样不确定性下的重建能力。")

    add_heading(doc, "3.2.2 数据集描述与预处理", 2)
    add_paragraph(doc, "本文在一个具有代表性的海洋表层流数据集上评估所提出模型。该数据集来自 Copernicus Marine Environment Monitoring Service（CMEMS）全球物理海流产品。我们使用日平均值构建渤海及邻近海域的表层流场，纬度范围为 35.50°N-41.17°N，经度范围为 117.17°E-126.83°E，空间分辨率约为 1/12°（纬向和经向步长均为 0.0833°）。时间跨度为 2022-09-01 至 2024-08-31，共 $N$=731 天。")
    add_paragraph(doc, "原始产品提供两个正交速度分量，分别记为 $u_o(x)$（纬向）和 $v_o(x)$（经向）。本文重建标量流速幅值")
    add_formula(doc, formulas[3], 3)
    add_paragraph(doc, "陆地点采用零填充，所选子区域中的海洋网格比例约为 0.503。由此得到的数据张量大小为")
    add_formula(doc, formulas[4], 4)
    add_paragraph(doc, "其中第一维对应时间，另外两维分别对应纬度和经度网格。该数据集具有明显季节变化、多尺度涡旋结构和复杂近岸边界，是检验稀疏观测神经场重建方法的挑战性试验平台。")

    add_heading(doc, "3.2.3 观测场景与稀疏度设置", 2)
    add_paragraph(doc, "为系统评估不同观测不确定性下的重建性能，本文构建四个采样随机性逐渐增强的任务：")
    for item in [
        "任务1（固定数量-固定位置）：所有时间步使用相同数量且位于固定网格位置的观测点，表示稳定传感器布局。",
        "任务2（随机数量-固定位置）：观测位置固定，但可用观测数量随时间随机变化，用于模拟仪器故障或质量控制导致的缺测。",
        "任务3（固定数量-随机位置）：每个时间步的观测数量固定，但观测位置随时间随机采样，用于模拟浮标或船载观测等漂移平台。",
        "任务4（随机数量-随机位置）：每个时间步的观测数量和位置均随机变化，代表最困难的情形，也近似长期业务化观测系统中的不确定性。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        add_text_with_math(p, item)
    add_paragraph(doc, "对于每个任务，进一步考虑三个稀疏度水平。给定某一天 $i$，观测点从有效海洋集合 $Ω_o$ 中按采样比例")
    add_formula(doc, formulas[5], 5)
    add_paragraph(doc, "进行抽取，即训练时仅保留 $Ω_o$ 中 5%、25% 或 50% 的有效海洋网格点作为观测，其余位置被视为缺失。评估时，重建误差在完整海洋网格上计算（等价于评估阶段 $s$=100%）。这种实验设计能够区分稀疏性（观测多少点）和采样不确定性（观测位于何处以及如何分布）的影响，从而在不同任务下公平比较鲁棒性和泛化能力。图1展示了任务1-4在代表性时间步上的空间采样模式。")
    add_picture(doc, "image6.png", width=6.2)
    add_caption(doc, "图1 四个重建任务（任务1-4）的空间采样模式。")

    add_heading(doc, "3.3 方法", 1)
    add_heading(doc, "3.3.1 概述", 2)
    add_paragraph(doc, "本文提出一种上下文条件化隐式神经场 M2GNet，用于从稀疏且时变的观测中重建连续海洋表层流速场。对于每个时空样本（天）$i∈{1,…,N}$ 以及任意查询位置 $x∈Ω_o$，模型将目标场表示为连续函数")
    add_formula(doc, formulas[6], 6)
    add_paragraph(doc, "其中 $Θ$ 表示网络参数，$ẑ_i$ 是样本级上下文编码，用于捕获第 $i$ 天的潜在动力状态和观测条件。函数 $F_Θ(·)$ 可以在任意 $x$ 处评估，从而实现无网格重建和连续空间查询。")
    add_picture(doc, "image8.png", width=6.4)
    add_caption(doc, "图2 M2GNet总体架构。为每个样本 $i$ 分配可学习上下文编码 $z_i$，并通过 Dropout 和投影映射为 $ẑ_i$；坐标 $x$ 经潜在门控混合频率编码器处理，再通过堆叠的乘性交互与 FiLM 调制块及稳定化残差连接，生成 $ŷ_i(x)$。")
    add_paragraph(doc, "上下文感知自动解码。与使用显式编码器从观测中推断上下文不同，本文采用自动解码方案：每个样本 $i$ 均关联一个可学习潜在向量 $z_i∈R^{d_z}$，并存储在上下文嵌入表中。训练时，潜在编码与网络参数 $Θ$ 一起通过稀疏观测 $D_i$ 上的重建损失联合优化。为增强鲁棒性，$z_i$ 进一步经过潜在 Dropout 正则化，并通过可学习线性投影映射为工作上下文表示 $ẑ_i$。")
    add_paragraph(doc, "多尺度谱表示。为减轻谱偏置并提升多尺度表达能力，M2GNet 首先使用潜在门控混合频率编码器对坐标输入 $x$ 进行变换，融合 Gabor、Fourier 和线性映射。融合权重由 $ẑ_i$ 通过 softmax 门控产生，使模型能够按样本自适应选择频率成分。")
    add_paragraph(doc, "上下文依赖调制与稳定深层堆叠。编码特征随后经过 $L$ 个乘性交互块传播，其中隐藏特征与 $ẑ_i$ 双线性融合，并与混合频率响应相乘。每个块进一步应用以 $ẑ_i$ 为条件的 FiLM 式特征级仿射调制，从而增强上下文依赖频率选择。为稳定深层架构中的优化过程，本文使用带有 LayerScale、DropPath 和 LayerNorm 的残差堆叠，以控制特征漂移并提升高稀疏、高随机采样下的泛化能力。")

    add_heading(doc, "3.3.2 基于自动解码的上下文嵌入", 2)
    add_paragraph(doc, "为在保持跨所有天共享统一解码器的同时建模样本特异的时空状态，本文采用自动解码形式，使每个样本 $i∈{1,…,N}$ 关联一个可学习潜在向量 $z_i∈R^{d_z}$。将所有潜在向量收集起来可得到上下文嵌入表")
    add_formula(doc, formulas[7], 7)
    add_paragraph(doc, "潜在编码与解码器参数 $Θ$ 通过最小化稀疏观测上的平方重建误差联合优化。对于第 $i$ 个样本，其观测集定义为")
    add_formula(doc, formulas[8], 8)
    add_paragraph(doc, "训练目标为")
    add_formula(doc, formulas[9], 9)
    add_paragraph(doc, "其中 $F_Θ(·)$ 表示隐式解码器，$ẑ_i$ 是后续模块使用的有效上下文表示。该自动解码策略能够用紧凑潜在编码表示每一天，同时保持解码器共享，因而尤其适合不规则且时变的观测布局。")
    add_paragraph(doc, "潜在 Dropout 正则化。为防止逐样本编码过拟合并提升观测不确定性下的鲁棒性，本文直接在潜在空间应用 Dropout。具体地，采样 Bernoulli 掩膜 $m$，形成扰动后的潜在向量")
    add_formula(doc, formulas[10], 10)
    add_paragraph(doc, "其中 $⊙$ 表示逐元素乘法，$p$ 为保留概率（实现中由 latent_dropout 配置）。")
    add_paragraph(doc, "线性投影（潜在压缩）。为控制门控和调制中上下文特征的工作维度，将 $tilde(z)_i$ 通过可学习线性投影矩阵 $P$ 映射到投影潜在空间：")
    add_formula(doc, formulas[11], 11)
    add_paragraph(doc, "其中 $d_p<d_z$ 为投影维度（实现中由 latent_proj 配置）。该投影有两方面作用：一是降低样本级条件化的有效自由度以改善泛化；二是为所有条件模块（如混合门控和 FiLM）提供稳定共享接口，使其不依赖原始潜在维度。")
    add_paragraph(doc, "式（9）-（11）定义了贯穿 M2GNet 的上下文嵌入机制。样本特异上下文编码 $z_i$ 充当状态描述符，并通过随机优化隐式推断；$ẑ_i$ 则是经过正则化和投影后的上下文表示，用于条件化后续谱编码与调制模块。")

    add_heading(doc, "3.3.3 带潜在门控融合的混合频率编码", 2)
    add_paragraph(doc, "地球物理场重建中的隐式神经表示面临一个关键挑战，即谱偏置：模型倾向于先拟合低频成分，而对近岸梯度和小尺度涡旋等局地高频结构表示不足。为扩展谱覆盖范围并保持对不同流动状态的适应性，本文采用混合频率编码器，将 Gabor、Fourier 和线性三个互补谱映射通过潜在条件门控机制进行融合。")
    add_paragraph(doc, "分支定义。令输入坐标为 $x∈R^d$（本文中 $d$=2，对应经纬度坐标）。对于通道 $k$，分支参数为：Gabor 分支含 $w_k^(g)∈R^d$、偏置 $b_k^(g)$、中心 $μ_k∈R^d$ 和带宽系数 $γ_k>0$；Fourier 分支含 $w_k^(f)∈R^d$ 及整体频率缩放因子 $α_f$；线性分支含 $w_k^(l)∈R^d$ 和偏置 $b_k^(l)$。令 $H$ 表示通道数，定义")
    add_formula(doc, formulas[12], 12)
    add_paragraph(doc, "以及")
    add_formula(doc, formulas[13], 13)
    add_paragraph(doc, "则三个分支分别为")
    add_formula(doc, formulas[14], 14)
    add_formula(doc, formulas[15], 15)
    add_formula(doc, formulas[16], 16)
    add_paragraph(doc, "沿特征维拼接各通道，可得")
    add_formula(doc, formulas[17], 17)
    add_paragraph(doc, "需要说明的是，式（15）每个通道产生一组正弦-余弦对；在实现中，该对被展平为特征通道，使三个分支均产生可兼容的特征维度以便融合。等价地，维度对齐也可吸收到后续线性层中，而不改变函数形式。")
    add_paragraph(doc, "潜在门控融合。给定投影上下文嵌入 $ẑ_i$，轻量门控网络产生混合权重")
    add_formula(doc, formulas[18], 18)
    add_paragraph(doc, "其中 $w_g(ẑ_i)$、$w_f(ẑ_i)$ 和 $w_l(ẑ_i)$ 分别对应 Gabor、Fourier 和线性分支。混合频率编码定义为")
    add_formula(doc, formulas[19], 19)
    add_paragraph(doc, "其第 $k$ 个通道形式为")
    add_formula(doc, formulas[20], 20)
    add_paragraph(doc, "该设计使模型能够根据推断出的上下文 $ẑ_i$，自适应强调低频背景结构（通过线性/Fourier 分量）或高频局地变化（通过 Gabor 分量），从而提升对异构和时变观测模式的鲁棒性。")
    add_paragraph(doc, "跨深度的层级尺度。混合频率编码器在每个隐藏层 $ℓ$ 处独立实例化并拥有自身参数集。为鼓励由粗到细的谱层级结构，有效频率尺度 $σ_ℓ$ 随深度衰减：")
    add_formula(doc, formulas[21], 21)
    add_paragraph(doc, "其中 $σ_0$ 为输入层初始尺度。该调度使浅层偏向高频细节，同时允许深层关注低频、大尺度结构，从而在单一条件 INR 中形成多分辨率谱分解。")

    add_heading(doc, "3.3.4 乘性交互与层级条件调制", 2)
    add_paragraph(doc, "混合频率编码 $f_mix(x,z_i)$ 提供了上下文自适应的谱基，但仍有必要显式耦合层级隐藏表示、混合频率响应和样本级上下文嵌入，使网络能够在不同深度动态选择与组合频率成分。设网络包含 $L$ 个隐藏层，令第 $ℓ$ 层输入特征为")
    add_formula(doc, formulas[22], 22)
    add_paragraph(doc, "并初始化为")
    add_formula(doc, formulas[23], 23)
    add_paragraph(doc, "该初始化意味着第一阶段交互由上下文嵌入和混合频率响应驱动，而更深层则逐渐融合层间特征演化。")
    add_paragraph(doc, "隐藏特征与上下文的线性融合。在第 $ℓ$ 层，首先通过线性组合当前隐藏特征和投影潜在变量 $ẑ_i$ 构造中间系数字段：")
    add_formula(doc, formulas[24], 24)
    add_paragraph(doc, "其中 $W_ℓ^(h)∈R^{d_h}$×$R^{d_h}$、$W_ℓ^(z)∈R^{d_h}$×$R^{d_p}$ 和 $b_ℓ∈R^{d_h}$ 为可学习参数。式（24）生成一组层特异系数，用于按通道调节谱响应。")
    add_paragraph(doc, "乘性交互核。令 $ϕ(·)$ 为平滑逐元素非线性函数（实现中为 SiLU）。定义")
    add_formula(doc, formulas[25], 25)
    add_paragraph(doc, "并在同一层与混合频率响应执行按通道乘性交互：")
    add_formula(doc, formulas[26], 26)
    add_paragraph(doc, "其中 $⊙$ 表示逐元素乘法，$f_ℓ(x,ẑ_i)$ 表示第 $ℓ$ 层的混合频率特征。该乘性核可视为上下文依赖的谱滤波器：对于通道 $k$，较大的 $s̃_ℓ,k$ 会放大对应频率成分，而较小的值则会抑制该成分，从而根据 $ẑ_i$ 和当前特征状态选择性增强有用尺度。")
    add_paragraph(doc, "基于 FiLM 的条件调制。为进一步适应异构流动状态，在乘性交互之后应用 FiLM。对于每个层 $ℓ$，使用一个小型 MLP $M_ℓ(·)$ 将投影潜在变量 $ẑ_i$ 映射为按通道的仿射参数：")
    add_formula(doc, formulas[27], 27)
    add_paragraph(doc, "随后计算 FiLM 调制后的特征：")
    add_formula(doc, formulas[28], 28)
    add_paragraph(doc, "其中 1 为全 1 向量，确保缩放项初始时接近恒等映射。相比直接对 $h_ℓ$ 应用 FiLM，式（28）调制的是已经完成频率选择的交互输出；当观测模式和流动状态在样本间变化时，这种做法更有优势，因为模型可同时调节有效谱强调和结果特征统计。")

    add_heading(doc, "3.3.5 稳定化深度残差堆叠与输出投影", 2)
    add_paragraph(doc, "第 3.3.4 节中的乘性-FiLM 块增强了上下文自适应表达能力，但也提高了深层隐式解码器训练不稳定的风险，特别是在存在高频成分且观测掩膜显著变化时。令 $v_ℓ(x,ẑ_i)∈R^{d_h}$ 表示式（28）得到的 FiLM 调制中间特征。本文通过以下三阶段稳定化形成残差更新。")
    add_paragraph(doc, "LayerScale。为避免深层网络中残差分支产生过强扰动，引入可学习逐通道缩放因子 $α_ℓ∈R^{d_h}$，并应用")
    add_formula(doc, formulas[29], 29)
    add_paragraph(doc, "其中 $⊙$ 表示逐元素乘法。实际中，$α_ℓ$ 初始化为较小正数（例如 $10^-2$ 或更小），使训练早期接近稳定的近恒等状态，同时仍允许模型逐渐学习适当的残差幅度。")
    add_paragraph(doc, "DropPath（随机深度）。为进一步改善泛化并抑制深层协同适应，对残差分支应用 DropPath。对于每个训练样本，采样")
    add_formula(doc, formulas[30], 30)
    add_paragraph(doc, "并计算")
    add_formula(doc, formulas[31], 31)
    add_paragraph(doc, "其中 $p_ℓ$ 为保留概率。该机制在训练中随机关闭整个残差路径，相当于训练一组具有不同深度的子网络，从而提升在不同采样模式导致的分布变化下的鲁棒性。")
    add_paragraph(doc, "残差相加与 LayerNorm。随后将稳定化残差加入隐藏表示，并应用 LayerNorm：")
    add_formula(doc, formulas[32], 32)
    add_formula(doc, formulas[33], 33)
    add_paragraph(doc, "LayerNorm 约束各深度处的特征统计；当乘性交互和条件仿射调制可能引起特征尺度漂移和梯度不稳定时，这一点尤为重要。经过 $L$ 层稳定化残差更新后，得到最终隐藏特征 $h_L(x,ẑ_i)$。解码器输出通过线性投影映射到物理量空间：")
    add_formula(doc, formulas[34], 34)
    add_paragraph(doc, "其中 $W_out∈R^{d_out}$×$R^{d_h}$ 和 $b_out∈R^{d_out}$ 为可学习参数。本文设置中，$d_out$=1，用于标量流速重建。")

    add_heading(doc, "3.3.6 训练目标与模型配置", 2)
    add_paragraph(doc, "总体模型。给定时空样本索引 $i∈{1,…,N}$ 与空间查询坐标 $x∈Ω_o$，所提出解码器通过")
    add_formula(doc, formulas[35], 35)
    add_paragraph(doc, "预测目标标量场，其中 $Θ$ 表示所有网络参数，$ẑ_i$ 是由潜在路径产生的样本特异上下文编码。核心解码器由 $L$ 个堆叠的条件乘性块构成，隐藏宽度为 $d_h$。")
    add_paragraph(doc, "重建损失。对于每个样本 $i$，稀疏观测集为")
    add_formula(doc, formulas[36], 36)
    add_paragraph(doc, "并在观测位置上优化均方重建误差：")
    add_formula(doc, formulas[37], 37)
    add_paragraph(doc, "其中 $Z={z_1,…,z_N}$ 收集 Dropout/投影之前的所有可学习样本嵌入。该目标直接对应稀疏观测设置，并允许解码器学习可在测试阶段于完整海洋网格上评估的连续映射。")
    add_paragraph(doc, "潜在编码正则化。为缓解严重稀疏和高度随机掩膜下样本级嵌入的过拟合，本文对潜在路径应用 Dropout，并可选地对潜在编码加入轻微 $ℓ_2$ 惩罚：")
    add_formula(doc, formulas[38], 38)
    add_paragraph(doc, "其中 $λ_z$ 控制潜在先验强度。如果实现中未使用 $λ_z$，可省略式（38）中的该项，同时保留基于 Dropout 的正则化。")

    add_heading(doc, "3.4 实验", 1)
    add_heading(doc, "3.4.1 实验协议", 2)
    add_paragraph(doc, "本文在第 3.2.2 节描述的渤海表层流速数据集上评估所有方法。按照第 3.2.3 节，训练阶段考虑四种观测场景（任务1-4）和三个稀疏度水平 $s∈{5%,25%,50%}$。对于每一天 $i$，模型使用稀疏观测 $D_i={(x_j,y_i(x_j))|x_j∈S_i}$ 进行训练，并在完整海洋网格 $Ω_o$ 上评估。除非另有说明，所有模型采用相同优化设置，并在完全一致的任务和稀疏度条件下比较。")
    add_heading(doc, "3.4.2 基线方法", 2)
    add_paragraph(doc, "为系统评估所提出模型在海洋流场重建上的性能，本文与多类代表性 INR 基线进行比较，包括普通坐标 MLP、显式频率编码模型以及基于 Gabor 的高频网络：")
    for item in [
        "ResMLP：带残差连接的多层感知机，将坐标映射为标量，以缓解深层网络中的梯度消失。该模型不使用显式频率编码或上下文条件化，代表标准坐标型 INR 基线。",
        "SIREN：采用正弦激活的 INR，能够增强高频结构表示，但不显式建模样本级上下文。",
        "FFN-P / FFN-G：Fourier 特征 INR，其中 FFN-P 使用周期位置编码，FFN-G 使用高斯随机 Fourier 特征。二者通过向坐标输入注入多尺度频率成分来缓解谱偏置。",
        "MMGN：本文方法的前身，利用 Gabor 核和乘性结构增强坐标特征与频率响应之间的耦合，但不包含本文提出的上下文嵌入和层级条件调制。",
        "M2GNet（本文）：完整提出模型，在 MMGN 基础上引入上下文嵌入、潜在门控混合频率编码、带 FiLM 条件化的乘性交互以及稳定化深度残差堆叠。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        add_text_with_math(p, item)

    add_heading(doc, "3.4.3 评价指标", 2)
    add_paragraph(doc, "本文将均方误差（MSE）作为主要评价指标：")
    add_formula(doc, formulas[39], 39)
    add_paragraph(doc, "在消融实验中，进一步报告归一化均方误差（NMSE），用于考察不同潜在维度下的误差趋势。所有指标均在完整海洋网格 $Ω_o$ 上计算。")

    add_heading(doc, "3.4.4 定量比较", 2)
    add_paragraph(doc, "表1汇总了渤海数据集上任务1-4及稀疏度 $s∈{5%,25%,50%}$ 下的 MSE 结果。对于每个任务-稀疏度组合，最佳结果以加粗表示，次优结果以下划线表示。")
    add_caption(doc, "表1 渤海流速数据集上，任务1-4在不同稀疏度水平 $s$ 下的 MSE 比较。最佳结果加粗，次优结果加下划线。")
    add_table_1(doc)
    add_paragraph(doc, "为量化 M2GNet 相对每个设置下最强基线的相对提升，本文报告如下指标：")
    add_formula(doc, formulas[40], 40)
    add_paragraph(doc, "总体性能。如表1所示，M2GNet 在所有任务-稀疏度组合中均取得最佳或并列最佳 MSE。当采样比例从 50% 降至 5% 时，所有方法都会因观测信息减少而性能下降；然而 M2GNet 的退化幅度更小，表明其在极端稀疏条件下具有更强鲁棒性。")
    add_paragraph(doc, "采样不确定性的影响。随着采样不确定性从任务1增加到任务4，M2GNet 与其他基线之间的性能差距逐渐扩大。特别是在最具挑战性的随机位置和/或随机数量设置（任务3-4）下，即使 $s$=5%，M2GNet 仍保持明显优势，说明其对时变异构观测模式具有更强泛化能力。这一趋势与上下文嵌入、潜在门控混合频率编码和层级条件调制的设计动机一致。")

    add_heading(doc, "3.4.5 定性比较", 2)
    add_paragraph(doc, "本文进一步通过代表性快照和相对误差图比较重建质量。总体来看，普通坐标 MLP 倾向于过度平滑细尺度结构；正弦激活模型可能产生振荡伪影；Fourier 特征基线在严重稀疏条件下可能出现结构化混叠；而基于 Gabor 的模型更能保持多尺度纹理。M2GNet 对大尺度环流模式和局地高梯度结构（如近岸剪切和涡旋边界）均能给出更忠实的恢复，高误差区域显著减少且空间上更加局地化。")
    add_picture(doc, "image43.png", width=6.4)
    add_caption(doc, "图3 渤海数据集上的定性重建结果：参考场、不同方法的重建结果以及代表性时间步的相对误差图。")

    add_heading(doc, "3.4.6 上下文索引与潜在空间消融", 2)
    add_paragraph(doc, "为研究上下文感知索引机制的作用，本文通过改变潜在维度开展消融实验。令 $T$ 为时间步数（渤海数据集为 $T$=731），并用 $z_{t_k}^{(d_z)}∈R^{d_z}$ 表示潜在维度为 $d_z$ 时在时间步 $t_k$ 学到的上下文编码。")
    add_paragraph(doc, "潜在编码相似性（相异性）分析。本文训练如下潜在维度设置下的模型：")
    add_formula(doc, formulas[41], 41)
    add_paragraph(doc, "对于每个 $d_z$，计算时间序列潜在编码 $z_{t_k}^{(d_z)}$（$k=1,…,T$）之间的相关系数，并以相关系数集合的标准差度量潜在编码相异性。经验结果表明，随着 $d_z$ 增大，相异性单调降低，说明更高维潜在空间可产生更有结构且更一致的几何关系。")
    add_picture(doc, "image45.png", width=5.4)
    add_picture(doc, "image46.png", width=5.4)
    add_picture(doc, "image47.png", width=5.4)
    add_caption(doc, "图4 不同潜在维度 $d_z$ 下的潜在空间分析：（a）相异性（相关系数标准差），（b）NMSE 随 $d_z$ 的变化，（c）相关结构不匹配 $MSE(C_z,C_y)$。")
    add_paragraph(doc, "潜在维度与重建误差。本文在相同训练配置下评估不同 $d_z$ 的 NMSE。当 $d_z$ 从非常小的值（如 2 或 4）增加到中等值（如 64）时，NMSE 迅速下降；当 $d_z≈128-256$ 后，误差趋于饱和，说明继续增大潜在维度的收益逐渐减弱。")
    add_paragraph(doc, "潜在结构与物理动力的对齐。为检验潜在编码是否捕获了原始流场的时间相关结构，本文由时间索引潜在矩阵 $Z∈R^{T×d_z}$ 计算潜在相关矩阵 $C_z$，并由时间索引物理场矩阵 $Y∈R^{T×P}$（仅海洋点）计算数据相关矩阵 $C_y$。结构对齐程度通过")
    add_formula(doc, formulas[42], 42)
    add_paragraph(doc, "衡量。该指标随 $d_z$ 增加而下降，说明更高维潜在编码能够更好保留底层动力过程的时间相关结构。")
    add_picture(doc, "image49.png", width=5.8)
    add_caption(doc, "图5 不同潜在维度下学习到的上下文编码的 t-SNE 可视化。较大的 $d_z$ 可得到更有组织、语义更一致的嵌入。")
    add_paragraph(doc, "t-SNE 可视化。为进一步增强可解释性，本文使用 t-SNE 将潜在向量嵌入到二维空间。随着 $d_z$ 增大，嵌入点由纠缠散点逐渐演化为更连贯、连续的聚类结构。")

    add_heading(doc, "3.5 结论", 1)
    add_paragraph(doc, "本文提出 M2GNet，一种上下文条件化隐式神经表示，用于从稀疏且不规则观测中连续重建海洋表层流速场。通过结合样本级潜在上下文、混合频率 Gabor/Fourier/线性编码、乘性 FiLM 调制以及稳定化残差堆叠，该模型能够在统一的基于坐标的解码器中表示多尺度环流和局地近岸梯度。")
    add_paragraph(doc, "基于 731 天渤海日平均流场的实验表明，M2GNet 在四种采样场景和三种稀疏度水平下，相比代表性 INR 基线均能持续降低 MSE。当观测位置和数量随时间变化时，增益最为明显，说明上下文感知谱条件化对布局不稳定的业务化观测网络具有实际价值。未来工作将把该框架扩展到矢量速度重建和物理约束建模。")

    add_heading(doc, "作者贡献声明", 2)
    add_paragraph(doc, "Yingjie Mei 和 Shuo Jin 对本文贡献相同。Yingjie Mei：概念化、方法、调查、形式分析、可视化、初稿写作。Shuo Jin：概念化、方法、调查、形式分析、可视化、初稿写作。Jing Hao：调查、数据整理、验证、审阅与编辑。Zuotao Ni：调查、数据整理、验证、审阅与编辑。Jingbo Jiang：监督、资源、经费获取、审阅与编辑。Tao Jin：监督、资源、经费获取、项目管理、审阅与编辑。Hongjuan Zhou：监督、资源、经费获取、审阅与编辑。Zhiquan Zhou、Chenxu Wang、Yunzhou Li 和 Juncheng Wang：调查、数据整理、验证、审阅与编辑。所有作者均阅读并批准最终稿。")
    add_heading(doc, "利益冲突声明", 2)
    add_paragraph(doc, "作者声明不存在可能影响本文工作的已知竞争性经济利益或个人关系。")
    add_heading(doc, "资助", 2)
    add_paragraph(doc, "本研究得到国家重点研发计划（2023YFC3107503、2023YFC3107501）、泰山学者青年专家计划（tsqn202507139）、山东省自然科学基金（ZR2025MS927）以及山东省重大科技创新工程（2022ZLGX04）资助。")
    add_heading(doc, "生成式 AI 与 AI 辅助技术声明", 2)
    add_paragraph(doc, "在本工作准备过程中，作者使用 OpenAI Codex 辅助语言润色和期刊投稿格式整理。使用该工具后，作者对内容进行了必要审阅和编辑，并对提交稿件的全部内容负责。")
    add_heading(doc, "数据可用性", 2)
    add_paragraph(doc, "支持本文研究发现的数据可根据合理请求从通信作者处获得。")

    add_heading(doc, "参考文献", 2)
    references = [
        "Champenois B, Sapsis T, 2024. Machine learning framework for the real-time reconstruction of regional 4d ocean temperature fields from historical reanalysis data and real-time satellite and buoy surface measurements. Physica D: Nonlinear Phenomena, 459:134026. https://doi.org/10.1016/j.physd.2023.134026",
        "Chen F, Xiao Y, Yu L, et al., 2025. Frequency-domain physics-informed neural network for accurate reconstruction of 3d acoustic fields under sparse and multi-frequency measurements. Neural Networks, 197:108476. https://doi.org/10.1016/j.neunet.2025.108476",
        "Chitturi SR, Ji Z, Petsch AN, et al., 2023. Capturing dynamical correlations using implicit neural representations. Nature Communications, 14(1):5852. https://doi.org/10.1038/s41467-023-41378-4",
        "Cutolo E, Pascual A, Ruiz S, et al., 2024. Cloinet: ocean state reconstructions through remote-sensing, in-situ sparse observations and deep learning. Frontiers in Marine Science, 11:1151868. https://doi.org/10.3389/fmars.2024.1151868",
        "Dalaiden Q, Goosse H, Holland PR, et al., 2025. Dynamical reconstruction of southern ocean and antarctic climate variability since 1700. Scientific Data, 12(1):1574. https://doi.org/10.1038/s41597-025-05808-w",
        "Essakine A, Cheng Y, Cheng CW, et al., 2024. Where do we stand with implicit neural representations? a technical and performance survey. arXiv, abs/2411.03688. https://doi.org/10.48550/arXiv.2411.03688",
        "Grande D, Buizza R, Storto A, 2026. Machine learning in ocean data assimilation: Advances, gaps and the road to operations. Ocean Modelling, 200:102678. https://doi.org/10.1016/j.ocemod.2026.102678",
        "Gu M, Li J, Wu Y, et al., 2025. 3d human avatar reconstruction with neural fields: A recent survey. Image and Vision Computing, 154:105341. https://doi.org/10.1016/j.imavis.2024.105341",
        "Hendriks T, Arends G, Versteeg E, et al., 2025. Implicit neural representations for accurate estimation of the standard model of white matter. Communications Biology, 9(1):120. https://doi.org/10.1038/s42003-025-09399-5",
        "Irshad MZ, Comi M, Lin YC, et al., 2024. Neural fields in robotics: A survey. arXiv, abs/2410.20220. https://doi.org/10.48550/arXiv.2410.20220",
        "Lentz S, Brune S, Kadow C, et al., 2025. Improving ocean reanalyses of observationally sparse regions with transfer learning. Scientific Reports, 15(1):2640. https://doi.org/10.1038/s41598-025-86374-4",
        "Li H, Liu Y, Li M, et al., 2025. A deep learning-based reconstruction model for 3d sound speed field combining underwater vertical information. Frontiers in Marine Science, 12:1551823. https://doi.org/10.3389/fmars.2025.1551823",
        "Ma Z, Huang J, Zhang X, et al., 2025a. Deep learning-based reconstruction of monthly antarctic surface air temperatures from 1979 to 2023. Scientific Data, 12(1):847. https://doi.org/10.1038/s41597-025-05175-6",
        "Ma Z, Yang Q, Xu Y, et al., 2025b. Reconstruction of antarctic sea ice thickness from sparse satellite laser altimetry data via deep learning. Scientific Data, 13(1):81. https://doi.org/10.1038/s41597-025-06390-x",
        "Martin SA, Manucharyan GE, Klein P, 2025. Generative data assimilation for surface ocean state estimation from multi-modal satellite observations. Journal of Advances in Modeling Earth Systems, 17(8):e2025MS005063. https://doi.org/10.1029/2025MS005063",
        "Meng Z, Hakim GJ, 2024. Reconstructing the tropical pacific upper ocean using online data assimilation with a deep learning model. Journal of Advances in Modeling Earth Systems, 16(11):e2024MS004422. https://doi.org/10.1029/2024MS004422",
        "Mo Y, Magri L, 2025. Reconstructing unsteady flows from sparse, noisy measurements with a physics-constrained convolutional neural network. Physical Review Fluids, 10(3):034901. https://doi.org/10.1103/PhysRevFluids.10.034901",
        "Mostajeran F, Pawar NM, Villarreal JM, et al., 2025. Context-aware implicit neural representations to compress earth systems model data. Scientific Reports, 15(1):25932. https://doi.org/10.1038/s41598-025-11092-w",
        "Nguyen PCH, Choi JB, Luu QT, 2026. Deep learning with fourier features for regressive flow field reconstruction from sparse sensor measurements. Scientific Reports, 16(1):5980. https://doi.org/10.1038/s41598-026-36301-y",
        "Schiodt M, Mucke NT, Velte CM, 2026. Generative super-resolution of turbulent flows via stochastic interpolants. Scientific Reports, 16(1):4229. https://doi.org/10.1038/s41598-025-34363-y",
        "Schirmer L, Novello T, da Silva V, et al., 2024. Geometric implicit neural representations for signed distance functions. Computers & Graphics, 125:104085. https://doi.org/10.1016/j.cag.2024.104085",
        "Smith LT, Horrocks T, Akhtar N, et al., 2025. Implicit neural representation for potential field geophysics. Scientific Reports, 15(1):9799. https://doi.org/10.1038/s41598-024-83979-z",
        "Smith PAH, Sorensen KA, Buongiorno Nardelli B, et al., 2023. Reconstruction of subsurface ocean state variables using convolutional neural networks with combined satellite and in situ data. Frontiers in Marine Science, 10:1218514. https://doi.org/10.3389/fmars.2023.1218514",
        "Su H, Zhang F, Teng J, et al., 2024. Reconstructing high-resolution subsurface temperature of the global ocean using deep forest with combined remote sensing and in situ observations. ISPRS Journal of Photogrammetry and Remote Sensing, 218:389-404. https://doi.org/10.1016/j.isprsjprs.2024.09.022",
        "Xu B, Zou Y, Sha G, et al., 2025. Sparse wavefield reconstruction based on physics-informed neural networks. Ultrasonics, 149:107582. https://doi.org/10.1016/j.ultras.2025.107582",
        "Xu Y, Sha Y, Wang C, et al., 2024. Adaptive estimation model: Robust full-state prediction through sparse observations with variable layout and quantity. Ocean Engineering, 301:117617. https://doi.org/10.1016/j.oceaneng.2024.117617",
        "Yu X, Yi DL, Wang P, 2025. Enhancing ocean temperature and salinity reconstruction with deep learning: The role of surface waves. Journal of Marine Science and Engineering, 13(5):910. https://doi.org/10.3390/jmse13050910",
        "Zhang L, Li R, Shao C, et al., 2026. Macro-micro feature driven digital twin for assembly planning of aero-engine rotor under weak correlation. Mechanical Systems and Signal Processing, 249:114060. https://doi.org/10.1016/j.ymssp.2026.114060",
        "Zhao Q, Peng S, Wang J, et al., 2024a. Applications of deep learning in physical oceanography: a comprehensive review. Frontiers in Marine Science, 11:1396322. https://doi.org/10.3389/fmars.2024.1396322",
        "Zhao X, Chen X, Gong Z, et al., 2024b. Recfno: A resolution-invariant flow and heat field reconstruction method from sparse observations via fourier neural operator. International Journal of Thermal Sciences, 195:108619. https://doi.org/10.1016/j.ijthermalsci.2023.108619",
        "Zheng X, Xia H, Wang H, et al., 2025. A fourier neural operator-enhanced parabolic equation framework for highly efficient underwater acoustic field prediction. Frontiers in Marine Science, 12:1692899. https://doi.org/10.3389/fmars.2025.1692899",
    ]
    for ref in references:
        add_paragraph(doc, ref)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build_doc()
