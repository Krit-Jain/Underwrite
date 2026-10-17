"""Synthetic Indian GST invoice generator.

Ground truth is known by construction: each invoice is generated from a
:class:`InvoiceSpec`, so the rendered image and its gold fields can never drift
apart. Layouts, date formats, money formats and tax splits are deliberately
varied so the extractor sees realistic heterogeneity rather than one template.

Tax follows the Indian convention: intra-state supply splits the tax into equal
CGST and SGST halves, inter-state supply charges a single IGST line.
"""

from __future__ import annotations

import io
import json
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterator

from underwrite.signals.gstin import build_gstin

TWO_PLACES = Decimal("0.01")
TAX_RATES = (Decimal("5"), Decimal("12"), Decimal("18"), Decimal("28"))

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y")
MONEY_STYLES = ("plain", "grouped", "symbol", "rs")

_TEMPLATE_COUNT = 6

VENDORS = (
    ("SUNRISE TRADERS", "42 MG Road, Bengaluru", "560001", "07"),
    ("SHAKTI ENTERPRISES", "18 Nehru Street, Coimbatore", "641001", "33"),
    ("GANESH PROVISION STORES", "7 Lamington Road, Mumbai", "400004", "27"),
    ("DEVI ELECTRICALS", "221 Anna Salai, Chennai", "600002", "33"),
    ("KRISHNA HARDWARE MART", "9 Station Road, Pune", "411001", "27"),
    ("ANAND STATIONERS", "55 Ashram Road, Ahmedabad", "380009", "24"),
    ("LAKSHMI TEXTILES", "3 Commercial Street, Bengaluru", "560001", "29"),
    ("OM SAI TRADING CO", "76 Sector 18, Noida", "201301", "09"),
)

BUYERS = (
    ("Bluewave Analytics Pvt Ltd", "27"),
    ("Northgate Logistics LLP", "07"),
    ("Pinnacle Retail India", "33"),
    ("Cobalt Studios", "29"),
    ("Vega Components", "24"),
)

PRODUCTS = (
    ("A4 Copier Paper 500 sheets", "4802"),
    ("Ballpoint Pen blue (box of 50)", "9608"),
    ("LED Panel Light 18W", "9405"),
    ("Stainless Steel Bottle 1L", "7323"),
    ("Cotton Bath Towel", "6302"),
    ("USB-C Cable 1m", "8544"),
    ("Hand Sanitiser 500ml", "3808"),
    ("Printer Toner Cartridge", "8443"),
    ("Extension Board 6 socket", "8536"),
    ("Notebook ruled 200 pages", "4820"),
    ("Ceramic Mug 350ml", "6912"),
    ("Desk Organiser plastic", "3924"),
)

UNIT_PRICES = (
    Decimal("45.00"),
    Decimal("120.50"),
    Decimal("249.00"),
    Decimal("399.99"),
    Decimal("575.00"),
    Decimal("1240.00"),
    Decimal("89.25"),
    Decimal("1875.50"),
)


@dataclass(frozen=True)
class LineItem:
    description: str
    hsn: str
    quantity: int
    unit_price: Decimal
    amount: Decimal


@dataclass(frozen=True)
class InvoiceSpec:
    doc_id: str
    template: int
    invoice_no: str
    date: date
    date_format: str
    money_style: str
    vendor_name: str
    vendor_address: str
    vendor_gstin: str
    buyer_name: str
    buyer_gstin: str | None
    inter_state: bool
    items: tuple[LineItem, ...]
    subtotal: Decimal
    tax_rate: Decimal
    cgst: Decimal
    sgst: Decimal
    igst: Decimal
    tax: Decimal
    total: Decimal

    @property
    def gold(self) -> dict[str, str]:
        return {
            "vendor": self.vendor_name,
            "date": self.date.isoformat(),
            "subtotal": f"{self.subtotal:.2f}",
            "tax": f"{self.tax:.2f}",
            "total": f"{self.total:.2f}",
        }


def _money(amount: Decimal) -> Decimal:
    return amount.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def _random_pan(rng: random.Random) -> str:
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}{digits}{rng.choice(string.ascii_uppercase)}"


def _indian_group(digits: str) -> str:
    """Group digits the Indian way: last three, then pairs (1,23,456)."""
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    pairs: list[str] = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    if head:
        pairs.insert(0, head)
    return ",".join(pairs + [tail])


def format_money(amount: Decimal, style: str = "grouped") -> str:
    """Render an amount in one of the invoice money styles."""
    sign = "-" if amount < 0 else ""
    value = abs(amount)
    whole, _, fraction = f"{value:.2f}".partition(".")
    if style == "plain":
        return f"{sign}{whole}.{fraction}"
    grouped = f"{_indian_group(whole)}.{fraction}"
    if style == "grouped":
        return f"{sign}{grouped}"
    if style == "symbol":
        return f"{sign}\u20b9{grouped}"
    if style == "rs":
        return f"{sign}Rs. {grouped}"
    raise ValueError(f"unknown money style: {style!r}")


def format_date(value: date, date_format: str) -> str:
    return value.strftime(date_format)


def format_rate(rate: Decimal) -> str:
    """Render a GST rate as it appears on an invoice, e.g. ``18`` or ``2.5``."""
    return f"{rate.normalize():f}%"


def page_tokens(spec: InvoiceSpec) -> list[str]:
    """The words a perfect OCR pass would return for this invoice.

    Synthetic invoices have no OCR annotations, so grounding is measured against the
    text the renderer drew. This is oracle text: it removes OCR error from the
    picture, which is documented as a limitation of the synthetic shift set.
    """
    style = spec.money_style
    tokens: list[str] = []
    tokens += spec.vendor_name.split()
    tokens += spec.vendor_address.split()
    tokens.append(spec.vendor_gstin)
    tokens += spec.buyer_name.split()
    if spec.buyer_gstin:
        tokens.append(spec.buyer_gstin)
    tokens.append(spec.invoice_no)
    tokens += format_date(spec.date, spec.date_format).split()
    for item in spec.items:
        tokens += item.description.split()
        tokens.append(item.hsn)
        tokens.append(str(item.quantity))
        tokens.append(format_money(item.unit_price, style))
        tokens.append(format_money(item.amount, style))
    tokens.append(format_money(spec.subtotal, style))
    if spec.inter_state:
        tokens.append(format_money(spec.igst, style))
    else:
        tokens.append(format_money(spec.cgst, style))
        tokens.append(format_money(spec.sgst, style))
    tokens.append(format_money(spec.total, style))
    return tokens


def build_spec(rng: random.Random, doc_id: str) -> InvoiceSpec:
    """Build one internally consistent invoice specification."""
    vendor_name, vendor_address, _pin, state_code = rng.choice(VENDORS)
    buyer_name, buyer_state = rng.choice(BUYERS)
    inter_state = buyer_state != state_code

    pan = _random_pan(rng)
    vendor_gstin = build_gstin(pan, state_code)
    buyer_gstin = build_gstin(_random_pan(rng), buyer_state) if rng.random() < 0.7 else None

    invoice_date = date(2026, 10, 17) - timedelta(days=rng.randint(0, 300))
    invoice_no = (
        f"{rng.choice(('INV', 'GST', 'TAX'))}/{state_code}/{rng.randint(100, 9999)}"
    )

    item_count = rng.randint(1, 5)
    items: list[LineItem] = []
    for _ in range(item_count):
        description, hsn = rng.choice(PRODUCTS)
        quantity = rng.randint(1, 6)
        unit_price = rng.choice(UNIT_PRICES)
        items.append(
            LineItem(
                description=description,
                hsn=hsn,
                quantity=quantity,
                unit_price=unit_price,
                amount=_money(unit_price * quantity),
            )
        )

    subtotal = _money(sum((item.amount for item in items), Decimal("0")))
    tax_rate = rng.choice(TAX_RATES)
    tax = _money(subtotal * tax_rate / Decimal("100"))

    if inter_state:
        cgst = sgst = Decimal("0.00")
        igst = tax
    else:
        # Split so the halves always re-sum to the tax exactly.
        cgst = _money(tax / 2)
        sgst = tax - cgst
        igst = Decimal("0.00")

    return InvoiceSpec(
        doc_id=doc_id,
        template=rng.randrange(_TEMPLATE_COUNT),
        invoice_no=invoice_no,
        date=invoice_date,
        date_format=rng.choice(DATE_FORMATS),
        money_style=rng.choice(MONEY_STYLES),
        vendor_name=vendor_name,
        vendor_address=vendor_address,
        vendor_gstin=vendor_gstin,
        buyer_name=buyer_name,
        buyer_gstin=buyer_gstin,
        inter_state=inter_state,
        items=tuple(items),
        subtotal=subtotal,
        tax_rate=tax_rate,
        cgst=cgst,
        sgst=sgst,
        igst=igst,
        tax=tax,
        total=_money(subtotal + tax),
    )


def _view(spec: InvoiceSpec) -> dict[str, Any]:
    """Pre-format every display value once."""
    style = spec.money_style

    def show(amount: Decimal) -> str:
        return format_money(amount, style)

    return {
        "invoice_no": spec.invoice_no,
        "date_text": format_date(spec.date, spec.date_format),
        "vendor_name": spec.vendor_name,
        "vendor_address": spec.vendor_address,
        "vendor_gstin": spec.vendor_gstin,
        "buyer_name": spec.buyer_name,
        "buyer_gstin": spec.buyer_gstin or "-",
        "rows": [
            {
                "index": index + 1,
                "description": item.description,
                "hsn": item.hsn,
                "quantity": item.quantity,
                "rate": show(item.unit_price),
                "amount": show(item.amount),
            }
            for index, item in enumerate(spec.items)
        ],
        "tax_rate": format_rate(spec.tax_rate),
        # Intra-state supply charges half the rate as CGST and half as SGST.
        "half_rate": format_rate(spec.tax_rate / 2),
        "subtotal": show(spec.subtotal),
        "cgst": show(spec.cgst),
        "sgst": show(spec.sgst),
        "igst": show(spec.igst),
        "total": show(spec.total),
        "inter_state": spec.inter_state,
    }


_BASE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; font-family: "DejaVu Sans", Arial, sans-serif;
       color: #1b1b1b; background: #ffffff; font-size: 14px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 7px 9px; }
.num { text-align: right; }
.totals { width: 300px; margin-left: auto; }
.totals td { padding: 5px 9px; }
"""


def render_html(spec: InvoiceSpec) -> str:
    """Render an invoice spec to a standalone HTML document."""
    v = _view(spec)
    body = _TEMPLATES[spec.template](v)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{_BASE_CSS}{_TEMPLATE_CSS.get(spec.template, '')}</style>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


def _rows_html(v: dict[str, Any]) -> str:
    return "".join(
        f"<tr><td>{row['index']}</td><td>{row['description']}</td>"
        f"<td>{row['hsn']}</td><td class='num'>{row['quantity']}</td>"
        f"<td class='num'>{row['rate']}</td><td class='num'>{row['amount']}</td></tr>"
        for row in v["rows"]
    )


def _totals_html(v: dict[str, Any]) -> str:
    if v["inter_state"]:
        tax_rows = f"<tr><td>IGST @ {v['tax_rate']}</td><td class='num'>{v['igst']}</td></tr>"
    else:
        tax_rows = (
            f"<tr><td>CGST @ {v['half_rate']}</td><td class='num'>{v['cgst']}</td></tr>"
            f"<tr><td>SGST @ {v['half_rate']}</td><td class='num'>{v['sgst']}</td></tr>"
        )
    return (
        f"<table class='totals'>"
        f"<tr><td>Subtotal</td><td class='num'>{v['subtotal']}</td></tr>"
        f"{tax_rows}"
        f"<tr class='grand'><td>Total</td><td class='num'>{v['total']}</td></tr>"
        f"</table>"
    )


_TABLE_HEAD = (
    "<tr><th>#</th><th>Description</th><th>HSN</th>"
    "<th class='num'>Qty</th><th class='num'>Rate</th><th class='num'>Amount</th></tr>"
)


def _template_classic(v):
    return (
        "<h1 style='text-align:center;margin:0 0 4px'>TAX INVOICE</h1>"
        f"<div style='text-align:center;margin-bottom:18px'>{v['vendor_name']}, "
        f"{v['vendor_address']}</div>"
        f"<div>GSTIN: {v['vendor_gstin']} &nbsp;&nbsp; Invoice No: {v['invoice_no']} "
        f"&nbsp;&nbsp; Date: {v['date_text']}</div>"
        f"<div style='margin-bottom:14px'>Billed To: {v['buyer_name']} "
        f"(GSTIN {v['buyer_gstin']})</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:16px'>{_totals_html(v)}</div>"
    )


def _template_minimal(v):
    return (
        f"<div style='font-size:22px;font-weight:600'>{v['vendor_name']}</div>"
        f"<div style='color:#666'>{v['vendor_address']} &middot; GSTIN {v['vendor_gstin']}</div>"
        "<hr>"
        f"<div>Invoice {v['invoice_no']}</div><div>Date {v['date_text']}</div>"
        f"<div style='margin-bottom:14px'>To {v['buyer_name']} ({v['buyer_gstin']})</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:16px'>{_totals_html(v)}</div>"
    )


def _template_boxed(v):
    return (
        "<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
        f"<div><div style='font-weight:700;font-size:18px'>{v['vendor_name']}</div>"
        f"<div>{v['vendor_address']}</div><div>GSTIN {v['vendor_gstin']}</div></div>"
        "<div style='border:1px solid #333;padding:8px 12px'>"
        f"<div>Invoice: {v['invoice_no']}</div><div>Date: {v['date_text']}</div></div></div>"
        f"<div style='margin:14px 0'>Buyer: {v['buyer_name']} &middot; {v['buyer_gstin']}</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:16px'>{_totals_html(v)}</div>"
    )


def _template_shaded(v):
    return (
        f"<div style='background:#274060;color:#fff;padding:12px 16px;font-size:18px'>"
        f"{v['vendor_name']} &mdash; TAX INVOICE</div>"
        f"<div style='padding:10px 0'>{v['vendor_address']} | GSTIN {v['vendor_gstin']}</div>"
        f"<div style='background:#eef2f7;padding:8px;margin-bottom:12px'>"
        f"Invoice {v['invoice_no']} &nbsp; Date {v['date_text']} &nbsp; "
        f"Buyer {v['buyer_name']}</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:16px'>{_totals_html(v)}</div>"
    )


def _template_banner(v):
    return (
        f"<div style='border:2px solid #1b1b1b;padding:10px;text-align:center;"
        f"letter-spacing:3px;font-size:20px'>TAX INVOICE</div>"
        f"<div style='display:flex;justify-content:space-between;margin-top:12px'>"
        f"<div><b>{v['vendor_name']}</b><br>{v['vendor_address']}<br>"
        f"GSTIN: {v['vendor_gstin']}</div>"
        f"<div style='text-align:right'>No: {v['invoice_no']}<br>{v['date_text']}</div></div>"
        f"<div style='margin:12px 0'>Consignee: {v['buyer_name']} ({v['buyer_gstin']})</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:16px'>{_totals_html(v)}</div>"
    )


def _template_sidebar(v):
    return (
        "<div style='display:flex;gap:18px'>"
        "<div style='width:210px;background:#f2f2f2;padding:12px'>"
        f"<div style='font-weight:700'>{v['vendor_name']}</div>"
        f"<div style='margin-top:8px'>{v['vendor_address']}</div>"
        f"<div style='margin-top:8px'>GSTIN<br>{v['vendor_gstin']}</div></div>"
        "<div style='flex:1'>"
        "<div style='font-size:18px;font-weight:600'>Invoice</div>"
        f"<div>Number: {v['invoice_no']}</div><div>Date: {v['date_text']}</div>"
        f"<div style='margin:10px 0'>Billed to {v['buyer_name']} &middot; {v['buyer_gstin']}</div>"
        f"<table border='1'>{_TABLE_HEAD}{_rows_html(v)}</table>"
        f"<div style='margin-top:14px'>{_totals_html(v)}</div>"
        "</div></div>"
    )


_TEMPLATES = (
    _template_classic,
    _template_minimal,
    _template_boxed,
    _template_shaded,
    _template_banner,
    _template_sidebar,
)

_TEMPLATE_CSS: dict[int, str] = {
    3: "th { background:#274060; color:#fff; }",
    4: "th { background:#ececec; }",
    1: "th { background:#fafafa; }",
    5: "th { background:#f2f2f2; }",
}


def render_image(html: str, width: int = 900, height: int = 1300) -> Any:
    """Render an HTML document to a PIL image using a fresh Chromium instance."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_content(html, wait_until="load")
            png = page.screenshot(full_page=True)
        finally:
            browser.close()

    from PIL import Image

    return Image.open(io.BytesIO(png)).convert("RGB")


def render_text(html: str) -> str:
    """Return the *visible* text of a rendered document.

    Unlike substring-matching the HTML source, this reflects what the layout
    engine actually displays, so it catches content hidden or clipped by CSS.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 900, "height": 1300})
            page.set_content(html, wait_until="load")
            return page.inner_text("body")
        finally:
            browser.close()


def augment(image: Any, rng: random.Random) -> Any:
    """Apply a mild, realistic capture degradation to a rendered invoice."""
    from PIL import Image, ImageFilter

    if rng.random() < 0.5:
        image = image.rotate(
            rng.uniform(-3.0, 3.0), resample=Image.BICUBIC, expand=True, fillcolor="white"
        )
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        image = _shade(image, rng)
    if rng.random() < 0.4:
        image = _warp(image, rng)
    if rng.random() < 0.6:
        image = _recompress(image, rng)
    return image


def _shade(image: Any, rng: random.Random) -> Any:
    """Overlay a soft directional shadow, built at low resolution for speed."""
    from PIL import Image

    width, height = image.size
    darkest = rng.randint(180, 235)
    ramp = Image.new("L", (64, 1))
    for x in range(64):
        ramp.putpixel((x, 0), int(darkest + (255 - darkest) * x / 63))
    if rng.random() < 0.5:
        ramp = ramp.transpose(Image.ROTATE_90)
    mask = ramp.resize((width, height))
    return Image.blend(image, Image.merge("RGB", (mask, mask, mask)), 0.30)


def _warp(image: Any, rng: random.Random) -> Any:
    """Apply a small perspective warp by remapping the source quadrilateral."""
    from PIL import Image

    width, height = image.size
    pull = 0.03
    quad = (
        rng.uniform(0, pull) * width, rng.uniform(0, pull) * height,
        rng.uniform(0, pull) * width, height - rng.uniform(0, pull) * height,
        width - rng.uniform(0, pull) * width, height - rng.uniform(0, pull) * height,
        width - rng.uniform(0, pull) * width, rng.uniform(0, pull) * height,
    )
    return image.transform(
        (width, height), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor="white"
    )


def _recompress(image: Any, rng: random.Random) -> Any:
    """Round-trip through JPEG to add compression artifacts."""
    from PIL import Image

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(60, 92))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def generate_specs(count: int, seed: int = 0, prefix: str = "synth") -> list[InvoiceSpec]:
    """Generate ``count`` deterministic invoice specifications."""
    rng = random.Random(seed)
    return [build_spec(rng, f"{prefix}_{index:04d}") for index in range(count)]


def generate_batch(
    count: int,
    out_dir: str | Path,
    *,
    seed: int = 0,
    prefix: str = "synth",
    augment_images: bool = True,
) -> Path:
    """Render a batch of invoices to ``out_dir`` and write their ground truth."""
    from playwright.sync_api import sync_playwright

    target = Path(out_dir)
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    specs = generate_specs(count, seed=seed, prefix=prefix)
    rng = random.Random(seed + 1)
    ground_truth: dict[str, dict[str, str]] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 900, "height": 1300})
            for spec in specs:
                page.set_content(render_html(spec), wait_until="load")
                image = page.screenshot(full_page=True)

                from PIL import Image

                picture = Image.open(io.BytesIO(image)).convert("RGB")
                if augment_images:
                    picture = augment(picture, rng)
                picture.save(images_dir / f"{spec.doc_id}.png")
                ground_truth[spec.doc_id] = spec.gold
        finally:
            browser.close()

    truth_path = target / "ground_truth.json"
    truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return truth_path
