"""Generate a small, fully synthetic manual corpus for the field-technician RAG assistant.

The folder tree is the schema the ingestion CLI expects:  client / machine-or-line / [component] / file.
File names follow the default document-type rules in backend/app/ingest/rules.py:
    "...Service Manual..."  -> operating_manual
    "...Wiring ... WD_EN"   -> wiring_diagram (drawing page: 300+ line objects -> keyword-only chunk)

Everything is fictional ("Acme" products). Two things are deliberate:
  * the W-100 manual is byte-for-byte the same text for two different clients, so the scope-isolation
    behaviour (identical text never crosses a client/machine boundary) is demonstrable;
  * alarm code E4 means something different on the W-100 washer and the D-200 dryer, so a question
    without the right scope is genuinely ambiguous.

    python make_sample_corpus.py --out sample_corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

rl_config.invariant = 1  # reproducible bytes: no creation timestamps

FOOTER = "SYNTHETIC SAMPLE DOCUMENT - fictional product, written for demonstration only."

WASHER_ALARMS = [
    ("E1", "Door lock fault", "Door switch open while a cycle is running.", "Check the door latch and the switch connector."),
    ("E2", "Water inlet timeout", "Fill level not reached within 8 minutes.", "Check the supply valve, inlet filter and level sensor tubing."),
    ("E3", "Drain pump overload", "Pump current above limit for 10 seconds.", "Isolate power, clear the drain filter, then test the pump."),
    ("E4", "Motor inverter fault", "Inverter reports over-current or over-temperature.", "Isolate power, wait 5 minutes for the DC bus to discharge, inspect the inverter heat sink."),
    ("E5", "Heater over-temperature", "Bath temperature above 95 C.", "Check the temperature probe and the heater relay."),
    ("E6", "Imbalance in extraction", "Drum vibration above limit at spin speed.", "Redistribute the load and check the suspension dampers."),
    ("E7", "Level sensor out of range", "Pressure reading outside 0.2-4.8 V.", "Check the sensor cable and the air tube for water."),
    ("E8", "Communication timeout", "No reply from the inverter for 3 seconds.", "Check the bus cable between the control unit and the inverter."),
]

DRYER_ALARMS = [
    ("E1", "Airflow restriction", "Pressure drop across the exhaust duct above limit.", "Clean the lint filter and inspect the exhaust duct."),
    ("E2", "Exhaust over-temperature", "Exhaust air above 85 C.", "Check the thermostat and the fan belt."),
    ("E3", "Heater ignition failure", "No flame signal within 10 seconds.", "Isolate gas and power, then inspect the ignition electrode."),
    ("E4", "Drum drive overload", "Drive motor current above limit.", "Isolate power and check the drum bearings and the drive belt tension."),
    ("E5", "Lint filter blocked", "Filter differential pressure above limit.", "Empty and clean the lint filter."),
    ("E6", "Moisture sensor fault", "Sensor value constant for 5 minutes.", "Clean the sensor bars and check the connector."),
]

SAFETY = [
    "Isolate the machine from mains power and apply lockout/tagout before opening any panel.",
    "Wait 5 minutes after isolation before touching inverter or capacitor terminals.",
    "Never bypass the door lock switch or the thermal cut-out.",
    "Steam and hot water lines stay pressurised after shutdown. Let them cool and vent them first.",
]


def _manual(path: Path, code: str, name: str, alarms: list[tuple[str, str, str, str]]) -> None:
    styles = getSampleStyleSheet()
    body, h1, h2 = styles["BodyText"], styles["Heading1"], styles["Heading2"]
    story = [
        Paragraph(f"Acme {code} {name} - Service Manual", styles["Title"]),
        Paragraph("Revision A. For trained service technicians.", body),
        Spacer(1, 8 * mm),
        Paragraph("1. Safety precautions", h1),
        Paragraph("Read this section before any repair step. Safety steps always come first.", body),
    ]
    story += [Paragraph(f"- {line}", body) for line in SAFETY]
    story += [PageBreak(), Paragraph("2. Alarm codes", h1)]
    table = Table(
        [["Code", "Meaning", "Condition", "First action"]] + [list(map(lambda s: Paragraph(s, body), row)) for row in alarms],
        colWidths=[16 * mm, 38 * mm, 55 * mm, 65 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story += [table, PageBreak(), Paragraph("3. Troubleshooting procedures", h1)]
    for alarm_code, meaning, condition, action in alarms[:4]:
        story += [
            Paragraph(f"3.{alarm_code[1:]} Alarm {alarm_code}: {meaning}", h2),
            Paragraph("Safety: isolate the machine and apply lockout/tagout before opening the panel.", body),
            Paragraph(f"Condition: {condition}", body),
            Paragraph(f"Procedure: {action} If the alarm returns after the check, escalate to the service desk with the alarm code and the cycle step.", body),
        ]
    story += [PageBreak(), Paragraph("4. Maintenance schedule", h1)]
    schedule = Table(
        [["Interval", "Task"], ["Daily", "Clean door seal and filters"], ["Monthly", "Inspect belts and drain filter"], ["Yearly", "Test safety switches and thermal cut-outs"]],
        colWidths=[30 * mm, 120 * mm],
    )
    schedule.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(schedule)

    def footer(canv: canvas.Canvas, doc) -> None:
        canv.setFont("Helvetica", 7)
        canv.drawString(18 * mm, 10 * mm, f"{FOOTER}  Page {doc.page}")

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title=f"Acme {code} service manual").build(story, onFirstPage=footer, onLaterPages=footer)


def _wiring(path: Path, code: str) -> None:
    """One drawing page with 400+ straight lines, so the chunker classifies it as a schematic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, height - 20 * mm, f"Acme {code} - wiring diagram (WD_EN)")
    c.setFont("Helvetica", 6)
    for col in range(20):
        x = 20 * mm + col * 8.5 * mm
        for row in range(12):
            y = height - 40 * mm - row * 14 * mm
            c.line(x, y, x + 6 * mm, y)
            c.line(x, y, x, y - 8 * mm)
            if row % 3 == 0:
                c.drawString(x, y + 1.5 * mm, f"X{col + 1}:{row + 1}")
    c.drawString(20 * mm, 18 * mm, "Terminals X1:1-X20:10. Inverter bus on X7. Door switch on X3:2.")
    c.setFont("Helvetica", 7)
    c.drawString(20 * mm, 10 * mm, FOOTER)
    c.showPage()
    c.save()


def build(out: Path) -> list[Path]:
    made: list[Path] = []

    def add(path: Path, fn, *args) -> None:
        fn(path, *args)
        made.append(path)

    acme = out / "Acme Laundry Services"
    add(acme / "W-100 Washer Line" / "W-100 Service Manual.pdf", _manual, "W-100", "washer-extractor", WASHER_ALARMS)
    add(acme / "W-100 Washer Line" / "Control Unit" / "CU-1 Control Unit Service Manual.pdf", _manual, "CU-1", "control unit", WASHER_ALARMS[3:])
    add(acme / "W-100 Washer Line" / "Control Unit" / "CU-1 Wiring WD_EN.pdf", _wiring, "CU-1")
    add(acme / "D-200 Dryer Line" / "D-200 Service Manual.pdf", _manual, "D-200", "tumble dryer", DRYER_ALARMS)

    borealis = out / "Borealis Textile Care"
    add(borealis / "W-100 Washer Line" / "W-100 Service Manual.pdf", _manual, "W-100", "washer-extractor", WASHER_ALARMS)  # same text on purpose
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("sample_corpus"))
    args = parser.parse_args()
    for path in build(args.out):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
