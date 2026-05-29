import csv
import json
import os
import zipfile
from xml.sax.saxutils import escape


RESULT_COLUMNS = [
    "Method",
    "Params(M)",
    "FLOPs(G)",
    "FPS",
    "Dice",
    "Precision",
    "Recall",
    "Specificity",
    "HD95",
]


def export_results(rows, metrics_dir, summary_path):
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, "comparison_results.csv")
    json_path = os.path.join(metrics_dir, "comparison_results.json")
    xlsx_path = os.path.join(metrics_dir, "comparison_results.xlsx")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    write_xlsx(rows, xlsx_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Comparison Results\n\n")
        f.write("| " + " | ".join(RESULT_COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(RESULT_COLUMNS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_format_cell(row.get(col, "")) for col in RESULT_COLUMNS) + " |\n")


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_xlsx(rows, path):
    sheet_rows = [RESULT_COLUMNS] + [[row.get(col, "") for col in RESULT_COLUMNS] for row in rows]
    sheet_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(sheet_rows, start=1):
        sheet_xml.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_excel_col(c_idx)}{r_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sheet_xml.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                sheet_xml.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_xml.append("</row>")
    sheet_xml.extend(["</sheetData>", "</worksheet>"])

    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="comparison_results" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        "xl/worksheets/sheet1.xml": "\n".join(sheet_xml),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _excel_col(index):
    result = ""
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result
