from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np


SENSOR_COLUMNS = tuple(f"sensor_{i}" for i in range(1, 16))
TARGET_COLUMNS = tuple(f"target_{i}" for i in range(1, 7))
_CELL_REF = re.compile(r"([A-Z]+)")


@dataclass(frozen=True, slots=True)
class CapturedSample:
    sample_index: int
    pose: str
    pose_sequence_index: int
    pose_frame_index: int
    elapsed_ms: float
    sensors_deg: np.ndarray
    targets_deg: np.ndarray | None
    raw: dict[str, str]


def resolve_delivery_path(path: str | Path) -> Path:
    """Resolve the teacher-delivery name, including browser-added ``(1)``."""
    path = Path(path)
    if path.is_file():
        return path
    suffixed = path.with_name(f"{path.stem}(1){path.suffix}")
    if suffixed.is_file():
        return suffixed
    raise FileNotFoundError(f"data-glove delivery file not found: {path} (or {suffixed.name})")


def _column_index(cell_ref: str) -> int:
    match = _CELL_REF.match(cell_ref)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_ref!r}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    doc_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{ns}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{ns}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find(f"{ns}sheets")
        if sheets is None or not list(sheets):
            raise ValueError(f"XLSX contains no worksheet: {path}")
        relationship_id = list(sheets)[0].attrib[f"{doc_rel}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            rel.attrib["Target"]
            for rel in relationships.findall(f"{rel_ns}Relationship")
            if rel.attrib["Id"] == relationship_id
        )
        sheet_path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        sheet = ET.fromstring(archive.read(sheet_path))

        matrix: list[list[str]] = []
        sheet_data = sheet.find(f"{ns}sheetData")
        if sheet_data is None:
            raise ValueError(f"XLSX worksheet contains no rows: {path}")
        for row in sheet_data.findall(f"{ns}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{ns}c"):
                index = _column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{ns}v")
                if cell_type == "inlineStr":
                    inline = cell.find(f"{ns}is")
                    value = "" if inline is None else "".join(
                        node.text or "" for node in inline.iter(f"{ns}t")
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text or 0)]
                else:
                    value = value_node.text or ""
                values[index] = value
            width = max(values, default=-1) + 1
            matrix.append([values.get(i, "") for i in range(width)])

    if not matrix:
        return
    headers = matrix[0]
    for values in matrix[1:]:
        yield {name: values[i] if i < len(values) else "" for i, name in enumerate(headers)}


def _csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def load_captured_samples(path: str | Path) -> list[CapturedSample]:
    path = resolve_delivery_path(path)
    if path.suffix.lower() == ".xlsx":
        raw_rows = _xlsx_rows(path)
    elif path.suffix.lower() == ".csv":
        raw_rows = _csv_rows(path)
    else:
        raise ValueError(f"captured_samples must be .xlsx or .csv, got: {path}")

    samples: list[CapturedSample] = []
    for row_number, raw in enumerate(raw_rows, start=2):
        missing = [name for name in SENSOR_COLUMNS if not raw.get(name, "").strip()]
        if missing:
            raise ValueError(f"captured_samples row {row_number} missing sensors: {missing}")
        target_values = [raw.get(name, "").strip() for name in TARGET_COLUMNS]
        targets = None
        if any(target_values):
            if not all(target_values):
                raise ValueError(f"captured_samples row {row_number} has incomplete targets")
            targets = np.asarray([float(value) for value in target_values], dtype=float)
        samples.append(
            CapturedSample(
                sample_index=int(float(raw.get("sample_index", len(samples)) or len(samples))),
                pose=str(raw.get("pose", "") or ""),
                pose_sequence_index=int(float(raw.get("pose_sequence_index", 0) or 0)),
                pose_frame_index=int(float(raw.get("pose_frame_index", 0) or 0)),
                elapsed_ms=float(raw.get("elapsed_ms", 0.0) or 0.0),
                sensors_deg=np.asarray([float(raw[name]) for name in SENSOR_COLUMNS], dtype=float),
                targets_deg=targets,
                raw=dict(raw),
            )
        )
    if not samples:
        raise ValueError(f"captured_samples is empty: {path}")
    return samples
