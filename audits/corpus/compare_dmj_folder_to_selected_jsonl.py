#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

PATH_FIELDS = ("_pdf_path","pdf_path","source_path","file_path","filepath","path","filename")
ID_FIELDS = ("paperId","paper_id")
YEAR_FIELDS = ("year","publication_year","publicationYear","published_year","publishedYear")

def parse_year(v):
    if v in (None, "") or isinstance(v, bool):
        return None
    try:
        y = int(v)
        if 1800 <= y <= 2100:
            return y
    except Exception:
        pass
    m = re.search(r"\\b(19|20)\\d{2}\\b", str(v))
    return int(m.group()) if m else None

def extract_filename(rec):
    for f in PATH_FIELDS:
        v = rec.get(f)
        if v not in (None, ""):
            s = str(v).strip()
            if s:
                return Path(s).name, s, f
    for f in ID_FIELDS:
        v = rec.get(f)
        if v not in (None, ""):
            s = str(v).strip()
            return f"{s}.pdf", s, f
    return "", "", ""

def extract_year(rec):
    for f in YEAR_FIELDS:
        if f in rec:
            y = parse_year(rec[f])
            if y is not None:
                return y, f
    return None, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, type=Path)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    jsonl = args.jsonl.expanduser().resolve()

    pdfs = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.casefold()==".pdf"],
        key=lambda p:p.name.casefold()
    )
    by_name = {p.name.casefold():p for p in pdfs}

    records = []
    with jsonl.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            try:
                rec=json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or "_metadata" in rec:
                continue
            fn, src, src_field = extract_filename(rec)
            y, y_field = extract_year(rec)
            records.append({
                "line_no": line_no,
                "filename": fn,
                "filename_key": fn.casefold(),
                "source_value": src,
                "source_field": src_field,
                "year": y,
                "year_field": y_field,
            })

    matched = [r for r in records if r["filename_key"] in by_name and r["filename_key"]]
    refs = defaultdict(list)
    for r in matched:
        refs[r["filename_key"]].append(r)

    folder_not_in_jsonl = [p for p in pdfs if p.name.casefold() not in refs]

    folder_name = folder.name.casefold()
    jsonl_refs_missing = []
    for r in records:
        src = r["source_value"].replace("\\","/").casefold()
        if folder_name in src and r["filename_key"] not in by_name:
            jsonl_refs_missing.append(r)

    out = args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    unmatched_csv = Path(str(out)+".folder_not_in_jsonl.csv")
    with unmatched_csv.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["filename","size_bytes","mtime_epoch"])
        for p in folder_not_in_jsonl:
            st=p.stat()
            w.writerow([p.name,st.st_size,st.st_mtime])

    summary = {
        "folder_pdf_count": len(pdfs),
        "matching_jsonl_records": len(matched),
        "matching_records_unknown_year": sum(r["year"] is None for r in matched),
        "matching_records_known_year": sum(r["year"] is not None for r in matched),
        "folder_pdfs_not_in_jsonl": len(folder_not_in_jsonl),
        "folder_pdfs_not_in_jsonl_names": [p.name for p in folder_not_in_jsonl],
        "duplicate_jsonl_filename_refs": {
            by_name[k].name:[r["line_no"] for r in v]
            for k,v in refs.items() if len(v)>1
        },
        "jsonl_refs_to_this_folder_missing_now": [
            {"line_no":r["line_no"],"filename":r["filename"],"source_value":r["source_value"],"year":r["year"]}
            for r in jsonl_refs_missing
        ],
    }
    Path(str(out)+".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("=== DMJ FOLDER vs SELECTED JSONL ===")
    print(f"Folder PDFs:                      {len(pdfs):,}")
    print(f"Matching JSONL records:           {len(matched):,}")
    print(f"  unknown year:                   {summary['matching_records_unknown_year']:,}")
    print(f"  known year:                     {summary['matching_records_known_year']:,}")
    print(f"Folder PDFs NOT in JSONL:         {len(folder_not_in_jsonl):,}")
    for p in folder_not_in_jsonl:
        print("  -", p.name)
    print(f"JSONL refs to folder missing now: {len(jsonl_refs_missing):,}")
    print(f"Duplicate JSONL filename refs:    {len(summary['duplicate_jsonl_filename_refs']):,}")
    print("Summary:", str(out)+".summary.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
