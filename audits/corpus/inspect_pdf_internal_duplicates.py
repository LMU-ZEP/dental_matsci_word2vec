#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def journal_name(rec: dict) -> str:
    j = rec.get("journal")
    return str((j or {}).get("name") or "").strip() if isinstance(j, dict) else str(j or "").strip()

ap = argparse.ArgumentParser()
ap.add_argument("--pdf-jsonl", required=True, type=Path)
ap.add_argument("--duplicate-keys-csv", required=True, type=Path)
ap.add_argument("--out-prefix", required=True, type=Path)
args = ap.parse_args()

records_by_line = {}
with args.pdf_jsonl.open("r", encoding="utf-8-sig") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and "_metadata" not in rec:
            records_by_line[line_no] = rec

groups = defaultdict(set)
with args.duplicate_keys_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        groups[(row["key_type"], row["key_value"])].add(int(row["pdf_line_no"]))

args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
group_rows, record_rows = [], []

for group_no, ((key_type, key_value), line_numbers) in enumerate(sorted(groups.items()), 1):
    hashes, ids, titles, journals, paths = set(), set(), set(), set(), set()
    missing = 0
    rows_this = []
    for line_no in sorted(line_numbers):
        rec = records_by_line.get(line_no, {})
        paper_id = str(rec.get("paperId") or rec.get("paper_id") or "")
        title = str(rec.get("title") or "")
        doi = str(rec.get("doi") or "")
        year = rec.get("year")
        journal = journal_name(rec)
        pages = str((rec.get("journal") or {}).get("pages") or "") if isinstance(rec.get("journal"), dict) else ""
        path_str = str(rec.get("_pdf_path") or rec.get("pdf_path") or rec.get("source_path") or "")
        path = Path(path_str) if path_str else None
        exists = bool(path and path.is_file())
        size = path.stat().st_size if exists else ""
        sha = sha256_file(path) if exists else ""
        if exists: hashes.add(sha)
        else: missing += 1
        if paper_id: ids.add(paper_id)
        if title: titles.add(title)
        if journal: journals.add(journal)
        if path_str: paths.add(path_str)
        row = dict(group_no=group_no,key_type=key_type,key_value=key_value,pdf_line_no=line_no,
                   paperId=paper_id,year=year,doi=doi,title=title,journal=journal,pages=pages,
                   pdf_path=path_str,file_exists=exists,size_bytes=size,sha256=sha)
        record_rows.append(row); rows_this.append(row)

    n = len(rows_this)
    if missing:
        relation = "incomplete_missing_files"
    elif len(hashes) == 1 and n > 1:
        relation = "all_exact_same_pdf_bytes"
    elif len(hashes) == n:
        relation = "all_pdf_bytes_distinct"
    else:
        relation = "mixed_some_exact_pdf_duplicates"

    group_rows.append(dict(group_no=group_no,key_type=key_type,key_value=key_value,
                           n_records=n,n_distinct_paperIds=len(ids),
                           n_distinct_original_titles=len(titles),
                           n_distinct_journals=len(journals),
                           n_distinct_paths=len(paths),n_distinct_sha256=len(hashes),
                           missing_pdf_files=missing,hash_relation=relation))

groups_csv = Path(str(args.out_prefix)+".groups.csv")
records_csv = Path(str(args.out_prefix)+".records.csv")

with groups_csv.open("w", encoding="utf-8", newline="") as f:
    fn = ["group_no","key_type","key_value","n_records","n_distinct_paperIds","n_distinct_original_titles",
          "n_distinct_journals","n_distinct_paths","n_distinct_sha256","missing_pdf_files","hash_relation"]
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(group_rows)

with records_csv.open("w", encoding="utf-8", newline="") as f:
    fn = ["group_no","key_type","key_value","pdf_line_no","paperId","year","doi","title","journal","pages",
          "pdf_path","file_exists","size_bytes","sha256"]
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(record_rows)

print("=== INTERNAL PDF DUPLICATE GROUP INSPECTION ===")
print("Groups:", len(group_rows))
for rel in ["all_exact_same_pdf_bytes","mixed_some_exact_pdf_duplicates","all_pdf_bytes_distinct","incomplete_missing_files"]:
    print(f"{rel}: {sum(r['hash_relation']==rel for r in group_rows)}")
print("Group summary:", groups_csv)
print("Record details:", records_csv)
