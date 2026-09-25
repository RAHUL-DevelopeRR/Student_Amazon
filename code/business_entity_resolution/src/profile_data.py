"""Exact full-file audit with DuckDB spilling to disk, checkpointed per file."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from config import ROOT, connect, save_json, log_experiment
from normalize import normalize


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def rows(con, sql):
    cursor = con.execute(sql)
    names = [x[0] for x in cursor.description]
    return [dict(zip(names, r)) for r in cursor.fetchall()]


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        if f.read(128).startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"Git LFS pointer, not data: {path}")
        f.seek(0)
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_stats(con, table):
    n = scalar(con, f"SELECT count(*) FROM {table}")
    result = {"rows": n, "schema": rows(con, f"DESCRIBE {table}"),
              "input_columns": ["entity_id", "business_name", "business_address", "country"],
              "dtype_policy": "All input columns explicitly read as VARCHAR, not inferred",
              "duplicate_entity_ids": n - scalar(con, f"SELECT count(DISTINCT entity_id) FROM {table}"),
              "countries": rows(con, f"SELECT country, count(*) n FROM {table} GROUP BY country ORDER BY n DESC")}
    for field, norm in [("business_name", "name_norm"), ("business_address", "address_norm")]:
        missing = scalar(con, f"SELECT count(*) FROM {table} WHERE trim(coalesce({field},'')) = ''")
        duplicate = scalar(con, f"SELECT count(*)-count(DISTINCT {norm}) FROM {table} WHERE {norm} <> ''")
        result[field] = {"missing": missing, "missing_rate": missing / n if n else 0,
                         "duplicate_normalized_excess_rows_excluding_empty": duplicate}
        for label, expr in [("characters", f"length(coalesce({field},''))"),
                            ("tokens", f"CASE WHEN {norm} = '' THEN 0 ELSE len(string_split({norm},' ')) END")]:
            result[field][label] = rows(con, f"SELECT min(x) minimum, avg(x) mean, max(x) maximum, quantile_cont(x,[0.5,0.9,0.99]) quantiles FROM (SELECT {expr} x FROM {table})")[0]
    return result


def import_source(con, path, table, artifacts):
    # Native scalar Python UDF calls proved much slower than a streaming pass on
    # this Windows runtime. A temporary TSV also avoids retaining all strings.
    temp = artifacts / f"{table}.normalized.tmp.tsv"
    with path.open(encoding="utf-8-sig", newline="") as src, temp.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        header = next(reader)
        if header != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected source schema: {header}")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(header + ["name_norm", "address_norm"])
        for i, row in enumerate(reader, 1):
            if len(row) != 4:
                raise ValueError(f"Malformed {path.name} row {i+1}")
            writer.writerow(row + [normalize(row[1]), normalize(row[2])])
            if i % 250000 == 0:
                print(f"  {table}: normalized {i:,} rows", flush=True)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv(?, delim='\t', header=true, all_varchar=true, nullstr='')", [str(temp)])
    # Explicit empty normalized fields simplify consistent missingness handling.
    con.execute(f"UPDATE {table} SET name_norm=coalesce(name_norm,''), address_norm=coalesce(address_norm,'') WHERE name_norm IS NULL OR address_norm IS NULL")
    temp.unlink()


def write_report(report, path):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
    gt = report["ground_truth"]
    lines = ["# Full dataset audit", "", "All seven files scanned; original TSVs unchanged. Exact counts, source SHA-256 fingerprints and schemas are in data_profile.json.",
             "", "## Source inventory", "", "| File | Rows | Bytes | Duplicate IDs | Missing names | Missing addresses |", "|---|---:|---:|---:|---:|---:|"]
    for name, info in report["files"].items():
        lines.append(f"| {name} | {info['rows']:,} | {info['bytes']:,} | {info['duplicate_entity_ids']} | {info['business_name']['missing']:,} | {info['business_address']['missing']:,} |")
    lines += ["", "All four input columns are read as VARCHAR to preserve exact text; normalized name/address are additional columns.",
              "", "## Ground truth", "", f"- S1 entities: {gt['rows']:,}.",
              f"- Singletons: {gt['singletons']:,} ({gt['singleton_rate']:.4%}); all-empty macro F0.5 = {gt['all_empty_macro_f05']:.6f}.",
              f"- Exactly one match: {gt['exactly_one']:,} ({gt['exactly_one_rate']:.2%}).",
              f"- Multiple matches: {gt['more_than_one']:,} ({gt['more_than_one_rate']:.2%}).",
              "", "| True matches per S1 | S1 count |", "|---:|---:|"]
    lines += [f"| {x['matches']} | {x['entities']:,} |" for x in gt["match_count_distribution"]]
    lines += ["", "## Integrity checks", "", "| Check | Count |", "|---|---:|"]
    for key in ["targets_with_multiple_owners", "duplicate_truth_ids", "duplicate_edges", "missing_truth_rows", "unknown_truth_ids", "unknown_target_ids"]:
        lines.append(f"| {key} | {gt[key]:,} |")
    lines += ["", "## Countries", "", "| File | Country | Count | Share |", "|---|---|---:|---:|"]
    for name, info in report["files"].items():
        for item in info["countries"]:
            lines.append(f"| {name} | {cell(item['country'])} | {item['n']:,} | {item['n']/info['rows']:.2%} |")
    lines += ["", "### True-link country consistency", "", "| S1 country | Target country | Links |", "|---|---|---:|"]
    lines += [f"| {cell(x['source_country'])} | {cell(x['target_country'])} | {x['positives']:,} |" for x in gt["country_match_breakdown"]]
    lines += ["", "## Text statistics", "", "Duplicate text counts are excess nonempty rows, not proven duplicate businesses.", "", "| File | Field | Duplicate normalized rows | Mean characters | Char P50/P90/P99 | Mean tokens | Token P50/P90/P99 |", "|---|---|---:|---:|---|---:|---|"]
    for name, info in report["files"].items():
        for field in ["business_name", "business_address"]:
            s=info[field]
            lines.append(f"| {name} | {field} | {s['duplicate_normalized_excess_rows_excluding_empty']:,} | {s['characters']['mean']:.2f} | {s['characters']['quantiles']} | {s['tokens']['mean']:.2f} | {s['tokens']['quantiles']} |")
    lines += ["", "## Matched examples with both fields changed", "", "Deterministic sample for inspection; not a representative noise-frequency estimate."]
    for example in gt["noisy_examples"]:
        lines += ["", f"### {example['source1_id']} -> {example['target_id']} ({cell(example['country'])})",
                  f"- S1 name: {cell(example['source_name'])}", f"- Target name: {cell(example['target_name'])}",
                  f"- S1 address: {cell(example['source_address'])}", f"- Target address: {cell(example['target_address'])}"]
    lines += ["", "## Resources", "", f"Database bytes: {report['database_bytes']:,}. Configured buffer limit: {report['configured_memory_limit']}.",
              report["memory_note"], f"Elapsed seconds for latest invocation: {report['elapsed_seconds_this_run']:.1f}; resumed invocations reuse completed sources."]
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    p.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    p.add_argument("--memory", default="512MB")
    a = p.parse_args()
    a.artifacts.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    con = connect(a.artifacts / "audit.duckdb", a.memory)
    report_path = a.artifacts / "data_profile.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"files": {}}
    report["status"] = "running"
    report["normalization"] = "Python Unicode NFKC + casefold + punctuation/whitespace; empty excluded from duplicate counts"
    for split in ["train", "test"]:
        for source in range(1, 4):
            table = f"{split}_source{source}"
            path = a.data_dir / split / f"{table}.tsv"
            sha = fingerprint(path)
            old = report["files"].get(table, {})
            exists = scalar(con, f"SELECT count(*) FROM information_schema.tables WHERE table_name='{table}'")
            if old.get("sha256") == sha and exists:
                print(f"Reusing verified {table}: {old['rows']:,} rows", flush=True)
                continue
            print(f"Importing and normalizing {table} ({path.stat().st_size:,} bytes)", flush=True)
            import_source(con, path, table, a.artifacts)
            result = source_stats(con, table)
            result.update(bytes=path.stat().st_size, sha256=sha)
            report["files"][table] = result
            save_json(report_path, report)
            con.execute("CHECKPOINT")
            print(f"Profiled {table}: {result['rows']:,} rows", flush=True)
    gt_path = a.data_dir / "train/train_ground_truth.tsv"
    print("Auditing ground truth and ownership", flush=True)
    con.execute("CREATE OR REPLACE TABLE truth AS SELECT source1_entity_id, coalesce(matched_entity_ids,'') matched_entity_ids FROM read_csv(?, delim='\t', header=true, all_varchar=true)", [str(gt_path)])
    con.execute("CREATE OR REPLACE TABLE edges AS SELECT source1_entity_id, unnest(string_split(matched_entity_ids,',')) target_id FROM truth WHERE matched_entity_ids <> ''")
    con.execute("CREATE OR REPLACE VIEW targets AS SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3")
    con.execute("CREATE OR REPLACE VIEW cardinality AS SELECT source1_entity_id, CASE WHEN matched_entity_ids='' THEN 0 ELSE len(string_split(matched_entity_ids,',')) END n FROM truth")
    n = scalar(con, "SELECT count(*) FROM truth")
    singleton = scalar(con, "SELECT count(*) FROM cardinality WHERE n=0")
    gt = {"rows": n, "bytes": gt_path.stat().st_size, "sha256": fingerprint(gt_path),
          "schema": rows(con, "DESCRIBE truth"), "singletons": singleton,
          "singleton_rate": singleton / n, "all_empty_macro_f05": singleton / n,
          "exactly_one": scalar(con, "SELECT count(*) FROM cardinality WHERE n=1"),
          "more_than_one": scalar(con, "SELECT count(*) FROM cardinality WHERE n>1"),
          "match_count_distribution": rows(con, "SELECT n matches, count(*) entities FROM cardinality GROUP BY n ORDER BY n"),
          "positive_by_source": rows(con, "SELECT left(target_id,2) AS target_source, count(*) positives FROM edges GROUP BY target_source"),
          "targets_with_multiple_owners": scalar(con, "SELECT count(*) FROM (SELECT target_id FROM edges GROUP BY target_id HAVING count(DISTINCT source1_entity_id)>1)"),
          "duplicate_truth_ids": n - scalar(con, "SELECT count(DISTINCT source1_entity_id) FROM truth"),
          "duplicate_edges": scalar(con, "SELECT count(*) FROM edges") - scalar(con, "SELECT count(*) FROM (SELECT DISTINCT * FROM edges)"),
          "missing_truth_rows": scalar(con, "SELECT count(*) FROM train_source1 s ANTI JOIN truth t ON s.entity_id=t.source1_entity_id"),
          "unknown_truth_ids": scalar(con, "SELECT count(*) FROM truth t ANTI JOIN train_source1 s ON s.entity_id=t.source1_entity_id"),
          "unknown_target_ids": scalar(con, "SELECT count(*) FROM edges e ANTI JOIN targets t ON e.target_id=t.entity_id"),
          "country_match_breakdown": rows(con, "SELECT s.country source_country,t.country target_country,count(*) positives FROM edges e JOIN train_source1 s ON s.entity_id=e.source1_entity_id JOIN targets t ON t.entity_id=e.target_id GROUP BY s.country,t.country"),
          "country_cardinality": rows(con, "SELECT s.country,count(*) entities,sum(c.n=0) singletons,avg(c.n) mean_matches FROM cardinality c JOIN train_source1 s ON s.entity_id=c.source1_entity_id GROUP BY s.country"),
          "noisy_examples": rows(con, "SELECT s.entity_id source1_id,t.entity_id target_id,s.business_name source_name,t.business_name target_name,s.business_address source_address,t.business_address target_address,s.country FROM edges e JOIN train_source1 s ON s.entity_id=e.source1_entity_id JOIN targets t ON t.entity_id=e.target_id WHERE s.name_norm<>t.name_norm AND s.address_norm<>t.address_norm ORDER BY md5(e.source1_entity_id || e.target_id) LIMIT 12")}
    gt["exactly_one_rate"] = gt["exactly_one"] / n
    gt["more_than_one_rate"] = gt["more_than_one"] / n
    report["ground_truth"] = gt
    report["status"] = "complete"
    report["elapsed_seconds_this_run"] = time.monotonic() - started
    con.execute("CHECKPOINT")
    report["database_bytes"] = (a.artifacts / "audit.duckdb").stat().st_size
    report["configured_memory_limit"] = a.memory
    report["memory_note"] = "DuckDB buffer limit, not a process RSS bound; original strings persist on disk"
    save_json(report_path, report)
    write_report(report, a.artifacts / "data_profile.md")
    log_experiment("full_data_audit", {"singletons": singleton, "s1": n}, {"memory": a.memory, "data_dir": str(a.data_dir)})
    print(f"DONE: {n:,} truth rows; singleton rate {singleton/n:.4%}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
