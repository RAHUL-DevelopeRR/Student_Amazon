"""Bounded-memory, multi-channel retrieval and explicit train-only evaluation.

Pilot mode includes known positives in the *search corpus*, never directly in the
returned candidate set. Its easier negative population makes recall optimistic.
Use --full-corpus to evaluate sampled S1 queries against every training target.
"""
import argparse
import csv
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from config import ROOT, SEED, connect, save_json, log_experiment
from metric import entity_f05
from normalize import expanded, normalize


def compatible(a, b):
    return not a or not b or normalize(a) == normalize(b)


def chunks(cursor, size):
    while batch := cursor.fetchmany(size):
        yield batch


def retain(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def retrieve(queries, batch_factory, fit_rows, k=20, max_features=80000,
             block_cap=100, rare_max_df=30, progress=False):
    """Rows are (ID, normalized name, normalized address, country).

    Global top K per field and source, not K per shard. Sparse multiplication is
    restricted to compatible countries before top K, retaining missing-country
    fallbacks. Source-specific heaps prevent S2 from crowding all S3 candidates.
    """
    if k < 1 or not queries or not fit_rows:
        raise ValueError("Need nonempty queries, fit corpus and positive k")
    qnames = {q[1] for q in queries if q[1]}
    qtokens = {t for q in queries for t in q[1].split() if len(t) >= 3}
    exact, tokens, overflow_exact, overflow_token = defaultdict(list), defaultdict(list), set(), set()
    # Only query-relevant postings are retained; discard overflowing blocks whole.
    # This gives global frequency caps independent of target shard boundaries.
    for batch in batch_factory():
        for row in batch:
            ident, name, address, country = row
            if name in qnames and name not in overflow_exact:
                exact[name].append((ident, country))
                if len(exact[name]) > block_cap:
                    del exact[name]
                    overflow_exact.add(name)
            for token in set(name.split()) & qtokens:
                if token in overflow_token:
                    continue
                tokens[token].append((ident, country))
                if len(tokens[token]) > rare_max_df:
                    del tokens[token]
                    overflow_token.add(token)
    methods = {m: [set() for _ in queries] for m in ["exact_name", "rare_name_token", "name_tfidf", "address_tfidf"]}
    for i, q in enumerate(queries):
        methods["exact_name"][i].update(x for x, c in exact.get(q[1], []) if compatible(q[3], c))
        for token in set(q[1].split()):
            methods["rare_name_token"][i].update(x for x, c in tokens.get(token, []) if compatible(q[3], c))
    del exact, tokens
    for field, label in [(1, "name_tfidf"), (2, "address_tfidf")]:
        kind = "name" if field == 1 else "address"
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                    lowercase=False, dtype=np.float32, max_features=max_features,
                                    sublinear_tf=True)
        fit_text = [expanded(r[field], kind) for r in fit_rows]
        if not any(len(t) >= 3 for t in fit_text):
            continue
        vectorizer.fit(fit_text)
        qmat = vectorizer.transform([expanded(q[field], kind) for q in queries])
        heaps = defaultdict(list)
        qgroups = defaultdict(list)
        for i, q in enumerate(queries):
            qgroups[normalize(q[3])].append(i)
        processed = 0
        for batch in batch_factory():
            for source in ("S2", "S3"):
                for country, qi in qgroups.items():
                    selected = [r for r in batch if r[0].startswith(source + "-") and compatible(country, r[3])]
                    if not selected:
                        continue
                    target = vectorizer.transform([expanded(r[field], kind) for r in selected])
                    similarities = sp_matmul_topn(qmat[qi], target.T.tocsr(), top_n=min(k, len(selected)),
                                                  threshold=0.0, sort=True, n_threads=1)
                    for local_i, original_i in enumerate(qi):
                        start, end = similarities.indptr[local_i:local_i+2]
                        for j, score in zip(similarities.indices[start:end], similarities.data[start:end]):
                            retain(heaps[original_i, source], (float(score), selected[j][0]), k)
            processed += len(batch)
            if progress and processed % 25000 < len(batch):
                print(f"{label}: scanned {processed:,} targets", flush=True)
        for (i, source), heap in heaps.items():
            methods[label][i].update(ident for score, ident in heap)
    candidates = [set().union(*(method[i] for method in methods.values())) for i in range(len(queries))]
    return candidates, methods, {"overflow_exact_blocks": len(overflow_exact), "overflow_rare_tokens": len(overflow_token)}


def metrics(queries, truth, candidates, target_count):
    positives = sum(len(truth[q[0]]) for q in queries)
    found = sum(len(truth[q[0]] & c) for q, c in zip(queries, candidates))
    nonempty = [(truth[q[0]], c) for q, c in zip(queries, candidates) if truth[q[0]]]
    counts = np.array([len(c) for c in candidates])
    return {"queries": len(queries), "target_corpus_size": target_count,
            "true_pairs": positives, "retrieved_true_pairs": found,
            "candidate_recall": found / positives if positives else None,
            "all_true_covered_non_singletons": sum(t <= c for t, c in nonempty) / len(nonempty) if nonempty else None,
            "all_true_covered_all_queries": sum(truth[q[0]] <= c for q, c in zip(queries, candidates)) / len(queries),
            "average_candidates": float(counts.mean()),
            "candidate_count_p50_p90_p99": np.quantile(counts, [.5, .9, .99]).tolist(),
            "candidate_pairs": int(counts.sum()),
            "reduction_ratio_vs_unrestricted_cartesian": 1 - int(counts.sum()) / (len(queries) * target_count),
            "oracle_macro_f05_ceiling": sum(entity_f05(truth[q[0]], truth[q[0]] & c) for q, c in zip(queries, candidates)) / len(queries)}


def exact_diagnostic(con, queries, cap=100):
    """Fast full-corpus control: exact name/address, no TF-IDF or rare tokens."""
    lookup = {q[0]: i for i,q in enumerate(queries)}
    channels = {name: [set() for _ in queries] for name in ["exact_name", "exact_address"]}
    caps = {}
    for field, method in [("name_norm","exact_name"),("address_norm","exact_address")]:
        # Count target block sizes before joining to queries (shared query names
        # must not inflate those counts). Oversized blocks are skipped whole.
        con.execute(f"CREATE OR REPLACE TEMP TABLE exact_blocks AS SELECT t.{field} AS block_key, count(*) n FROM corpus t SEMI JOIN queries q ON t.{field}=q.{field} WHERE t.{field}<>'' GROUP BY t.{field}")
        caps[method] = con.execute(f"SELECT count(*) FROM exact_blocks WHERE n>{cap}").fetchone()[0]
        cursor = con.execute(f"SELECT q.entity_id,t.entity_id FROM queries q JOIN corpus t ON q.{field}=t.{field} JOIN exact_blocks b ON b.block_key=t.{field} WHERE b.n<={cap} AND (q.country='' OR t.country='' OR lower(q.country)=lower(t.country))")
        for batch in chunks(cursor,5000):
            for qid,tid in batch:
                channels[method][lookup[qid]].add(tid)
    return [set().union(*(v[i] for v in channels.values())) for i in range(len(queries))], channels, caps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--distractors", type=int, default=30000)
    parser.add_argument("--fit-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--exact-only", action="store_true", help="Exact name/address diagnostic; skips fuzzy retrieval")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--db", type=Path, default=ROOT / "artifacts/audit.duckdb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/blocking_pilot")
    a = parser.parse_args()
    if min(a.queries, a.distractors, a.fit_size, a.batch_size, a.top_k) < 1:
        parser.error("Counts must be positive")
    started = time.monotonic()
    con = connect(a.db)
    # Sampling uses only IDs and a fixed seed; labels do not select query rows.
    con.execute(f"CREATE OR REPLACE TEMP TABLE queries AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM train_source1 ORDER BY md5(entity_id || '{a.seed}') LIMIT {a.queries}")
    queries = con.execute("SELECT * FROM queries ORDER BY entity_id").fetchall()
    truth = {i: set(v.split(',')) if v else set() for i, v in con.execute("SELECT t.source1_entity_id,t.matched_entity_ids FROM truth t JOIN queries q ON t.source1_entity_id=q.entity_id").fetchall()}
    if len(truth) != len(queries):
        raise ValueError("Missing/duplicate ground truth for evaluation queries")
    con.execute("CREATE OR REPLACE TEMP VIEW all_targets AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM train_source2 UNION ALL SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM train_source3")
    if a.full_corpus:
        con.execute("CREATE OR REPLACE TEMP VIEW corpus AS SELECT * FROM all_targets")
        scope = "sampled_queries_full_target_corpus"
    else:
        con.execute(f"CREATE OR REPLACE TEMP TABLE corpus AS SELECT * FROM (SELECT * FROM all_targets ORDER BY md5(entity_id || '{a.seed}') LIMIT {a.distractors}) UNION SELECT t.* FROM all_targets t JOIN edges e ON e.target_id=t.entity_id JOIN queries q ON q.entity_id=e.source1_entity_id")
        scope = "optimistic_pilot_true_targets_plus_sampled_distractors_NOT_validation"
    count = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    def batches():
        return chunks(con.execute("SELECT * FROM corpus"), a.batch_size)
    print(f"Scope: {scope}; {len(queries):,} queries against {count:,} targets", flush=True)
    if a.exact_only:
        candidates, channels, caps = exact_diagnostic(con, queries)
    else:
        fit_rows = con.execute(f"SELECT * FROM corpus ORDER BY md5(entity_id || 'fit{a.seed}') LIMIT {a.fit_size}").fetchall()
        candidates, channels, caps = retrieve(queries, batches, fit_rows, k=a.top_k, progress=True)
    result = metrics(queries, truth, candidates, count)
    result.update(scope=scope, seed=a.seed, caps=caps,
                  seconds=time.monotonic() - started,
                  tfidf_fit_note="Not used" if a.exact_only else "Unsupervised fixed ID-hash sample from evaluated corpus; vocabulary/IDF approximate full corpus",
                  parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    result["channel_metrics"] = {name: metrics(queries, truth, c, count) for name, c in channels.items()}
    result["country_metrics"] = {}
    for country in sorted({q[3] for q in queries}):
        indexes = [i for i,q in enumerate(queries) if q[3] == country]
        result["country_metrics"][country] = metrics([queries[i] for i in indexes], truth, [candidates[i] for i in indexes], count)
    result["source_recall"] = {}
    for source in ["S2", "S3"]:
        total = sum(sum(x.startswith(source + '-') for x in truth[q[0]]) for q in queries)
        hits = sum(sum(x.startswith(source + '-') for x in truth[q[0]] & c) for q,c in zip(queries,candidates))
        result["source_recall"][source] = {"positives": total, "hits": hits, "recall": hits/total if total else None}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(a.output_dir / "metrics.json", result)
    save_json(a.output_dir / "missed_pairs.json", [{"source1": q[0], "country": q[3], "missed": sorted(truth[q[0]]-c)} for q,c in zip(queries,candidates) if truth[q[0]]-c])
    with (a.output_dir / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        writer.writerows((q[0], ','.join(sorted(c))) for q,c in zip(queries,candidates))
    log_experiment("blocking", {k: result[k] for k in ["scope", "candidate_recall", "average_candidates", "seconds"]}, result["parameters"])
    print(json.dumps({k: v for k,v in result.items() if k not in ["channel_metrics", "parameters"]}, indent=2), flush=True)
    con.close()


if __name__ == "__main__":
    main()
