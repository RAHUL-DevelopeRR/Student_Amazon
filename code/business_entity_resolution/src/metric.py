"""Macro F0.5 from the official formula, including singleton special handling."""
import argparse
import csv


def entity_f05(truth, predicted):
    truth, predicted = set(truth), set(predicted)
    if not truth:
        return float(not predicted)
    tp = len(truth & predicted)
    # Equivalent to 1.25*TP/(1.25*TP + FP + .25*FN).
    return 1.25 * tp / (0.25 * len(truth) + len(predicted))


def macro_f05(truth, predicted):
    if not truth:
        raise ValueError("Cannot score an empty evaluation population")
    if truth.keys() != predicted.keys():
        raise ValueError("Prediction keys must exactly cover the evaluation population")
    return sum(entity_f05(v, predicted[k]) for k, v in truth.items()) / len(truth)


def read_mapping(path, column="matched_entity_ids"):
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", column]:
            raise ValueError(f"Unexpected TSV schema: {reader.fieldnames}")
        for row in reader:
            key, value = row["source1_entity_id"], row[column]
            if not key or key in result or value is None or None in row:
                raise ValueError(f"Duplicate or malformed row: {key}")
            ids = value.split(",") if value else []
            if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                raise ValueError(f"Invalid target list: {key}")
            result[key] = set(ids)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("truth")
    parser.add_argument("predictions")
    args = parser.parse_args()
    print(macro_f05(read_mapping(args.truth), read_mapping(args.predictions)))
