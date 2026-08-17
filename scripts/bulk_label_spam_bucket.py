"""One-time bulk classification of the remaining "spam"-bucket rows in
data/manual_labels/, per the user's request after hand-labeling a first
sample (185 rows: 178 reklam, 6 spam, 1 bilgilendirme - see
src/spamdet/manual_labels.py's docstring for the discovery that this
bucket is overwhelmingly real ads, not fraud).

Approach, dry-run-reviewed before writing anything (see the conversation
this script came from): rather than trust a broad keyword rule (an
earlier draft using "nakit avans"/"son 6 hane" as spam markers had a bad
false-positive rate - those are completely normal, legitimate Turkish
bank card campaign mechanics, e.g. real VakifBank Worldcard/Paraf
promotions use the exact same "send your card's last 6 digits" pattern),
this uses a narrow, word-boundary-anchored, hand-verified list of
unambiguous gambling-site/betting-tip and adult-scam-product terms -
every one of its ~25 matches against the real data was individually
eyeballed and confirmed before this script was written. Everything else
in the spam bucket defaults to reklam, matching the ~96% base rate found
in the user's own hand-labeled sample. Garbled MIME quoted-printable
email fragments (turkish_spam_dataset was repurposed from an email
corpus - see docs/datasets.md) are skipped outright, not usable text for
either label.

This only touches the SPAM bucket, not ham - ham was already confirmed
to be genuinely mostly bilgilendirme, no bulk-relabeling case for it.

Usage:
    python scripts/bulk_label_spam_bucket.py            # dry run, prints counts only
    python scripts/bulk_label_spam_bucket.py --commit    # actually writes decisions
"""

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spamdet.manual_labels import (  # noqa: E402
    already_decided_texts,
    append_decision,
    build_candidate_pool,
    default_output_path,
)

# Hand-verified against every match in the real data (see the commit
# this script was introduced in) - word-boundary-anchored to avoid the
# earlier draft's false positives (e.g. an unanchored "bet[0-9]" matched
# "SOHBET400", a telecom talk-time campaign code; an unanchored
# "bet.*\.com" matched "betashoes.com", a shoe store).
GAMBLING = re.compile(
    r"\b(bahis|casino|iddaa|jojobet|betist|betkid\w*|intbet\w*|restbet\w*|safirbet\w*|tipobet\w*|mybahis\w*|superbahis|1xbet)\b",
    re.IGNORECASE,
)
ADULT_SCAM = re.compile(
    r"\b(cinsel|penis|geciktirici\w*|azdirici\w*|viagra|ereksiyon\w*|buyutucu\w*|cialis)\b", re.IGNORECASE
)
MIME_GARBLE = re.compile(r"=[0-9A-F]{2}=[0-9A-F]{2}", re.IGNORECASE)


def classify(text: str) -> str:
    if MIME_GARBLE.search(text):
        return "skip"
    if GAMBLING.search(text) or ADULT_SCAM.search(text):
        return "spam"
    return "reklam"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="actually write decisions (default: dry run)")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args(argv)

    output_path = default_output_path(PROJECT_ROOT)
    pool = build_candidate_pool(args.raw_dir)
    done = already_decided_texts(output_path)
    remaining_spam_bucket = [item for item in pool if item["original_label"] == "spam" and item["text"] not in done]

    counts = {"spam": 0, "reklam": 0, "skip": 0}
    for item in remaining_spam_bucket:
        label = classify(item["text"])
        counts[label] += 1
        if args.commit:
            append_decision(
                output_path,
                text=item["text"],
                label=label,
                original_label=item["original_label"],
                original_source=item["original_source"],
                method="bulk_rule",
            )

    print(f"islenen satir: {len(remaining_spam_bucket)}")
    print(f"  reklam (varsayilan): {counts['reklam']}")
    print(f"  spam (kural eslesti): {counts['spam']}")
    print(f"  skip (bozuk kodlama): {counts['skip']}")
    if not args.commit:
        print("\n[dry run] hicbir sey yazilmadi. Gercekten uygulamak icin --commit ekleyin.")
    else:
        print(f"\nyazildi: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
