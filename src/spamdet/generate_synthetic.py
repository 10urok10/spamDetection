import argparse
import json
from pathlib import Path

from .schema import Record
from .synthetic.adversarial import generate_adversarial_set
from .synthetic.augment import TemplateParaphraser, augment_examples
from .synthetic.seeds import load_all_seeds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "synthetic" / "seeds"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "generated"


def _write_jsonl(path: Path, records: list[Record]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {"text": r.text, "label": r.label.value, "source": r.source, "lang": r.lang.value, "extra": r.extra},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate augmented + adversarial synthetic training data from hand-authored seeds.")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-per-seed", type=int, default=3, help="paraphrase variants generated per seed example")
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args(argv)

    seeds = load_all_seeds(args.seed_dir)
    if not seeds:
        print(f"No seed examples found under {args.seed_dir}")
        return 1

    paraphraser = TemplateParaphraser(rng_seed=args.rng_seed)
    augmented = augment_examples(seeds, paraphraser, n_per_seed=args.n_per_seed)
    # Adversarial corruption is applied to both the hand-authored seeds and
    # their paraphrases, so the model sees evasion variants across the
    # whole surface-form distribution, not just the original 61 examples.
    adversarial = generate_adversarial_set(seeds + augmented, rng_seed=args.rng_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "seeds.jsonl", seeds)
    _write_jsonl(args.out_dir / "augmented.jsonl", augmented)
    _write_jsonl(args.out_dir / "adversarial.jsonl", adversarial)

    print(f"seeds: {len(seeds)}, augmented: {len(augmented)}, adversarial: {len(adversarial)}")
    print(f"written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
