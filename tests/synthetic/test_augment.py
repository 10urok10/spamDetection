from spamdet.schema import Label, Lang, Record
from spamdet.synthetic.augment import TemplateParaphraser, augment_examples


def test_paraphrase_returns_exactly_n_distinct_variants():
    paraphraser = TemplateParaphraser(rng_seed=1)
    text = "Bonus kazandiniz hemen tiklayin"
    variants = paraphraser.paraphrase(text, n=3)
    assert len(variants) == 3
    assert len(set(variants)) == 3
    assert text not in variants


def test_paraphrase_deterministic_given_same_seed():
    a = TemplateParaphraser(rng_seed=7).paraphrase("hesabinizi dogrulayin", n=3)
    b = TemplateParaphraser(rng_seed=7).paraphrase("hesabinizi dogrulayin", n=3)
    assert a == b


def test_paraphrase_differs_with_different_seed():
    a = TemplateParaphraser(rng_seed=1).paraphrase("bonus kazandiniz tiklayin", n=3)
    b = TemplateParaphraser(rng_seed=2).paraphrase("bonus kazandiniz tiklayin", n=3)
    assert a != b


def test_paraphrase_handles_text_with_no_known_trigger_words():
    paraphraser = TemplateParaphraser(rng_seed=1)
    variants = paraphraser.paraphrase("xyz abc qwe", n=2)
    assert len(variants) == 2
    assert len(set(variants)) == 2


def test_augment_examples_preserves_label_and_tags_original_text():
    seeds = [Record(text="bonus kazandiniz", label=Label.GAMBLING_SCAM, source="synthetic_seed", lang=Lang.TR)]
    augmented = augment_examples(seeds, TemplateParaphraser(rng_seed=5), n_per_seed=3)
    assert len(augmented) == 3
    for rec in augmented:
        assert rec.label is Label.GAMBLING_SCAM
        assert rec.source == "synthetic_augmented"
        assert rec.extra["original_text"] == "bonus kazandiniz"
        assert rec.text != "bonus kazandiniz"
