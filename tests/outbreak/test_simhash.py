from spamdet.outbreak.simhash import hamming_distance, similarity, simhash
from spamdet.synthetic.augment import TemplateParaphraser


def test_identical_text_produces_identical_fingerprint():
    text = "Bonus kazandiniz hemen tiklayin bit.ly/kazan"
    assert simhash(text) == simhash(text)


def test_fingerprint_deterministic_across_calls():
    text = "hesabinizi dogrulayin"
    fingerprints = {simhash(text) for _ in range(5)}
    assert len(fingerprints) == 1


def test_empty_text_returns_zero_fingerprint():
    assert simhash("") == 0


def test_hamming_distance_zero_for_identical_fingerprints():
    fp = simhash("ayni metin")
    assert hamming_distance(fp, fp) == 0
    assert similarity(fp, fp) == 1.0


def test_hamming_distance_symmetric():
    a = simhash("birinci metin buradadir")
    b = simhash("tamamen farkli bir cumle bu")
    assert hamming_distance(a, b) == hamming_distance(b, a)


def test_similarity_is_between_zero_and_one():
    a = simhash("bir metin ornegi")
    b = simhash("baska bir metin ornegi daha")
    score = similarity(a, b)
    assert 0.0 <= score <= 1.0


def test_paraphrased_near_duplicate_is_more_similar_than_unrelated_text():
    original = "Tebrikler bonus kazandiniz hemen tiklayin bit.ly/bonus500"
    paraphraser = TemplateParaphraser(rng_seed=3)
    near_duplicate = paraphraser.paraphrase(original, n=1)[0]
    unrelated = "Yarin saat 15:00 te toplantimiz var lutfen unutma"

    fp_original = simhash(original)
    fp_near_dup = simhash(near_duplicate)
    fp_unrelated = simhash(unrelated)

    sim_near = similarity(fp_original, fp_near_dup)
    sim_unrelated = similarity(fp_original, fp_unrelated)
    assert sim_near > sim_unrelated


def test_text_shorter_than_shingle_size_does_not_raise_and_is_in_range():
    fp = simhash("ab", shingle_size=4)
    assert 0 <= fp < (1 << 64)
