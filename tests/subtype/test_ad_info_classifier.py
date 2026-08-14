from spamdet.subtype.ad_info_classifier import BILGILENDIRME, REKLAM, AdInfoClassifier

REKLAM_EXAMPLES = [
    "Bu hafta magazamizda tum urunlerde yuzde 40 indirim var, kacirmayin!",
    "Sana ozel 100 TL indirim kuponu tanimlandi, hemen kullan.",
    "Yeni sezon urunlerinde ilk alisverisine yuzde 20 indirim firsati.",
    "Uyelere ozel kampanya basladi, secili urunlerde 2 al 1 ode.",
    "Simdi kayit ol, ilk siparişte 50 TL hediye cekin seni bekliyor.",
    "Kis kampanyamiz basladi, ucretsiz kargo firsatini kacirma.",
]
BILGILENDIRME_EXAMPLES = [
    "Kargonuz dagitima cikmistir, yarin teslim edilecektir.",
    "Faturaniz hazir, tutar 245,60 TL, son odeme 15.08.2026.",
    "Randevunuz 05.08.2026 saat 10:30 icin onaylanmistir.",
    "Hesabiniza 1.500,00 TL EFT girisi gerceklesmistir.",
    "Basvurunuz incelemeye alinmistir, sonuc 5 is gunu icinde bildirilecektir.",
    "Su faturaniz otomatik odeme talimatiniz geregi tahsil edilmistir.",
]


def _fit_classifier(*, reklam_threshold: float = 0.5, **kwargs) -> AdInfoClassifier:
    # default threshold=0.5 (neutral) here so basic prediction tests aren't
    # sensitive to the production default's recall-favoring bias (0.4) -
    # that bias is exercised deliberately in its own dedicated test below.
    clf = AdInfoClassifier(reklam_threshold=reklam_threshold, **kwargs)
    texts = REKLAM_EXAMPLES + BILGILENDIRME_EXAMPLES
    labels = [REKLAM] * len(REKLAM_EXAMPLES) + [BILGILENDIRME] * len(BILGILENDIRME_EXAMPLES)
    clf.fit(texts, labels)
    return clf


def test_predict_returns_reklam_for_promotional_text():
    clf = _fit_classifier()
    result = clf.predict("Bu hafta sonu tum ayakkabilarda yuzde 30 indirim, kacirma!")
    assert result.subtype == REKLAM


def test_predict_returns_bilgilendirme_for_transactional_text():
    clf = _fit_classifier()
    result = clf.predict("Siparisiniz kargoya verildi, teslimat 2 gun icinde.")
    assert result.subtype == BILGILENDIRME


def test_reklam_probability_always_reported_regardless_of_prediction():
    clf = _fit_classifier()
    result = clf.predict("Kargonuz dagitima cikti.")
    assert 0.0 <= result.reklam_probability <= 1.0
    assert result.confidence >= 0.0


def test_lower_threshold_favors_reklam_recall():
    strict = AdInfoClassifier(reklam_threshold=0.9)
    lenient = AdInfoClassifier(reklam_threshold=0.1)
    for clf in (strict, lenient):
        clf.fit(
            REKLAM_EXAMPLES + BILGILENDIRME_EXAMPLES,
            [REKLAM] * len(REKLAM_EXAMPLES) + [BILGILENDIRME] * len(BILGILENDIRME_EXAMPLES),
        )
    # a borderline text: with a very low threshold, lenient must call it
    # reklam at least as often as strict does across a batch
    borderline_texts = BILGILENDIRME_EXAMPLES + REKLAM_EXAMPLES
    strict_reklam_count = sum(1 for t in borderline_texts if strict.predict(t).subtype == REKLAM)
    lenient_reklam_count = sum(1 for t in borderline_texts if lenient.predict(t).subtype == REKLAM)
    assert lenient_reklam_count >= strict_reklam_count


def test_save_and_load_round_trip(tmp_path):
    clf = _fit_classifier(reklam_threshold=0.35)
    path = tmp_path / "model.joblib"
    clf.save(path)

    loaded = AdInfoClassifier.load(path)
    assert loaded.reklam_threshold == 0.35
    for text in REKLAM_EXAMPLES + BILGILENDIRME_EXAMPLES:
        assert loaded.predict(text).subtype == clf.predict(text).subtype
