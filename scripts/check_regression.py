"""Committed, durable regression check against a trained model - accumulates
every (label, text) pair that got explicit human confirmation (mostly the
project owner, live-testing real/user-authored SMS) across this project's
fix-and-retrain sessions. Unlike the scratchpad probe scripts used during
those sessions (gitignored, session-only), this file is meant to survive
across sessions so a future retrain has something concrete to check itself
against instead of re-deriving "what should this message classify as" from
conversation history.

This is NOT part of `pytest` (loading the model takes a few seconds and
runs 60+ inferences - too slow/heavy for the fast offline unit-test suite,
and CLAUDE.md's "Status" section explains why one run's pass/fail here
shouldn't be over-trusted in isolation: training has real run-to-run
variance even with identical data - see docs/model.md's 2026-08-18 update).
Run it by hand after any retrain:

    python scripts/check_regression.py

Exits non-zero if anything fails, so it's also usable as a CI gate later
if training is ever automated.

IMPORTANT: cases are written with real Turkish diacritics (ç ğ ı ö ş ü İ),
not ASCII-transliterated - testing the formal-register cases with ASCII
was found to mask a real gap (see docs/model.md). Keep it that way when
adding more cases.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spamdet.api.config import get_model_dir  # noqa: E402
from spamdet.api.pipeline import ClassificationPipeline  # noqa: E402
from spamdet.model.inference import SpamClassifier  # noqa: E402

# (expected_label, text, note) - note is free text: what this case is
# guarding against / where it came from.
CASES: list[tuple[str, str, str]] = [
    # --- established "known regression" set (built up across many rounds) ---
    ("reklam", "Hepsiburada'da bugüne özel %40 indirim kuponu seni bekliyor! Kodu kullan: HEPSI40. Mersis: 0123456789. STOP yazarak reklam mesajlarından çıkabilirsiniz.", "branded coupon + Mersis + STOP opt-out"),
    ("reklam", "Trendyol'da seçili ürünlerde yüzde 50'ye varan indirim başladı! Kaçırma, hemen incele.", "plain retail discount"),
    ("reklam", "Migros'ta bu hafta seçili gıda ürünlerinde yüzde 25 indirim var, kaçırma!", "plain retail discount"),
    ("reklam", "Bugüne özel %40 indirim kuponu seni bekliyor! Kodu kullan: HEPSI40.", "KNOWN FRAGILE - brandless/bare discount, has flipped to spam across multiple retrains"),
    ("otp", "GÜVENLİĞİNİZ İÇİN KİMSEYLE PAYLAŞMAYINIZ! Hızlı Giriş şifreniz 865458 ile GELECEGIYAZANLAR uygulamasına giriş yapabilirsiniz. Referans: Y4YX6 B002", "real user OTP, ALL CAPS"),
    ("bilgilendirme", "WTS0303142395 nolu siparişiniz HepsiJET'e ulaşmış olup tahmini olarak 29.09.2026 tarihinde teslim edilecektir. Teslimatınızı takip etmek için: hpj.im/gm0fzrtj", "real cargo tracking, branded shortlink"),
    ("bilgilendirme", "Pluxee dünyasına hoş geldin! Restaurant Pass Yemek Kartın tanımlandı. Hemen Pluxee Türkiye mobil uygulamasını indir ve kullanmaya başla! bit.ly/pluxee_mobil", "real onboarding message, generic shortlink"),
    ("bilgilendirme", "VDAS aboneliğiniz talebiniz üzerine iptal edilmiştir. Ödemesini yapmış olduğunuz döneme ait faydalardan ilgili tarihe kadar yararlanmaya devam edebilirsiniz.", "real subscription cancellation"),
    ("bilgilendirme", "Hediye Çarkından kazandığınız 1 gün geçerli 1 GB internet paketiniz hesabınıza yüklenmiştir. Güle güle kullanın!", "real reward-credit confirmation, 'kazandiginiz' in a non-scam context"),
    ("bilgilendirme", "Değerli yolcumuz, ne düşündüğünüzü önemsiyor, geri bildirimlerinizi alarak size daha iyi hizmet vermek için çalışıyoruz. Bizimle yaptığınız son yolculuğu değerlendirmek için tıklayınız! SMS almak istemiyorsanız RET yazıp 6235'e SMS gönderebilirsiniz.", "real survey invite - hard negative, has opt-out phrase but is not an ad"),
    ("bilgilendirme", "Hesabınıza 1.500,00 TL EFT girişi gerçekleşmiştir. Bakiye: 3.240,00 TL. Garanti BBVA", "real bank transaction notice"),
    ("reklam", "Apple iPhone için 1250 TL kuponun var. Aklında telefonunu yenileme düşüncesi varsa, şimdi tam zamanı. Sınırlı süre geçerli sana özel hediyeni şimdi kullan. https://app.hb.biz/0L3Wl Ret için HBSMS yaz 3172'ye ilet. Mersis:0265017991000011", "real Hepsiburada coupon, Mersis + opt-out"),
    ("spam", "Tebrikler! Bahis hesabınıza 500 TL bonus tanımlandı. Hemen çekmek için: bit.ly/bonus500", "gambling"),
    ("spam", "Bankanız güvenlik nedeniyle hesabınızı kısıtladı. Doğrulamak için: guvenlik-bankam.com/dogrula", "phishing with link"),
    ("spam", "Şimdi doğrulama yapmazsanız hesabınız 24 saat içinde kapatılacaktır!", "KNOWN FRAGILE - URL-less threat/urgency phishing, has flipped to bilgilendirme across multiple retrains"),
    ("spam", "Anne, telefonum bozuldu bu numaradan yazıyorum, acil 2000 TL gönderir misin, akşam öderim", "financial_urgency hijacked-relative scam - regressed twice in one session from unrelated reklam.yaml edits"),

    # --- novel-brand generalization set (brands/wording not in any seed file) ---
    ("reklam", "CarrefourSA'da bu hafta et ürünlerinde yüzde 20 indirim var, kaçırma!", "novel brand"),
    ("reklam", "Türk Telekom'dan yeni yıl hediyesi: 10 GB ek internet hesabına tanımlandı, hemen kullan!", "novel brand"),
    ("reklam", "Bellona'da yatak odası takımlarında yüzde 35 indirim! Kaçırma.", "novel brand"),
    ("reklam", "Passo'da futbol maçı biletlerinde erken rezervasyon indirimi başladı!", "novel brand"),
    ("reklam", "Cinemaximum'da bu hafta bilet fiyatları yüzde 50 indirimli! Hemen bilet al.", "novel brand"),
    ("reklam", "Yurtiçi Kargo'dan sana özel: bu ay gönderi ücretlerinde yüzde 15 indirim kampanyası!", "novel brand"),
    ("reklam", "Klima bakım hizmetimizde bu ay özel fırsatlar sizi bekliyor, hemen bizi arayın.", "novel brand, service sector"),
    ("reklam", "Arkadaşını uygulamaya davet et, sen ve arkadaşın 50 TL indirim kazanın! Davet kodun: FR8823.", "KNOWN FRAGILE - referral-bonus ad, structurally close to gambling referral scams, has flipped to spam repeatedly - a fix was tried and reverted after it regressed 3 other cases, see reklam.yaml's note"),
    ("reklam", "Tebrikler! Alışveriş programımızda 750 TL değerinde puan kazandınız, hesabınızda kullanıma hazır.", "loyalty-points hard negative, 'kazandiniz' in non-scam context"),
    ("reklam", "Milli Piyango yeni yıl çekilişi biletleri satışta! Bayilerden hemen temin edin.", "state lottery ad, gambling-adjacent but legal"),

    # --- real reklam messages from the user's phone ---
    ("reklam", "İlk çiçek siparişin öğretmenine yakışır! Semtinin çiçekçilerinden vereceğin ilk siparişinde 150 TL indirimi kaçırma! https://getircarsi.onelink.me/Eaz7/f8727f49 Mersis:0394048265800010 SMS reddi için: http://onay.li/PU34", "real Getir Carsi message"),
    ("reklam", "Ağustos ayında QNB Kredi Kartı, First kredi kartı veya vadesiz hesabınızla vereceğiniz ilk 3 yeni otomatik fatura ödeme talimatına 150'şer TL, sonraki 2 talimata 275'er TL olmak üzere, toplamda 25 gün geçerli 1000 TL'ye varan ParaPuan hediye! Detay: https://qnb.mn/43095qc 08502220900 SMSRET->3639 Mersis:0388002333400576", "real QNB message, long/formal banking-campaign style"),
    ("reklam", "Kampanya: Sevgili Money'li üniversite öğrencileri Migros'ta ayrıcalıklı! Migros Hemen'de ücretsiz teslimat, en sevilen kategori ve markalardan fırsatlar seni bekliyor! Ayrıca tek seferde 750 TL ve üzeri yapacağın ilk alışverişinde 100 Money hediye! Kampanya UniMoney'ye ilk üye olma tarihinden itibaren yalnızca bir kere kullanılabilir. UniMoney'ye ücretsiz üye olmak ve detaylı bilgi almak için tıkla: https://mgrs.link/e607d Migros Tic. SMS Ret: 7447'ye RET MG yaz MN:0622052951300016", "real Migros UniMoney message"),
    ("reklam", "ARNAK'TAN İZMİT'E ÖZEL 2 ADET EV TÜPÜ SADECE 1600 TL 3 TEMMUZ SON 4443727 05413154141 ÇIKIŞ için AR54 yaz 4609 Gönder. MRS:0080001324800017", "real ARNAK LPG message, terse ALL-CAPS local-business register"),

    # --- gambling / phishing / cargo-scam real messages ---
    ("spam", "200 FREE EGE HESABINDA CEVRIM YOK! 10.000 TL HOSGELDIN BONUSU SADECE EGE OZEL JACKPOT 100 TL ANLIK HAVALE 25 MILYON GUVENLI CEKIM https://dub.sh/egecom", "real EGE gambling message - modern betting-site vocabulary (cevrim yok/hosgeldin bonusu/jackpot)"),
    ("spam", "GULMTAL arafindan adiniza, AP20231222167 barkodlu alici odemeli gonderi kabul edilmistir. Odeme: K.Karti Tutar: 24.251,65 TL Bilgi: https://bit.ly/487moOS", "real fake-cargo-COD-payment phishing message"),
    ("reklam", "TIKLA KAZAN! Samsung TV sepette sadece 1 TL https://app.hb.biz/iJd2N Ret icin HBSMS yaz 3172'ye ilet. Mersis:0265017991000011", "real Hepsiburada message, clickbait opener + Mersis"),
    ("reklam", "BITIYOR! Buyuk Agustos Indirimleri'nde Buyuk Final'i kacirma! https://app.hb.biz/BlwkG Ret icin HBSMS yaz 3172'ye ilet. Mersis:0265017991000011", "real Hepsiburada message, ALL-CAPS urgency opener"),
    ("reklam", "Seni Özledik! Mağazalarımızdan veya Cepte ŞOK'tan 1000 TL ve üzeri yapacağınız alışverişe 100 win para hediye! Avantajı kaçırmayın! Son gün: 19.08.2026 Alışverişe başlamak için tıklayın: https://ceptesok.go.link/gPKXm SMS almak istemiyorsanız SOK RET yazın 4933'e gönderin. Mersis No:0814013189988099", "real SOK message, spend-threshold-reward framing"),

    # --- Oruntu A/B: formal-register bilgilendirme (existing-benefit-status /
    #     calm security notices), confirmed correct by the user across a
    #     137-message stress-test batch - a representative slice, not all 137 ---
    ("bilgilendirme", "Hesabınızdaki kampanya avantajı otomatik olarak yenilenmeyecektir. Kullanmak istiyorsanız geçerlilik tarihini kontrol ediniz.", "Oruntu A: existing-benefit status, not a new offer"),
    ("bilgilendirme", "Kampanya mesajlarını almak istemiyorsanız iletişim tercihlerinizi hesabınızdan değiştirebilirsiniz.", "Oruntu A: opt-out settings notice, no promotional content itself"),
    ("bilgilendirme", "Kampanya koşullarında değişiklik yapılmıştır. Güncel koşulları resmi internet sitemizden inceleyebilirsiniz.", "Oruntu A: the exact case that first exposed the ASCII-vs-diacritics testing gap"),
    ("bilgilendirme", "Önemli bilgilendirme: Yeni dönem hizmet bedelleri 1 Eylül itibarıyla uygulanacaktır.", "Oruntu A: fee-change notice"),
    ("bilgilendirme", "Randevunuzu iptal etmek için bu mesaja yanıt vermeyiniz; müşteri hizmetleri üzerinden işlem yapabilirsiniz.", "Oruntu B: calm account-process notice, had scored spam before the fix"),
    ("bilgilendirme", "Hesabınızın bazı özellikleri güvenlik nedeniyle geçici olarak devre dışı bırakılmıştır.", "Oruntu B: shares vocabulary with phishing.yaml almost verbatim but has no link/urgency"),
    ("bilgilendirme", "Siparişinizde teslimat adresi değişikliği talep edilmiştir. Bu işlem size ait değilse destek ekibimizle iletişime geçiniz.", "Oruntu B: e-commerce fraud-alert pattern, passive"),
    ("bilgilendirme", "Kartınız güvenlik amacıyla geçici olarak kullanıma kapatılmıştır. Yeniden aktifleştirme seçenekleri için müşteri hizmetlerini arayınız.", "Oruntu B: directs to generic customer service, not a specific untrusted number"),
    ("bilgilendirme", "Hesabınızda kullanılmamış bir hediye bakiyesi bulunmaktadır. Son kullanım tarihini kontrol etmek için hesabınızı ziyaret ediniz.", "Oruntu A: existing balance status"),
    ("bilgilendirme", "Randevunuzu hatırlatmak isteriz. 21 Ağustos saat 14:30'daki randevunuz için değişiklik yapmak isterseniz bizi arayabilirsiniz.", "clearest single miss in the original 17-message batch, user-confirmed bilgilendirme"),
    ("bilgilendirme", "Sayın müşterimiz, hesabınızla ilgili önemli bir bilgilendirme bulunmaktadır. Detaylar için 0850 000 00 00 numaralı müşteri hizmetlerimizi arayabilirsiniz.", "Oruntu B: generic 0850 customer-service number, not suspicious"),

    # --- new-offer reklam, confirmed correct across the same batch/discussion ---
    ("reklam", "Tebrikler, başvurunuz kapsamında size özel bir teklif oluşturuldu. Teklif detaylarını hesabınızdan görebilirsiniz.", "new-offer pattern: contrast pair against the Oruntu A bilgilendirme cases above (existing-benefit status vs. brand-new offer)"),
    ("reklam", "Hesabınıza 1.500 TL limitli yeni bir teklif tanımlandı. Teklifi değerlendirmek için başvuru ekranına geçebilirsiniz.", "new-offer pattern, from the 137-message batch"),
    ("reklam", "Size tanımlanan 500 TL avantajı kullanmak için son gün. Ayrıntılı koşullar kampanya sayfamızda yer almaktadır.", "user's explicit call: urgency-pushing use of an expiring benefit is promotional, despite reading close to Oruntu A on the surface"),
    ("reklam", "Size özel hazırlanan finansman teklifinin süresi bugün sona ermektedir. Detaylı bilgi için başvuru ekranını inceleyiniz.", "new-offer pattern"),
    ("reklam", "Son fırsat! Hesabınıza tanımlanan avantajın kullanım süresi bugün saat 23:59'da sona erecektir.", "new-offer/urgency pattern, ALL-CAPS-clickbait-idiom family"),
    ("reklam", "Kullanım alışkanlıklarınıza göre size uygun yeni bir teklif hazırlandı. Ayrıntıları hesabınızdan inceleyebilirsiniz.", "new-offer pattern"),
]


def main() -> int:
    classifier = SpamClassifier(get_model_dir())
    pipeline = ClassificationPipeline(classifier)

    passed = 0
    failed: list[tuple[str, str, str, float]] = []
    for expected, text, note in CASES:
        result = pipeline.process("regression-check", text)
        got = result.prediction.label
        ok = got == expected
        passed += ok
        status = "OK  " if ok else "MISS"
        print(f"{status} exp={expected:14s} got={got:14s} {result.prediction.confidence:.3f}  {text[:60]}")
        if not ok:
            failed.append((expected, got, note, result.prediction.confidence))

    print(f"\n{passed}/{len(CASES)} correct")
    if failed:
        print("\nFAILURES:")
        for expected, got, note, confidence in failed:
            print(f"  expected {expected}, got {got} ({confidence:.3f}) - {note}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
