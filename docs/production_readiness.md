# Uretime Gecis Notu

Bu belge, mevcut MVP'nin gercek uretim ortamina tasinmasi icin gereken ek
calismayi, hukuki/operasyonel degerlendirmeleri ve tahmini maliyetleri
ozetler. Bu belgedeki hicbir madde koda yansitilmamistir - MVP kapsami
bilinclli olarak bunlari icermez; asagidakiler bir sonraki asamalarin
girdisi olarak dusunulmelidir.

## 1. KVKK / Veri Saklama

- Sistem ucuncu kisilere ait SMS/e-posta icerigini isleyecek (KVKK
  kapsaminda kisisel veri isleme faaliyeti).
- MVP'de yapilmayan, uretim oncesi gerekli isler:
  - Aydinlatma metni / hukuki isleme sebebinin netlestirilmesi (ornegin
    operator/banka ile sozlesme iliskisi ya da mesru menfaat).
  - **Veri saklama suresi politikasi**: outbreak fingerprint kaydi icin
    `RedisLSHIndex.add(ttl_seconds=...)` parametresi zaten var, ama
    `OutbreakDetector` varsayilan olarak `ttl_seconds=None` (sinirsiz)
    kullaniyor. Uretimde mutlaka bir TTL (ornegin 30-90 gun) set
    edilmeli.
  - `data/review/confirmed.jsonl` dosyasi insan-onayli gercek mesaj
    metinlerini duz metin olarak, sifrelenmemis ve erisim kontrolsuz
    bicimde biriktiriyor. Uretimde: sifreli depolama, erisim loglari,
    ve sinirli saklama suresi gerekir.
  - Veri sahibi haklari (erisim/silme talebi) icin bir mekanizma yok -
    `message_id` bazli silme islevi eklenmeli.
  - Bulut saglayici/veri lokasyonu secimi KVKK'nin yurt disina veri
    aktarimi hukumleri acisindan degerlendirilmeli (model agirliklari
    Hugging Face'ten indiriliyor ama kullanici verisi disariya
    gonderilmiyor - bu iyi, ama barindirma lokasyonu ayri bir konu).

## 2. Tahmini Altyapi Maliyeti

Kaba bir buyukluk-mertebesi tahmini - gercek trafik hacmine gore
degisir. Ornek senaryo: gunde ~100.000 mesaj.

| Kalem | Tahmini aylik maliyet | Not |
|---|---|---|
| API compute (ONNX CPU inference) | ~$30-50 | Model kucuk, GPU sart degil; 2 vCPU/4GB yeterli olmali |
| Redis (managed, tek node) | ~$15-30 | Mevcut docker-compose Redis'i tek sunucuda HA'siz calisir |
| Periyodik GPU retraining | ~$20-50 | Haftada birkac saat spot/on-demand GPU instance |
| Depolama (model + veri) | <$5 | Checkpoint'ler ~0.5-1GB, veri seti kucuk |
| **Toplam (kaba)** | **~$70-150/ay** | Kucuk-orta trafik icin; trafik artisiyla dogrusal degil daha hizli buyuyebilir |

Bu tablo sadece buyukluk mertebesi vermek icindir - gercek karar icin
bulut saglayici fiyat hesaplayicilariyla ve gercek trafik verisiyle
detayli calisma yapilmalidir.

## 3. Model Drift / Retraining Stratejisi

- **Drift izleme**: `/metrics`'teki `spamdet_confidence` histogramindaki
  ortalama guven skorunun zamanla dusmesi, modelin gordugu mesaj
  dagiliminin egitim verisinden uzaklastiginin bir isareti olabilir.
  `spamdet_review_queue_total` oranindaki artis da benzer bir sinyal.
- **Retraining tetikleyicileri**: (a) belirli bir insan-onayli veri
  hacmine ulasildiginda (ornegin her 500 yeni onayli ornek), (b) sabit
  bir zaman araliginda (ornegin aylik), (c) confidence dagiliminda
  belirgin bir kayma tespit edildiginde.
- **Retraining sureci**: `data/review/confirmed.jsonl` icindeki veriler
  su an `build_training_dataframe()` pipeline'ina otomatik dahil
  DEGIL - bu, bir sonraki gelistirme adimi olarak not edilmelidir (yeni
  bir "human_reviewed" loader/kaynak eklenerek). Sonrasinda
  `scripts/train_model.py` yeniden calistirilir, yeni model bir staging
  ortaminda mevcut modelle karsilastirilir, metrikler gecerse uretime
  alinir.
  Ayrica bkz. `docs/model.md`'deki egitim/test split uyarisi - retraining
  oncesi split stratejisi (seed-gruplu split) gozden gecirilmeli.
- **Model versiyonlama**: su an yok - bir model registry (MLflow ya da
  basitce versiyonlu depolama klasorleri) ile hangi modelin ne zaman/ne
  veriyle egitildiginin izlenebilir olmasi gerekir.
- **Rollback plani**: yeni model uretimde beklenmedik sekilde kotu
  performans gosterirse `SPAMDET_MODEL_DIR` ortam degiskenini onceki
  surume isaret edecek sekilde degistirip servisi yeniden baslatmak
  yeterli olacak sekilde tasarlandi - ama bu akis henuz otomatiklestirilmedi.

## 4. Insan-Dongu Operasyon Sureci

- **Kim inceleyecek?**: Mevcut dashboard tek kullanicili/kimliksiz.
  Uretimde rol bazli erisim ve "hangi karari kim verdi" loglamasi
  (hesap verebilirlik icin) gerekir.
- **SLA**: Inceleme kuyrugu ne kadar surede bosaltilmali? Bahis/phishing
  gibi zaman-kritik dolandiriciliklar icin saatler icinde inceleme
  idealdir; kuyruk buyurse otomatik uyari (ornegin
  `spamdet_review_queue_total` artis hizina dayali bir Prometheus
  alert'i) kurulmalidir.
- **Kalite kontrolu**: Farkli reviewer'larin ayni mesaja ne siklikla ayni
  karari verdigini olcen bir mekanizma yok - kucuk ekiplerde onemsiz
  olabilir ama buyudukce eklenmelidir.
- **Geri besleme dongusu**: Onaylanan/duzeltilen etiketler otomatik
  olarak `confirmed.jsonl`'a ekleniyor (`append_confirmed_record`), ama
  bu veri henuz otomatik olarak yeniden egitime dahil edilmiyor (bkz.
  yukaridaki "Retraining sureci"). Yanlis-pozitif/yanlis-negatif
  oranlarini zaman icinde izleyen periyodik bir rapor operasyon ekibine
  gitmelidir.
- **Escalation**: Yuksek benzerlikli bir outbreak uyarisi (ornegin
  >0.95, cok sayida benzer mesaj) geldiginde, bu su an sadece API
  yanitinda/metrics'te pasif olarak goruluyor - insan incelemesini
  beklemeden aktif bir bildirim kanalina (ornegin Slack webhook)
  yonlendirilmesi uretim icin eklenmelidir.

## Kapsam disi birakilanlar (bilinçli MVP kararlari, hatirlatma)

- SBERT + vektor DB ikincil outbreak katmani, periyodik HDBSCAN batch
  clustering (bkz. `docs/outbreak.md`)
- URL unshortening'in senkron `/classify` yoluna entegre edilmemesi (bkz.
  `pipeline.py` docstring'i) - uretimde ayri bir async zenginlestirme
  adimi olarak eklenebilir
- CharBERT / ikinci "robustness" modeli
- Tam ag izolasyonu (SSRF korumasi su an sadece IP-bazli engelleme,
  izole proxy/network degil - bkz. `docs/datasets.md` ve Stage 1 README
  notlari)
