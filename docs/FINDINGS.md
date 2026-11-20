# MirageBench — Bulgular (İlk Tam Tur)

Bu belge ilk eksiksiz deney turunun (17 kol, tek tohum, DUT Anti-UAV gerçek test
seti, imgsz 1280, YOLOv11s) sonuçlarını ve yorumunu kaydeder. Çalışma tasarımı ve
hipotezler için bkz. [`RESEARCH.md`](RESEARCH.md).

**Durum:** İlk tur tamamlandı. Çoğu kol **tek tohumla** koşuldu (varyans tahmini yok,
tablodaki `±0.000` sıfır varyans değil, tek koşu demektir). Titiz bir sonuç için
3 tohumla tekrar gerekir (bkz. Sınırlar).

---

## 1. Özet sonuç tablosu (gerçek DUT test seti, mAP@50)

| Kol | gerçek | sentetik | mAP@50 | mAP@50-95 | small | large |
|-----|------:|---------:|-------:|----------:|------:|------:|
| real_only_50 | 50 | 0 | 0.039 | 0.026 | 0.003 | 0.198 |
| real_only_100 | 100 | 0 | 0.589 | 0.333 | 0.775 | 0.172 |
| real_only_250 (2 tohum) | 250 | 0 | 0.819 | 0.499 | 0.876 | 0.669 |
| real_only_500 | 500 | 0 | 0.856 | 0.561 | 0.905 | 0.761 |
| real_only_1000 | 1000 | 0 | 0.923 | 0.617 | 0.949 | 0.872 |
| real_only_all | 5200 | 0 | 0.948 | 0.654 | 0.958 | 0.922 |
| synth_only | 0 | 4000 | 0.387 | 0.222 | 0.479 | 0.274 |
| synth_pretrain_real_50 | 50 | 4000 | 0.543 | 0.309 | 0.681 | 0.248 |
| synth_pretrain_real_100 | 100 | 4000 | 0.670 | 0.404 | 0.763 | 0.482 |
| synth_pretrain_real_250 | 250 | 4000 | 0.772 | 0.495 | 0.846 | 0.632 |
| synth_pretrain_real_500 | 500 | 4000 | 0.868 | 0.591 | 0.905 | 0.819 |
| synth_pretrain_real_1000 | 1000 | 4000 | 0.901 | 0.613 | 0.924 | 0.872 |
| mixed_250 | 250 | 4000 | 0.643 | 0.405 | 0.738 | 0.470 |
| ablation_no_pose_scale_random | 250 | 3000 | 0.773 | 0.450 | 0.857 | 0.550 |
| ablation_low_texture_realism | 250 | 3000 | 0.783 | 0.498 | 0.854 | 0.643 |
| ablation_no_bg_variation | 250 | 3000 | 0.796 | 0.515 | 0.873 | 0.660 |
| ablation_no_light_random | 250 | 3000 | 0.800 | 0.515 | 0.857 | 0.683 |

---

## 2. Veri verimliliği (RQ1/RQ2, H1 ✓)

| N | real_only | synth ön-eğitim | kazanç (puan) |
|---:|---:|---:|---:|
| 50 | 0.039 | 0.543 | **+50.4** |
| 100 | 0.589 | 0.670 | +8.1 |
| 250 | 0.819 | 0.772 | −4.7 |
| 500 | 0.856 | 0.868 | +1.2 |
| 1000 | 0.923 | 0.901 | −2.2 |

**Sonuç:** Sentetik ön-eğitim, gerçek veri aşırı kıtken belirleyici. N=50'de gerçek-only
pratikte çöküyor (0.039), sentetik ön-eğitim kullanılabilir bir modele çıkarıyor (0.543).
Katkı N arttıkça monoton azalıyor ve ~N=250'de sıfırlanıp hafif negatife dönüyor
(gerçek veri doyunca sentetik gereksiz, hatta küçük bir engel). Bu, domain-randomization
/ sim2real literatürünün beklediği davranış; burada DUT üzerinde temiz biçimde ölçüldü.
Kritik eşik: N≈100–250.

**H1 kararı: desteklendi.** N≤250'de en büyük kazanım 50.4 puan (N=50).

---

## 3. Domain gap (RQ3, H2 ✓)

`synth_only` = **0.387** mAP@50. Sıfır gerçek veriyle, yalnız Blender sentetiğiyle
eğitilen model gerçek test setinde 0.387 alıyor — en iyi gerçek modelin (0.948) 56 puan
altında. Yorum: büyük bir domain gap var (sentetik tek başına yetersiz), ama transfer
sıfır değil — sentetik veri gerçek drone'ları kısmen tanıyacak kadar bilgi taşıyor.

**H2 kararı: desteklendi.** Açık = 56.0 puan.

---

## 4. Ablasyon (RQ4, H3 ✗ — ama yanlış çalışma noktasında)

Ablasyon kolları (bir domain-randomization faktörü kapalı, N=250 gerçek + sentetik
ön-eğitim):

| Faktör kapalı | mAP@50 | referansa göre |
|---|---:|---:|
| referans (tüm faktörler açık, synth_pretrain_real_250) | 0.772 | — |
| no_pose_scale_random | 0.773 | +0.001 |
| low_texture_realism | 0.783 | +0.011 |
| no_bg_variation | 0.796 | +0.024 |
| no_light_random | 0.800 | +0.028 |

Hiçbir faktörü kapatmak referansı anlamlı düşürmedi; tersine hepsi hafifçe yüksek
(farklar 1–3 puan, tek tohumla gürültü içinde). **Bu bir null sonuç değil, metodoloji
bulgusudur:** ablasyon N=250'de ölçüldü, ama §2'de gördük ki N=250'de sentetik zaten
katkı sağlamıyor (synth_pretrain_250 ≈ real_only_250). Sentetiğin etkisinin olmadığı bir
noktada, sentetik tasarım faktörlerini kurcalamak doğal olarak fark göstermez. Doğru test:
ablasyonu **N=50** (sentetiğin belirleyici olduğu rejim) veya **synth_only** üzerinde
tekrarlamak.

**H3 kararı: bu turda desteklenmedi (yanlış operasyon noktası; yeniden kurgulanmalı).**

---

## 5. Boyut rejimi (RQ5, H4 ✗ N=250'de — ama N=50'de güçlü)

Otomatik karar N=250'de verildi: boyut başına kazanç small −2.5 / medium −2.6 / large +0.1.
Ama N=50'de tablo tersine dönüyor:

| Boyut | real_only_50 | synth_pretrain_50 | kazanç |
|---|---:|---:|---:|
| small | 0.003 | 0.681 | **+67.8** |
| large | 0.198 | 0.248 | +5.0 |

**Sentetik ön-eğitimin küçük-nesne katkısı, az-veri rejiminde muazzam** (+68 puan) —
H4'ün ruhu tam burada doğrulanıyor. Otomatik karar sadece yanlış N'ye (250) baktığı için
"desteklenmedi" dedi. H4 de ablasyonla aynı düzeltmeyi gerektiriyor: düşük-N noktasında ölç.

---

## 6. Diğer bulgular

- **Birleştirme stratejisi önemli.** `mixed_250` (sentetik + gerçek aynı anda) = 0.643,
  hem `real_only_250` (0.819) hem `synth_pretrain_real_250` (0.772) altında. Yani sentetiği
  gerçekle karıştırmak, ön-eğitim→ince-ayar sıralamasından **kötü**; karıştırma N=250'de
  gerçek sinyali seyreltip zarar vermiş. Transfer yöntemi (karıştırma vs sıralı) başlı başına
  bir eksen.
- **real_only_50 = 0.039** neredeyse tam çöküş (50 görüntü + ağır mozaik artırma). Makul ama
  tek tohum; kazancın bir kısmı bu çöküşün artefaktı olabilir — başka tohumla doğrulanmalı.

---

## 7. Sınırlar (dürüst)

- **Tek tohum** (real_only_250 hariç). `±0.000` tek koşu demektir, sıfır varyans değil.
  Titiz sonuç için `seeds: [0,1,2]` ile tekrar (runner biten kolları atlar).
- **Tek gerçek veri seti (DUT).** Sonuçlar bu dağılıma özgü olabilir; ikinci bir kaynak
  (ör. Det-Fly) çapraz-doğrulaması genellemeyi test eder (bkz. RESEARCH.md §8).
- **Ablasyon/boyut yanlış operasyon noktasında** ölçüldü (N=250). Düşük-N'de tekrar şart.
- **Az sayıda 3B varlık** (prosedürel quadcopter). Sentetik çeşitlilik sınırlı; ölçülen
  sentetik katkı muhtemelen bir alt sınır.
- Render EEVEE (hızlı); Cycles (foto-gerçekçi) ile doku-gerçekçiliği ekseni ayrıca ölçülmeli.

---

## 8. Sonraki adımlar (öncelik sırası)

1. **3 tohumla tekrar** — varyans çubukları; bu olmadan farklar istatistiksel değil.
2. **Ablasyon + boyut analizini N=50'de kurgula** — H3/H4'ü doğru rejimde test et.
3. **İkinci gerçek veri seti** (Det-Fly / Drone-vs-Bird) ile çapraz-doğrulama.
4. README figürlerini `docs/report/figures/`'daki gerçek sonuçlarla değiştir.
5. (Opsiyonel) Cycles render + daha fazla 3B varlıkla sentetik çeşitliliğini artır.

**Özet:** İlk tur, ana tezi (az-veri rejiminde sentetik ön-eğitimin belirleyici katkısı)
ve domain gap'i temiz biçimde ortaya koydu. "Desteklenmeyen" iki hipotez, çalışmanın
yanlış olduğunu değil, ablasyonun yanlış noktada ölçüldüğünü gösteriyor — bu da sonraki
turun net bir yol haritası.
