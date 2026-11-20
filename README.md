# MirageBench

**🔗 Canlı pano:** https://mrpmoffical.github.io/sim2real-drone-detection/


**Küçük İHA/drone tespitinde sentetik eğitim verisinin gerçek veri açığını ne kadar
kapattığını ölçen kontrollü bir benchmark.**

MirageBench "bir model eğittim" deposu değil, küçük bir **ampirik çalışmadır**. Soru
nettir: Blender ile üretilen sentetik drone görüntüleri, gerçek küçük-nesne tespit
başarısını ne kadar artırır; hangi koşulda ve hangi tasarım seçimiyle? Kod, bu soruyu
tekrar üretilebilir biçimde yanıtlamak için vardır.

Çalışma tasarımı, hipotezler ve karar kuralları: [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## Neden bu proje

"Sentetik veri yardımcı olur" iddiası literatürde sık; ama *ne kadar*, *hangi
rejimde* ve *hangi domain-randomization faktörü sayesinde* sorularına temiz,
referansla doğrulanmış metrikle yanıt veren açık bir çalışma azdır. MirageBench bu
boşluğu doldurur:

- **Kontrollü deney**: model, artırma ve eğitim bütçesi tüm kollarda sabit; yalnız
  eğitim verisinin bileşimi değişir. Gözlenen fark veriye atfedilebilir.
- **Referansla doğrulanmış ölçüm**: boyut-katmanlı mAP motoru sıfırdan yazıldı ve
  `pycocotools` ile **4 ondalık basamağa kadar** uyumlu (bkz. `tests/test_eval_map.py`).
- **Kusursuz sentetik etiket**: bbox render pikselinden değil, 3B mesh köşelerinin
  kamera matrisiyle projeksiyonundan üretilir; gölge/blur/saydamlık etiketi bozmaz.
- **Sızıntısız protokol**: grup-farkında split (aynı videodan kareler train/test'e
  bölünmez), deterministik veri-bütçesi alt örnekleme, çok-tohumlu belirsizlik.

---

## Sistem mimarisi

```mermaid
flowchart LR
    subgraph DATA[Veri üretimi]
        B[Blender / bpy<br/>domain randomization] -->|kusursuz bbox| S[(Sentetik<br/>YOLO seti)]
        R[(Gerçek setler<br/>Anti-UAV / MAV-VID / VOC / COCO)] -->|adaptörler| N[ImageRecord]
    end
    subgraph PREP[Hazırlık]
        N --> G[grup-farkında split<br/>+ bütçe alt örnekleme]
        S --> M[materialize<br/>data.yaml]
        G --> M
    end
    subgraph TRAIN[Eğitim - GPU]
        M --> T[YOLOv11 sarmalayıcı<br/>adım-bütçesi eşitleme<br/>synth ön-eğitim -> gerçek ince ayar]
    end
    subgraph EVAL[Ölçüm]
        T --> P[çıkarım<br/>opsiyonel SAHI] --> E[boyut-katmanlı mAP<br/>pycocotools-doğrulanmış]
    end
    E --> REP[rapor: tablo + hipotez kararları<br/>+ veri-verimliliği / ablasyon grafikleri]
```

Deney kolları (`configs/arms/`): `real_only_{N}`, `synth_only`,
`synth_pretrain_real_{N}`, `mixed_{N}`, `ablation_{faktör}`.

---

## Sonuçlar

İlk tam tur: 17 kol, DUT Anti-UAV gerçek test seti (2200 görüntü), YOLOv11s @ 1280px.
Tam analiz ve tablolar: [`docs/FINDINGS.md`](docs/FINDINGS.md).

**Öne çıkan bulgu — veri verimliliği (mAP@50):** Sentetik ön-eğitim, gerçek veri
aşırı kıtken belirleyici; katkı veri arttıkça eriyor.

| N (gerçek görüntü) | gerçek-only | sentetik ön-eğitim | kazanç |
|---:|---:|---:|---:|
| 50 | 0.039 | 0.543 | **+50.4** |
| 100 | 0.589 | 0.670 | +8.1 |
| 250 | 0.819 | 0.772 | −4.7 |
| 500 | 0.856 | 0.868 | +1.2 |
| 1000 | 0.923 | 0.901 | −2.2 |

**Domain gap:** yalnız sentetikle eğitim gerçek testte 0.387 mAP@50 — en iyi gerçek
modelin (0.948) 56 puan altında (büyük ama sıfır olmayan transfer).

![veri verimliliği](docs/report/figures/data_efficiency.png)

Sentetik üreticinin mesh projeksiyonundan ürettiği kusursuz bbox (prosedürel quadcopter):

![sentetik bbox](docs/figures/example_synth_bbox.png)

> Not: Bu tur çoğu kolda tek tohumla koşuldu (varyans tahmini yok). Titiz bir sonuç
> için 3 tohumla tekrar önerilir — bkz. `docs/FINDINGS.md` §7-8.

---

## Kurulum

```bash
# Çekirdek (CPU): veri hattı, değerlendirme, rapor
pip install -e ".[report,dev]"

# Sentetik üretim (Blender modülü)
pip install bpy            # ya da sistemdeki `blender` çalıştırılabiliri kullanılır

# Eğitim/çıkarım (GPU, ör. RTX 4070)
pip install -e ".[train]"
```

Docker (CUDA + Ultralytics + bpy): `docker build -t miragebench -f docker/Dockerfile .`

---

## Hızlı başlangıç

```bash
# 1) Sentetik veri üret (full + ablasyon faktörleri)
python -m miragebench.synth.cli --out data/synth/full            --n 20000 --factor full
python -m miragebench.synth.cli --out data/synth/no_light_random --n 20000 --factor no_light_random

# 2) Gerçek seti içeri al (grup-farkında split)
python -m miragebench.data.cli --format yolo --root /veri/anti_uav --out data/real --merge-classes

# 3) Veri kompozisyonunu GPU'suz doğrula
python -m miragebench.experiment.cli --dry-run \
  --arm configs/arms/*.yaml --real-root data/real --synth full=data/synth/full --seeds 0

# 4) Kolları çalıştır (GPU)
python -m miragebench.experiment.cli \
  --arm configs/arms/real_only_250.yaml configs/arms/synth_pretrain_real_250.yaml \
  --real-root data/real --synth full=data/synth/full no_light_random=data/synth/no_light_random \
  --seeds 0 1 2 --results results/

# 5) Rapor + grafikler + hipotez kararları
python -m miragebench.report.cli --results results/ --out docs/report
```

`make help` yaygın iş akışlarını listeler.

---

## Depo yapısı

```
miragebench/
  config/      pydantic şema + base/arm yükleme (kontrollü deney doğrulaması)
  synth/       geometry (projeksiyon) · factors (ablasyon) · assets · blender_generator
  data/        adapters (YOLO/COCO/VOC) · records · split · materialize
  eval/        metrics (boyut-katmanlı mAP, pycocotools-doğrulanmış) · io
  experiment/  train_wrapper (YOLOv11, adım-bütçesi) · runner · cli
  report/      aggregate (hipotez kararları) · plots
configs/       base.yaml + arms/*.yaml
tests/         42 test; mAP çapraz-doğrulama, projeksiyon, split sızıntısı, hipotez mantığı
docs/          RESEARCH.md (çalışma tasarımı) · figures/
docker/        CUDA + Ultralytics + bpy imajı
```

---

## Doğrulama

```bash
pytest tests/ -q        # 42 test
```

Öne çıkan testler:
- `test_eval_map.py` — boyut-katmanlı mAP'in `pycocotools` ile sayısal uyumu (≤1e-3).
- `test_geometry.py` — kamera projeksiyonunun analitik referansla uyumu.
- `test_data_pipeline.py` — grup-farkında split'in sızıntısızlığı, alt örnekleme determinizmi.
- `test_blender_smoke.py` — render edilen karede bbox'ın drone piksellerini sarması (bpy gerekir).
- `test_experiment_and_report.py` — adım-bütçesi eşitleme ve hipotez karar mantığı.

---

## Kapsam ve sınırlar

- Sonuçlar seçilen gerçek sete özgü olabilir; en az iki kaynakta çapraz doğrulama
  önerilir (gelecek iş).
- Az sayıda 3B varlık sentetik çeşitliliği sınırlar — bu, ölçülen sentetik katkının
  muhtemelen bir **alt sınır** olduğu anlamına gelir.
- Prototip render EEVEE (hızlı) ile; foto-gerçekçilik etkisi Cycles ile tekrar
  ölçülmeli. Doku-gerçekçiliği ablasyonu bu farkı doğrudan hedefler.
- **Negatif sonuç da sonuçtur**: sentetik veri beklendiği kadar yardımcı olmadıysa bu
  da raporlanır. Değer, bulgunun yönünde değil, ölçümün güvenilirliğindedir.

## Lisans

MIT — bkz. [`LICENSE`](LICENSE).
