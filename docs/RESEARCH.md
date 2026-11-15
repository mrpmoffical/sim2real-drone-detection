# MirageBench — Araştırma Tasarımı

Küçük İHA/drone tespitinde **sentetik eğitim verisinin gerçek veri açığını ne kadar
kapattığını** ölçen kontrollü bir çalışma. Bu belge projeyi bir yazılım deposundan
ayıran kısmı tanımlar: sorulan soru, hipotezler, deney kolları, ölçütler ve
geçerliliği tehdit eden faktörler.

Bu bir mühendislik vitrini değil, küçük bir ampirik çalışmadır. Kod, deneyi
tekrar üretilebilir kılmak için vardır.

---

## 1. Problem ve motivasyon

Küçük nesne tespiti (nesnenin görüntüdeki alanı < 32×32 piksel) zor bir rejimdir:
az piksel, düşük kontrast, yüksek yanlış-pozitif oranı. Drone tespiti bunun uç
örneğidir. Gerçek, etiketli drone verisi az ve pahalıdır; sentetik veri (Blender ile
render) bol ve bedava etiketlidir, ama **domain gap** (sentetik ile gerçek görüntü
dağılımı arasındaki fark) nedeniyle doğrudan işe yarayıp yaramadığı belirsizdir.

Literatürde "sentetik veri yardımcı olur" iddiası sık; ama *ne kadar*, *hangi
koşulda* ve *hangi tasarım seçimiyle* sorularına temiz, tekrar üretilebilir yanıt
veren açık bir çalışma azdır. MirageBench bu boşluğu doldurur.

---

## 2. Araştırma soruları

- **RQ1 (Açık):** Sentetik ön-eğitim + küçük gerçek veriyle ince ayar, aynı
  miktarda gerçek veriyle sıfırdan eğitime kıyasla tespiti ne kadar iyileştirir?
- **RQ2 (Veri verimliliği):** Sentetik ön-eğitim, belirli bir hedef başarıya
  ulaşmak için gereken gerçek etiketli görüntü sayısını kaç kat azaltır?
- **RQ3 (Sıfır-gerçek):** Yalnızca sentetik veriyle eğitilen model, gerçek test
  setinde kullanılabilir bir başarıya ulaşır mı? Ulaşamıyorsa açık ne kadar?
- **RQ4 (Tasarım ablasyonu):** Sentetik verinin hangi tasarım faktörü en çok
  katkı sağlar — arka plan çeşitliliği, ışık randomizasyonu, poz/ölçek
  çeşitliliği, yoksa doku gerçekçiliği?
- **RQ5 (Boyut rejimi):** Sentetik verinin katkısı en küçük nesnelerde (en zor
  rejim) orantısız biçimde daha mı yüksek, yoksa daha mı düşük?

---

## 3. Hipotezler (önceden kayıt)

Deneyleri çalıştırmadan önce beklentileri yazıyoruz; böylece sonuçları sonradan
anlatıya uydurma (HARKing) riskini azaltırız.

- **H1:** Sentetik ön-eğitim + ince ayar, düşük gerçek-veri bütçesinde (≤ birkaç yüz
  görüntü) gerçek-only'i anlamlı biçimde geçer; bütçe büyüdükçe fark kapanır.
- **H2:** Sıfır-gerçek (sentetik-only) modelin gerçek sette mAP@50'si, eşdeğer
  gerçek-only modelin belirgin altında kalır (büyük domain gap).
- **H3:** Arka plan + ışık randomizasyonu, doku gerçekçiliğinden daha çok katkı
  sağlar (domain randomization hipotezi: çeşitlilik > foto-gerçekçilik).
- **H4:** Sentetik verinin göreli katkısı en küçük boyut katmanında en yüksektir,
  çünkü gerçek küçük-nesne örneği en kıttır.

Her hipotez için önceden bir **karar ölçütü** tanımlanır (bkz. §6).

---

## 4. Deney kolları (arms)

Her kol bir YAML config'idir (`configs/arms/`). Model, artırma ve eğitim
hiperparametreleri **koldan koşa sabittir**; sadece eğitim verisinin bileşimi değişir.
Bu, gözlenen farkın veriye atfedilebilmesi için zorunludur (tek değişken ilkesi).

| Kol | Eğitim verisi | Amaç |
|-----|---------------|------|
| `real_only_{N}` | N gerçek görüntü (N ∈ {50,100,250,500,1000,tümü}) | Veri-verimliliği taban eğrisi |
| `synth_only` | Yalnız sentetik | RQ3, domain gap tavanı |
| `synth_pretrain_real_{N}` | Sentetik ön-eğitim → N gerçek ince ayar | RQ1, RQ2 |
| `mixed_{N}` | Sentetik + N gerçek birlikte | Alternatif birleştirme stratejisi |
| `ablation_{factor}` | Bir faktörü kapatılmış sentetik + sabit N gerçek | RQ4 |

Ablasyon faktörleri (`synth_only` tam randomizasyona göre tek tek **kapatılır**):
`no_bg_variation`, `no_light_random`, `no_pose_scale_random`, `low_texture_realism`.

**Değişmezlik kontrolü:** Tüm kollar aynı gerçek **test** setinde değerlendirilir.
Test seti hiçbir eğitim kolunda kullanılmaz ve sentetik hiç içermez.

---

## 5. Veri

- **Gerçek (hedef dağılım):** Açık drone tespit setleri (ör. Anti-UAV, MAV-VID,
  Drone-vs-Bird). Tek bir birleşik test seti sabitlenir; eğitim bölütleri buradan
  ayrı tutulur. Adaptörler `miragebench/data/adapters/` içinde.
- **Sentetik (kaynak dağılım):** `miragebench/synth/` Blender üreticisi. Faktörler
  config'ten kontrol edilir (ablasyon için).
- **Sızıntı önleme:** Split, **grup-farkında**dır — aynı video/çekimden kareler
  train ve test'e bölünmez (küçük-nesne video setlerinde en yaygın sızıntı kaynağı).
- **Boyut katmanı:** Her kutu COCO eşiğiyle etiketlenir (small < 32², medium < 96²,
  large). RQ5 için metrikler bu katmanlara ayrıştırılır.

---

## 6. Ölçütler ve karar kuralları

Birincil ölçüt: gerçek test setinde **mAP@50** ve **mAP@50-95**, ayrıca
**boyut-katmanlı mAP** (small/medium/large). İkincil: precision/recall@sabit-eşik,
küçük-nesne recall'u.

- **H1 kararı:** `synth_pretrain_real_N`, `real_only_N`'i N≤250'de mAP@50'de
  ≥ 3 puan (mutlak) geçerse H1 desteklenir.
- **H2 kararı:** `real_only` eğrisinde hedef mAP@50'ye ulaşan N ile
  `synth_pretrain` eğrisindeki N oranı > 2× ise anlamlı veri tasarrufu.
- **H3/H4 kararı:** Ablasyonda bir faktörün kapatılması mAP@50'yi en çok düşüren
  faktör = en çok katkı sağlayan faktör.

**Belirsizlik:** Her kol ≥ 3 tohum (seed) ile tekrarlanır; ortalama ± standart sapma
raporlanır. Tohumlar arası sapmadan küçük farklar "sonuç" olarak sunulmaz.

---

## 7. Sabitlenen değişkenler (confounder kontrolü)

Model mimarisi, girdi çözünürlüğü, artırma politikası, optimizer, epoch sayısı,
batch boyutu, erken durdurma ölçütü — hepsi `configs/base.yaml`'de sabittir ve
kollar arası değişmez. Yalnızca **veri bileşimi** manipüle edilir. Eğitim
bütçesi (toplam optimizer adımı) kollar arası eşitlenir; aksi halde "daha çok veri"
ile "daha çok eğitim" karışır.

---

## 8. Geçerliliğe yönelik tehditler

- **Tek hedef dağılımı:** Sonuçlar seçilen gerçek sete özgü olabilir. En az iki
  farklı gerçek kaynakta çapraz doğrulama yapılması önerilir (gelecek iş).
- **Sentetik varlık sayısının azlığı:** Az sayıda 3B model, sentetik çeşitliliği
  sınırlar; bu, domain randomization lehine değil aleyhine bir yanlılıktır
  (yani ölçtüğümüz katkı muhtemelen bir alt sınırdır).
- **Render motoru:** Prototip EEVEE (hızlı) ile; nihai çalışma Cycles (foto-gerçekçi)
  ile tekrarlanmalı. Doku-gerçekçiliği ablasyonu bu farkı doğrudan ölçer.
- **Değerlendirme metriği uyumu:** mAP motorumuz COCO protokolüne göre birim
  testleriyle doğrulanır (bkz. `tests/test_eval_map.py`); referans implementasyonla
  sayısal uyum kontrol edilir.

---

## 9. Çıktılar

- Kol × tohum sonuç tablosu (CSV + markdown).
- Veri-verimliliği eğrisi (mAP@50 vs gerçek görüntü sayısı; real_only vs
  synth_pretrain).
- Boyut-katmanlı katkı grafiği (RQ5).
- Ablasyon çubuk grafiği (RQ4).
- Her hipotez için "desteklendi / desteklenmedi / sonuçsuz" kararı.

Negatif sonuç da sonuçtur: sentetik veri beklendiği kadar yardımcı olmadıysa, bu da
raporlanır. Çalışmanın değeri yönü değil, ölçümün güvenilirliğindedir.
