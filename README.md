# ⚔️ LoL Win Predictor

**2026 Profesyonel League of Legends Maçlarında 10. Dakika İstatistikleriyle Galibiyet Tahmini**
BİL4011 Makine Öğrenmesi – Final Projesi

Profesyonel LoL maçlarının ilk 10 dakikasındaki altın, XP, CS farkı ve kill sayılarından **Blue takımının maçı kazanma olasılığını** tahmin eden bir makine öğrenmesi sistemi ve Streamlit arayüzü.

> Konuyu baştan öğrenmek için: [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)

---

## İlk teslimin model seçimi (`reports/metrics.json`)

| Model | CV ROC-AUC (5 kat) | Test Accuracy | Precision | Recall | Specificity | F1 | Test ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.7788 ± 0.012** | **0.7035** | 0.7004 | 0.7605 | 0.6405 | 0.7292 | 0.7794 | 0.1912 |
| Random Forest | 0.7765 ± 0.014 | 0.6973 | 0.6961 | 0.7517 | 0.6372 | 0.7228 | 0.7770 | 0.1924 |
| XGBoost | 0.7754 ± 0.014 | 0.6984 | 0.6960 | 0.7556 | 0.6351 | 0.7246 | 0.7791 | 0.1913 |

- Seçim **yalnızca doğrulama (CV) ROC-AUC'una** göre yapıldı → Logistic Regression.
- Referanslar (test): "her zaman Blue" → 0.525 accuracy; "altın farkı > 0 ise Blue" → 0.693 accuracy; yalnız altın farkı skor olarak → ROC-AUC 0.772.
- **Modelin altın-işareti kuralına göre ek faydası küçüktür:** accuracy +0.010 (%95 bootstrap GA −0.005 … +0.025), ROC-AUC +0.007 (GA −0.001 … +0.015). İki aralık da 0'ı içerdiği için fark **istatistiksel olarak anlamlı değildir**. Modelin asıl katkısı yaklaşık kalibre bir **olasılık** üretmesidir (kusursuz değil, aşağıya bakın).
- Karışıklık matrisi (LR, test, 1956 maç): TN=595, FP=334, FN=246, TP=781.
- **Kalibrasyon:** sabit olasılık referansı (her maça eğitimdeki Blue oranı 0.540) Brier 0.2496 / log-loss 0.692; model Brier 0.1912 / log-loss 0.560 → **Brier Skill Score 0.234**. Ortalama tahmin 0.546, gerçek oran 0.525: test döneminde Blue avantajı azaldığı için model Blue'yu ~2 puan fazla tahmin ediyor.

## Tarihsel olasılık karşılaştırması (`reports/esports/esports_metrics.json`)

İki lojistik regresyon modeli aynı maçlarda karşılaştırıldı. Yalnız altın modeli 10. dakika altın farkını, beş girdili model ayrıca XP, CS ve kill bilgilerini kullanır. Hiperparametre seçimi, beş kronolojik doğrulama katında ortalama Brier skoruna göre yapıldı.

| Model | Brier ↓ | ROC-AUC ↑ | Log-loss ↓ |
|---|---:|---:|---:|
| Yalnız altın | 0,19397 | 0,77211 | 0,56671 |
| Beş girdi | 0,19122 | 0,77944 | 0,56034 |

1.956 test maçında Brier iyileşmesi **0,00276**, seri bazlı %95 güven aralığı **[−0,00055, +0,00606]**. Aralık sıfırı içeriyor. Bu dönemde beş girdinin istatistiksel olarak anlamlı ek faydası gösterilemedi.

Tarihsel veri penceresi `2026-01-01 00:00:00`–`2026-11-21 17:02:48`, test başlangıcı `2026-08-25 17:19:21`. Testle aynı tahmini seriye düşen 10 maç eğitimden çıkarıldı ve 7.810 eğitim maçı kaldı. Seri eşleştirmesi lig, takvim günü ve takım çiftine dayanır. Gece yarısını aşan seriler ayrılabilir, aynı gün oynanan ayrı seriler birleşebilir. Bu test dönemi ilk teslimde de kullanıldı.

## Takım gücü ve diğer özellikler (keşifsel)

İlk teslimin %70,35 doğruluğu, 25 Ağustos 2026 saat 17:19:21 ve sonrasındaki 1.956 maçtan ölçüldü. Yeni adayların seçiminde bu maçlar kullanılmadı. ChainCC'nin 23 Kasım 2026 tarihinde sunulan 2026 takım dosyası yerel kopyayla aynı SHA-256 özetine sahip. Ayrı bir yeni test dönemi bulunamadığı için bu sonuçlar keşifseldir. [ChainCC açık veri](https://chaincc.lol/free/data)

İlk dört kronolojik doğrulama katı adayları seçmek için kullanıldı. Beşinci kat, 27 Temmuz–25 Ağustos arasındaki 1.300 maçta seçilen adayı ve referansları değerlendirdi. Bu dönem ilk teslimin doğrulamasında da yer alıyordu. Aynı lig, gün ve sırasız takım çiftiyle eşleşen tahmini seriler bölme sınırlarından ayıklandı. Güven aralıkları iki modelin aynı maçlardaki tahminlerini seri grupları halinde 2.000 kez yeniden örnekler. Aday taramasındaki aralıklar çoklu karşılaştırma için düzeltilmedi.

Elo ve şampiyon güçleri yalnız maç gününden en az iki takvim günü önceki sonuçlardan hesaplandı. Böylece önceki gün dahil olmak üzere maç günü ve gelecekteki sonuçlar başlangıç özelliklerine girmez. Elo, 1.500 başlangıç puanı ve K=24 ile güncellenir. Şampiyon galibiyet oranları %50 önsel ortalamaya doğru 20 maç ağırlığıyla düzeltilir. Draft kodlaması, Blue ve Red'in seçtiği şampiyonların imzalı varlık göstergeleridir. Ek maç değişkenleri yalnız 10. dakika kaynaklarına dayanır. Geçmiş takım gücü ve draft bağlamı için araştırma örnekleri bulunur. [Profesyonel LoL maç önü tahmin modeli](https://arxiv.org/abs/2609.08060) · [Geçmiş performans ve draft çalışması](https://arxiv.org/abs/2001.11274)

| Geliştirme, 5.203 maç | Doğruluk | ROC-AUC | F1 | Brier | Log-loss |
|---|---:|---:|---:|---:|---:|
| Beş girdi | %70,38 | 0,77635 | 0,72575 | 0,19167 | 0,56205 |
| Beş girdi + Elo | %71,82 | 0,79253 | 0,74007 | 0,18511 | 0,54603 |
| Beş girdi + draft seçimleri | %68,90 | 0,75256 | 0,71066 | 0,20715 | 0,63216 |
| Beş girdi + geçmiş şampiyon gücü | %70,75 | 0,78045 | 0,72841 | 0,19001 | 0,55822 |
| Beş girdi + ek 10. dk özellikleri | %70,33 | 0,77528 | 0,72300 | 0,19211 | 0,56289 |
| Beş girdi + 2025 eğitim verisi | %70,34 | 0,77601 | 0,72743 | 0,19196 | 0,56266 |
| Beş girdi + eşik ayarı | %70,44 | 0,77635 | 0,72368 | 0,19167 | 0,56205 |
| Elo + eşik ayarı (ek kontrol) | %71,36 | 0,79253 | 0,73731 | 0,18511 | 0,54603 |

2025 dosyasında 9.236 temiz maç vardı ve aynı beş girdinin 2026 eğitim verisine eklenmesi doğruluğu artırmadı. Eksik 10. dakika kaydı nedeniyle 2025'te 817/10.053, 2026'da 175/9.951 maç çıkarıldı. Sezonlar farklı yamaları kapsıyor. Eşikler her geliştirme katından önce yalnız önceki doğrulama tahminleriyle seçildi. Elo tek başına geliştirmede pozitif güven aralığına sahip tek özellik ailesiydi. Elo için ek eşik ayarı da 0,5 eşiğinden daha düşük doğruluk verdi.

| Son zaman dilimi, 1.300 maç | Doğruluk | ROC-AUC | F1 | Brier | Log-loss |
|---|---:|---:|---:|---:|---:|
| Yalnız altın, lojistik regresyon | %70,54 | 0,77365 | 0,74951 | 0,19113 | 0,56099 |
| Beş girdi, lojistik regresyon | %71,85 | 0,78732 | 0,75858 | 0,18568 | 0,54898 |
| Beş girdi + Elo, lojistik regresyon | %73,00 | 0,79456 | 0,76490 | 0,18265 | 0,54073 |
| Random Forest | %71,69 | 0,78618 | 0,75467 | 0,18646 | 0,55056 |
| XGBoost | %71,77 | 0,78673 | 0,75615 | 0,18611 | 0,54961 |

Ağaç modelleri ilk teslimin sabit ayarlarıyla aynı eğitim döneminde yeniden eğitildi ve aday seçiminde kullanılmadı. Elo'nun doğruluk farkı **+1,15 yüzde puan** ve seri bazlı %95 güven aralığı **[−0,62, +2,90] yüzde puan**. Gün bloklarıyla aralık **[−1,03, +3,11]**. Brier iyileşmesi **0,00303**, %95 güven aralığı **[−0,00278, +0,00840]**. Bu sonuçlarla mevcut Logistic Regression modelini kullanmaya devam ediyoruz. Elo yeni bir test dönemi için öncelikli aday. [Deney metrikleri](reports/improvements/metrics.json)

ChainCC'nin 2025 ve 2026 takım dosyalarını `data/raw/` altında tutarak deneyi çalıştır. Oluşturulan yerel aday modeli `data/processed/exploratory_model.joblib` yoluna kaydedilir. Elo tahmininde 2026 takım dosyası, tahmin tarihinden önceki takım geçmişini hesaplamak için de kullanılır.

```powershell
python -m src.evaluate_improvements --tree-references
python -m src.evaluate_improvements --predict-json match.json
```

Elo modeli için örnek `match.json`. Takım adları kaynak CSV ile uyumlu olmalı. Geçmişi bulunmayan takımlar 1.500 puanla başlar.

```json
{
  "date": "2026-11-23", "team_name": "Gen.G", "opponent_team_name": "T1",
  "golddiffat10": 1000, "xpdiffat10": 500, "csdiffat10": 10,
  "killsat10": 3, "deathsat10": 2
}
```

## Mahkeme deneyi

Mahkeme, beş girdili lojistik regresyon, Elo'lu lojistik regresyon, Random Forest ve XGBoost olasılıklarını birleştirir. Elo, beş girdili lojistik regresyonla maçların %9,92'sinde ayrıştı. Ağaç modelleri birbirleriyle yalnız %2,08 ayrıştı ve olasılık korelasyonları 0,997 oldu. Yalnız altın modeli dört uzmanın tamamının yanıldığı 25 son dönem maçında doğruydu. Beşinci uzman olarak eklenmesi, seçim katlarında sabit ağırlıklı ortalamanın log-loss değerini 0,55778'den 0,55783'e çıkardı. Bu ek kontrol son dönemde yöntem seçmek için kullanılmadı.

- Uzmanlar yalnız önceki maçlarla eğitildi. Mahkeme'nin ilk eğitimi için 2.598 kronolojik OOF tahmini kullanıldı. Sonraki iki katın 2.597 maçındaki log-loss yöntem seçimini belirledi. Her seçim katının üst modeli yalnız daha erken katların OOF tahminlerini gördü. Beşinci kattaki 1.300 maçta seçim dondurulduktan sonra ölçüm yapıldı.
- Kat sınırlarında tahmini seriler ayrıldı. Elo, maç günü ve bir önceki günün sonuçlarını kullanmadı. Tarihsel 1.956 maç aday seçimine girmedi. 1.300 maçlık dönem önceki deneylerde görüldüğü için bütün sonuçlar keşifseldir.
- Basit ortalama, Brier ile öğrenilen sabit ağırlıklar, lojistik stacking ve koşullu ağırlıklar karşılaştırıldı. Koşullu model, 10. dakika farkları, Elo ve uzmanlar arasındaki olasılık dağılımından maç başına softmax ağırlıkları çıkarır. Ağaç uzmanlarının hiperparametreleri ilk teslimden sabit alındı ve tekrarlanabilirlik için tek iş parçacığı kullanıldı. [Wolpert, Stacked Generalization](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf) · [Jacobs ve diğerleri, Adaptive Mixtures of Local Experts](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf)
- Duruşmada Blue oyu verenler savunan, Red oyu verenler saldıran tarafı oluşturur. Her uzmanda altın, XP, CS, kill veya Elo farkı tek tek eşitlendiğinde tahminin ne kadar değiştiği ölçülür. Lojistik kanıt değerlendiricileri bu tepkilerden, maç koşullarından, oy güveninden ve desteğinden her uzmana 0–1 arasında güven katsayısı verir. Tarafların gücü `güven × |2p − 1|` değerlerinin toplamıdır. Bunlar model tepkileridir ve nedensel açıklama olarak yorumlanmaz. Bir hâkim kanıtı kendi maç istatistiği tahminiyle, diğeri stacking tahminiyle eşit ağırlıkta birleştirir. İkisi de tarafların kararını değiştirebilir.

| Son dönem, 1.300 maç | Doğruluk | ROC-AUC | F1 | Brier ↓ | Log-loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Lojistik regresyon, beş girdi | %71,85 | 0,78732 | 0,75858 | 0,18568 | 0,54898 | 0,03501 |
| Lojistik regresyon, beş girdi + Elo | %73,00 | 0,79456 | 0,76490 | 0,18265 | 0,54073 | 0,04315 |
| Lojistik regresyon, yalnız altın | %70,54 | 0,77365 | 0,74951 | 0,19113 | 0,56099 | 0,03976 |
| Random Forest | %71,69 | 0,78618 | 0,75467 | 0,18646 | 0,55056 | 0,03861 |
| XGBoost | %71,46 | 0,78698 | 0,75414 | 0,18612 | 0,54976 | 0,03164 |
| Basit ortalama | %72,54 | 0,79400 | 0,76216 | 0,18304 | 0,54221 | 0,03524 |
| Sabit ağırlıklı ortalama | %72,85 | 0,79696 | 0,76293 | 0,18144 | 0,53774 | 0,03327 |
| Lojistik stacking | %72,92 | 0,79709 | 0,76596 | 0,18157 | 0,54040 | 0,03329 |
| Koşullu ağırlık | %72,54 | 0,79474 | 0,76152 | 0,18275 | 0,54160 | 0,03237 |
| Yalnız kanıt | %72,46 | 0,79450 | 0,76037 | 0,18289 | 0,54192 | 0,03408 |
| Kanıt + maç istatistiği hâkimi | %72,92 | 0,79780 | 0,76439 | 0,18115 | 0,53729 | 0,03150 |
| Kanıt + stacking hâkimi | %73,08 | 0,79734 | 0,76573 | 0,18148 | 0,53927 | 0,03349 |

Seçim katlarında sabit ağırlıklı ortalama kazandı. Son dönem için öğrenilen ağırlıkların yaklaşık %74,6'sı Elo'ya, %17,0'ı Random Forest'a ve %8,5'i beş girdili lojistik regresyona gitti. Elo'ya göre 11 hatayı düzeltti ve 13 doğru kararı bozdu. Doğruluk farkı **−0,15 yüzde puan**, seri bazlı %95 güven aralığı **[−0,93, +0,67] yüzde puan**. Log-loss iyileşmesi **0,00299**, aralık **[−0,00076, +0,00706]**. Elo çıkarıldığında aynı yöntemin log-loss değeri 0,54873'e yükseldi. Elo'nun bu birliktelikte log-loss katkısı **0,01099**, aralık **[+0,00137, +0,02026]**.

Dört uzman 278 maçta birlikte yanıldı. Maç istatistiği hâkimi bunlardan birini düzeltti ve genel olarak Elo'ya göre 21 maçı düzeltti, 22 maçı bozdu. Stacking hâkimi 27 maçı düzeltti, 26 maçı bozdu. 35 adet 2–2 bölünmede Elo 24, stacking hâkimi 25 maçı doğru bildi. Örneğin `LOLTMNT02_442438` maçında Blue Elo'da 125 puan önde ama 10. dakikada 1.504 altın gerideydi. Elo %52,0 Blue derken maç istatistiği hâkimi %47,1 Blue verdi ve Red kazandı. `LOLTMNT02_442117` maçında aynı hâkim %50,8 Blue vererek Elo'nun doğru Red kararını bozdu. [Kalibrasyon dilimleri, güven aralıkları ve örnek maçlar](reports/court/metrics.json)

Mahkeme'nin tekil Elo'ya karşı tutarlı bir avantajı gösterilemedi. Üretimdeki Streamlit ve kayıtlı model korunuyor. Deney, yerel 2026 takım dosyasıyla çalıştırılabilir.

```powershell
python -m src.evaluate_court
```

## İlk teslimin teknik denetimi

| # | Kontrol | Durum | Not |
|---|---|---|---|
| 1 | Yalnızca 10. dk özellikleri | PASS | 5 özelliğin hepsi `*at10`; maç sonu, kimlik, 15. dk sütunları yok (testli) |
| 2 | Eksik sütun raporu | PASS | `bo_format`, `match_in_series`, `game_in_match` %100 boş; `split` 2.964 satır (EWC, LJL, MSI, Worlds, promotion…); ban'lar 315 satır / 233 maç. Hiçbiri kullanılmıyor → işlem yapılmadı |
| 3a | Kronolojik bölme | PASS (düzeltildi) | 8 aynı zaman damgası vardı; sıralama `game_id` ile deterministik yapıldı ve bölme eşit zaman damgalarını ayırmayacak şekilde değiştirildi. Sonuç değişmedi (sınırda çakışma yoktu) |
| 3b | TimeSeriesSplit sırası | PASS | İlk teslimde eğitim tarihi ≤ doğrulama tarihi. Tarihsel deneyde eşit zaman damgaları aynı tarafta tutulur. |
| 3c | Seri (Bo3/Bo5) çakışması | Saptandı | İlk teslimde yaklaşık seri anahtarıyla 7 test maçı eşleşti. Tarihsel deney sınırı aşan eşleşmeleri eğitimden çıkarır. |
| 4 | 175 maçın çıkarılması | PASS | İki tarafta da aynı 175 maç, maç sonu verisi dolu; geçerli sıfırlar (5 maç altın farkı 0, 1.272 maç Blue kill 0) korundu |
| 5 | Patch/lig performansı | PASS | `reports/group_performance.csv`; n ≥ 100 olan 4 patch ve 3 lig dışında sonuç çıkarılmadı |
| 6 | Altın işaretiyle kıyas | PASS | Fark küçük ve anlamlı değil (yukarıda) |
| 7 | Sabit olasılık referansı | PASS | BSS 0.234 |

## Veri

- **Kaynak:** ChainCC Open LoL Esports Dataset 2026 – <https://chaincc.lol/free/data>
- **Dosya:** `chaincc-teams-2026.csv.gz` → `data/raw/` altına konur
- **SHA-256:** `6f801f27ed58b0f5fc6e15f4556076c5fdd794179eb8548eac22ea98c31919ed`
- Patch sürümleri metin olarak okunur. `16.1` ve `16.10` ayrı sürümler olarak raporlanır.
- **Lisans:** CC BY 4.0 (veri sağlayıcının beyanı). Atıf: *"Data: ChainCC Open LoL Esports Dataset 2026, chaincc.lol, CC BY 4.0"*
- Doğrulanan boyut: 19.902 satır (maç başına Blue + Red), 9.951 maç, 74 sütun, 2026-01-01 – 2026-11-21.
- Ham veri **Git'e eklenmez** (`.gitignore`), çünkü upstream yeniden dağıtım koşulları kesinleştirilmedi. Projeyi klonlayan kişi dosyayı kaynaktan indirip `data/raw/` içine koymalıdır.

### Temizleme özeti
| Adım | Maç |
|---|---|
| Ham veri | 9.951 |
| Tüm 10. dk değerleri 0 (eksik kayıt, çoğu LPL/DCGI) → çıkarıldı | −175 |
| Eksik değer (kullanılan sütunlarda) | 0 |
| **Son veri seti** | **9.776** |

Her maç **tek satır** (Blue bakış açısı). Kronolojik bölme: en eski %80 eğitim (7.820 maç, 2026-01-08 → 2026-08-25), en yeni %20 test (1.956 maç, 2026-08-25 → 2026-11-21).

### Özellikler
| Sütun | Anlam |
|---|---|
| `golddiffat10` | 10. dk altın farkı (Blue − Red) |
| `xpdiffat10` | 10. dk XP farkı |
| `csdiffat10` | 10. dk minyon (CS) farkı |
| `killsat10` | Blue'nun 10. dk kill sayısı |
| `deathsat10` | Blue'nun 10. dk ölüm sayısı (≈ Red'in kill sayısı) |

Hedef: `result` (1 = Blue kazandı). Maç sonu sütunları, `game_id`, takım adları ve 15. dk verileri **kullanılmadı** (sızıntı / genellenemezlik).

## Proje yapısı

```
LoL Win Predictor/
├── app.py                     # Streamlit arayüzü
├── data/raw/                  # ham veri (Git dışı)
├── data/processed/            # blue_at10.csv (temiz veri, train.py üretir)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA (çalıştırılmış, çıktılı)
│   └── build_eda_notebook.py  # notebook'u yeniden üretir
├── src/
│   ├── data.py                # yükleme, temizleme, kronolojik bölme
│   ├── train.py               # GridSearchCV, değerlendirme, grafikler, model kaydı
│   └── predict.py             # kayıtlı modelle tahmin
├── models/best_model.joblib   # Pipeline (ölçekleyici + model)
├── reports/metrics.json       # tüm metrikler (baseline, kalibrasyon, bootstrap GA dahil)
├── reports/group_performance.csv  # patch ve lig bazında test performansı
├── reports/figures/*.png      # EDA ve model grafikleri
├── tests/                     # pytest testleri
├── requirements.txt           # doğrudan bağımlılıklar (sabit sürüm)
├── requirements-lock.txt      # pip freeze (tam ortam)
├── PROJECT_EXPLANATION.md     # öğrenme dokümanı
└── PRESENTATION_NOTES.md      # sunum planı ve demo sırası
```

## Kurulum ve çalıştırma (Windows, PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Veri dosyasını `data\raw\chaincc-teams-2026.csv.gz` olarak yerleştirin, ardından:

```powershell
.\.venv\Scripts\python.exe -m src.train          # veri işleme + 3 model + metrikler + grafikler (~35 sn)
.\.venv\Scripts\python.exe -m src.predict        # örnek tahminler
.\.venv\Scripts\python.exe -m src.evaluate_esports --raw data/raw/chaincc-teams-2026.csv.gz --output-dir reports/esports
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider   # testler
.\.venv\Scripts\python.exe -m streamlit run app.py              # uygulama → http://localhost:8501
```

EDA notebook'unu yeniden üretip çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe notebooks\build_eda_notebook.py
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\01_eda.ipynb
```

Yeniden üretilebilirlik: `random_state=42`, sabit paket sürümleri, deterministik kronolojik bölme.

## Sınırlılıklar

- Yalnızca 5 sayısal özellik; şampiyon seçimi, takım gücü, lig, objektifler modelde yok.
- 10. dakikada maç hâlâ açık: ~%70 doğruluk bu problem için makul bir tavandır.
- Meta/patch değiştikçe performans düşebilir; model 2026 verisiyle sınırlı.
- Altın/XP/CS farkları güçlü korelasyonlu; LR katsayıları tek tek nedensel yorumlanmamalı.
- Kaynak veride Blue `killsat10` ile Red `deathsat10` maçların ~%1'inde uyuşmuyor.
- Test dönemi yalnızca patch 16.16–16.23'ü kapsar; erken patch'lerdeki performans testte ölçülmedi. Lig bazında yalnız LAS, EM ve HC'de ≥100 test maçı var.
- Model, altın farkının işaretine bakan basit kuraldan anlamlı ölçüde daha doğru **değildir**; değeri yaklaşık kalibre (kusursuz olmayan) bir olasılık vermesidir.

## Atıf ve yasal not

Veri: **ChainCC Open LoL Esports Dataset 2026** (<https://chaincc.lol/free/data>), CC BY 4.0.

LoL Win Predictor, Riot Games tarafından onaylanmamıştır ve Riot Games'in ya da League of Legends'ın üretiminde veya yönetiminde resmi olarak yer alan herhangi bir kişinin görüşlerini yansıtmaz. League of Legends ve Riot Games, Riot Games, Inc.'in ticari markaları veya tescilli ticari markalarıdır. Projede Riot Games'e ait görsel kullanılmamıştır.
