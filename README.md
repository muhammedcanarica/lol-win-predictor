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
