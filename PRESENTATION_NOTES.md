# Sunum Planı – LoL Win Predictor (~12 dk + soru)

Tüm sayılar `reports/metrics.json` dosyasından alınmıştır. Ayrıntılı açıklamalar için `PROJECT_EXPLANATION.md`.

## 1. Amaç (1 dk)
- Soru: Profesyonel bir LoL maçında **ilk 10 dakikaya** bakarak Blue takımının kazanma olasılığını tahmin edebilir miyiz?
- İkili sınıflandırma + olasılık çıktısı.

## 2. Veri seti (1 dk)
- ChainCC Open LoL Esports Dataset 2026 (CC BY 4.0), 19.902 satır, 74 sütun, **9.951 maç**, Ocak–Kasım 2026, 40+ lig.
- Her maç 2 satır (Blue + Red). Hedef: `result`.

## 3. EDA (2 dk) – `notebooks/01_eda.ipynb`
- Tekrar yok; eksikler yalnızca kullanılmayan sütunlarda (3 sütun %100 boş, `split` 2.964, ban'lar 315 satır).
- 175 maçta tüm 10. dk değerleri 0 → eksik kayıt.
- Blue galibiyet oranı %53.7 (hafif avantaj).
- Göster: `winrate_by_golddiff.png` → ≤ −3000 altın: %5.5, > +3000 altın: %97.5.
- Göster: `correlation.png` → altın/XP/CS farkları birbirine güçlü bağlı.

## 4. Preprocessing (1.5 dk)
- Maç başına **tek satır** (Blue); fark sütunları simetrik.
- 175 eksik maç çıkarıldı → **9.776 maç**. Geçerli sıfırlar korundu.
- Özellikler: `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `deathsat10`. Maç sonu sütunları yok (sızıntı).
- **Kronolojik bölme:** eğitim 7.820 maç (Oca–25 Ağu), test 1.956 maç (25 Ağu–21 Kas).
- `StandardScaler` Pipeline içinde, sızıntı yok.

## 5. Üç model (1.5 dk)
- Logistic Regression (doğrusal + sigmoid), Random Forest (ağaçların oylaması), XGBoost (sıralı boosting).

## 6. Cross-validation ve tuning (1 dk)
- `TimeSeriesSplit(5)`: doğrulama verisi her zaman eğitimden sonraki tarihlerden.
- `GridSearchCV` ROC-AUC ile: LR 5, RF 12, XGB 12 kombinasyon.
- Seçim **yalnızca CV** ile; test seti sadece son değerlendirmede.

## 7. Sonuçlar (2 dk) – `model_comparison.png`, `roc_curves.png`, `calibration.png`

| Model | CV ROC-AUC | Test Acc | Test ROC-AUC | Brier |
|---|---|---|---|---|
| **Logistic Regression** | **0.7788** | **0.7035** | 0.7794 | 0.1912 |
| Random Forest | 0.7765 | 0.6973 | 0.7770 | 0.1924 |
| XGBoost | 0.7754 | 0.6984 | 0.7791 | 0.1913 |

- Üç model pratikte eşit → en basit ve yorumlanabilir olan seçildi.
- **Dürüst kıyas:** "altın farkı > 0 ise Blue" kuralı 0.693 doğruluk veriyor. Modelin farkı +0.010, %95 GA [−0.005, +0.025] → **istatistiksel olarak anlamlı değil.**
- Kalibrasyon: sabit olasılık referansına göre Brier Skill Score 0.234. **Yaklaşık kalibre, kusursuz değil:** ortalama tahmin 0.546, gerçek oran 0.525.
- En önemli özellik: altın farkı (permütasyon önemi 0.157).

### Tarihsel olasılık karşılaştırması

Üç model ve ROC-AUC ile seçim ilk teslime ait. Tarihsel deney, yalnız altın farkını ve beş girdiyi kullanan iki lojistik regresyonu karşılaştırır. Beş kronolojik katta seçim Brier skoruyla yapılır ve sınırı aşan tahmini seri eşleşmeleri eğitimden çıkarılır. Seri anahtarı lig, gün ve takım çiftidir. Bu yaklaşık anahtar, gece yarısını aşan seriyi ayırabilir veya aynı gün oynanan ayrı serileri birleştirebilir.

Veri penceresi `2026-01-01 00:00:00`–`2026-11-21 17:02:48`. İlk teslimde de görülen test dönemi `2026-08-25 17:19:21` tarihinde başlıyor ve 1.956 maç içeriyor. Brier skoru yalnız altınla **0,19397**, beş girdiyle **0,19122**. %95 güven aralığı **[−0,00055, +0,00606]** sıfırı içeriyor. Sonucu sunarken ek faydanın istatistiksel olarak gösterilemediğini söyle. Performans sekmesinde iki modelin ROC-AUC ve log-loss değerleri de yer alır.

Ham dosya SHA-256 özeti `6f801f27ed58b0f5fc6e15f4556076c5fdd794179eb8548eac22ea98c31919ed`.

```powershell
python -m src.evaluate_esports --raw data/raw/chaincc-teams-2026.csv.gz --output-dir reports/esports
```

### Ek slayt · Elo ve Mahkeme (süre uygunsa)

- 1.300 maçlık son zaman diliminde beş girdili model %71,85, Elo ekli model %73,00 doğruluk verdi. **+1,15 yüzde puan** için %95 güven aralığı **[−0,62, +2,90]**. Elo'yu doğrulanmış yükseltme yerine keşifsel aday olarak sun.
- Mahkeme seçim katlarında sabit ağırlıklı ortalamayı seçti. Son dönemde doğruluk %72,85 ve log-loss 0,53774 oldu. Tekil Elo'nun log-loss'u 0,54073'tü. Mahkeme Elo'ya göre 11 hatayı düzeltti, 13 doğru kararı bozdu ve farkın güven aralığı sıfırı içerdi.
- Sızıntı önlemini bir cümlede anlat. Uzmanlar önceki maçlardan, üst model de önceki katların uzman tahminlerinden öğrenir. İlk teslimin 1.956 test maçı aday seçimine girmedi. Son 1.300 maçlık dönem önceden görülmüştü.
- Demoda Streamlit'in mevcut modeli ve tarihsel olasılık karşılaştırmasını göster. Elo ve Mahkeme için kayıtlı deney tablolarını kullan. [Elo raporu](reports/improvements/metrics.json) · [Mahkeme raporu](reports/court/metrics.json)

## 8. Sınırlılıklar (1 dk)
- Yayındaki model beş girdi kullanır. Takım gücü ve draft keşifsel deneylerde ölçüldü.
- İlk teslimde 10. dakika girdileriyle doğruluk %70,35 olarak ölçüldü.
- Test dönemi sadece patch 16.16–16.23; ≥100 test maçı olan yalnızca 3 lig (LAS, EM, HC).
- Model basit kurala göre anlamlı doğruluk üstünlüğü sağlamıyor; katkısı olasılık vermesi.

## 9. Canlı demo sırası (2 dk)
Önceden çalıştır: `.\.venv\Scripts\python.exe -m streamlit run app.py` → <http://localhost:8501>

1. **Tahmin** sekmesi, varsayılan değerler (3–2 kill, +1000 altın) → Blue önde.
2. Hepsini 0 yap → ~%51 (Blue tarafının küçük avantajı), "dengeli" uyarısı.
3. Blue 5, Red 1, +3000 altın, +1500 XP, +30 CS → ~%91 Blue.
4. Blue 1, Red 6, −4000 altın, −2500 XP, −50 CS → ~%4 Blue, kazanan Red.
5. **Performans** sekmesi → ilk teslimin model tablosu, baseline ve kalibrasyon grafikleri, tarihsel Gold-only / Five-features tablosu.
6. **Hakkında** sekmesi → veri kaynağı, sınırlılıklar, Riot yasal notu.

Yedek plan: uygulama açılmazsa `.\.venv\Scripts\python.exe -m src.predict` ve `reports/figures/` grafikleri.

## 10. Muhtemel sorular (hazır cevaplar `PROJECT_EXPLANATION.md` §13'te)
Neden %70? · Sızıntı nasıl önlendi? · Neden kronolojik bölme? · Neden LR? · Kill katsayısı neden negatif? · Basit kuraldan iyi mi? · Olasılıklar güvenilir mi?
