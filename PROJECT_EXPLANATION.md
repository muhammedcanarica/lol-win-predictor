# LoL Win Predictor – Projeyi Baştan Öğrenme Rehberi

Bu doküman, projeyi hiç bilmeyen birinin (ya da sunumdan önce senin) her adımı anlaması için yazıldı. Bütün sayılar projenin gerçek çalıştırmasından (`reports/metrics.json`) alınmıştır.

---

## 1. Projemiz hangi problemi çözüyor?

Bir League of Legends maçı ortalama 30+ dakika sürer. Biz şu soruyu soruyoruz:

> *Maçın sadece ilk 10 dakikasına bakarak, Blue takımının maçı kazanma olasılığını tahmin edebilir miyiz?*

Bu bir **ikili sınıflandırma** problemidir: çıktı ya "Blue kazanır" (1) ya da "Red kazanır" (0). Ama biz sadece sınıfı değil, **olasılığı** da veriyoruz (ör. "%78 Blue kazanır"). Bu, yayıncıların canlı kazanma olasılığı göstergelerine ve analistlerin "erken oyun ne kadar belirleyici?" sorusuna benzer.

## 2. Veri setimiz ne içeriyor?

- **ChainCC Open LoL Esports Dataset 2026** (CC BY 4.0), dosya `chaincc-teams-2026.csv.gz`.
- **19.902 satır, 74 sütun, 9.951 maç.** Her maç iki satırdır: biri Blue, biri Red takımı için.
- Tarih aralığı: 2026-01-01 → 2026-11-21; 40'tan fazla lig (LPL, LCK, LEC, LAS, …), patch 15.24 – 16.23.
- Sütun grupları:
  - Kimlik/bağlam: `game_id`, `date`, `league`, `patch`, `side`, `team_name` …
  - Draft: `ban1..5`, `pick1..5`
  - **Maç sonu** istatistikleri: `kills`, `towers`, `dragons`, `barons`, `totalgold`, `gamelength` …
  - **10. dakika**: `goldat10`, `killsat10`, `deathsat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`
  - 15. dakika: aynı sütunların `at15` versiyonları
  - Hedef: `result` (True = o satırdaki takım kazandı)

## 3. EDA nedir ve neden yaptık?

**EDA (Exploratory Data Analysis / Keşifsel Veri Analizi)**, modele başlamadan önce veriyi tanımaktır: kaç satır var, eksik var mı, değerler mantıklı mı, hangi değişken hedefle ilişkili? EDA yapmadan model kurmak, haritaya bakmadan yola çıkmaya benzer.

EDA'mız `notebooks/01_eda.ipynb` içinde. Önemli bulgular:

| Bulgu | Sonuç |
|---|---|
| Her maçta tam 2 satır, tam 1 kazanan | Veri tutarlı |
| Tekrarlanan satır: 0 | Temizlik gerekmedi |
| Eksik değerler yalnız ban, split, bo_format gibi kullanmadığımız sütunlarda | Silme/doldurma gerekmedi |
| 175 maçta **tüm** 10. dk değerleri 0 | Eksik veri → çıkarıldı |
| `golddiffat10` Blue ve Red satırlarında tam simetrik (Blue = −Red) | Sadece Blue satırı yeterli |
| Blue kazanma oranı %53.7 | Hafif Blue avantajı, ciddi dengesizlik yok |
| Altın farkı ile hedef korelasyonu 0.46, XP 0.44, CS 0.39, kill 0.22 | Altın farkı en güçlü sinyal |

Altın farkına göre Blue galibiyet oranı (temiz veri):

| 10. dk altın farkı | Blue galibiyet oranı | Maç |
|---|---|---|
| ≤ −3000 | %5.5 | 255 |
| −1000 … −500 | %38.2 | 1202 |
| 0 … +500 | %55.5 | 1303 |
| +2000 … +3000 | %88.7 | 708 |
| > +3000 | %97.5 | 354 |

Grafikler: `reports/figures/` (`target_distribution`, `boxplots`, `diff_distributions`, `winrate_by_golddiff`, `correlation`, `date_patch`).

## 4. Veri temizleme nasıl yapıldı?

Kod: `src/data.py` → `clean()`.

1. **Sadece Blue satırları alındı.** Fark sütunları simetrik olduğu için Red satırı yeni bilgi taşımaz. İki satırı da alsaydık aynı maç hem eğitimde hem testte olabilirdi (sızıntı).
2. **Tüm 10. dk değerleri 0 olan 175 maç çıkarıldı.** Neden? `goldat10` gerçek bir maçta asla 0 olamaz (oyuncular başlangıç altınıyla başlar ve pasif altın kazanır). Bu maçların maç sonu verisi dolu (ortalama 14.9 kill) olduğu için "maç oynanmadı" değil, "10. dk verisi kaydedilmedi" demektir. 0'ı gerçek değer sanarsak model "eşit oyun" diye yanlış öğrenirdi.
   ```python
   all_zero = (blue[AT10_COLS] == 0).all(axis=1)   # 175 maç
   blue = blue[~all_zero]
   ```
3. **Aykırı değerler tutuldu.** IQR yöntemiyle ~130–200 aykırı görünen değer var (ör. 10. dk'da +10.446 altın, 24 kill). Bunlar hata değil, gerçek ezici maçlar ve modelin öğrenmesi gereken örnekler.
4. **Kullanılmayan sütunlar:** maç sonu istatistikleri (`kills`, `towers`, `gamelength`…) → 10. dakikada bilinmez, kullanmak **veri sızıntısı** olurdu. `game_id`, takım adları → kimlik bilgisi. 15. dk verileri → problem 10. dk ile sınırlı.
5. Veri **tarihe göre sıralandı** ve kronolojik bölündü: eski %80 (7.820 maç) eğitim, yeni %20 (1.956 maç) test. Veride aynı zaman damgasına sahip 8 maç olduğu için sıralamada ikinci anahtar olarak `game_id` kullanıldı (her çalıştırmada aynı sıra) ve bölme bir tarih kesimiyle yapılıyor: aynı anda başlayan maçlar asla eğitim/test arasında bölünmez.

**Geçerli sıfırlar korundu mu?** Evet. Sadece altı 10. dk sütununun *hepsi aynı anda* 0 olan satırlar silindi. Tek bir sütunun 0 olması normaldir ve korunur: 5 maçta altın farkı tam 0, 1.272 maçta Blue 10. dakikada hiç kill almamış, 329 maçta hiç kill yok. Silinen 175 maçın her ikisi tarafı da sıfır (bir taraf sıfır diğeri dolu olan maç yok) ve hepsinde maç sonu kill'leri dolu – testlerle doğrulandı.

**Kullanılmayan sütunlardaki eksikler** (dokunulmadı, çünkü modele girmiyorlar): `bo_format`, `match_in_series`, `game_in_match` %100 boş; `split` 2.964 satırda boş (EWC, LJL, MSI, Worlds, promotion turnuvaları gibi "split" kavramı olmayan etkinlikler); ban sütunları 315 satırda / 233 maçta eksik (çoğu CD Qualifier).

Sonuç: **9.776 maç, 5 özellik, 1 hedef.**

## 5. Logistic Regression nasıl çalışır?

Özelliklerin ağırlıklı toplamını alır ve bunu **sigmoid** fonksiyonuyla 0–1 arasına sıkıştırır:

```
z = b + w1·gold + w2·xp + w3·cs + w4·kills + w5·deaths
P(Blue kazanır) = 1 / (1 + e^(−z))
```

Eğitim, gerçek sonuçlarla tahminler arasındaki **log-loss**'u en aza indiren ağırlıkları bulmaktır. `C` parametresi **düzenlileştirmeyi** (regularization) kontrol eder: küçük C → ağırlıklar küçük tutulur, aşırı öğrenme azalır.

Projemizde özellikler önce `StandardScaler` ile ölçeklendi (ortalama 0, std 1), çünkü altın farkı binlerle, kill ise tek haneli sayılarla ölçülür. Öğrenilen (ölçeklenmiş) katsayılar:

| Özellik | Katsayı |
|---|---|
| golddiffat10 | **+0.847** |
| xpdiffat10 | +0.383 |
| csdiffat10 | +0.256 |
| deathsat10 | +0.093 |
| killsat10 | −0.063 |
| sabit (b) | +0.214 (→ Blue avantajı) |

İlginç nokta: kill katsayısı **negatif**. Bu "kill almak kötü" demek değildir! Kill'in değeri zaten altın farkının içinde. Altın farkı sabitken daha fazla kill + daha fazla ölüm "kaotik ama dengeli" bir oyunu gösterir. Bu, **çoklu doğrusallığın** (özellikler arası güçlü korelasyon) klasik bir örneğidir.

## 6. Random Forest nasıl çalışır?

Bir **karar ağacı**, "golddiff > 1200 mü? Evet → xpdiff > 300 mü? …" gibi soru zinciriyle karar verir. Tek ağaç kolayca ezber yapar.

**Random Forest**, yüzlerce ağacı her biri verinin farklı bir rastgele örneğiyle (bootstrap) ve her bölmede farklı rastgele özellik alt kümesiyle eğitir; sonra hepsinin oylamasını/ortalamasını alır. Bu "kalabalığın bilgeliği" varyansı azaltır.

Projemizde en iyi ayarlar: `n_estimators=300`, `max_depth=4`, `min_samples_leaf=20` → sığ ağaçlar en iyisiydi, yani ilişki basit.

## 7. XGBoost nasıl çalışır?

**Gradient Boosting** ağaçları **sırayla** kurar: her yeni küçük ağaç, önceki ağaçların hatalarını (gradyanlarını) düzeltmeye çalışır. `learning_rate` her ağacın katkısını küçültür; `max_depth` ağaçların karmaşıklığını sınırlar. **XGBoost**, bunun düzenlileştirmeli ve hızlı bir uygulamasıdır.

En iyi ayarlar: `n_estimators=200`, `max_depth=2`, `learning_rate=0.03`, `subsample=0.8`. Yine çok sığ ağaçlar seçildi.

## 8. Cross-validation neden kullanıldı?

Tek bir doğrulama bölmesi şansa bağlıdır. **Cross-validation (çapraz doğrulama)** eğitim verisini birkaç parçaya bölüp modeli birkaç kez farklı parçalarda doğrular ve ortalama alır → daha güvenilir bir tahmin.

Biz **`TimeSeriesSplit(n_splits=5)`** kullandık: her katta doğrulama verisi, eğitim verisinden **daha sonraki** tarihlerdendir. Normal (karışık) k-fold, modelin "geleceği görüp geçmişi tahmin etmesine" izin verirdi; maç verisi zamana bağlı (patch/meta) olduğu için bu yanıltıcı olurdu.

Ayrıca `StandardScaler` bir **Pipeline** içinde olduğundan her katta sadece o katın eğitim kısmına fit edilir → doğrulama verisinden bilgi sızmaz.

## 9. Hiperparametre optimizasyonu nedir?

Modelin veriden öğrendiği değerlere **parametre** (ör. LR ağırlıkları), bizim eğitimden önce seçtiğimiz ayarlara **hiperparametre** (ör. `C`, `max_depth`) denir. `GridSearchCV`, verilen her kombinasyonu cross-validation ile dener ve en iyi ortalama skoru vereni seçer.

| Model | Denenen ızgara | Kombinasyon |
|---|---|---|
| Logistic Regression | C ∈ {0.001, 0.01, 0.1, 1, 10} | 5 |
| Random Forest | max_depth ∈ {4, 6, 8, None} × min_samples_leaf ∈ {1, 20, 50} | 12 |
| XGBoost | n_estimators ∈ {200, 400} × max_depth ∈ {2, 3, 4} × learning_rate ∈ {0.03, 0.1} | 12 |

Skor: ROC-AUC (eşikten bağımsız, olasılık sıralamasını ölçer).

## 10. Performans metrikleri ne anlama geliyor?

"Pozitif" = Blue kazandı. Seçilen modelin test karışıklık matrisi:

|  | Tahmin: Red | Tahmin: Blue |
|---|---|---|
| **Gerçek: Red** | TN = 595 | FP = 334 |
| **Gerçek: Blue** | FN = 246 | TP = 781 |

| Metrik | Formül | Değer | Anlamı |
|---|---|---|---|
| Accuracy | (TP+TN)/Toplam | 0.704 | Maçların %70'ini doğru bildik |
| Precision | TP/(TP+FP) | 0.700 | "Blue kazanır" dediğimizde %70 haklıyız |
| Recall (duyarlılık) | TP/(TP+FN) | 0.760 | Blue'nun kazandığı maçların %76'sını yakaladık |
| Specificity | TN/(TN+FP) | 0.640 | Red'in kazandığı maçların %64'ünü yakaladık |
| F1 | 2·P·R/(P+R) | 0.729 | Precision ve recall dengesi |
| ROC-AUC | ROC eğrisi altı alan | 0.779 | Rastgele bir Blue-kazandı maçına, Red-kazandı maçından daha yüksek olasılık verme şansı %78 |
| Brier | ortalama (p − y)² | 0.191 | Olasılık hatası; küçük = iyi (hep 0.5 demek 0.25 verir) |

Recall'un specificity'den yüksek olması, modelin hafif Blue yanlılığını (veride Blue %53.7 kazanıyor) yansıtır.

**Kalibrasyon:** Model "%80" dediğinde gerçekten ~%80 kazanılıyor mu? `calibration.png` grafiğinde üç modelin eğrisi de köşegene yakın; olasılıklar **yaklaşık** kalibre. Kusursuz değil: test döneminde model Blue'yu ortalama ~2 puan fazla tahmin ediyor (aşağıda). Ek kalibrasyon (Platt/isotonic) uygulanmadı; sapma zaman kaymasından geldiği için eski veriyle yeniden kalibre etmek bunu düzeltmezdi.

**Sabit olasılık referansıyla kalibrasyon:** Hiçbir şey bilmeyen ama "dürüst" bir tahminci her maça eğitimdeki Blue galibiyet oranını (0.540) verir. Bunun Brier'ı 0.2496, log-loss'u 0.692'dir. Modelimiz Brier 0.1912, log-loss 0.560 → **Brier Skill Score = 1 − 0.1912/0.2496 = 0.234**, yani sabit referansa göre olasılık hatası %23 azaldı. Küçük bir sapma var: modelin ortalama tahmini 0.546, testteki gerçek Blue oranı 0.525. Test döneminde Blue avantajı biraz azaldı ve model bunu bilemez (zaman kayması).

**Referanslarla kıyas (dürüst sonuç):**

| Yöntem | Test accuracy | Test ROC-AUC |
|---|---|---|
| Her zaman Blue | 0.525 | – |
| Altın farkı > 0 ise Blue | 0.693 | 0.772 (altın farkı skor olarak) |
| **Logistic Regression (5 özellik)** | **0.704** | **0.779** |

Aynı 1.956 test maçından 2.000 kez bootstrap örneği alıp farkın %95 güven aralığını hesapladık (`paired_bootstrap` in `src/train.py`):
- Accuracy farkı **+0.010**, GA **[−0.005, +0.025]**
- ROC-AUC farkı **+0.007**, GA **[−0.001, +0.015]**

İki aralık da 0'ı içeriyor → model, altın farkının işaretinden **istatistiksel olarak anlamlı biçimde daha doğru değil**. Bu bir hata değil, verinin söylediği şey: 10. dakikadaki bilginin neredeyse tamamı altın farkında. Modelin gerçek katkısı "kim kazanır?" yerine "**ne kadar olası?**" sorusuna yaklaşık kalibre bir cevap vermesidir (+500 altın ile +5000 altın arasındaki farkı işaret kuralı göremez).

**Patch ve lig bazında performans** (`reports/group_performance.csv`). Yalnızca test setinde **en az 100 maçı** olan gruplar yorumlandı:

| Grup | n | Accuracy | ROC-AUC |
|---|---|---|---|
| Patch 16.17 | 781 | 0.691 | 0.764 |
| Patch 16.18 | 539 | 0.718 | 0.805 |
| Patch 16.20 | 320 | 0.716 | 0.781 |
| Patch 16.16 | 280 | 0.686 | 0.760 |
| LAS | 347 | 0.749 | 0.842 |
| EM | 240 | 0.713 | 0.799 |
| HC | 102 | 0.608 | 0.684 |

Patch'ler arasında belirgin bir bozulma yok. HC'deki düşük skor 102 maçla dikkat çekici ama kesin değil. LCK, LPL, LEC gibi büyük liglerin test döneminde yalnızca ~50 maçı var (sezon sonu); bunlar için sonuç çıkarılmadı. Test dönemi sadece patch 16.16–16.23'ü kapsıyor.

## 11. En iyi model neden seçildi?

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Logistic Regression** | **0.7788** | 0.7794 |
| Random Forest | 0.7765 | 0.7770 |
| XGBoost | 0.7754 | 0.7791 |

- Seçim **sadece CV skoruna** göre yapıldı; test seti seçimde hiç kullanılmadı (yoksa test skoru iyimser olurdu).
- Farklar küçük (standart sapma ~0.012 içinde), yani üç model pratikte eşit. Bu durumda **en basit, en hızlı ve en yorumlanabilir** model olan Logistic Regression tercih edilmeye de değer (Occam'ın usturası).
- Ağaç modellerinin sığ ağaçları seçmesi de ilişkinin büyük ölçüde doğrusal/monoton olduğunu gösteriyor (altın farkı ↑ → kazanma ↑).
- Permütasyon önemi (test ROC-AUC düşüşü): golddiff 0.157, xpdiff 0.028, csdiff 0.009, kills 0.001, deaths ≈ 0 → model esasen altın farkına dayanıyor.

## 12. Streamlit modeli nasıl kullanıyor?

`app.py`:

1. `models/best_model.joblib` yüklenir (`@st.cache_resource` ile bir kez). Bu dosya **ölçekleyici + model** içeren tam Pipeline'dır; girdiye eğitimdeki aynı ön işlem uygulanır.
2. Kullanıcı Blue/Red kill sayılarını ve altın/XP/CS farklarını girer.
3. `src/predict.py` → `predict_blue_win()` girdileri doğru sütun sırasıyla bir DataFrame'e çevirir ve `predict_proba` çağırır.
4. Blue olasılığı p, Red olasılığı 1 − p olarak gösterilir; p ≥ 0.5 ise tahmini kazanan Blue'dur.
5. "Performans" sekmesi `reports/metrics.json` ve `reports/figures/` içeriğini okur; hiçbir sayı kodda sabit yazılı değildir.

Örnek gerçek çıktılar:

| Girdi | Blue olasılığı |
|---|---|
| Her şey eşit, 2–2 kill | 0.521 |
| +3000 altın, +1500 XP, +30 CS, 5–1 kill | 0.912 |
| −3000 altın, −1500 XP, −30 CS, 1–5 kill | 0.105 |

## 13. Sunumda sorulabilecek sorular ve cevaplar

**S: Neden doğruluk sadece %70? Düşük değil mi?**
C: 10. dakikada maçın büyük kısmı daha oynanmadı; geri dönüşler (comeback) sık. Altın farkının işaretine bakan basit kural 0.693 veriyor; problemin doğal tavanı yakın. Daha fazla bilgi (draft, takım gücü) ile artabilir.

**S: Veri sızıntısını nasıl önlediniz?**
C: (1) Sadece 10. dk'da bilinen özellikler; maç sonu sütunları yok. (2) Maç başına tek satır, train/test'te aynı maç yok (testle doğrulandı). (3) Ölçekleme Pipeline içinde, her CV katında sadece eğitim kısmına fit. (4) Kronolojik bölme, geleceğin verisi eğitimde yok.

**S: Modeliniz "altın farkı pozitifse Blue kazanır" kuralından daha mı iyi?**
C: Doğrulukta +1 puan, ama bootstrap %95 güven aralığı 0'ı içeriyor; anlamlı değil. Modelin katkısı yaklaşık kalibre olasılık (Brier Skill Score 0.234; kusursuz değil, Blue ~2 puan fazla tahmin ediliyor). Bunu saklamak yerine açıkça raporladık.

**S: Aynı Bo3/Bo5 serisinin maçları hem eğitimde hem testte olabilir mi?**
C: Seriyi "aynı gün + aynı takım çifti" olarak tanımladık (veride seri sütunları tamamen boş). Kronolojik sınırı aşan yalnızca 7 test maçı var ve model takım adı/kimliği görmediği için seriyi ezberleyemez. Risk ihmal edilebilir.

**S: Neden random değil kronolojik bölme?**
C: Gerçekte model geçmiş maçlarla eğitilip gelecek maçları tahmin eder. Meta/patch zamanla değişir; kronolojik bölme bu gerçek senaryoyu taklit eder ve daha dürüst bir skor verir.

**S: 0 değerli satırları neden sildiniz, gerçekten 0 olamaz mı?**
C: Altın farkı tek başına 0 olabilir, ama `goldat10` (takımın toplam altını) 0 olamaz. 175 maçta altı sütunun hepsi aynı anda 0 ve maç sonu verisi dolu → kayıt eksikliği.

**S: Neden en iyi model Logistic Regression? Karmaşık modeller daha iyi olmaz mı?**
C: CV ROC-AUC'da LR en yüksekti (0.7788). İlişki neredeyse monoton; ağaçlar ek bir yapı bulamadı. Basit model daha az ezber yapar, hızlıdır ve yorumlanabilir.

**S: Kill katsayısı neden negatif?**
C: Çoklu doğrusallık. Kill'in etkisi altın farkına zaten yansıyor; altın sabitken ek kill/ölüm sadece oyunun kaotikliğini gösteriyor. Katsayılar tek tek nedensellik göstermez.

**S: Sınıf dengesizliği var mı?**
C: Blue %53.7 – Red %46.3. Hafif; sınıf ağırlığı veya yeniden örnekleme gerekmedi. Bu yüzden recall > specificity.

**S: Olasılıklara güvenebilir miyiz?**
C: Kalibrasyon eğrisi köşegene yakın, Brier 0.191 (rastgele 0.25). Makul ölçüde evet, ama kusursuz değil: ortalama tahmin 0.546, gerçek Blue oranı 0.525. Olasılıklar kesin değil, yaklaşık değerlendirilmeli.

**S: Test setine kaç kez baktınız?**
C: Model seçimi tamamen CV ile yapıldı; test seti yalnızca son değerlendirmede kullanıldı.

**S: Hangi özellik en önemli?**
C: Altın farkı (permütasyon önemi 0.157); onu XP (0.028) ve CS (0.009) izliyor.

**S: Projeyi nasıl geliştirirsiniz?**
C: Draft/şampiyon bilgisi, takım Elo/geçmiş form, ejderha ve void grub gibi erken objektifler, lig bazlı etkiler; 15. dk modeli ile karşılaştırma.

**S: Neden ROC-AUC ile seçtiniz, accuracy ile değil?**
C: Amacımız olasılık üretmek. ROC-AUC eşikten bağımsızdır ve sıralama kalitesini ölçer; accuracy 0.5 eşiğine bağlıdır.

---

### Dosya haritası (hangi soru hangi kodda?)

| Konu | Dosya |
|---|---|
| Temizleme, özellik listesi, bölme | `src/data.py` |
| Modeller, ızgaralar, CV, metrikler, grafikler | `src/train.py` |
| Tahmin fonksiyonu | `src/predict.py` |
| Arayüz | `app.py` |
| EDA | `notebooks/01_eda.ipynb` |
| Testler | `tests/` |
