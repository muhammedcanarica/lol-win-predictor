"""LoL Win Predictor – Streamlit arayüzü. Çalıştırma: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_blue_win
from src.train import FIG_DIR, METRICS_PATH

ESPORTS_METRICS_PATH = METRICS_PATH.parent / "esports" / "esports_metrics.json"

st.set_page_config(page_title="LoL Win Predictor", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
.team {border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-weight: 700; font-size: 1.15rem;}
.blue {background: linear-gradient(90deg, #0a3a6b, #0a6ebd33); border-left: 6px solid #2a8cf0; color: #cfe6ff;}
.red  {background: linear-gradient(90deg, #6b0a14, #c6282833); border-left: 6px solid #e5484d; color: #ffd6d6;}
.winner {text-align: center; font-size: 1.4rem; font-weight: 700; padding: 0.6rem; border-radius: 10px;
         border: 1px solid #c89b3c; color: #c89b3c;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


st.title("⚔️ LoL Win Predictor")
st.caption("2026 profesyonel League of Legends maçlarında **10. dakika istatistikleriyle** Blue takımının kazanma olasılığı")

try:
    bundle, metrics = get_model(), get_metrics()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

tab_pred, tab_perf, tab_about = st.tabs(["🎯 Tahmin", "📊 Performans", "ℹ️ Hakkında"])

with tab_pred:
    with st.form("stats"):
        c_blue, c_red = st.columns(2)
        with c_blue:
            st.markdown('<div class="team blue">🔵 BLUE TAKIMI</div>', unsafe_allow_html=True)
            blue_kills = st.number_input("Blue kill sayısı (10. dk)", 0, 16, 3,
                                         help="Modeldeki karşılığı: killsat10")
        with c_red:
            st.markdown('<div class="team red">🔴 RED TAKIMI</div>', unsafe_allow_html=True)
            red_kills = st.number_input("Red kill sayısı (10. dk)", 0, 16, 2,
                                        help="Modeldeki karşılığı: deathsat10 (Blue'nun ölüm sayısı)")
        st.markdown("**Farklar (Blue − Red)** – pozitif değer Blue'nun önde olduğunu gösterir")
        c1, c2, c3 = st.columns(3)
        # Sınırlar eğitim verisindeki min/max değerlerine yakın: dışarıda model ekstrapolasyon yapardı.
        gold = c1.number_input("Altın farkı", -7500, 7500, 1000, step=100, help="golddiffat10")
        xp = c2.number_input("XP farkı", -6500, 6500, 500, step=100, help="xpdiffat10")
        cs = c3.number_input("CS (minyon) farkı", -160, 160, 10, help="csdiffat10")
        st.caption("Girdi sınırları eğitim verisinde görülen aralıktır (kill 0–16, altın ±7500, XP ±6500, CS ±160).")
        st.form_submit_button("Tahmin et", type="primary", width="stretch")

    stats = dict(golddiffat10=gold, xpdiffat10=xp, csdiffat10=cs, killsat10=blue_kills, deathsat10=red_kills)
    p_blue = predict_blue_win(bundle, stats)
    p_red = 1 - p_blue

    m1, m2 = st.columns(2)
    m1.metric("🔵 Blue kazanma olasılığı", f"%{p_blue * 100:.1f}")
    m2.metric("🔴 Red kazanma olasılığı", f"%{p_red * 100:.1f}")
    st.progress(p_blue, text=f"Çubuk = Blue olasılığı (%{p_blue * 100:.1f}); kalan kısım Red")
    winner = "🔵 Blue" if p_blue >= 0.5 else "🔴 Red"
    st.markdown(f'<div class="winner">Tahmini kazanan: {winner}</div>', unsafe_allow_html=True)
    if 0.4 < p_blue < 0.6:
        st.info("Olasılık %40–60 arasında: maç 10. dakikada hâlâ dengeli, tahmin belirsiz.")
    st.caption(f"Model: {bundle['name']} · Tahminler gerçek eğitilmiş modelden üretilir. Olasılıklar yaklaşık "
               "kalibredir, kusursuz değildir: test döneminde model Blue'yu ortalama ~2 puan fazla tahmin etti.")

with tab_perf:
    best = metrics["best_model"]
    t = metrics["test"][best]
    st.subheader(f"Seçilen model: {best}")
    k = st.columns(4)
    k[0].metric("Accuracy", f"{t['accuracy']:.3f}")
    k[1].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    k[2].metric("F1", f"{t['f1']:.3f}")
    k[3].metric("Brier", f"{t['brier']:.3f}")

    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
              "f1": "F1", "roc_auc": "Test ROC-AUC", "brier": "Brier"}
    rows = {n: {"CV ROC-AUC": metrics["cv"][n]["cv_roc_auc"], **{lbl: metrics["test"][n][k] for k, lbl in labels.items()}}
            for n in metrics["test"]}
    st.markdown("**Üç modelin karşılaştırması** (seçim CV ROC-AUC ile, metrikler test setinde)")
    st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")
    b = metrics["baselines"]
    st.caption(f"Referans: Her zaman Blue tahmini → accuracy {b['always_blue_accuracy']:.3f} · "
               f"Sadece altın farkının işareti → accuracy {b['golddiff_sign_accuracy']:.3f} · "
               f"Modelin ek faydası {b['best_vs_sign_bootstrap']['accuracy_diff']:+.3f} "
               f"(%95 GA {b['best_vs_sign_bootstrap']['accuracy_diff_ci95'][0]:+.3f} – "
               f"{b['best_vs_sign_bootstrap']['accuracy_diff_ci95'][1]:+.3f}; istatistiksel olarak anlamlı değil)")
    c = metrics["calibration"]
    st.caption(f"Kalibrasyon: Brier {c['brier_model']:.3f} vs sabit olasılık ({c['constant_prob']:.3f}) referansı "
               f"{c['brier_constant']:.3f} → Brier Skill Score {c['brier_skill_score']:.3f}. Kalibrasyon iyi ama "
               f"kusursuz değil: ortalama tahmin {c['mean_predicted']:.3f}, gerçek Blue oranı {c['observed_rate']:.3f}.")

    g1, g2 = st.columns(2)
    g1.image(str(FIG_DIR / "feature_importance.png"), caption="Özellik önemi (permütasyon, test seti)")
    g2.image(str(FIG_DIR / "roc_curves.png"), caption="ROC eğrileri")
    g3, g4 = st.columns(2)
    g3.image(str(FIG_DIR / "confusion_matrix.png"), caption="Karışıklık matrisi")
    g4.image(str(FIG_DIR / "calibration.png"), caption="Olasılık kalibrasyonu")

    st.subheader("Tarihsel olasılık karşılaştırması")
    try:
        esports = json.loads(ESPORTS_METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.caption("Tarihsel değerlendirme raporu henüz mevcut değil.")
    else:
        historical_models = {"Yalnız altın": "gold_only", "Beş girdi": "five_features"}
        historical_metrics = {"Brier ↓": "brier", "ROC-AUC ↑": "roc_auc", "Log-loss ↓": "log_loss"}
        rows = []
        for label, name in historical_models.items():
            scores = esports["models"][name]["test"]
            rows.append({"Model": label, **{title: scores[key] for title, key in historical_metrics.items()}})
        st.dataframe(pd.DataFrame(rows).round(5), hide_index=True, width="stretch")
        comparison = esports["brier_comparison"]
        low, high = comparison["ci95"]
        st.caption(f"{esports['split']['test']['games']:,} test maçı · Brier farkı (yalnız altın − beş girdi) "
                   f"{comparison['gold_only_minus_five_features']:+.5f} · Seri bazlı %95 GA [{low:+.5f}, {high:+.5f}]")
        st.write(esports["interpretation"])

with tab_about:
    s, c = metrics["split"], metrics["cleaning"]
    st.markdown(f"""
### Proje
Profesyonel LoL maçlarının ilk 10 dakikasındaki **altın, XP, CS farkı ve kill sayıları** ile Blue takımının
maçı kazanma olasılığını tahmin eden bir makine öğrenmesi sistemi (BİL4011 final projesi).

### Veri
- Kaynak: **ChainCC Open LoL Esports Dataset 2026** – [chaincc.lol/free/data](https://chaincc.lol/free/data), lisans **CC BY 4.0**
- {c['raw_games']} maç okundu, 10. dk verisi eksik olan {c['dropped_all_zero_at10']} maç çıkarıldı → **{c['final_games']} maç**
- Kronolojik bölme: eğitim {s['train_games']} maç ({s['train_period'][0][:10]} – {s['train_period'][1][:10]}),
  test {s['test_games']} maç ({s['test_period'][0][:10]} – {s['test_period'][1][:10]})

### Sınırlılıklar
- Yalnızca 5 sayısal özellik kullanılır; şampiyon seçimi, takım gücü, ejderha/kule gibi bilgiler modelde yok.
- 10. dakikada maç büyük ölçüde açıktır: doğruluk ~%70 civarındadır, %100'e yaklaşması beklenmez.
- Model, "altın farkı pozitifse Blue kazanır" basit kuralından doğrulukta **istatistiksel olarak anlamlı biçimde
  daha iyi değildir** (+1 puan, %95 güven aralığı 0'ı içerir). Katkısı, kazanma ihtimalini olasılık olarak vermesidir.
- Olasılıklar yaklaşık kalibredir ama kusursuz değildir (Blue hafifçe fazla tahmin ediliyor).
- Model 2026 patch'leriyle eğitildi; meta değiştikçe performans düşebilir.
- Altın/XP/CS farkları birbiriyle güçlü ilişkilidir; tek tek katsayılar nedensellik göstermez.
- Girdiler eğitim aralığıyla sınırlandı, ama birbiriyle çelişen kombinasyonlar (ör. 0–10 kill ile +5000 altın)
  hâlâ girilebilir; bu durumda tahmin güvenilir değildir.
- "Red kill sayısı", modelde Blue'nun ölüm sayısı (`deathsat10`) olarak kullanılır; kaynak veride bu iki değer
  maçların ~%1'inde küçük farklılık gösterir.

### Yasal not
LoL Win Predictor, Riot Games tarafından onaylanmamıştır ve Riot Games'in ya da League of Legends'ın
üretiminde veya yönetiminde resmi olarak yer alan herhangi bir kişinin görüşlerini yansıtmaz.
League of Legends ve Riot Games, Riot Games, Inc.'in ticari markaları veya tescilli ticari markalarıdır.
Bu projede Riot Games'e ait görsel kullanılmamıştır.
""")
