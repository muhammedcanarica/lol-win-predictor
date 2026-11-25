"""LoL Win Predictor – Streamlit arayüzü. Çalıştırma: streamlit run app.py"""
import json
from base64 import b64encode
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_blue_win
from src.train import FIG_DIR, METRICS_PATH

ESPORTS_METRICS_PATH = METRICS_PATH.parent / "esports" / "esports_metrics.json"
ASSETS = Path(__file__).resolve().parent / "assets"
INPUTS = ("killsat10", "deathsat10", "golddiffat10", "xpdiffat10", "csdiffat10")
DEFAULT_STATS = dict(zip(INPUTS, (3, 2, 1000, 500, 10)))
PRESETS = {
    "Dengeli": (2, 2, 0, 0, 0),
    "Blue önde": (5, 1, 3000, 1500, 30),
    "Red önde": (1, 6, -4000, -2500, -50),
}
INPUT_LABELS = ("Blue kill", "Red kill", "Altın farkı", "XP farkı", "Minyon farkı")

st.set_page_config(page_title="LoL Win Predictor", page_icon=str(ASSETS / "lol" / "score.png"), layout="wide")
st.html(ASSETS / "dashboard.css")


@st.cache_data
def asset_uri(relative_path, media_type):
    return f"data:{media_type};base64,{b64encode((ASSETS / relative_path).read_bytes()).decode('ascii')}"


def image_uri(filename):
    media_type = "image/webp" if filename.endswith(".webp") else "image/png"
    return asset_uri(f"lol/{filename}", media_type)


def stat_icon(filename, label, color=""):
    st.html(f'<div class="stat-icon {color}"><img src="{image_uri(filename)}" alt="">{label}</div>')


def select_preset(values):
    for key, value in zip(INPUTS, values):
        st.session_state[key] = value
    st.session_state.match_stats = dict(zip(INPUTS, values))


def sensitivity(bundle, stats, probability):
    equal_kills = (stats["killsat10"] + stats["deathsat10"]) / 2
    alternatives = {
        "Altın": {"golddiffat10": 0}, "XP": {"xpdiffat10": 0}, "Minyon": {"csdiffat10": 0},
        "Kill": {"killsat10": equal_kills, "deathsat10": equal_kills},
    }
    return pd.DataFrame([
        {"Eşitlenen girdi": label, "Blue olasılığındaki değişim (yüzde puan)":
         round((predict_blue_win(bundle, stats | overrides) - probability) * 100, 2)}
        for label, overrides in alternatives.items()
    ])


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def optional_report(directory):
    try:
        return json.loads((METRICS_PATH.parent / directory / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


st.html(f'''<style>
    :root {{ --score-icon: url("{image_uri("score.png")}"); }}
    @font-face {{
        font-family: "Instrument Serif";
        src: url("{asset_uri("fonts/instrument-serif.woff2", "font/woff2")}") format("woff2");
        font-weight: 400;
        font-display: swap;
    }}
</style>''')
st.html(f'''<header class="match-hero">
    <img src="{image_uri("kindred.webp")}" alt="">
    <h1>LoL Win Predictor</h1>
    <p>Profesyonel League of Legends maçlarında 10. dakika verileriyle galibiyet olasılığını hesapla.</p>
</header>''')

try:
    bundle, metrics = get_model(), get_metrics()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

tab_pred, tab_perf, tab_about = st.tabs(["Tahmin", "Performans", "Hakkında"])

with tab_pred:
    for key, value in DEFAULT_STATS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("match_stats", DEFAULT_STATS.copy())
    with st.container(horizontal=True, horizontal_alignment="center", key="match-presets"):
        for label, values in PRESETS.items():
            st.button(label, on_click=select_preset, args=(values,))

    stats = st.session_state.match_stats
    with st.container(key="scoreboard"):
        st.html(f'''<div class="sb-header">
            <span class="sb-team team-blue">MAVİ TARAF</span>
            <span class="sb-score"><b class="team-blue">{stats["killsat10"]}</b>
            <img src="{image_uri("score.png")}" alt="kill"><b class="team-red">{stats["deathsat10"]}</b></span>
            <span class="sb-team team-red">KIRMIZI TARAF</span></div>
            <p class="sb-hint">10. dakika verileri · Farkları Blue − Red olarak gir. Pozitif değer Blue avantajını gösterir.</p>''')
        with st.form("stats", border=False):
            with st.container(key="sb-row-kill"):
                c_blue, c_mid, c_red = st.columns([1, 0.7, 1], gap=None)
                with c_blue:
                    st.number_input("Blue kill sayısı", 0, 16, key="killsat10")
                with c_mid:
                    stat_icon("score.png", "Kill", "sb-label")
                with c_red:
                    st.number_input("Red kill sayısı", 0, 16, key="deathsat10")
            for icon, label, unit, bound, step, key in zip(
                ("gold.png", "xp.png", "minion.png"), ("Altın farkı", "XP farkı", "Minyon farkı"),
                ("altın", "XP", "minyon"), (7500, 6500, 160), (100, 100, 1), INPUTS[2:],
            ):
                value = stats[key]
                lead = f'<div class="sb-lead" style="--w:{abs(value) / bound * 100:.1f}%">+{abs(value):,} {unit} önde</div>'.replace(",", ".")
                with st.container(key=f"sb-row-{key}"):
                    c_blue, c_mid, c_red = st.columns([1, 0.7, 1], gap=None)
                    with c_blue:
                        st.html(lead if value > 0 else '<div class="sb-lead sb-none">—</div>')
                    with c_mid:
                        stat_icon(icon, label, "sb-label")
                        st.number_input(label, -bound, bound, step=step, key=key, label_visibility="collapsed")
                    with c_red:
                        st.html(lead if value < 0 else '<div class="sb-lead sb-none">—</div>')
            if st.form_submit_button("Tahmin et", type="primary", width="stretch"):
                st.session_state.match_stats = stats = {key: st.session_state[key] for key in INPUTS}
                st.rerun()  # başlık ve önde-çubukları yeni değerlerle çizilsin

    with st.container(key="match-result"):
        p_blue = predict_blue_win(bundle, stats)
        blue_pct, red_pct = p_blue * 100, (1 - p_blue) * 100
        winner = "Blue" if p_blue >= 0.5 else "Red"
        m1, m_mid, m2 = st.columns([1, 0.9, 1], vertical_alignment="center")
        with m1, st.container(key="blue-result"):
            st.metric("Blue", f"%{blue_pct:.1f}")
        with m_mid:
            st.markdown(f'<div class="match-verdict"><small>GALİBİYET OLASILIĞI</small>Tahmini kazanan '
                        f'<span class="team-{winner.lower()}">{winner}</span></div>', unsafe_allow_html=True)
        with m2, st.container(key="red-result"):
            st.metric("Red", f"%{red_pct:.1f}")
        st.html(f'<div class="duel-track" role="img" aria-label="Blue yüzde {blue_pct:.1f}, Red yüzde {red_pct:.1f}">'
                f'<span style="width:{blue_pct:.4f}%"></span></div>')
        if 0.4 < p_blue < 0.6:
            st.caption("Maç dengeli görünüyor. İki tarafın da kazanma olasılığı %40–60 aralığında.")
        st.caption(f"{bundle['name']} · Son hesaplanan senaryo")

        pin_label = "Referansı güncelle" if "reference_stats" in st.session_state else "Referans olarak kaydet"
        if st.button(pin_label, key="pin-match"):
            st.session_state.reference_stats = stats.copy()
        if "reference_stats" in st.session_state:
            reference = st.session_state.reference_stats
            reference_pct = 100 * predict_blue_win(bundle, reference)
            st.metric("Kaydedilen Blue olasılığı", f"%{reference_pct:.1f}", f"{blue_pct - reference_pct:+.1f} yüzde puan",
                      delta_description="Güncel tahmin − kaydedilen referans")
            with st.expander("Senaryo girdilerini karşılaştır"):
                st.table(pd.DataFrame({"Referans": [reference[key] for key in INPUTS],
                                       "Güncel": [stats[key] for key in INPUTS]}, index=INPUT_LABELS))
            if st.button("Referansı kaldır", key="remove-reference"):
                del st.session_state.reference_stats
                st.rerun()

        with st.expander("Girdilerin etkisini incele"):
            st.dataframe(sensitivity(bundle, stats, p_blue), hide_index=True, width="stretch")
            st.caption("Her satırda yalnız bir fark eşitlenir, diğer girdiler aynı kalır. Değerler modelin tepkisini gösterir.")

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

    with st.expander("Elo ve Mahkeme deneyleri"):
        improvements = optional_report("improvements")
        court = optional_report("court")
        if improvements or court:
            st.caption("Keşifsel sonuçlar · Ayrı bir yeni test dönemiyle doğrulanmadı.")
        if improvements:
            elo = improvements["audit"]["elo"]
            baseline = improvements["audit"]["five_features"]
            st.write(f"**Elo** · Aynı {improvements['audit_fold']['validation']['games']} maçta "
                     f"doğruluk %{baseline['accuracy'] * 100:.2f} → %{elo['accuracy'] * 100:.2f}.")
        if court:
            chosen = court["audit_metrics"][court["selected_method"]]
            changes = court["audit_cases"]["changes"][court["selected_method"]]
            st.write(f"**Mahkeme** · Doğruluk %{chosen['accuracy'] * 100:.2f}. Elo'ya göre "
                     f"{changes['rescued_vs_elo']} hata düzeldi, {changes['spoiled_vs_elo']} doğru karar bozuldu.")
        if not improvements and not court:
            st.caption("Keşifsel deney raporları henüz mevcut değil.")

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
- İlk teslimin tarihsel test döneminde doğruluk %{t['accuracy'] * 100:.2f} olarak ölçüldü.
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
Altın, minyon, skor ikonları ve Kindred görseli Riot Games Data Dragon'dan alınmıştır.
XP ikonu CommunityDragon üzerinden alınan League of Legends istemci varlığıdır.
[Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon) · [CommunityDragon](https://communitydragon.org/docs/assets)
""")
