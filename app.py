"""LoL Win Predictor – Streamlit arayüzü. Çalıştırma: streamlit run app.py"""
import json
import time
from base64 import b64encode
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_blue_win, xp_levels
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


def cast_prediction():
    """Tahmini güncel girdilerle hesaplar; efektler sırayla ışık huzmesi / bıçak lotusu olur."""
    st.session_state.match_stats = {key: st.session_state[key] for key in INPUTS}
    count = st.session_state.get("casts", 0)
    st.session_state.casts = count + 1
    st.session_state.pending_cast = "lux" if count % 2 == 0 else "kata"


def fmt(value):
    return f"{abs(value):,.0f}".replace(",", ".")


def lead_html(value, unit, side):
    ahead = value > 0 if side == "blue" else value < 0
    text = f"{side.capitalize()} +{fmt(value)} {unit} önde" if ahead else "—"
    return f'<div class="sb-lead {side}{"" if ahead else " muted"}">{text}</div>'


def lead_text(value, unit, even_text):
    side = "blue" if value > 0 else "red" if value < 0 else "even"
    text = even_text if value == 0 else f"{side.capitalize()} +{fmt(value)} {unit}"
    return f'<p class="lead-text {side}">{text}</p>'


def team_badge(side):
    name, sub = ("MAVİ TARAF", "Blue · sol üs") if side == "blue" else ("KIRMIZI TARAF", "Red · sağ üs")
    return (f'<div class="team-badge {side}"><span class="team-portrait"><img src="{image_uri(f"minion-{side}.webp")}" '
            f'alt="{name} minyonu"></span><div><strong>{name}</strong><span>{sub}</span></div></div>')


def portrait(side, level):
    # st.html <svg> etiketlerini temizlediği için halka CSS conic-gradient ile çizilir.
    face = image_uri("portrait-lux.webp" if side == "blue" else "portrait-katarina.webp")
    return (f'<div class="xp-side {side}"><div class="xp-portrait" style="--p:{level / 18 * 100:.1f}%" aria-hidden="true">'
            f'<img class="xp-face" src="{face}" alt=""><span class="xp-badge">{level}</span></div>'
            f'<div class="xp-level"><span>{side.capitalize()} seviyesi</span><b>{level}</b></div></div>')


def cs_track(cs):
    pos = 50 + max(-1, min(1, cs / 160)) * 46
    side = "blue" if cs > 0 else "red" if cs < 0 else "even"
    return (f'<div class="cs-track {side}" aria-hidden="true"><span class="cs-line"></span><span class="cs-mid"></span>'
            f'<span class="cs-fill" style="left:{min(50, pos):.1f}%;width:{abs(pos - 50):.1f}%"></span>'
            f'<span class="cs-marker" style="left:{pos:.1f}%"><img src="{image_uri("minion.png")}" alt=""></span></div>')


CAST_SECONDS = {"lux": 1.7, "kata": 1.9}


def cast_effect(kind):
    """Tam ekran, yalnız CSS ile oynayan efekt katmanı (assets/dashboard.css: .cast-fx)."""
    if kind == "lux":
        sparks = "".join(f'<i style="left:{6 + i * 7.5}%;top:{20 + i * 37 % 60}%;animation-delay:{450 + i * 45}ms"></i>'
                         for i in range(12))
        return (f'<div class="cast-fx lux" aria-hidden="true"><div class="fx-veil"></div><div class="fx-orb"></div>'
                f'<div class="fx-beam"><div class="fx-glow"></div><div class="fx-core"></div>{sparks}</div>'
                f'<div class="fx-flash"></div></div>')
    blades = "".join(f'<i style="--r:{i * 30}deg"></i>' for i in range(12))
    petals = "".join(f'<i style="--r:{i * 30 + 15}deg"></i>' for i in range(12))
    return (f'<div class="cast-fx kata" aria-hidden="true"><div class="fx-veil"></div><div class="fx-bloom"></div>'
            f'<div class="fx-shock"></div><div class="fx-shock late"></div>'
            f'<div class="fx-lotus">{blades}</div><div class="fx-petals">{petals}</div></div>')


# Fareyi izleyen parıltı: imleç konumunu --gx/--gy olarak yazar, CSS radial-gradient çizer.
GLOW_SCRIPT = """<script>
if (!window.__lolGlow) {
  window.__lolGlow = true;
  document.addEventListener('pointermove', (e) => {
    const el = e.target.closest && e.target.closest('.st-key-cast button, .st-key-gold-row');
    document.querySelectorAll('.glow-on').forEach((x) => { if (x !== el) x.classList.remove('glow-on'); });
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--gx', `${e.clientX - r.left}px`);
    el.style.setProperty('--gy', `${e.clientY - r.top}px`);
    el.classList.add('glow-on');
  }, { passive: true });
}
</script>"""


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
    :root {{
        --score-icon: url("{image_uri("score.png")}");
        --lux-art: url("{image_uri("lux.webp")}");
        --kata-art: url("{image_uri("katarina.webp")}");
    }}
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
    live = {key: st.session_state[key] for key in INPUTS}  # anlık girdiler: görseller bunları izler
    st.html(GLOW_SCRIPT, unsafe_allow_javascript=True)
    with st.container(key="scoreboard"):
        left, mid, right = st.columns([1, 1.05, 1], vertical_alignment="center")
        with left:
            st.html(team_badge("blue"))
        with mid, st.container(key="kill-score"):
            k_blue, k_icon, k_red = st.columns([1, 0.8, 1], vertical_alignment="center", gap="small")
            with k_blue:
                st.number_input("Blue kill sayısı", 0, 16, key="killsat10", label_visibility="collapsed")
            with k_icon:
                st.html(f'<div class="kill-medal"><img src="{image_uri("score.png")}" alt="kill"></div>')
            with k_red:
                st.number_input("Red kill sayısı", 0, 16, key="deathsat10", label_visibility="collapsed")
            st.html('<p class="sb-caption">KILL · 10. DAKİKA</p>')
        with right:
            st.html(team_badge("red"))

        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", key="match-presets"):
            st.html('<span class="sb-caption">SENARYO</span>', width="content")
            for (label, values), key in zip(PRESETS.items(), ("preset-even", "preset-blue", "preset-red")):
                st.button(label, key=key, on_click=select_preset, args=(values,))
        st.html('<p class="sb-hint">Farkları Blue − Red olarak gir. Pozitif değer Blue, negatif değer Red avantajıdır.</p>')

        gold = live["golddiffat10"]
        tint = {side: f"{0.15 + min(1, abs(gold) / 7500) * 0.5:.2f}" if ahead else "0.06"
                for side, ahead in (("blue", gold > 0), ("red", gold < 0))}
        st.html(f'<style>.st-key-gold-row{{--blue-a:{tint["blue"]};--red-a:{tint["red"]}}}</style>')
        with st.container(key="gold-row"):
            g_blue, g_mid, g_red = st.columns(3, vertical_alignment="center")
            with g_blue:
                st.html(lead_html(gold, "altın", "blue"))
            with g_mid, st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", key="gold-input"):
                st.html(f'<span class="coin-medal"><img src="{image_uri("gold.png")}" alt=""></span>', width="content")
                st.number_input("Altın farkı", -7500, 7500, step=100, key="golddiffat10")
            with g_red:
                st.html(lead_html(gold, "altın", "red"))

        xp = live["xpdiffat10"]
        blue_lvl, red_lvl = xp_levels(xp)
        with st.container(key="xp-row"):
            x_blue, x_mid, x_red = st.columns([1, 0.8, 1], vertical_alignment="center")
            with x_blue:
                st.html(portrait("blue", blue_lvl))
            with x_mid:
                stat_icon("xp.png", "XP farkı", "sb-label")
                st.number_input("XP farkı", -6500, 6500, step=100, key="xpdiffat10", label_visibility="collapsed")
                st.html(lead_text(xp, "XP", "Seviyeler eşit"))
            with x_red:
                st.html(portrait("red", red_lvl))

        cs = live["csdiffat10"]
        with st.container(key="cs-row"):
            c_blue, c_mid, c_red = st.columns([1, 1.2, 1], vertical_alignment="center")
            with c_blue:
                st.html(f'<span class="lane-label{" blue" if cs > 0 else ""}">Blue koridoru</span>')
            with c_mid, st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", key="cs-input"):
                st.html(f'<span class="stat-icon sb-label"><img src="{image_uri("minion.png")}" alt="">Minyon farkı</span>',
                        width="content")
                st.number_input("Minyon farkı", -160, 160, step=1, key="csdiffat10", label_visibility="collapsed")
            with c_red:
                st.html(f'<span class="lane-label right{" red" if cs < 0 else ""}">Red koridoru</span>')
            st.html(cs_track(cs))
            st.html(lead_text(cs, "minyon", "Koridor dengede"))

        with st.container(key="match-result"):
            cast = st.session_state.pop("pending_cast", None)
            if cast:
                st.html(cast_effect(cast))
                time.sleep(CAST_SECONDS[cast])  # efekt ekrandan kalkınca sonuç çizilsin
            stats = st.session_state.match_stats
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
        with st.container(horizontal=True, vertical_alignment="center", key="cast-row"):
            st.html(f'<div class="cast-emblem" aria-hidden="true"><img src="{image_uri("score.png")}" alt=""></div>',
                    width="content")
            with st.container(key="cast"):
                st.button("Tahmin et", key="cast-button", width="stretch", on_click=cast_prediction)
            pinned = st.button(pin_label, key="pin-match")

    with st.container(key="match-extra"):
        if pinned:
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
Tahmin sekmesindeki Lux ve Katarina görselleri League of Legends kostüm çizimleridir ve hakları Riot Games'e aittir.
[Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon) · [CommunityDragon](https://communitydragon.org/docs/assets)
""")
