"""Veri yükleme ve temizleme: her maç = Blue takımının tek satırı."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "chaincc-teams-2026.csv.gz"
PROCESSED_PATH = ROOT / "data" / "processed" / "blue_at10.csv"

# Yalnızca 10. dakikada bilinen değerler (Blue bakış açısı). deathsat10 = Red'in 10. dk kill sayısı.
FEATURES = ["golddiffat10", "xpdiffat10", "csdiffat10", "killsat10", "deathsat10"]
AT10_COLS = ["goldat10", "killsat10", "deathsat10", "golddiffat10", "xpdiffat10", "csdiffat10"]
TARGET = "result"


def load_raw(path=RAW_PATH, *, patch_as_text=False) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Ham veri bulunamadı: {path}. Dosyayı data/raw/ içine koyun.")
    return pd.read_csv(path, parse_dates=["date"], dtype={"patch": "string"} if patch_as_text else None)


def clean(df: pd.DataFrame, *, keep_team_names=False) -> tuple[pd.DataFrame, dict]:
    """Blue satırlarını alır, eksik 10. dk kayıtlarını atar, tarihe göre sıralar.

    Dönen rapor her adımda kaç maç kaldığını gösterir (README/sunum için).
    """
    report = {"raw_rows": len(df), "raw_games": int(df["game_id"].nunique())}

    blue = df[df["side"] == "Blue"].drop_duplicates("game_id")
    report["blue_games"] = len(blue)

    # goldat10 == 0 imkânsız (başlangıç altını bile >0); bu satırlarda tüm 10. dk sütunları 0 -> eksik veri.
    all_zero = (blue[AT10_COLS] == 0).all(axis=1)
    report["dropped_all_zero_at10"] = int(all_zero.sum())
    blue = blue[~all_zero]

    na = blue[FEATURES + [TARGET, "date"]].isna().any(axis=1)
    report["dropped_na"] = int(na.sum())
    blue = blue[~na]

    columns = ["game_id", "date", "league", "patch"] + FEATURES + [TARGET]
    if keep_team_names:
        columns += ["team_name", "opponent_team_name"]
    out = blue[columns].copy()
    # Seri kimliği (aynı gün + aynı takım çifti) – özellik DEĞİL, yalnız sızıntı denetimi için.
    # shortcut: bo_format/match_in_series veride tamamen boş; gece yarısını aşan seriler iki seri sayılır.
    pair = blue[["team_name", "opponent_team_name"]].apply(lambda r: " vs ".join(sorted(r.astype(str))), axis=1)
    out["series"] = blue["date"].dt.strftime("%Y-%m-%d") + " " + pair
    out[TARGET] = out[TARGET].astype(int)
    # Aynı zaman damgalı maçlar var (8): game_id ikincil anahtar -> deterministik sıra.
    out = out.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    report["final_games"] = len(out)
    report["blue_win_rate"] = round(float(out[TARGET].mean()), 4)
    return out, report


def chrono_split(df: pd.DataFrame, test_size=0.2):
    """Kronolojik bölme: en yeni ~%20 maç test. Aynı zaman damgalı maçlar aynı tarafta kalır."""
    cut_date = df["date"].sort_values().iloc[int(len(df) * (1 - test_size))]
    return df[df["date"] < cut_date], df[df["date"] >= cut_date]


if __name__ == "__main__":
    data, rep = clean(load_raw())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    print(rep)
