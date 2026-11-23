"""10. dakika deneyleri için maç girdileri ve yalnız geçmişe dayalı özellikler."""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import FEATURES, TARGET, clean, load_raw

PICKS = [f"pick{i}" for i in range(1, 6)]
STARTING_ELO = 1500.0
ELO_K = 24.0
CHAMPION_PRIOR_GAMES = 20
HISTORY_DELAY = pd.Timedelta(days=2)


def load_matches(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = load_raw(path)
    blue = raw.loc[raw.side == "Blue"].set_index("game_id", verify_integrity=True)
    red = raw.loc[raw.side == "Red"].set_index("game_id", verify_integrity=True)
    if not blue.index.difference(red.index).empty or not red.index.difference(blue.index).empty:
        raise ValueError("Her maçta bir Blue ve bir Red satırı bulunmalı.")
    red = red.reindex(blue.index)
    if (not blue[TARGET].isin([0, 1]).all() or not red[TARGET].isin([0, 1]).all()
            or not (blue[TARGET] + red[TARGET]).eq(1).all()):
        raise ValueError("Maç sonuçları ikili ve iki taraf için tutarlı olmalı.")
    matches, cleaning = clean(raw, keep_team_names=True)
    identities = blue[["date", "team_name", "opponent_team_name", TARGET, "goldat10"]].copy()
    identities["blue_picks"] = blue[PICKS].apply(tuple, axis=1)
    identities["red_picks"] = red[PICKS].apply(tuple, axis=1)
    history = identities.dropna(subset=["date"]).reset_index().sort_values(["date", "game_id"])
    matches = matches.merge(identities[["goldat10", "blue_picks", "red_picks"]], on="game_id", validate="one_to_one")
    return matches, history, cleaning


def name_key(value) -> str:
    return str(value).strip().casefold() if pd.notna(value) else ""


def draft_keys(picks) -> set[str]:
    return {name for pick in picks if (name := name_key(pick))}


@dataclass
class History:
    elo: dict = field(default_factory=dict)
    champion_games: Counter = field(default_factory=Counter)
    champion_wins: Counter = field(default_factory=Counter)

    def update(self, day: pd.DataFrame) -> None:
        """Aynı gündeki Elo değişimleri gün başındaki puanlardan hesaplanır."""
        deltas: Counter = Counter()
        for row in day.itertuples(index=False):
            blue, red = name_key(row.team_name), name_key(row.opponent_team_name)
            if blue and red and blue != red:
                difference = self.elo.get(blue, STARTING_ELO) - self.elo.get(red, STARTING_ELO)
                expected = 1 / (1 + 10 ** (-difference / 400))
                change = ELO_K * (row.result - expected)
                deltas[blue] += change
                deltas[red] -= change
            for picks, result in ((row.blue_picks, row.result), (row.red_picks, 1 - row.result)):
                for name in draft_keys(picks):
                    self.champion_games[name] += 1
                    self.champion_wins[name] += result
        for name, change in deltas.items():
            self.elo[name] = self.elo.get(name, STARTING_ELO) + change

    def champion_strength(self, picks: set[str]) -> float:
        if not picks:
            return 0.5
        return sum((self.champion_wins[name] + CHAMPION_PRIOR_GAMES / 2)
                   / (self.champion_games[name] + CHAMPION_PRIOR_GAMES) for name in sorted(picks)) / len(picks)


def match_features(stats: dict, history: History) -> dict[str, dict]:
    """Özellik ailelerini ayrı tutarak tekli deney ve kombinasyonları destekler."""
    base = {key: float(stats[key]) for key in FEATURES}
    blue, red = name_key(stats.get("team_name")), name_key(stats.get("opponent_team_name"))
    blue_elo, red_elo = history.elo.get(blue, STARTING_ELO), history.elo.get(red, STARTING_ELO)
    blue_picks = draft_keys(stats.get("blue_picks", ()))
    red_picks = draft_keys(stats.get("red_picks", ()))
    blue_strength = history.champion_strength(blue_picks)
    red_strength = history.champion_strength(red_picks)
    gold = base["golddiffat10"] / 1000
    xp = base["xpdiffat10"] / 1000
    kill_gap = base["killsat10"] - base["deathsat10"]
    gold_blue = float(stats.get("goldat10", 0))
    return {
        "base": base,
        "gold_only": {"golddiffat10": base["golddiffat10"]},
        "elo": {"elo_diff": blue_elo - red_elo, "elo_mean": (blue_elo + red_elo) / 2},
        "draft_picks": {f"pick={name}": int(name in blue_picks) - int(name in red_picks)
                        for name in sorted(blue_picks | red_picks)},
        "draft_strength": {"draft_strength_diff": blue_strength - red_strength,
                           "draft_strength_mean": (blue_strength + red_strength) / 2},
        "extra_at10": {"goldat10": gold_blue, "total_goldat10": 2 * gold_blue - base["golddiffat10"],
                       "gold_signed_square": gold * abs(gold), "xp_signed_square": xp * abs(xp),
                       "gold_xp": gold * xp, "gold_kill_gap": gold * kill_gap},
    }


def feature_rows(matches: pd.DataFrame, history_rows: pd.DataFrame) -> list[dict]:
    """Maç günü ve bir önceki günün sonuçları hiçbir geçmiş özelliğine girmez."""
    states = History()
    days = iter(history_rows.groupby(history_rows.date.dt.normalize(), sort=True))
    next_day = next(days, None)
    scheduled = matches.reset_index(drop=True)
    rows = [{} for _ in range(len(scheduled))]
    for day, same_day in scheduled.groupby(scheduled.date.dt.normalize(), sort=True):
        while next_day is not None and next_day[0] <= day - HISTORY_DELAY:
            states.update(next_day[1])
            next_day = next(days, None)
        for i, row in zip(same_day.index, same_day.to_dict("records")):
            rows[i] = match_features(row, states)
    return rows
