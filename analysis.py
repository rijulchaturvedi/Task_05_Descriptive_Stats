"""analysis.py

Reusable helpers that solve the **10 intermediate-to-hard Formula‑1 questions**
posed in our Task 05 notebook. Call them from `Task_05_Descriptive_Stats.ipynb`:

```python
import analysis as f1
f1.gain_positions(2024)
```

Each helper loads raw Ergast CSVs from *Data/* and returns a tidy
`pandas` object.  Functions now rely on **`results.csv`** (rather than
`driver_standings.csv`) when a driver‑to‑constructor mapping is needed,
because `results.csv` always carries a `constructorId` column.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

DATA_DIR = Path(__file__).resolve().parent / "Data"

def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", **kw)

def _season_race_ids(season: int) -> pd.Series:
    return _csv("races", usecols=["raceId", "year"]).query("year == @season").raceId

# ──────────────────────────────────────────────────────────────────────────────
# Q1  Grid‑gain champions
# ──────────────────────────────────────────────────────────────────────────────

def gain_positions(season: int = 2024, top_n: int = 5) -> pd.DataFrame:
    rids   = _season_race_ids(season)
    quali  = (_csv("qualifying", usecols=["raceId","driverId","position"])
              .query("raceId in @rids")
              .groupby(["raceId","driverId"], as_index=False).position.min()
              .rename(columns={"position":"grid"}))
    finish = (_csv("driver_standings", usecols=["raceId","driverId","position"])
              .query("raceId in @rids")
              .rename(columns={"position":"finish"}))
    df = quali.merge(finish,on=["raceId","driverId"]).dropna()
    df["gain"] = df.grid - df.finish
    avg = df.groupby("driverId",as_index=False).gain.mean().sort_values("gain",ascending=False)
    return avg.merge(_csv("drivers",usecols=["driverId","code","forename","surname"]),on="driverId").head(top_n)

# ──────────────────────────────────────────────────────────────────────────────
# Q2  Pit‑stop consistency per constructor
# ──────────────────────────────────────────────────────────────────────────────

def consistent_pit_stop_team(season: int = 2024) -> pd.DataFrame:
    rids = _season_race_ids(season)
    pit  = (_csv("pit_stops", usecols=["raceId","driverId","duration"])
            .query("raceId in @rids"))
    pit["duration"] = pd.to_numeric(pit.duration, errors="coerce")
    # keep only realistic service stops (< 60 s); removes drive‑throughs & long repairs
    pit = pit.dropna(subset=["duration"]).query("duration < 60")

    # Map driver → constructor using *results.csv* for the **same season**
    mapping = (_csv("results", usecols=["raceId", "driverId", "constructorId"])
               .query("raceId in @rids")
               .drop_duplicates("driverId"))
    # Ensure same dtype for merge safety
    pit["driverId"] = pit.driverId.astype(int)
    mapping["driverId"] = mapping.driverId.astype(int)
    pit = pit.merge(mapping[["driverId", "constructorId"]], on="driverId", how="left")
    pit = pit.dropna(subset=["constructorId"])
    agg = pit.groupby("constructorId",as_index=False).duration.std().rename(columns={"duration":"std_sec"})
    return (agg.merge(_csv("constructors",usecols=["constructorId","name"]),on="constructorId")
               .sort_values("std_sec"))

# ──────────────────────────────────────────────────────────────────────────────
# Q3  Circuit with highest overtake rate  (same as before)
# ──────────────────────────────────────────────────────────────────────────────

def circuit_highest_overtake_rate(season: int = 2024) -> pd.DataFrame:
    warnings.warn("Overtake‑rate calc is slow; consider subsampling lap_times.")
    rids = _season_race_ids(season)
    top10 = (_csv("driver_standings", usecols=["raceId","driverId","position"])
             .query("raceId in @rids")
             .sort_values(["raceId","position"])   # top‑10 finishers
             .groupby("raceId").head(10))
    laps = _csv("lap_times").query("raceId in @rids and driverId in @top10.driverId")
    laps = laps.sort_values(["raceId","lap","position"])
    laps["prev"] = laps.groupby(["raceId","driverId"]).position.shift(1)
    laps["overtake"] = (laps.prev.notna() & (laps.position < laps.prev)).astype(int)
    per_gp = (laps.groupby("raceId")["overtake"].sum().to_frame("ovtk")
                    .join(laps.groupby("raceId").lap.max().to_frame("laps")))
    per_gp["rate"] = per_gp.ovtk / per_gp.laps
    out = per_gp.reset_index().merge(_csv("races",usecols=["raceId","name","circuitId"]),on="raceId")
    return (out.merge(_csv("circuits",usecols=["circuitId","name"]).rename(columns={"name":"circuit"}),on="circuitId")
              .sort_values("rate",ascending=False)[["name","circuit","rate"]])

# ──────────────────────────────────────────────────────────────────────────────
# Q4  Biggest single‑lap improvement  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def biggest_single_lap_improvement(season: int = 2024) -> pd.Series:
    """
    Return the cleanest, largest lap‑to‑lap time gain for *season*.

    Rules
    -----
    • improvement_ms = previous_lap_time – current_lap_time  (positive = faster)  
    • Compare laps **within the same race**.  
    • Exclude the lap on which a pit‑stop occurs *and* the following out‑lap.  
    • Discard any ‘gains’ ≥ 6000 ms (likely SC / red‑flag artefacts).  
    """
    rids = _season_race_ids(season)

    # ------ load lap times & pit‑stop info -----------------------------------
    laps = (_csv("lap_times")
            .query("raceId in @rids")
            .sort_values(["driverId", "raceId", "lap"]))

    pits = (_csv("pit_stops", usecols=["raceId", "driverId", "lap"])
            .query("raceId in @rids"))

    # Build a set of (raceId, driverId, lap) to exclude: pit‑in & out‑lap
    bad_laps = pd.concat([
        pits[["raceId", "driverId", "lap"]],
        pits.assign(lap=pits.lap + 1)[["raceId", "driverId", "lap"]]
    ])
    bad_set = set(map(tuple, bad_laps.values))

    # ------ compute lap‑to‑lap gain ------------------------------------------
    laps["prev_ms"] = laps.groupby(["driverId", "raceId"]).milliseconds.shift(1)
    laps["gain"] = laps.prev_ms - laps.milliseconds

    gains = (laps.dropna(subset=["gain"])
                 .query("gain > 0 and gain < 3000"))          # 0 – 3 s gains only

    # discard if previous lap was >30 % slower than that driver's median lap in the race
    median = laps.groupby(["driverId", "raceId"]).milliseconds.transform("median")
    gains = gains[gains.prev_ms / median[gains.index] < 1.30]

    # remove pit‑in / out‑laps
    # boolean mask: True if this (raceId, driverId, lap) is a pit‑in/out lap
    mask_bad = gains.apply(lambda r: (r.raceId, r.driverId, r.lap) in bad_set, axis=1)
    gains = gains[~mask_bad]

    if gains.empty:
        raise ValueError(f"No clean lap‑to‑lap improvements found for {season}")

    best = gains.loc[gains.gain.idxmax()]

    # ------ pretty output -----------------------------------------------------
    drv = _csv("drivers").set_index("driverId").loc[int(best.driverId)]
    gp_name = (_csv("races", usecols=["raceId", "name"])
               .set_index("raceId")
               .loc[int(best.raceId), "name"])

    return pd.Series({
        "driver":        f"{drv.forename} {drv.surname}",
        "grand_prix":    gp_name,
        "lap":           int(best.lap),
        "improvement_ms": int(best.gain)
    })



# -----------------------------------------------------------------------------
# Q5 – Largest points swing between consecutive rounds
# -----------------------------------------------------------------------------

def largest_points_swing(season: int = 2024) -> pd.Series:
    race_ids = _season_race_ids(season)

    # driver standings for the season
    ds = _csv("driver_standings", usecols=["driverId", "raceId", "points"]) \
         .query("raceId in @race_ids")

    # race lookup with round number & GP name
    races = _csv("races", usecols=["raceId", "round", "name"]) \
            .query("raceId in @race_ids")

    # merge to attach round & name
    ds = ds.merge(races, on="raceId", how="left")

    # order by driver & round
    ds = ds.sort_values(["driverId", "round"])

    # delta to previous race
    ds["prev_points"] = ds.groupby("driverId")["points"].shift(1)
    ds["swing"] = ds.points - ds.prev_points

    # drop first race for each driver (prev_points = NaN)
    ds = ds.dropna(subset=["swing"])

    if ds.empty:
        raise ValueError(f"No consecutive race points found for season {season}")

    best = ds.loc[ds.swing.abs().idxmax()]

    driver = _csv("drivers").set_index("driverId").loc[int(best.driverId)]
    prev_round = best["round"] - 1
    gp_prev_name = races.loc[races["round"] == prev_round, "name"].iloc[0]

    return pd.Series({
        "driver": f"{driver.forename} {driver.surname}",
        "from_gp": gp_prev_name,
        "to_gp": best["name"],         # current GP name
        "points_swing": int(best.swing)
    })


# -----------------------------------------------------------------------------
# Q6 – Constructor efficiency (points per pit-stop second)
# -----------------------------------------------------------------------------

def constructor_efficiency(season: int = 2024) -> pd.DataFrame:
    race_ids = _season_race_ids(season)

    pts = (
        _csv("constructor_standings", usecols=["constructorId", "raceId", "points"])
        .query("raceId in @race_ids")
        .groupby("constructorId", as_index=False)["points"].max()  # end-of-season tally
    )

    pit = _csv("pit_stops", usecols=["raceId", "driverId", "duration"])
    pit = pit.query("raceId in @race_ids")
    pit["duration"] = pd.to_numeric(pit.duration, errors="coerce")
    pit = pit.dropna(subset=["duration"])

    # Map drivers → constructors via results.csv
    results = (_csv("results", usecols=["driverId", "constructorId"])
               .drop_duplicates("driverId"))
    pit = pit.merge(results, on="driverId")
    pit_sec = pit.groupby("constructorId", as_index=False)["duration"].sum()

    df = pts.merge(pit_sec, on="constructorId")
    df["pts_per_sec"] = df.points / df.duration

    constructors = _csv("constructors", usecols=["constructorId", "name"])
    return (
        df.merge(constructors, on="constructorId")
        .sort_values("pts_per_sec", ascending=False)
    )


# -----------------------------------------------------------------------------
# Q7 – Simple linear regression for pit-stop duration
# -----------------------------------------------------------------------------

def pit_stop_duration_model(season: int = 2024, random_state: int | None = 0):
    """Return (R², fitted model) for a one-hot-encoded linear regression."""
    race_ids = _season_race_ids(season)
    pit = _csv("pit_stops", usecols=["raceId", "driverId", "lap", "stop", "duration"])
    pit = pit.query("raceId in @race_ids")
    pit["duration"] = pd.to_numeric(pit.duration, errors="coerce")
    pit = pit.dropna(subset=["duration"])

    # Map driver → constructor
    dr_const = (_csv("results", usecols=["driverId", "constructorId"])
                .drop_duplicates("driverId"))
    pit = pit.merge(dr_const, on="driverId")

    # ------------------------------------------------------------------
    X = pit[["lap", "stop", "constructorId"]]
    y = pit["duration"]

    numeric_features = ["lap", "stop"]
    categorical = ["constructorId"]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = Pipeline([
        ("pre", pre),
        ("lin", LinearRegression()),
    ])

    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))  # simple in-sample metric
    return r2, model


# -----------------------------------------------------------------------------
# Q8 – Fast-lap specialist
# -----------------------------------------------------------------------------

def fast_lap_specialist(start_year: int = 2020, end_year: int = 2024, top_n: int = 5):
    races = _csv("races", usecols=["raceId", "year"])
    race_ids = races.loc[races.year.between(start_year, end_year), "raceId"]

    laps = _csv("lap_times")
    laps = laps[laps.raceId.isin(race_ids)]

    # Fastest lap per race
    idx = laps.groupby("raceId")["milliseconds"].idxmin()
    winners = laps.loc[idx, ["raceId", "driverId"]]

    # Count awards & entries
    awards = winners.groupby("driverId").size()
    entries = laps.groupby("driverId").raceId.nunique()
    rate = (awards / entries).sort_values(ascending=False)

    drivers = _csv("drivers")[["driverId", "code", "forename", "surname"]].set_index("driverId")
    df = drivers.join(rate.rename("fastlap_rate"), how="inner").dropna()
    return df.sort_values("fastlap_rate", ascending=False).head(top_n)


# -----------------------------------------------------------------------------
# Q9 – Team-mate qualifying duel
# -----------------------------------------------------------------------------

def teammate_qualifying_duel(season: int = 2024) -> pd.DataFrame:
    """
    For every constructor in *season*, compute the average absolute grid‑position
    advantage between its two drivers (qualifying). Positive abs_diff means one
    driver consistently out‑qualifies their team‑mate.

    Implementation notes:
    • qualifying.csv does NOT include constructorId in some Ergast dumps, so we
      map driver → constructor via results.csv for each race.
    • Use the best session time (min position) to represent grid slot.
    """
    race_ids = _season_race_ids(season)

    # Best qualifying position per driver per race
    quali = (_csv("qualifying", usecols=["raceId", "driverId", "position"])
             .query("raceId in @race_ids")
             .groupby(["raceId", "driverId"], as_index=False)["position"].min())

    # Map driver → constructor (results.csv is reliable)
    drv_const = (_csv("results", usecols=["raceId", "driverId", "constructorId"])
                 .query("raceId in @race_ids"))

    quali = quali.merge(drv_const, on=["raceId", "driverId"], how="left").dropna(subset=["constructorId"])

    # Pairwise diff within constructor per race
    def _pairwise(df):
        if len(df) != 2:
            return None  # skip if not exactly two drivers classified
        return abs(df.iloc[0].position - df.iloc[1].position)  # absolute gap

    duels = (quali.groupby(["raceId", "constructorId"])
                   .apply(_pairwise)
                   .dropna()
                   .reset_index(name="abs_diff"))

    mean_adv = duels.groupby("constructorId", as_index=False)["abs_diff"].mean()

    constructors = _csv("constructors", usecols=["constructorId", "name"])
    return (mean_adv.merge(constructors, on="constructorId")
                    .sort_values("abs_diff", ascending=False))


# -----------------------------------------------------------------------------
# Q10 – YoY constructor improvement
# -----------------------------------------------------------------------------

def constructor_season_improvement(from_year: int = 2023, to_year: int = 2024):
    def _season_points(year):
        race_ids = _season_race_ids(year)
        pts = (
            _csv("constructor_standings", usecols=["constructorId", "raceId", "points"])
            .query("raceId in @race_ids")
            .groupby("constructorId", as_index=False)["points"].max()
        )
        races = _season_race_ids(year).nunique()
        pts["pts_per_race"] = pts.points / races
        return pts[["constructorId", "pts_per_race"]]

    a = _season_points(from_year).rename(columns={"pts_per_race": "prev"})
    b = _season_points(to_year).rename(columns={"pts_per_race": "curr"})
    df = a.merge(b, on="constructorId", how="outer").fillna(0)
    df["abs_change"] = df.curr - df.prev
    df["pct_change"] = np.where(df.prev == 0, np.nan, df.abs_change / df.prev)

    constructors = _csv("constructors", usecols=["constructorId", "name"])
    return (
        df.merge(constructors, on="constructorId")
        .sort_values("abs_change", ascending=False)
    )

