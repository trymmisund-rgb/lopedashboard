"""Analyse av løpedata.

All logikk samlet ett sted, så den kan brukes av:
  - notebookene (utforsking)
  - scripts/lag_stats.py (lager stats.json til ditt eget dashboard)
  - et API senere (der andre laster opp sin egen CSV)

Hovedfunksjonene er:
    df = les_garmin("data/Activities_løping.csv")
    stats = analyser(df)   # -> dict som kan lagres som JSON

    # Med brukerens egne opplysninger (alt valgfritt):
    stats = analyser(df, Profil(niva="intermediate", makspuls=195))
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Innlesing
# ---------------------------------------------------------------------------

# Garmin-kolonnenavn (norsk og engelsk eksport) -> våre egne navn.
# De engelske navnene er ikke testet på en ekte fil ennå.
KOLONNER = {
    "dato": ["Dato", "Date"],
    "type": ["Aktivitetstype", "Activity Type"],
    "tittel": ["Tittel", "Title"],
    "km": ["Distanse", "Distance"],
    "tid": ["Tid", "Time"],
    "snittpuls": ["Gjennomsnittlig puls", "Avg HR"],
    "makspuls": ["Makspuls", "Max HR"],
    "stigning_m": ["Total stigning", "Total Ascent"],
    "kadens": ["Gjennomsnittlig frekvens for løping", "Avg Run Cadence"],
    "runder": ["Antall runder", "Number of Laps"],
    "gap": ["Gjennomsnittlig GAP", "Avg GAP"],
}

PAAKREVD = ["dato", "km", "tid"]

LOPETYPER = ["løping", "løping på tredemølle", "running", "treadmill running", "trail running"]


class UgyldigFil(ValueError):
    """Filen ser ikke ut som en Garmin-eksport vi forstår."""


def tid_til_sek(s) -> float | None:
    """'01:01:12' eller '6:34' -> sekunder."""
    if s is None or (isinstance(s, float) and math.isnan(s)) or s == "":
        return None
    sek = 0.0
    try:
        for d in str(s).split(":"):
            sek = sek * 60 + float(d)
    except ValueError:
        return None
    return sek


def sek_til_tempo(sek: float | None) -> str | None:
    """394 -> '6:34'."""
    if sek is None or (isinstance(sek, float) and math.isnan(sek)):
        return None
    return f"{int(sek // 60)}:{int(sek % 60):02d}"


def _finn_kolonne(raw: pd.DataFrame, navn: str) -> pd.Series | None:
    for kandidat in KOLONNER[navn]:
        if kandidat in raw.columns:
            return raw[kandidat]
    return None


def les_garmin(fil: str | IO) -> pd.DataFrame:
    """Leser en Garmin Connect CSV-eksport og returnerer en ryddig tabell med løpeturer.

    `fil` kan være en filsti eller et filobjekt (f.eks. en opplastet fil i et API).
    Andre aktiviteter enn løping (styrke, fotball …) filtreres bort.
    """
    try:
        raw = pd.read_csv(fil, thousands=",", na_values=["--"], on_bad_lines="skip")
    except Exception as e:  # noqa: BLE001
        raise UgyldigFil(f"Klarte ikke å lese filen som CSV: {e}") from e

    kol = {navn: _finn_kolonne(raw, navn) for navn in KOLONNER}
    mangler = [n for n in PAAKREVD if kol[n] is None]
    if mangler:
        raise UgyldigFil(f"Fant ikke kolonnene {mangler}. Er dette en CSV fra Garmin Connect?")

    def tall(navn):
        s = kol[navn]
        return pd.to_numeric(s, errors="coerce") if s is not None else pd.Series(float("nan"), index=raw.index)

    df = pd.DataFrame({
        "dato": pd.to_datetime(kol["dato"], errors="coerce"),
        "type": kol["type"] if kol["type"] is not None else "Løping",
        "tittel": kol["tittel"] if kol["tittel"] is not None else "",
        "km": tall("km"),
        "tid_s": kol["tid"].map(tid_til_sek),
        "snittpuls": tall("snittpuls"),
        "makspuls": tall("makspuls"),
        "stigning_m": tall("stigning_m"),
        "kadens": tall("kadens"),
        "runder": tall("runder"),
        "gap_s": kol["gap"].map(tid_til_sek) if kol["gap"] is not None else float("nan"),
    })

    # Bare løping
    df = df[df["type"].astype(str).str.lower().isin(LOPETYPER)]
    df = df.dropna(subset=["dato", "km", "tid_s"])
    df = df[df["km"] > 0]
    if df.empty:
        raise UgyldigFil("Fant ingen løpeturer i filen.")

    df["tempo_s"] = df["tid_s"] / df["km"]
    # GAP (bakkejustert tempo) mangler på tredemølle – der er det flatt, så vanlig tempo duger
    df["gap_s"] = pd.to_numeric(df["gap_s"], errors="coerce").fillna(df["tempo_s"])
    # Formmål: meter per hjerteslag, med bakkejustert tempo. Høyere = bedre form.
    df["meter_per_slag"] = (1000 / df["gap_s"] * 60) / df["snittpuls"]

    return df.sort_values("dato").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Perioder og pauser
# ---------------------------------------------------------------------------

def legg_til_perioder(df: pd.DataFrame, pause_uker: int = 4) -> pd.DataFrame:
    """Ny periode starter når det har gått mer enn `pause_uker` siden forrige tur."""
    df = df.copy()
    df["dager_siden_forrige"] = df["dato"].diff().dt.days
    df["periode"] = (df["dato"].diff() > pd.Timedelta(weeks=pause_uker)).cumsum()
    return df


# ---------------------------------------------------------------------------
# 3. Økttyper
# ---------------------------------------------------------------------------

TYPER = ["rolig", "lang", "tempo", "hard"]

# Regulære uttrykk som leter etter økttypen i tittelen (engelsk og norsk).
# Sjekkes i rekkefølge – første treff vinner. Derfor kommer tydelige intervall-ord
# først, så tempo/terskel (så "Terskel 3x10" blir tempo), og så det generelle
# "4x4"-mønsteret. \b betyr ordgrense, så "lang" ikke treffer i "Langrenn".
MONSTRE = [
    ("hard", [
        r"interval", r"repeats?", r"pyramid", r"time trial", r"drop set", r"broken miles",
        r"\bhills?\b", r"fartlek", r"vo2", r"sprint", r"\brace\b", r"konkurranse",
        r"intervall", r"\bdrag\b", r"bakkeløp",
    ]),
    ("tempo", [r"tempo", r"threshold", r"terskel", r"\bmp\b", r"marathon pace"]),
    ("hard", [r"\b\d+\s*[x×]\s*\d+"]),  # f.eks. "4x4", "6 x 1000", "10×400"
    ("lang", [r"\blong\b", r"langtur", r"langkjøring", r"\blang tur\b", r"\blsd\b"]),
    ("rolig", [r"\beasy\b", r"recovery", r"\brolig", r"restitusjon", r"\bjogg", r"\blett\b", r"\bbase\b"]),
]
_MONSTRE_KOMPILERT = [(t, [re.compile(m) for m in ms]) for t, ms in MONSTRE]


def okttype_fra_tittel(tittel) -> str | None:
    t = str(tittel).lower()
    for okttype, monstre in _MONSTRE_KOMPILERT:
        if any(m.search(t) for m in monstre):
            return okttype
    return None


# --- Løperprofil: nivå, makspuls og ukesnitt ---------------------------------
# Det som er "langt" eller "hardt" avhenger av hvem som løper. Profilen fylles ut
# av brukeren (valgfritt) og resten regnes ut fra dataene.

# Grove tommelfingerregler – juster gjerne. Brukes til å foreslå nivå fra ukesnittet,
# og som standardverdier når en ny bruker har for lite data.
NIVAER = {
    #               fra km/uke   typisk tur (km)   typisk ukesnitt (km)
    "beginner":     {"fra_km_uke": 0,  "typisk_tur_km": 4,  "ukesnitt_km": 10},
    "amateur":      {"fra_km_uke": 15, "typisk_tur_km": 7,  "ukesnitt_km": 22},
    "intermediate": {"fra_km_uke": 30, "typisk_tur_km": 10, "ukesnitt_km": 40},
    "experienced":  {"fra_km_uke": 50, "typisk_tur_km": 13, "ukesnitt_km": 60},
}
MIN_TURER_FOR_DATA = 8  # færre gyldige turer enn dette -> bruk nivåets standardverdier


@dataclass
class Profil:
    """Det brukeren kan oppgi selv. Alt er valgfritt – det som mangler regnes ut fra dataene."""
    niva: str | None = None          # "beginner" | "amateur" | "intermediate" | "experienced"
    makspuls: float | None = None    # bedre enn å gjette fra dataene, hvis brukeren vet den
    ukesnitt_km: float | None = None


@dataclass
class BeregnetProfil:
    """Ferdig utfylt profil som resten av analysen bruker. `kilder` viser hvor hver verdi kom fra."""
    niva: str
    makspuls: float
    ukesnitt_km: float
    typisk_tur_km: float
    typisk_tempo_s: float
    foreslatt_niva: str              # det dataene tyder på, uansett hva brukeren valgte
    kilder: dict = field(default_factory=dict)


def _gyldige(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["km"] >= 2) & (df["tempo_s"] <= 9 * 60)]


def niva_fra_ukesnitt(km_uke: float) -> str:
    niva = "beginner"
    for navn, n in NIVAER.items():
        if km_uke >= n["fra_km_uke"]:
            niva = navn
    return niva


def beregn_ukesnitt(df: pd.DataFrame, siste_uker: int = 12) -> float | None:
    """Snitt km per uke over de siste `siste_uker` *aktive* ukene (pauser teller ikke)."""
    g = _gyldige(df)
    if g.empty:
        return None
    uker = g.set_index("dato")["km"].resample("W-MON").sum()
    aktive = uker[uker > 0]
    return float(aktive.tail(siste_uker).mean()) if not aktive.empty else None


def lag_profil(df: pd.DataFrame, profil: Profil | None = None) -> BeregnetProfil:
    """Fyller ut profilen: brukerens verdier først, så dataene, så nivåets standardverdier."""
    profil = profil or Profil()
    if profil.niva is not None and profil.niva not in NIVAER:
        raise ValueError(f"Ukjent nivå '{profil.niva}'. Velg en av {list(NIVAER)}")
    g = _gyldige(df)
    nok_data = len(g) >= MIN_TURER_FOR_DATA
    kilder = {}

    # Ukesnitt og nivå
    ukesnitt_data = beregn_ukesnitt(df)
    foreslatt = niva_fra_ukesnitt(ukesnitt_data) if ukesnitt_data else "beginner"
    niva = profil.niva or foreslatt
    kilder["niva"] = "bruker" if profil.niva else "data"

    if profil.ukesnitt_km:
        ukesnitt, kilder["ukesnitt_km"] = profil.ukesnitt_km, "bruker"
    elif nok_data and ukesnitt_data:
        ukesnitt, kilder["ukesnitt_km"] = ukesnitt_data, "data"
    else:
        ukesnitt, kilder["ukesnitt_km"] = NIVAER[niva]["ukesnitt_km"], "nivå"

    # Typisk turlengde og tempo
    if nok_data:
        typisk_km, kilder["typisk_tur_km"] = float(g["km"].median()), "data"
    else:
        typisk_km, kilder["typisk_tur_km"] = NIVAER[niva]["typisk_tur_km"], "nivå"
    basis = g if not g.empty else df
    typisk_tempo = float(basis["gap_s"].median())

    # Makspuls: brukerens verdi er best. Den høyeste i dataene er ofte for lav
    # (man er sjelden helt på maks), men det er det beste vi har.
    if profil.makspuls:
        makspuls, kilder["makspuls"] = float(profil.makspuls), "bruker"
    else:
        makspuls, kilder["makspuls"] = float(df["makspuls"].max()), "data"

    return BeregnetProfil(niva=niva, makspuls=makspuls, ukesnitt_km=round(ukesnitt, 1),
                          typisk_tur_km=round(typisk_km, 1), typisk_tempo_s=typisk_tempo,
                          foreslatt_niva=foreslatt, kilder=kilder)


# --- Features til maskinlæringsmodellen -------------------------------------
# Alt er *relativt* til løperens egen profil, så en modell trent på én person
# kan brukes på andre (som har annen makspuls, annet tempo og annet volum).

FEATURES = [
    "km_rel",         # turlengde / løperens typiske turlengde
    "andel_av_uke",   # turlengde / løperens ukesnitt – stor andel = langtur (avhenger av antall turer i uka)
    "puls_rel",       # snittpuls / makspuls
    "makspuls_rel",   # turens makspuls / løperens makspuls
    "puls_spenn",     # (makspuls - snittpuls) / makspuls – stor for intervaller
    "tempo_rel",      # bakkejustert tempo / løperens typiske tempo (<1 = raskere enn vanlig)
    "runder_per_km",  # mange korte runder = intervaller
]
# Uten runder: for løpere som bare har automatiske runder per km. Da sier rundene
# ingenting om økta, og modellen må klare seg med puls, tempo og distanse.
FEATURES_UTEN_RUNDER = [f for f in FEATURES if f != "runder_per_km"]


def runder_er_informative(X: pd.DataFrame) -> bool:
    """Har løperen runder som følger øktstrukturen (Runna, Garmin-økter, rundeknapp)?

    Med bare auto-runder per km ligger runder_per_km rett rundt 1 på alle turer.
    Har minst 10 % av turene 1,3 runder per km eller mer, regner vi rundene som nyttige.
    """
    r = X["runder_per_km"].dropna()
    return len(r) > 0 and (r >= 1.3).mean() >= 0.10


def lag_features(df: pd.DataFrame, profil: BeregnetProfil | None = None) -> pd.DataFrame:
    """Lager modell-features relativt til løperprofilen."""
    p = profil or lag_profil(df)
    X = pd.DataFrame(index=df.index)
    X["km_rel"] = df["km"] / p.typisk_tur_km
    X["andel_av_uke"] = df["km"] / p.ukesnitt_km
    X["puls_rel"] = df["snittpuls"] / p.makspuls
    X["makspuls_rel"] = df["makspuls"] / p.makspuls
    X["puls_spenn"] = (df["makspuls"] - df["snittpuls"]) / p.makspuls
    X["tempo_rel"] = df["gap_s"] / p.typisk_tempo_s
    X["runder_per_km"] = df["runder"] / df["km"]
    return X[FEATURES]


MODELL_STI = Path(__file__).parent / "okttype_modell.joblib"
_modell_cache = None


def last_modell() -> dict | None:
    """Laster de trente modellene hvis de finnes (lages av scripts/tren_modell.py).

    Returnerer {"med_runder": modell, "uten_runder": modell}.
    """
    global _modell_cache
    if _modell_cache is None and MODELL_STI.exists():
        import joblib
        _modell_cache = joblib.load(MODELL_STI)
    return _modell_cache


def _gjett_med_regler(df: pd.DataFrame, p: BeregnetProfil) -> pd.Series:
    """Reserveløsning hvis modellen ikke finnes. Alle grenser er relative til løperprofilen."""
    rolig_grense = 0.80 * p.makspuls

    def gjett(rad) -> str:
        # Lang: godt over løperens vanlige tur OG en stor del av ukevolumet.
        # Vi bruker den største grensen: med 3 turer i uka er *hver* tur ~30 % av uka,
        # og med 6 turer i uka kan en vanlig tur være kort sammenlignet med langturen.
        lang_grense = max(1.4 * p.typisk_tur_km, 0.35 * p.ukesnitt_km)
        if rad["km"] >= lang_grense:
            return "lang"
        if pd.notna(rad["runder"]) and rad["runder"] / rad["km"] >= 1.8:
            return "hard"
        if pd.notna(rad["snittpuls"]) and rad["snittpuls"] <= rolig_grense:
            return "rolig"
        return "tempo"

    return df.apply(gjett, axis=1)


def klassifiser(df: pd.DataFrame, profil: Profil | BeregnetProfil | None = None,
                bruk_modell: bool = True) -> pd.DataFrame:
    """Setter `okttype` (rolig/lang/tempo/hard) og `kilde`.

    1. Står typen i tittelen ("Easy Run", "4x4 intervall", "langtur" …) brukes den  -> kilde="tittel"
    2. Ellers predikerer ML-modellen typen                                          -> kilde="modell"
       (modellen med runder hvis løperens runder følger øktene, ellers den uten)
    3. Finnes ingen modell (eller pulsdata mangler), brukes enkle regler            -> kilde="regel"
    """
    p = profil if isinstance(profil, BeregnetProfil) else lag_profil(df, profil)
    df = df.copy()
    fra_tittel = df["tittel"].map(okttype_fra_tittel)
    df["okttype"] = fra_tittel
    df["kilde"] = None
    df.loc[fra_tittel.notna(), "kilde"] = "tittel"

    mangler = fra_tittel.isna()
    if mangler.any():
        regler = _gjett_med_regler(df, p)
        df.loc[mangler, "okttype"] = regler[mangler]
        df.loc[mangler, "kilde"] = "regel"

        modeller = last_modell() if bruk_modell else None
        if modeller is not None:
            X = lag_features(df, p)
            if runder_er_informative(X):
                modell, features = modeller["med_runder"], FEATURES
            else:
                modell, features = modeller["uten_runder"], FEATURES_UTEN_RUNDER
            kan_predikeres = mangler & X[features].notna().all(axis=1)
            if kan_predikeres.any():
                df.loc[kan_predikeres, "okttype"] = modell.predict(X.loc[kan_predikeres, features])
                df.loc[kan_predikeres, "kilde"] = "modell"

    # Småturer (oppvarming, gåturer) tas ikke med i analysen
    df["gyldig"] = (df["km"] >= 2) & (df["tempo_s"] <= 9 * 60)
    return df


# ---------------------------------------------------------------------------
# 4. Samlet analyse -> JSON-vennlig dict
# ---------------------------------------------------------------------------

def _rens(v):
    """Gjør om NaN/numpy-typer til vanlige Python-verdier som kan lagres som JSON."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"):  # numpy-tall
        v = v.item()
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 3)
    return v


def _poster(df: pd.DataFrame) -> list[dict]:
    return [{k: _rens(v) for k, v in rad.items()} for rad in df.to_dict(orient="records")]


def analyser(df: pd.DataFrame, profil: Profil | None = None, pause_uker: int = 4) -> dict:
    """Kjører hele analysen og returnerer alt nettsiden trenger.

    `profil` er det brukeren har oppgitt (nivå, makspuls, ukesnitt) – alt valgfritt.
    Inneholder bevisst ikke titler eller klokkeslett (personvern) – bare datoer og tall.
    """
    p = lag_profil(df, profil)
    df = klassifiser(legg_til_perioder(df, pause_uker), p)
    g = df[df["gyldig"]]

    # Nøkkeltall
    lengste = g.loc[g["km"].idxmax()]
    over5 = g[g["km"] >= 5]
    raskest = over5.loc[over5["tempo_s"].idxmin()] if not over5.empty else None
    nokkeltall = {
        "turer": len(g),
        "km": round(g["km"].sum(), 1),
        "timer": round(g["tid_s"].sum() / 3600, 1),
        "forste_tur": _rens(g["dato"].min()),
        "siste_tur": _rens(g["dato"].max()),
        "lengste_tur": {"dato": _rens(lengste["dato"]), "km": _rens(lengste["km"])},
        "raskeste_5km_pluss": None if raskest is None else {
            "dato": _rens(raskest["dato"]), "km": _rens(raskest["km"]),
            "tempo": sek_til_tempo(raskest["tempo_s"]),
        },
    }

    # Kilometer per uke (uker uten løping = 0, så pausene syns)
    uker = g.set_index("dato")["km"].resample("W-MON").sum()
    uker_liste = [{"uke": d.strftime("%Y-%m-%d"), "km": round(km, 1)} for d, km in uker.items()]

    # Kilometer per måned fordelt på økttype
    mnd = (g.groupby([g["dato"].dt.to_period("M"), "okttype"])["km"].sum()
             .unstack(fill_value=0).reindex(columns=TYPER, fill_value=0))
    mnd = mnd.reindex(pd.period_range(mnd.index.min(), mnd.index.max(), freq="M"), fill_value=0)
    maaneder = [{"maaned": str(p), **{t: round(float(r[t]), 1) for t in TYPER}} for p, r in mnd.iterrows()]

    andel = (g.groupby("okttype")["km"].sum().reindex(TYPER, fill_value=0) / g["km"].sum() * 100).round(0)

    # Perioder
    perioder = g.groupby("periode").agg(
        start=("dato", "min"), slutt=("dato", "max"), turer=("km", "size"),
        km=("km", "sum"), snittpuls=("snittpuls", "mean"), meter_per_slag=("meter_per_slag", "mean"),
    ).reset_index()

    # Form: glidende snitt på rolige og lange turer, innenfor hver periode
    rolige = g[g["okttype"].isin(["rolig", "lang"])].copy()
    rolige["form_snitt"] = (rolige.groupby("periode")["meter_per_slag"]
                                  .transform(lambda s: s.rolling(4, min_periods=2).mean()))
    beste_form = rolige["form_snitt"].max()
    na_form = rolige["form_snitt"].iloc[-1] if not rolige.empty else None

    turer = g[["dato", "km", "tempo_s", "gap_s", "snittpuls", "makspuls",
               "okttype", "kilde", "periode", "meter_per_slag"]]

    loperprofil = asdict(p)
    loperprofil["typisk_tempo"] = sek_til_tempo(loperprofil.pop("typisk_tempo_s"))

    return {
        "loperprofil": loperprofil,
        "nokkeltall": nokkeltall,
        "form": {
            "naa": _rens(na_form),
            "beste": _rens(beste_form),
            "prosent_av_beste": None if not beste_form else round(na_form / beste_form * 100),
            "serie": _poster(rolige[["dato", "okttype", "periode", "meter_per_slag", "form_snitt"]]),
        },
        "uker": uker_liste,
        "maaneder": maaneder,
        "andel_okttype": {t: _rens(a) for t, a in andel.items()},
        "perioder": _poster(perioder),
        "turer": _poster(turer),
    }
