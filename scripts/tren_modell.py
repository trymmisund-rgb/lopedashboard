"""Trener modellene som predikerer økttype for turer uten Runna-tittel.

Kjør fra prosjektmappen (på nytt når du har fått mer data):
    uv run scripts/tren_modell.py

To modeller trenes og lagres sammen i src/lopedashboard/okttype_modell.joblib:
  - "med_runder":  for løpere der rundene følger øktene (Runna, Garmin-økter, rundeknapp)
  - "uten_runder": for løpere som bare har automatiske runder per km
klassifiser() i analyse.py velger riktig modell automatisk.
Se notebooks/02_klassifisering.ipynb for hvorfor akkurat disse modellene ble valgt.
"""

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lopedashboard.analyse import (
    FEATURES,
    FEATURES_UTEN_RUNDER,
    MODELL_STI,
    klassifiser,
    lag_features,
    les_garmin,
)

ROT = Path(__file__).resolve().parent.parent

OPPSETT = {
    "med_runder": (FEATURES, lambda: DecisionTreeClassifier(max_depth=3, random_state=0)),
    "uten_runder": (FEATURES_UTEN_RUNDER, lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tren økttype-modellene")
    parser.add_argument("--inn", type=Path, default=ROT / "data" / "Activities_løping.csv")
    args = parser.parse_args()

    df = klassifiser(les_garmin(args.inn), bruk_modell=False)
    X_alle = lag_features(df)
    merket = (df["kilde"] == "tittel") & df["gyldig"] & X_alle.notna().all(axis=1)
    y = df.loc[merket, "okttype"]
    print(f"Treningsdata: {len(y)} turer ({', '.join(f'{t}: {n}' for t, n in y.value_counts().items())})")

    modeller = {}
    for navn, (features, lag_modell) in OPPSETT.items():
        X = X_alle.loc[merket, features]
        pred = cross_val_predict(lag_modell(), X, y, cv=LeaveOneOut())
        print(f"  {navn:12s} leave-one-out: {accuracy_score(y, pred):.0%} riktig")
        modeller[navn] = lag_modell().fit(X, y)

    joblib.dump(modeller, MODELL_STI)
    print(f"✓ Lagret modellene til {MODELL_STI.relative_to(ROT)}")


if __name__ == "__main__":
    main()
