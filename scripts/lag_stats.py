"""Lager stats.json til nettsiden fra Garmin-eksporten.

Kjør fra prosjektmappen:
    uv run scripts/lag_stats.py

Valgfritt med egne stier og løperprofil:
    uv run scripts/lag_stats.py --inn data/Activities_løping.csv --ut web/public/stats.json
    uv run scripts/lag_stats.py --niva intermediate --makspuls 195
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from lopedashboard.analyse import NIVAER, Profil, analyser, les_garmin

ROT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lag stats.json fra Garmin-CSV")
    parser.add_argument("--inn", type=Path, default=ROT / "data" / "Activities_løping.csv")
    parser.add_argument("--ut", type=Path, default=ROT / "web" / "public" / "stats.json")
    parser.add_argument("--niva", choices=list(NIVAER), help="ditt løpenivå (ellers foreslått fra data)")
    parser.add_argument("--makspuls", type=float, help="din makspuls, hvis du vet den")
    parser.add_argument("--ukesnitt", type=float, help="km per uke (ellers regnet ut fra data)")
    args = parser.parse_args()

    df = les_garmin(args.inn)
    profil = Profil(niva=args.niva, makspuls=args.makspuls, ukesnitt_km=args.ukesnitt)
    stats = analyser(df, profil)
    stats["oppdatert"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    args.ut.parent.mkdir(parents=True, exist_ok=True)
    args.ut.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    n = stats["nokkeltall"]
    print(f"✓ Skrev {args.ut}")
    print(f"  {n['turer']} turer, {n['km']} km, {n['timer']} timer ({n['forste_tur']} → {n['siste_tur']})")
    lp = stats["loperprofil"]
    print(f"  Nivå: {lp['niva']} ({lp['kilder']['niva']}), foreslått fra data: {lp['foreslatt_niva']}")
    print(f"  Ukesnitt {lp['ukesnitt_km']} km, typisk tur {lp['typisk_tur_km']} km, makspuls {lp['makspuls']:.0f}")
    if stats["form"]["prosent_av_beste"]:
        print(f"  Form nå: {stats['form']['prosent_av_beste']} % av beste registrerte")


if __name__ == "__main__":
    main()
