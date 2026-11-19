"""Lager stats.json til nettsiden fra Garmin-eksporten.

Kjør fra prosjektmappen:
    uv run scripts/lag_stats.py

Valgfritt med egne stier:
    uv run scripts/lag_stats.py --inn data/Activities_løping.csv --ut web/public/stats.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from lopedashboard.analyse import analyser, les_garmin

ROT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lag stats.json fra Garmin-CSV")
    parser.add_argument("--inn", type=Path, default=ROT / "data" / "Activities_løping.csv")
    parser.add_argument("--ut", type=Path, default=ROT / "web" / "public" / "stats.json")
    args = parser.parse_args()

    df = les_garmin(args.inn)
    stats = analyser(df)
    stats["oppdatert"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    args.ut.parent.mkdir(parents=True, exist_ok=True)
    args.ut.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    n = stats["nokkeltall"]
    print(f"✓ Skrev {args.ut.relative_to(ROT)}")
    print(f"  {n['turer']} turer, {n['km']} km, {n['timer']} timer ({n['forste_tur']} → {n['siste_tur']})")
    if stats["form"]["prosent_av_beste"]:
        print(f"  Form nå: {stats['form']['prosent_av_beste']} % av beste registrerte")


if __name__ == "__main__":
    main()
