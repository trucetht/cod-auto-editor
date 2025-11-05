import os
import psycopg2
from typing import List, Tuple, Dict
from .models import Asset, Trigger

def pg_connect_from_env():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "codassets"),
        user=os.environ.get("DB_USER", "coduser"),
        password=os.environ.get("DB_PASSWORD", "codpass"),
    )

def load_triggers(conn) -> Tuple[List[Trigger], Dict[int, Asset]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, kind, path, default_duration_sec, position, scale_pct, fade_in_ms, fade_out_ms
            FROM assets
        """)
        assets = {}
        for row in cur.fetchall():
            assets[row[0]] = Asset(
                id=row[0], kind=row[1], path=row[2],
                default_duration_sec=float(row[3]), position=row[4],
                scale_pct=float(row[5]), fade_in_ms=int(row[6]), fade_out_ms=int(row[7])
            )

        cur.execute("""
            SELECT id, phrase, match_type, asset_id, min_cooldown_sec, priority
            FROM triggers
            ORDER BY priority DESC
        """)
        triggers = [
            Trigger(
                id=r[0], phrase=(r[1] or "").lower(), match_type=r[2],
                asset_id=int(r[3]), min_cooldown_sec=float(r[4]), priority=int(r[5])
            )
            for r in cur.fetchall()
        ]
    return triggers, assets
