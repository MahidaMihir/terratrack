import json, os
from textwrap import dedent
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _normalize_stats(stats, start_year, latest_year, target_year):
    """
    Accepts either:
      - dict: {Class: {start, latest, proj}}
      - list of records: [{"Class":..., "<year>_km²":..., "<target>_proj_km²":...}, ...]
      - stringified JSON of either of the above
    Returns dict {Class: {start, latest, proj}}.
    """
    # If passed as string, load it
    if isinstance(stats, str):
        stats = json.loads(stats)

    # Already dict of dicts
    if isinstance(stats, dict) and all(isinstance(v, dict) for v in stats.values()):
        return stats

    # If list of row records, convert
    if isinstance(stats, list):
        out = {}
        for rec in stats:
            if not isinstance(rec, dict):
                continue
            cls = rec.get("Class") or rec.get("class") or rec.get("name")
            if not cls:
                continue
            def pick(*keys):
                for k in keys:
                    if k in rec and rec[k] is not None:
                        return rec[k]
                return 0.0
            start_val  = pick(f"{start_year}_km²", f"{start_year}_km2", str(start_year))
            latest_val = pick(f"{latest_year}_km²", f"{latest_year}_km2", str(latest_year))
            proj_val   = pick(f"{target_year}_proj_km²", f"{target_year}_Projected_km²",
                              f"{target_year}_km²", f"{target_year}_km2", str(target_year))
            out[cls] = {"start": float(start_val or 0),
                        "latest": float(latest_val or 0),
                        "proj": float(proj_val or 0)}
        return out

    return {}

def render_summary(area_label, start_year, latest_year, target_year, stats_json):
    # Normalize any input shape
    try:
        stats = _normalize_stats(stats_json, start_year, latest_year, target_year)
    except Exception:
        return "Invalid stats JSON."

    # Build bullets
    lines = []
    for k, v in stats.items():
        lines.append(f"- {k}: {v.get('start','?')} → {v.get('latest','?')} km²; projected {v.get('proj','?')} km² by {target_year}")
    bullets = "\n".join(lines) if lines else "No stats."

    prompt = dedent(f"""
    Write a 160–220 word **layman** summary for an environmental report.

    Context:
    - Area: {area_label}
    - Years: {start_year} (start) → {latest_year} (latest) → {target_year} (projection)
    - Key changes:
    {bullets}

    Explain what changed in simple terms, why it matters (flood risk, heat islands, habitat loss),
    and list 3–4 practical preventive measures (afforestation, waterbody buffers, zoning controls, permeable paving).
    Tone: neutral, informative, non-alarmist.
    """)

    # Offline fallback
    if not OPENAI_API_KEY:
        return ("[Offline draft]\n"
                f"Between {start_year} and {latest_year}, the selected area shows noticeable shifts in land use. "
                "Tree cover trends lower while built-up surfaces expand, increasing heat and runoff risks. "
                "Water areas reduce slightly, affecting biodiversity and recharge. "
                f"If current trends continue, by {target_year} these patterns could intensify.\n\n"
                "What helps: targeted tree-planting, wetland/lake buffer protection, roof rainwater harvesting, "
                "and permeable materials for new pavements.")

    # Minimal OpenAI call (optional)
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
            json={"model":"gpt-4o-mini",
                  "messages":[{"role":"user","content":prompt}],
                  "temperature":0.6}
        ).json()
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Failed to contact LLM API. ({e})"
