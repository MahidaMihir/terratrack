def linear_projection(start_stats, latest_stats, start_year, latest_year, target_year):
    """
    start_stats/latest_stats: dict[class_name] -> km2
    Returns dict[class_name] -> projected km2 at target_year (simple linear model).
    """
    if target_year <= latest_year:
        return {k: latest_stats[k] for k in latest_stats}
    years = latest_year - start_year
    horizon = target_year - latest_year
    proj = {}
    for k in latest_stats:
        dy = (latest_stats[k] - start_stats.get(k, latest_stats[k])) / max(1, years)
        proj[k] = round(latest_stats[k] + dy * horizon, 2)
    return proj
