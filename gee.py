# gee.py
"""
Utilities for talking to Google Earth Engine (GEE) from the Streamlit app.

Responsibilities:
- Initialize GEE safely (for local dev and deployment).
- Provide a simple API: compute_class_areas(aoi_geojson, year) -> {class_name: km2}.
- Use ESA WorldCover 10m land cover (2020: v100, 2021: v200).
"""

import os
import json
import ee

# ESA WorldCover palette mapping (simplified, matches ESA docs)
# Value -> Class name
CLASS_MAP = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    70: "Snow/Ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss/Lichen",
}

# ESA WorldCover collections: 2020 & 2021
WORLDCOVER_2020 = "ESA/WorldCover/v100"
WORLDCOVER_2021 = "ESA/WorldCover/v200"

_GEE_INITIALIZED = False


def _get_project_id() -> str:
    """
    Get the Earth Engine / Google Cloud project id.

    Priority:
    - EARTHENGINE_PROJECT env var
    - GEE_PROJECT env var
    - fallback: 'earthengine-legacy' (works for most personal EE accounts)
    """
    return (
        os.getenv("EARTHENGINE_PROJECT")
        or os.getenv("GEE_PROJECT")
        or "earthengine-legacy"
    )


def init_gee():
    """
    Safe initialization of the Earth Engine client.

    Call this once at the beginning of any request that needs GEE.
    - Uses default credentials or service account if provided.
    - Always passes a project id (required by newer EE API).
    """
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return

    project_id = _get_project_id()

    # 1) Try default credentials (works if you've run `earthengine authenticate`
    #    or ee.Authenticate() previously on this machine).
    try:
        ee.Initialize(project=project_id)
        _GEE_INITIALIZED = True
        return
    except Exception:
        pass

    # 2) Try service account JSON from env var (recommended for deployment)
    svc_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if svc_json:
        try:
            info = json.loads(svc_json)
            service_account = info["client_email"]
            credentials = ee.ServiceAccountCredentials(
                service_account, key_data=svc_json
            )
            ee.Initialize(credentials, project=project_id)
            _GEE_INITIALIZED = True
            return
        except Exception:
            # fall back to interactive auth if this fails
            pass

    # 3) Local interactive auth (for development)
    ee.Authenticate()
    ee.Initialize(project=project_id)
    _GEE_INITIALIZED = True


def _esa_worldcover_image(year: int) -> ee.Image:
    """
    Return a WorldCover land-cover image for the given year.

    WorldCover is only available for:
    - 2020: ESA/WorldCover/v100
    - 2021: ESA/WorldCover/v200
    We map any requested year <= 2020 -> 2020 image,
    and any year >= 2021 -> 2021 image.
    """
    year = int(year)
    if year <= 2020:
        coll_id = WORLDCOVER_2020
    else:
        coll_id = WORLDCOVER_2021

    collection = ee.ImageCollection(coll_id)
    img = collection.first().select("Map")  # band name is 'Map'
    return img


def compute_class_areas(aoi_geojson, year: int = 2020) -> dict:
    """
    Compute area in km² for each WorldCover class within the AOI.

    Parameters
    ----------
    aoi_geojson : dict or str
        A GeoJSON geometry (Polygon/MultiPolygon) as dict or JSON string.
    year : int
        Logical year label (we map to 2020/2021 WorldCover internally).

    Returns
    -------
    dict : {class_name: area_km2}
    """
    init_gee()

    # Normalize AOI
    if isinstance(aoi_geojson, str):
        aoi_geojson = json.loads(aoi_geojson)
    aoi = ee.Geometry(aoi_geojson)

    # Get appropriate WorldCover image (2020/2021)
    img = _esa_worldcover_image(year)

    # pixelArea() returns m² per pixel in band 'area'
    pixel_area = ee.Image.pixelArea()

    areas_km2 = {}

    for code, name in CLASS_MAP.items():
        # Mask to class
        class_mask = img.eq(code)
        # Area image for that class only
        class_area_img = pixel_area.updateMask(class_mask)

        # Sum m² within the AOI
        stats = class_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,          # 30m for faster computation (still good enough)
            maxPixels=1e13,
            bestEffort=True,
        )

        area_m2 = stats.get("area")  # band name from pixelArea is 'area'
        if area_m2 is None:
            areas_km2[name] = 0.0
            continue

        try:
            val = ee.Number(area_m2).getInfo()
        except Exception:
            val = 0.0

        areas_km2[name] = round(float(val) / 1e6, 2)  # m² -> km²

    return areas_km2
