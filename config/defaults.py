from dataclasses import dataclass, field
import os

@dataclass
class Defaults:
    seed: int = 42
    test_size: float = 0.30
    cv_folds: int = 5
    n_jobs: int = -1

    data_dir: str = "/content/FCT_FLOOD_data"
    out_dir: str = "/content/fct_flood_ml_results"
    fig_dir: str = ""
    map_dir: str = ""

    factor_dir: str = "flood_vulnerability_mapping-conditioning_factors"
    points_file: str = "geojson/Flood_points_1.geojson"
    boundary_file: str = "geojson/FCT_Boundary.geojson"
    label_field: str = "flooded"

    FACTORS = {
        "elevation": "fct_elevation.tif",
        "slope": "fct_slope.tif",
        "TWI": "fct_TWI.tif",
        "SPI": "fct_SPI.tif",
        "drainage_density": "fct_drainage_density.tif",
        "dist_to_water": "fct_dist_to_water.tif",
        "rainfall": "fct_rainfall.tif",
        "NDVI": "fct_NDVI.tif",
        "landcover": "fct_landcover.tif",
        "built_up": "fct_built_up.tif",
    }

    COPING_LAYERS = {
        "emergency_access": "fct_Emergency.tif",
        "health_access": "health.tif",
        "night_light": "fctlight.tif",
        "road_density": "fct_road_density.tif",
        "population": "fct_pop.tif",
    }

    INCLUDE_COPING_LAYERS: bool = False
    REFERENCE_FACTOR: str = "elevation"
    CATEGORICAL: list[str] = field(default_factory=lambda: ["landcover"])
    LANDCOVER_NODATA: int = 0
    WATER_CLASS: int = 80

    SMOOTH_RAINFALL: bool = True
    RAINFALL_CELL_DEG: float = 0.05
    DIST_TO_WATER_MODE: str = "recompute"
    MAX_PLAUSIBLE_DIST_M: float = 200_000

    CLASS_METHOD: str = "equal_interval"
    CLASS_NAMES: list[str] = field(default_factory=lambda: ["Very low", "Low", "Moderate", "High", "Very high"])
    CLASS_COLOURS: list[str] = field(default_factory=lambda: ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"])

    SPATIAL_BLOCK_KM: float = 10.0
    PRED_CHUNK_ROWS: int = 256
    OUT_NODATA: float = -9999.0

    def __post_init__(self):
        if not self.fig_dir:
            self.fig_dir = os.path.join(self.out_dir, "figures")
        if not self.map_dir:
            self.map_dir = os.path.join(self.out_dir, "maps")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.map_dir, exist_ok=True)

DEFAULTS = Defaults()
