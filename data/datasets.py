from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEO_FEATURES = {"latitude", "longitude", "site", "sensor"}
DEFAULT_FEATURES = [
    "pm25", "pm10", "temperature", "humidity", "wind", "hour",
    "site", "sensor", "latitude", "longitude", "unreliable",
]


def synthetic_air_quality(rows: int = 720, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    site = rng.choice(4, rows, p=[0.46, 0.27, 0.19, 0.08])
    hour = rng.integers(0, 24, rows)
    humidity = rng.uniform(28, 93, rows)
    wind = rng.uniform(0.2, 9, rows)
    latent = rng.gamma(2.7, 11.0, rows)
    unreliable = rng.random(rows) < 0.09
    pm25 = np.maximum(0.5, latent + rng.normal(0, 5, rows))
    pm25[unreliable] += rng.normal(0, 15, unreliable.sum())
    risk = latent + np.array([-5, 5, 12, 19])[site] + 0.09 * (humidity - 55)
    risk += 4 * ((hour >= 7) & (hour <= 9)) - 0.9 * wind
    risk += rng.normal(0, 7, rows)
    return pd.DataFrame({
        "example_id": [f"air-{i:04d}" for i in range(rows)],
        "included": True,
        "pm25": np.round(pm25, 2),
        "pm10": np.round(np.maximum(pm25, latent * 1.5 + rng.normal(0, 9, rows)), 2),
        "temperature": np.round(rng.normal(21, 6, rows), 2),
        "humidity": np.round(humidity, 2),
        "wind": np.round(wind, 2),
        "hour": hour,
        "site": np.array(["River", "Central", "Hills", "Harbor"])[site],
        "sensor": [f"sensor-{s}-{i % 2}" for i, s in enumerate(site)],
        "latitude": np.round(40.0 + site * 0.035 + rng.normal(0, 0.001, rows), 6),
        "longitude": np.round(-74.0 + site * 0.045 + rng.normal(0, 0.001, rows), 6),
        "unreliable": unreliable.astype(int),
        "condition": np.where(risk > 42, "alert", "normal"),
    })


def load_air_quality() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "air_quality.csv")


if __name__ == "__main__":
    destination = ROOT / "data" / "air_quality.csv"
    synthetic_air_quality().to_csv(destination, index=False)
    print(f"Generated {destination}")
