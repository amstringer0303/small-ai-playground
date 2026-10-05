# Air-quality scenario provenance

`air_quality.csv` is generated entirely by `datasets.py` using NumPy's seeded
random generator (seed 23, 720 observations). Regenerate with
`python -m data.datasets`. There are no imported observations, personal records,
external dataset licenses, or real sensor locations. The four site names and
coordinates are invented. Dataset and project licensing remain undecided.

PM2.5/PM10 are simulated concentrations; temperature, humidity, wind, hour,
site, sensor, and an unreliable-observation flag are simulated covariates. The
`alert` label is a synthetic latent rule with noise and a site-dependent effect,
not an official air-quality classification, health threshold, or regulatory label.
That deliberately creates an observable location/performance tradeoff.

The dataset is imbalanced by both class and site. Measurements have noise and
some flagged unreliable readings. These are experimental controls, not evidence
of how an actual community or physical monitoring system behaves.

Original held-out examples and their original labels remain the evaluation anchor.
Editing training labels does not silently edit the answer key. A different label
definition should also be evaluated with explicit custom tests; baseline accuracy
continues to measure agreement with the original synthetic definition.
