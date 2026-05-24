# ncar-nsf-neon

Containerized JupyterLab environment for NCAR / NSF / NEON work, built on top of
[`escomp/cesm-lab-neon`](https://hub.docker.com/r/escomp/cesm-lab-neon). Ships
the CTSM/CESM toolchain plus this project's analytics modules and notebooks,
with a GitHub Actions workflow that publishes the image to GHCR on every push.

## Published image

`ghcr.io/mariaob1201/ncar-nsf-neon`

Tags produced by CI:

| Tag                | When pushed                                          |
| ------------------ | ---------------------------------------------------- |
| `latest`           | every push to `main`                                 |
| `main`             | every push to `main`                                 |
| `sha-<short>`      | every push to `main` (immutable, commit-pinned)      |
| `vX.Y.Z` / `X.Y` / `X` | when a git tag `vX.Y.Z` is pushed                |
| `pr-<n>`           | on a pull request (built but **not** pushed)         |

## Quick start

### Pull and run the published image

```bash
docker run --rm -p 8888:8888 ghcr.io/mariaob1201/ncar-nsf-neon:latest
```

Then open <http://127.0.0.1:8888/lab>. The base image ships with JupyterLab
configured with **no token and no password** (it's intended for local dev), so
you'll land directly in the lab UI.

If the image is still private on GHCR, log in first:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <your-username> --password-stdin
```

### Build locally

```bash
docker build --platform linux/amd64 -t cesm-lab-neon-custom:dev .
docker run --rm -p 8888:8888 cesm-lab-neon-custom:dev
```

`--platform linux/amd64` matters on Apple Silicon: the upstream base image is
amd64-only, so the container runs through Rosetta emulation.

### Persist your notebooks across runs

```bash
docker run --rm -p 8888:8888 \
  -v "$PWD/notebooks:/home/user/notebooks" \
  ghcr.io/mariaob1201/ncar-nsf-neon:latest
```

## Project layout

```
.
├── Dockerfile                       # Extends escomp/cesm-lab-neon
├── requirements.txt                 # Extra Python deps installed on top of base
├── notebooks/                       # Custom JupyterLab notebooks (mounted at /home/user/notebooks)
├── analytics_modules/               # Reusable Python modules (installed at /opt/analytics_modules)
│   ├── __init__.py
│   ├── kalman_filter.py             # Kalman calibration for CTSM outputs
│   └── model_misfit.py              # Residual diagnostics used by kalman_filter
└── .github/workflows/
    └── docker-publish.yml           # Build + push to GHCR
```

## Using the analytics modules in a notebook

`analytics_modules/` is copied into `/opt/analytics_modules/` and `/opt` is
added to `PYTHONPATH`, so any notebook in the container can do:

```python
from analytics_modules.kalman_filter import kalman_filter, kalman_gain_bias, calibrate_and_evaluate

calibrated_df, report = calibrate_and_evaluate(df, col="LE", hour_col="hour")
```

To add another module, drop it into `analytics_modules/` and rebuild — it will
be importable as `analytics_modules.<your_module>`.

## Adding Python dependencies

Edit `requirements.txt` (one package spec per line, `#` for comments) and
rebuild. The Dockerfile skips `pip install` when the file only has comments,
so it's safe to leave empty.

## CI/CD

`.github/workflows/docker-publish.yml` runs on every push to `main`, every
`v*` tag, every pull request (build-only, no push), and on manual dispatch
from the Actions tab. It uses the built-in `GITHUB_TOKEN` to push to GHCR —
no secrets to configure.

Cache is stored in GitHub Actions cache (`type=gha`) so unchanged layers
skip on subsequent builds.

## License

[MIT](LICENSE.md)
