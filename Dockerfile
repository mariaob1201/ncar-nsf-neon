FROM escomp/cesm-lab-neon:latest

USER root

# Place to install extra OS packages if needed:
# RUN dnf install -y <packages> && dnf clean all

# Make conda binaries reachable even when the base image's entrypoint is
# bypassed (e.g. when JupyterHub's DockerSpawner launches jupyterhub-singleuser
# directly). Without this, jupyterhub-singleuser, jupyter, etc. would only
# resolve to /opt/ncar/conda/bin via the entrypoint's conda activation.
ENV PATH="/opt/ncar/conda/bin:${PATH}"

USER user

# Add Python deps on top of the base CESM conda environment.
# Keep requirements.txt at repo root; the install is skipped if the file is empty.
COPY --chown=user:cesm requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ] && grep -qvE '^\s*(#|$)' /tmp/requirements.txt; then \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Install reusable Python modules at /opt/analytics_modules so they can be
# imported from any notebook either as `from analytics_modules.<name> import ...`
# (preferred) or as `from <name> import ...` (legacy notebooks).
USER root
COPY --chown=user:cesm analytics_modules/ /opt/analytics_modules/
ENV PYTHONPATH="/opt:/opt/analytics_modules:${PYTHONPATH}"

# Drop extended NEON wrapper next to the upstream run_neon.py so it inherits
# the same CTSM/CIME relative imports (_CTSM_PYTHON resolves to /opt/ncar/cesm/python).
COPY --chown=user:cesm --chmod=0755 \
     cesm-tools/site_and_regional/run_neon_v2.py \
     /opt/ncar/cesm/tools/site_and_regional/run_neon_v2.py

# Expose both NEON runners as PATH-resolvable commands (no .py suffix) so
# notebook %%bash cells can call them as `run_neon` / `run_neon_v2`.
RUN ln -sf /opt/ncar/cesm/tools/site_and_regional/run_neon.py     /usr/local/bin/run_neon \
 && ln -sf /opt/ncar/cesm/tools/site_and_regional/run_neon_v2.py  /usr/local/bin/run_neon_v2
USER user

# Drop in custom notebooks / analysis code.
COPY --chown=user:cesm notebooks/ /home/user/notebooks/
