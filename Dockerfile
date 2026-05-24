FROM escomp/cesm-lab-neon:latest

USER root

# Place to install extra OS packages if needed:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       <packages> \
#  && rm -rf /var/lib/apt/lists/*

USER jovyan

# Add Python deps on top of the base environment.
# Keep requirements.txt at repo root; the COPY is a no-op if the file is empty.
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Drop in custom notebooks / analysis code.
COPY --chown=jovyan:users notebooks/ /home/jovyan/notebooks/
