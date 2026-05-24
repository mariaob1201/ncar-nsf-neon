FROM escomp/cesm-lab-neon:latest

USER root

# Place to install extra OS packages if needed:
# RUN dnf install -y <packages> && dnf clean all

USER user

# Add Python deps on top of the base CESM conda environment.
# Keep requirements.txt at repo root; the install is skipped if the file is empty.
COPY --chown=user:cesm requirements.txt /tmp/requirements.txt
RUN if [ -s /tmp/requirements.txt ] && grep -qvE '^\s*(#|$)' /tmp/requirements.txt; then \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Drop in custom notebooks / analysis code.
COPY --chown=user:cesm notebooks/ /home/user/notebooks/
