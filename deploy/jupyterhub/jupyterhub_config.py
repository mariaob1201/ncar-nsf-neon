"""JupyterHub configuration: NativeAuthenticator + DockerSpawner.

Designed for 1-5 users on a single VM. Each user logs in with a password
(signup is open; an admin must approve from /hub/authorize), then gets
their own isolated cesm-lab-neon container with persistent /home/user.
"""
import os

c = get_config()  # noqa: F821 (provided by JupyterHub at runtime)

# ---------------------------------------------------------------------------
# Spawner: one Docker container per logged-in user
# ---------------------------------------------------------------------------
c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"

c.DockerSpawner.image = os.environ.get(
    "DOCKER_SPAWN_IMAGE",
    "ghcr.io/mariaob1201/ncar-nsf-neon:latest",
)

# Shared Docker network for Hub <-> user containers
c.DockerSpawner.network_name = os.environ["DOCKER_NETWORK_NAME"]
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.hub_connect_ip = "jupyterhub"  # matches the hub container name

# Override the cesm-lab-neon image's default entrypoint (which starts
# standalone JupyterLab) so DockerSpawner can launch it as a Hub
# single-user server instead.
c.DockerSpawner.extra_create_kwargs = {"entrypoint": [""]}
c.DockerSpawner.cmd = ["jupyterhub-singleuser"]

# Working directory inside the user container
c.DockerSpawner.notebook_dir = "/home/user"

# Persistent per-user volume: jupyterhub-user-<username> -> /home/user
c.DockerSpawner.volumes = {
    "jupyterhub-user-{username}": "/home/user",
}

# Remove the spawned container on logout; the named volume survives,
# so the user's files persist across sessions.
c.DockerSpawner.remove = True

# Don't repeatedly pull the spawn image
c.DockerSpawner.pull_policy = "ifnotpresent"

# Per-user resource limits
c.Spawner.mem_limit = os.environ.get("JUPYTERHUB_MEM_LIMIT", "4G")
c.Spawner.cpu_limit = float(os.environ.get("JUPYTERHUB_CPU_LIMIT", "2"))

# Connections to single-user servers timeout after 10 minutes idle
c.Spawner.http_timeout = 120
c.Spawner.start_timeout = 600  # base image is large; allow time for first spawn


# ---------------------------------------------------------------------------
# Authenticator: NativeAuthenticator (signup + password)
# ---------------------------------------------------------------------------
c.JupyterHub.authenticator_class = "nativeauthenticator.NativeAuthenticator"

# Self-service signup at /hub/signup. Admin must approve at /hub/authorize.
c.NativeAuthenticator.open_signup = True
c.NativeAuthenticator.minimum_password_length = 12
c.NativeAuthenticator.check_common_password = True

# Admins are auto-approved on signup and can approve/manage other users
admins = os.environ.get("JUPYTERHUB_ADMIN_USERS", "admin").split(",")
c.Authenticator.admin_users = {a.strip() for a in admins if a.strip()}


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------
c.JupyterHub.bind_url = "http://:8000"

# Persist Hub state across restarts (users, tokens, sessions)
c.JupyterHub.db_url = "sqlite:////srv/jupyterhub/jupyterhub.sqlite"
c.JupyterHub.cookie_secret_file = "/srv/jupyterhub/jupyterhub_cookie_secret"
