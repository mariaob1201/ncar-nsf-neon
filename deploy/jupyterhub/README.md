# JupyterHub deployment for ncar-nsf-neon

Multi-user JupyterLab with password authentication. JupyterHub handles login
and spawns a fresh [`ghcr.io/mariaob1201/ncar-nsf-neon`](https://github.com/mariaob1201/ncar-nsf-neon/pkgs/container/ncar-nsf-neon)
container for each logged-in user with its own persistent home directory.

```
[Browser] --HTTPS--> [Caddy] --> [JupyterHub] -- DockerSpawner --> [cesm-lab-neon per user]
                                       |
                                       +-- NativeAuthenticator (passwords)
```

## What you get

- **Self-service signup** at `/hub/signup` with passwords stored hashed in the Hub's SQLite DB.
- **Admin approval** required before a new user can log in (manage at `/hub/authorize`).
- **One container per user**, isolated CPU/memory, persistent `/home/user` via named volumes.
- **Hub state survives restarts** — sessions, users, and volumes persist across `docker compose down`.

## Prerequisites

- A Linux VM with Docker + Docker Compose (DigitalOcean droplet, EC2, GCP, UW campus host, etc.)
- ≥4 GB RAM for the Hub alone, plus the per-user memory limit × expected concurrent users
- ≥40 GB disk (the base image is ~2 GB; each user's volume grows over time)
- Optional but recommended: a domain name pointing at the VM for HTTPS

## Quick start (HTTP only, for testing on a local network)

```bash
# On the VM:
git clone https://github.com/mariaob1201/ncar-nsf-neon.git
cd ncar-nsf-neon/deploy/jupyterhub

cp .env.example .env
# Edit .env: set JUPYTERHUB_ADMIN_USERS to your username

docker compose up -d --build
```

That's it. Open `http://<vm-ip>:8000` in a browser:
1. Click **Signup** and create your admin user (the username must match `JUPYTERHUB_ADMIN_USERS`).
2. Log in. Your `ghcr.io/mariaob1201/ncar-nsf-neon` container will spawn — give it 30-60s on first run.
3. To invite teammates, send them the URL. They sign up, then you approve them at `/hub/authorize`.

## Production (HTTPS via Caddy, recommended)

Passwords over HTTP is unsafe — do this once you have a domain.

1. Point a DNS A record at the VM: `hub.example.com → <vm-ip>`.
2. Edit `docker-compose.yml`:
   - Change the jupyterhub `ports:` line to `- "127.0.0.1:8000:8000"` (no longer public).
   - Add a Caddy service (template inside `Caddyfile`) and the two named volumes it needs.
3. Set the domain: `export DOMAIN=hub.example.com` (or put it in `.env`).
4. `docker compose up -d`.

Caddy will auto-fetch a Let's Encrypt cert on first request and renew it forever. Visit `https://hub.example.com`.

## Operating it

```bash
# View hub logs
docker compose logs -f jupyterhub

# Restart hub (user containers keep running)
docker compose restart jupyterhub

# Stop everything (user volumes preserved)
docker compose down

# Wipe user data (DESTRUCTIVE)
docker compose down -v

# Pull a newer cesm-lab-neon image (users get it on next login)
docker pull ghcr.io/mariaob1201/ncar-nsf-neon:latest

# See running user containers
docker ps --filter "label=jupyterhub.user.name"

# Inspect a specific user's persistent volume
docker volume inspect jupyterhub-user-<username>
```

## Adding / removing users

- **Add:** they sign up at `/hub/signup`. Admin approves at `/hub/authorize`.
- **Remove access:** at `/hub/authorize`, click the user → delete. Their volume stays — `docker volume rm jupyterhub-user-<username>` to wipe their data.
- **Reset password:** the admin can change a user's password from the same admin panel.

## Tuning

All in `.env`:

| Variable | What it does |
|---|---|
| `JUPYTERHUB_ADMIN_USERS` | Comma-sep usernames that get admin powers (auto-approved). |
| `HUB_PORT` | Host port for the Hub (when not using Caddy). |
| `DOCKER_SPAWN_IMAGE` | Image to spawn per user. Override for local dev (e.g. `cesm-lab-neon-custom:dev`). |
| `JUPYTERHUB_MEM_LIMIT` | Per-user RAM ceiling. CESM runs can spike — `4G` is safe for analysis, `8G` if users actually run simulations. |
| `JUPYTERHUB_CPU_LIMIT` | Per-user CPU ceiling (cores). |

## Why this works with the cesm-lab-neon image

The image's default entrypoint (`/opt/ncar/conda/scripts/start`) starts standalone JupyterLab — fine for `docker run`, but JupyterHub needs to launch the container as a `jupyterhub-singleuser` server instead. Two things make this work:

1. **`requirements.txt` includes `jupyterhub`** — so the `jupyterhub-singleuser` binary is on PATH inside the user container.
2. **`jupyterhub_config.py` overrides entrypoint and cmd:**
   ```python
   c.DockerSpawner.extra_create_kwargs = {"entrypoint": [""]}
   c.DockerSpawner.cmd = ["jupyterhub-singleuser"]
   ```

The image stays usable standalone — Hub just bypasses the default launcher.

## Cost estimates (cloud VM)

For 1–5 active users running notebooks (no simultaneous full CTSM simulations):

| Provider | Spec | Approx /mo |
|---|---|---|
| DigitalOcean | 4 vCPU / 8 GB / 80 GB | $48 |
| AWS EC2 | t3.large (2 vCPU / 8 GB) + 80 GB EBS | ~$70 |
| GCP | e2-standard-2 + 80 GB | ~$55 |
| UW-Madison campus host | free if available | $0 |

Bump RAM/disk if users will run actual CTSM simulations in their containers (each one is a multi-hour CPU-bound job).
