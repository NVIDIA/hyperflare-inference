# Flare monitoring example

This directory runs Prometheus and Grafana for one Flare server with Docker
Compose. Prometheus scrapes Flare's `/metrics` route every 10 seconds, and
Grafana opens on a dashboard of throughput, request load, KV cache capacity,
draft acceptance, and cache restores.

## Requirements

- Docker with the Docker Compose plugin (`docker compose`).
- A Flare server on the same host, listening on its default port 8000:
  `flare run CORE`. `/metrics` needs no API key.

## Run

```sh
cd examples/monitoring
docker compose up -d
```

Grafana listens on its default port, 3000. Open <http://127.0.0.1:3000> and
log in as `admin` with password `admin`. Change the password after the first
login. The home page is the "Flare inference" dashboard.



Stop the stack with `docker compose down`. Metric history and Grafana state
stay in named volumes; `docker compose down -v` deletes them.

## Several servers

The dashboard shows one Flare server at a time, selected in the "Server" list
above the panels. To monitor more than one, give each a target in
`prometheus.yaml`. `/metrics` does not report the model name, so the list shows
each target's address unless the target sets an `instance` label:

```yaml
    static_configs:
      - targets: ['127.0.0.1:8000']
        labels:
          instance: inkling-2-small-nvfp4
      - targets: ['127.0.0.1:8001']
        labels:
          instance: qwen3.8-flash
```

Apply the change with `docker compose restart prometheus`.
