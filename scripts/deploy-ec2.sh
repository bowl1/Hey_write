#!/usr/bin/env bash
# Runs on EC2 after CI uploads backend.tar.gz and this script to a release directory.
set -euo pipefail
revision=${1:?Pass the Git commit SHA}
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || exit 2
exec 9>/home/ubuntu/.heywrite-deploy.lock
flock -n 9 || { echo 'Another AWS deployment is running'; exit 1; }
release_dir=$(cd "$(dirname "$0")" && pwd)
env_file=/home/ubuntu/heywrite-production.env
upstream=/etc/nginx/snippets/heywrite-upstream.conf
candidate="heywrite-candidate-${revision:0:12}"
image="heywrite-backend:$revision"
old_port=$(docker port heywrite-backend 8000/tcp | sed 's/.*://')
[[ "$old_port" = 8000 || "$old_port" = 8001 ]] || exit 2
port=8001
if [ "$old_port" = 8001 ]; then port=8000; fi
test -s "$env_file"
test -f "$upstream"
cp "$upstream" "$release_dir/upstream.previous"
switched=false
promoted=false
rollback() {
  if [ "$promoted" = true ]; then return; fi
  if [ "$switched" = true ]; then
    sudo cp "$release_dir/upstream.previous" "$upstream"
    sudo nginx -t && sudo systemctl reload nginx
  fi
  if ! docker container inspect heywrite-backend >/dev/null 2>&1 && \
      docker container inspect heywrite-backend-previous >/dev/null 2>&1; then
    docker rename heywrite-backend-previous heywrite-backend
  fi
  docker rm -f "$candidate" >/dev/null 2>&1 || true
}
trap rollback EXIT
tar -xzf "$release_dir/backend.tar.gz" -C "$release_dir"
docker build --label "org.opencontainers.image.revision=$revision" \
  -t "$image" -f "$release_dir/chatbox-backend/dockerfile" "$release_dir/chatbox-backend"
docker run -d --name "$candidate" --restart unless-stopped \
  --env-file "$env_file" -p "127.0.0.1:$port:8000" \
  --log-opt max-size=10m --log-opt max-file=3 "$image"
ready=false
for ((attempt=0; attempt<30; attempt++)); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
[ "$ready" = true ] || { echo 'Candidate health check failed'; exit 1; }
# Checks runtime role, RLS and required migrations without creating tenant data.
docker exec -i "$candidate" python - <<'PY'
import uuid
from tenancy import tenant_context, ensure_workspace
try:
    with tenant_context(str(uuid.uuid4())):
        ensure_workspace()
except Exception as error:
    print('Database readiness failed:', type(error).__name__)
    raise SystemExit(1)
print('Database schema and tenant isolation ready')
PY
printf 'proxy_pass http://127.0.0.1:%s;\n' "$port" > "$release_dir/upstream.next"
switched=true
sudo cp "$release_dir/upstream.next" "$upstream"
sudo nginx -t
sudo systemctl reload nginx
curl -fsS --retry 5 --retry-delay 2 --max-time 20 \
  https://hey-write.vercel.app/api/health >/dev/null
curl -fsS --retry 3 --retry-delay 2 --max-time 20 \
  https://hey-write.vercel.app/api/public/templates >/dev/null
auth_status=$(curl -sS --max-time 20 -o /dev/null -w '%{http_code}' \
  https://hey-write.vercel.app/api/agent/memory)
[ "$auth_status" = 401 ] || { echo 'Authentication smoke check failed'; exit 1; }
# Retain one stopped container and its image for rollback.
if docker container inspect heywrite-backend-previous >/dev/null 2>&1; then
  docker rm -f heywrite-backend-previous >/dev/null
fi
docker rename heywrite-backend heywrite-backend-previous
docker rename "$candidate" heywrite-backend
promoted=true
docker update --restart=no heywrite-backend-previous >/dev/null
docker stop --time 130 heywrite-backend-previous >/dev/null
# Keep only the images behind the live and rollback containers; cleanup never fails the deploy.
keep=$(docker inspect --format '{{.Image}}' heywrite-backend heywrite-backend-previous)
for id in $(docker images -q --no-trunc heywrite-backend | sort -u); do
  grep -qx "$id" <<<"$keep" || docker rmi "$id" >/dev/null 2>&1 || true
done
docker image prune -f >/dev/null 2>&1 || true
echo "AWS deployment verified: $revision on port $port"
