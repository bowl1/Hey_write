# AWS backend deployment

The public application is https://hey-write.vercel.app. Vercel forwards `/api/*`
over HTTPS to the EC2 Elastic IP `16.170.1.179`. The Vercel project root is
`chatbox-frontend`; its `vercel.json` defines the rewrite.

## Production resources

- EC2: `i-07cfe6535836da6ec`, `eu-north-1`, Ubuntu 24.04.
- Elastic IP allocation: `eipalloc-0cca4ef00be742be6`.
- Docker runtime configuration: `/home/ubuntu/heywrite-production.env`, mode 600.
  Values must use Docker env-file syntax (no surrounding dotenv quotes).
- Nginx site: `/etc/nginx/sites-available/heywrite-ip`.
- Active upstream: `/etc/nginx/snippets/heywrite-upstream.conf`.
- IP certificate: `/etc/letsencrypt/live/heywrite-ip/`; snap Certbot renews it
  automatically and reloads Nginx. Its renewal dry run has been verified.
- A 2 GiB swap file supplements the instance's approximately 1 GiB RAM.

## Release flow

On a push to `master`, `.github/workflows/deploy-backend.yml` runs the full
regression gate, applies Supabase migrations, then deploys to EC2. Pull requests
run tests only. Deployment uses the existing `EC2_SSH_KEY`, `EC2_HOST`, and pinned
`EC2_KNOWN_HOSTS` GitHub secrets. Application secrets stay on EC2.

CI uploads the committed backend and `scripts/deploy-ec2.sh`. The server builds
an image tagged with the commit SHA, starts a candidate on the inactive localhost
port (8000 or 8001), and checks health, database schema and RLS readiness. Nginx
switches upstream only after those checks. Public Vercel health, templates, and
authentication checks must also pass. A failure before promotion restores the
previous upstream and removes the candidate. Successful releases retain the
previous stopped container as `heywrite-backend-previous`.

Vercel external rewrites have a 120-second timeout. Longer generation requests
need an asynchronous job API. Database migrations must remain compatible with
the previous image; rolling back a container does not undo database migrations.

## Manual rollback

On EC2, inspect `docker port heywrite-backend-previous 8000/tcp` to find its host
port. Start it with `docker start heywrite-backend-previous`, set its restart
policy to `unless-stopped`, and point the upstream snippet at that localhost
port. Run `sudo nginx -t` and reload Nginx. Verify `/api/health`, public templates,
and an authenticated writing request before stopping the failed container.
Restore the serving container name to `heywrite-backend` before another CI deploy.

Do not suspend Render until the first AWS workflow deployment and public smoke
checks have passed. Suspending preserves the service for recovery; deletion is
not required.
