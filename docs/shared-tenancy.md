# Shared-table tenancy deployment

All private data now lives in `heywrite.templates`, `heywrite.agent_sessions`,
`heywrite.agent_runs`, and `heywrite.messages`. Each row has a required UUID
`tenant_id`. For now, one Supabase user is one workspace, so `tenant_id` is the
verified user's ID; this does not add team sharing or membership management.

Primary keys include `tenant_id`, including session IDs, template IDs, run IDs,
and imported message IDs. Runs and messages reference sessions through composite
foreign keys, so references cannot cross tenants. RLS uses the transaction-local
`app.tenant_id` for both `USING` and `WITH CHECK`. A restrictive policy prevents an
additional permissive policy from widening access. Missing identity denies access.

`DATABASE_URL` must use a dedicated non-owner role without SUPERUSER or BYPASSRLS.
The application rejects a privileged connection instead of silently bypassing
isolation. Runtime requests do not execute DDL. Registration requires no database
schema changes. The trusted backend sets identity after validating the Supabase
token; custom SQL identity settings are not a defense against arbitrary SQL
execution. Keep parameterized queries and protect database credentials.

## Schema migrations

The schema, RLS policies, the `heywrite_app` runtime role, and its grants are
defined in versioned SQL files under `supabase/migrations/` and applied with the
[Supabase CLI](https://supabase.com/docs/guides/deployment/database-migrations).
The CLI records applied versions in `supabase_migrations.schema_migrations` and
runs each file in its own transaction: a failing file changes nothing and is not
recorded.

`20261025000000_shared_tenancy.sql` is the baseline. It reproduces the schema
first created by the retired `migrate_shared.py` exactly and is idempotent, so the
first push to an existing database only records the version.

### Changing the schema

1. Create a migration from the repository root and write the SQL in it:

   ```bash
   supabase migration new add_session_title
   ```

   Never edit a migration that has already been applied; add a new one.
2. Keep each change compatible with the backend version currently running,
   because migrations are applied before the new backend is deployed. Add
   columns with defaults, and drop or rename in a later release once no deployed
   code uses the old name.
3. A new private table must have a `tenant_id uuid not null` column, the same
   `tenant_access`/`tenant_guard` policies with `FORCE ROW LEVEL SECURITY`, and
   explicit grants to `heywrite_app`, and belongs in the RLS readiness check in
   `tenancy.ensure_workspace`. Do not add it to `TABLES` in
   `import_legacy_tenants.py`; legacy schemas never had it.
4. Run the tests locally (see below); they apply every migration to a disposable
   database before checking isolation.
5. Open a pull request. CI runs the tests and applies the migrations to a
   throwaway PostgreSQL. Merging to `master` applies them to production, then
   deploys the backend.

### Deployment pipeline

`.github/workflows/deploy-backend.yml` runs tests on every branch push and pull request. Production migration and deployment run only on pushes to `master`:

1. `test`: backend unit and RLS tests, frontend tests and production build,
   then `supabase db push` against a disposable pgvector PostgreSQL.
2. `migrate`: `supabase db push` against production using the
   `DATABASE_ADMIN_URL` repository secret.
3. `deploy`: triggers the Render deploy hook.

A failure stops the later jobs, so the backend is never deployed ahead of its
schema. Render auto-deploy is off (`render.yaml`) so the workflow controls the
order. Pull requests run only the `test` job.

`DATABASE_ADMIN_URL` is the `postgres` connection string. Use the Supabase
**session pooler** (port 5432) endpoint: GitHub runners have no IPv6, and the
direct connection host is IPv6-only. Store it only as a GitHub secret; never add
it to the Render service.

### Applying migrations manually

To apply migrations without CI (for example on a staging database):

```bash
read -r -s -p "Admin database URL: " DATABASE_ADMIN_URL; echo
supabase db push --db-url "$DATABASE_ADMIN_URL"
unset DATABASE_ADMIN_URL
```

`supabase migration list --db-url ...` shows which versions are applied.

## Runtime role

The baseline migration creates `heywrite_app` without a password. Set one once
with a secure administrative tool (for example psql's interactive
`\password heywrite_app`) and use it in the web service's `DATABASE_URL`, with the
connection format for your Supabase endpoint or pooler. Do not reuse the postgres
role. The role can read/write private tables and use the message sequence, but
cannot create tables, change policies, truncate tables, or read the import ledger.

Keep `heywrite` outside Supabase's exposed Data API schemas. Only the backend
should access it. The shared default-template embedding cache contains bundled
public content, is rebuilt as needed, and grants the runtime role SELECT/INSERT.

## Legacy import

`import_legacy_tenants.py` is a one-off data migration from the former
per-user `tenant_<uuidhex>` schemas. It requires the migrations to be applied,
and runs with `DATABASE_ADMIN_URL` during a write maintenance window with the old
backend stopped:

```bash
cd chatbox-backend
.venv/bin/python import_legacy_tenants.py
```

All copies run in one transaction and are checked for matching values before
commit; any conflict aborts. Source schemas remain intact and are not
synchronized afterwards. An import ledger prevents repeat copying. Existing
target data is never merged or overwritten.

Legacy `public` tables have no reliable tenant attribution and are never imported
automatically. To explicitly assign them to one confirmed user, use
`--public-owner USER_UUID`; importing into an already populated tenant fails.
`manage_workspace.py --user-id UUID` checks readiness; `--seed-templates` seeds that
workspace with the restricted runtime role.

If cutover fails before new writes, the old backend and preserved schemas can be
used again. Once shared tables receive new writes, switching back requires a
separate reconciliation/export plan; do not assume the old schemas are current.

## Automated verification

Use a disposable PostgreSQL database with pgvector installed. The test connection
needs administrative privileges: tests apply `supabase/migrations` inside a
transaction, then switch to the real `heywrite_app` role for application
operations. Everything, including the role if the migration created it, rolls
back. Tests do not use production accounts or call an embedding API.

```bash
cd chatbox-backend
# Set RLS_TEST_DATABASE_URL securely to the disposable test database.
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Without `RLS_TEST_DATABASE_URL`, database tests are explicitly skipped. Tests cover
cross-tenant reads/writes, forged tenant IDs, same-ID upserts, composite foreign
keys, denied DDL, missing identity, transaction cleanup, vector retrieval,
migration idempotency, legacy imports, repeat imports, and conflict rejection.
