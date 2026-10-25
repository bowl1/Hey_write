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

## Deployment order

1. Back up the database. Rehearse the migration on a staging copy. Stop all old
   application workers and CLI writers for the migration and cutover; retained
   tenant schemas are not synchronized after the import.
2. In a controlled administrative environment, configure `DATABASE_ADMIN_URL`
   with the existing database administrator credentials. Do not add this variable
   to the web service. Run from `chatbox-backend`:

   ```bash
   .venv/bin/python migrate_shared.py --runtime-role heywrite_app
   ```

   This creates the shared schema and pgvector extension if needed, applies RLS,
   creates a restricted login role if absent, and copies all `tenant_<uuidhex>`
   schemas. Copies are checked for matching values before commit. Source schemas
   remain intact. All DDL and copies run in one transaction; any conflict aborts.
   An import ledger prevents repeat copying. Existing target data is never merged
   or overwritten. Fix incomplete source schemas or orphaned session references
   before retrying a failed migration.
3. Set a strong password for `heywrite_app` using a secure administrative tool
   (for example psql's interactive `\password heywrite_app`). The script deliberately
   does not generate, print, or store a password. Set the web service's
   `DATABASE_URL` to the connection string for this role using the connection
   format appropriate to your Supabase endpoint/pooler. Do not reuse the postgres
   role. The role can read/write private tables and use the message sequence, but
   cannot create tables, change policies, truncate tables, or read the import ledger.
4. Keep `heywrite` outside Supabase's exposed Data API schemas. Only the backend
   should access it. The shared default-template embedding cache contains bundled
   public content, is rebuilt as needed, and grants the runtime role SELECT/INSERT.
5. Deploy the new backend. Verify A/B sign-in, template creation, generation,
   session restoration, and reading migrated data. Resume traffic only after these
   checks pass. Do not run the old and new backends concurrently.

For a fresh installation use `--skip-tenant-copy`. Legacy `public` tables have no
reliable tenant attribution and are never imported automatically. To explicitly
assign them to one confirmed user, use `--public-owner USER_UUID`; importing into
an already populated tenant fails. `manage_workspace.py --user-id UUID` now checks
readiness; `--seed-templates` seeds that workspace with the restricted runtime role.

If cutover fails before new writes, the old backend and preserved schemas can be
used again. Once shared tables receive new writes, switching back requires a
separate reconciliation/export plan; do not assume the old schemas are current.

## Automated verification

Use a disposable PostgreSQL database with pgvector installed. The test connection
needs administrative privileges to create transactional fixtures and test roles.
Tests switch to a real restricted role for application operations. They do not
use production accounts or call an embedding API.

```bash
cd chatbox-backend
# Set RLS_TEST_DATABASE_URL securely to the disposable test database.
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Without `RLS_TEST_DATABASE_URL`, database tests are explicitly skipped. Tests cover
cross-tenant reads/writes, forged tenant IDs, same-ID upserts, composite foreign
keys, denied DDL, missing identity, transaction cleanup, vector retrieval, legacy
imports, repeat imports, and conflict rejection. Fixture tables and roles roll
back at the end of each test.
