-- Shared-table tenancy: private rows carry tenant_id and are isolated by RLS.
--
-- Baseline of the schema first created by migrate_shared.py. Every statement is
-- idempotent, so applying it to that database only records the version.

create extension if not exists vector with schema public;

-- Runtime role for the web service. Its password is set out of band
-- (psql: \password heywrite_app) and never stored in a migration.
do $$
declare
  existing pg_roles%rowtype;
begin
  select * into existing from pg_roles where rolname = 'heywrite_app';
  if not found then
    create role heywrite_app login nosuperuser nobypassrls nocreatedb nocreaterole noinherit;
  elsif existing.rolsuper or existing.rolbypassrls or existing.rolcreaterole or existing.rolcreatedb then
    raise exception 'heywrite_app must not have administrative privileges';
  end if;
  if exists (select 1 from pg_auth_members where member = 'heywrite_app'::regrole) then
    raise exception 'heywrite_app must not inherit other roles';
  end if;
end
$$;

create schema if not exists heywrite;
revoke all on schema heywrite from public;

create table if not exists heywrite.templates (
  tenant_id uuid not null default nullif(current_setting('app.tenant_id', true), '')::uuid,
  id text not null,
  primary key (tenant_id, id),
  title text not null,
  category text not null default 'general',
  description text not null default '',
  tags text[] not null default '{}',
  style text not null default 'Formal',
  language text not null default 'English',
  use_cases jsonb not null default '[]'::jsonb,
  structure jsonb not null default '[]'::jsonb,
  content text not null default '',
  enabled boolean not null default true,
  version integer not null default 1,
  -- Must match TEMPLATE_EMBEDDING_DIMENSION; changing it needs a new migration.
  embedding public.vector(1536),
  embedding_text text,
  embedding_model text,
  embedding_updated_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists templates_enabled_idx on heywrite.templates (tenant_id, enabled);
create index if not exists templates_category_idx on heywrite.templates (tenant_id, category);
create index if not exists templates_tags_idx on heywrite.templates using gin (tags);
create index if not exists templates_embedding_idx on heywrite.templates
  using ivfflat (embedding public.vector_cosine_ops) with (lists = 100);

create table if not exists heywrite.agent_sessions (
  tenant_id uuid not null default nullif(current_setting('app.tenant_id', true), '')::uuid,
  session_id text not null,
  primary key (tenant_id, session_id),
  active_template_id text,
  current_draft text not null default '',
  style text not null default 'Formal',
  language text not null default 'English',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists heywrite.agent_runs (
  tenant_id uuid not null default nullif(current_setting('app.tenant_id', true), '')::uuid,
  run_id text not null,
  primary key (tenant_id, run_id),
  session_id text not null,
  foreign key (tenant_id, session_id) references heywrite.agent_sessions (tenant_id, session_id),
  action text not null,
  intent text not null,
  selected_template_id text,
  input_state jsonb not null default '{}'::jsonb,
  output_state jsonb not null default '{}'::jsonb,
  trace jsonb not null default '[]'::jsonb,
  reply text not null default '',
  created_at timestamptz not null default now()
);
create index if not exists agent_runs_session_idx on heywrite.agent_runs (tenant_id, session_id);

create table if not exists heywrite.messages (
  tenant_id uuid not null default nullif(current_setting('app.tenant_id', true), '')::uuid,
  message_id bigserial not null,
  primary key (tenant_id, message_id),
  session_id text not null,
  foreign key (tenant_id, session_id) references heywrite.agent_sessions (tenant_id, session_id),
  role text not null,
  content text not null,
  created_at timestamptz not null default now()
);
create index if not exists messages_session_idx on heywrite.messages (tenant_id, session_id);

-- Ledger for the one-off legacy import (import_legacy_tenants.py).
create table if not exists heywrite.schema_imports (
  source_schema text primary key,
  tenant_id uuid not null unique,
  imported_at timestamptz not null default now()
);

-- Embeddings of bundled public templates; contains no tenant data.
create table if not exists heywrite.default_template_vectors (
  template_id text not null,
  content_hash text not null,
  model text not null,
  dimension integer not null,
  embedding jsonb not null,
  primary key (template_id, content_hash, model, dimension)
);

-- RLS uses the transaction-local identity set by the backend. The restrictive
-- guard keeps any future permissive policy from widening access.
do $$
declare
  t text;
  predicate constant text := $p$tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid$p$;
begin
  foreach t in array array['templates', 'agent_sessions', 'agent_runs', 'messages'] loop
    execute format('alter table heywrite.%I enable row level security', t);
    execute format('alter table heywrite.%I force row level security', t);
    execute format('drop policy if exists tenant_access on heywrite.%I', t);
    execute format('drop policy if exists tenant_guard on heywrite.%I', t);
    execute format('create policy tenant_access on heywrite.%I as permissive for all to public using (%s) with check (%s)',
                   t, predicate, predicate);
    execute format('create policy tenant_guard on heywrite.%I as restrictive for all to public using (%s) with check (%s)',
                   t, predicate, predicate);
  end loop;
end
$$;

revoke all on schema heywrite from heywrite_app;
grant usage on schema heywrite to heywrite_app;

revoke all on heywrite.templates, heywrite.agent_sessions, heywrite.agent_runs, heywrite.messages
  from public, heywrite_app;
grant select, insert, update, delete
  on heywrite.templates, heywrite.agent_sessions, heywrite.agent_runs, heywrite.messages
  to heywrite_app;

revoke all on heywrite.schema_imports from public, heywrite_app;

revoke all on heywrite.default_template_vectors from public, heywrite_app;
grant select, insert on heywrite.default_template_vectors to heywrite_app;

revoke all on sequence heywrite.messages_message_id_seq from public, heywrite_app;
grant usage on sequence heywrite.messages_message_id_seq to heywrite_app;
