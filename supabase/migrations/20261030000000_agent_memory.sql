-- Cross-session memory and incremental summaries remain private to each tenant.
create table if not exists heywrite.memory_profiles (
  tenant_id uuid primary key default nullif(current_setting('app.tenant_id', true), '')::uuid,
  enabled boolean not null default true,
  facts jsonb not null default '[]'::jsonb check (jsonb_typeof(facts) = 'array' and jsonb_array_length(facts) <= 20),
  revision bigint not null default 0,
  updated_at timestamptz not null default now()
);

create table if not exists heywrite.session_summaries (
  tenant_id uuid not null default nullif(current_setting('app.tenant_id', true), '')::uuid,
  session_id text not null,
  primary key (tenant_id, session_id),
  foreign key (tenant_id, session_id) references heywrite.agent_sessions (tenant_id, session_id) on delete cascade,
  summary text not null default '' check (char_length(summary) <= 4000),
  through_message_id bigint not null default 0,
  through_offset integer not null default 0 check (through_offset >= 0),
  updated_at timestamptz not null default now()
);

do $$
declare
  t text;
  predicate constant text := $p$tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid$p$;
begin
  foreach t in array array['memory_profiles', 'session_summaries'] loop
    execute format('alter table heywrite.%I enable row level security', t);
    execute format('alter table heywrite.%I force row level security', t);
    execute format('drop policy if exists tenant_access on heywrite.%I', t);
    execute format('drop policy if exists tenant_guard on heywrite.%I', t);
    execute format('create policy tenant_access on heywrite.%I as permissive for all to public using (%s) with check (%s)', t, predicate, predicate);
    execute format('create policy tenant_guard on heywrite.%I as restrictive for all to public using (%s) with check (%s)', t, predicate, predicate);
    execute format('revoke all on heywrite.%I from public, heywrite_app', t);
    execute format('grant select, insert, update, delete on heywrite.%I to heywrite_app', t);
  end loop;
end
$$;
