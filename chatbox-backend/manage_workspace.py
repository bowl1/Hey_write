"""Maintenance of a shared-table workspace using the restricted runtime role."""
import argparse
from template_store import seed_templates_from_files
from tenancy import tenant_context, ensure_workspace

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', required=True, help='Confirmed Supabase Auth user UUID')
    parser.add_argument('--seed-templates', action='store_true')
    args = parser.parse_args()
    with tenant_context(args.user_id):
        ensure_workspace()
        if args.seed_templates:
            seed_templates_from_files(embed=True)
    print('Workspace ready; no per-user schema is needed. Legacy imports use migrate_shared.py.')
