# Deployment Checklist

## Pre-Deployment Validation Steps
1. Verify that all code changes are merged into the main branch.
2. Ensure that the deployment environment is configured correctly.
3. Check for any outstanding issues in the issue tracker that must be resolved before deployment.
4. Review the change logs for any significant changes that need to be communicated.
5. Confirm with the team that all relevant documentation and training materials are up-to-date.

## Build Verification
1. Trigger continuous integration (CI) pipelines to ensure all builds pass.
2. Review build logs for warnings or errors.
3. Verify that all unit tests pass successfully.
4. Ensure that integration tests are run against a staging environment.

## Data Pipeline Checks
1. Validate that all data migration scripts are tested in a development environment.
2. Ensure that proper backups are taken before data modification.
3. Check data integrity through validation rules post-migration.
4. Monitor data flow for any disruptions or anomalies post-deployment.
5. For local validation, run `npm run gen:data` (or `python scripts/generate_dashboard_data.py --data-dir public/data`) before `npm run dev` or `npm run build`.
6. Confirm `public/data/dashboard_payload.json`, `public/data/dashboard_state.json`, and `public/data/tables.json` exist after generation.
7. If typo files such as `dashoard_payload.json` or `dashoard_state.json` exist, rename/move them to `public/data/dashboard_payload.json` and `public/data/dashboard_state.json`, then regenerate.
8. Verify GitHub Actions deploy run triggered and completed the upstream `Basketball_prediction` pipeline before the dashboard build step.

## Testing Procedures
1. Conduct manual testing for major functionalities in the staging environment.
2. Perform automated regression tests.
3. Validate the user acceptance testing (UAT) results from stakeholders.
4. Check compatibility with various browsers and devices (if applicable).

## Post-Deployment Smoke Tests
1. Verify that the application is running without errors after deployment.
2. Check key functionalities to confirm they are operational.
3. Ensure that all services and integrations are functional,
4. Monitor application logs for any unexpected behavior during the initial hours post-deployment.

# Last Updated: 2026-03-26 21:01:07 UTC
