# Aegis Payments Deployment Guide

## Standard Deployment Procedure

Deployments to aegis-payments go through the standard pipeline: merge to main, automated
tests, canary rollout to five percent of instances, then a full rollout if the canary looks
healthy for fifteen minutes. Do not skip the canary stage even for a change that looks small;
several past incidents involved changes that looked unrelated to payment authorization but
affected it through a shared configuration path.

A deploy is considered complete once every instance in the fleet is running the new version
and GET /health/ready is returning healthy on all of them.

## Health Checks

aegis-payments exposes GET /health/ready, which the load balancer polls to decide whether an
instance should receive traffic. An instance reports not ready during startup until it has
finished warming its connection pool to the upstream card processor and to the idempotency
key store, and it reports not ready again during a graceful shutdown so in-flight requests
can finish before the instance stops receiving new ones.

If a newly deployed instance never becomes ready, it is almost always a configuration
problem specific to that environment, such as a missing credential for the upstream
processor, rather than a code bug, since the same code is already running successfully on
the instances that have not been replaced yet.

## Rollback

### Rollback Procedure

1. Identify the last known-good version from the deployment history.
2. Trigger a rollback deploy to that version through the same pipeline used for a forward
   deploy. Do not hand-edit running instances.
3. The rollback follows the same canary-then-full-rollout shape as a forward deploy, but the
   canary stage can be shortened to five minutes during an active incident, since the target
   version has already run successfully in production before.
4. Confirm every instance is on the rolled-back version and reporting healthy on
   GET /health/ready before considering the rollback complete.

Rolling back aegis-payments does not roll back any database schema migration that shipped
with the version being rolled back from. If the incident is caused by a migration rather than
application code, rolling back the application will not fix it; escalate immediately instead
of retrying the rollback.

### Rollback Verification

After a rollback completes, verify recovery using the same dashboards used during triage:
error rate back to baseline, p99 latency back to baseline, and no continuing
PaymentGatewayTimeoutError spike. Watch for at least fifteen minutes rather than declaring
recovery immediately, since some symptoms take a few minutes to clear even after the bad
version is fully removed from the fleet.

If error rate does not recover after a full rollback, the deployment was not the root cause
and the incident should go back to general triage, including checking the upstream card
processor's own status.

## Canary Deployment Notes

The canary stage exists specifically to catch problems before they reach the full fleet.
A canary that shows any statistically meaningful rise in 5xx responses or in
PaymentGatewayTimeoutError should be treated as a failed canary and rolled back immediately,
even if the absolute numbers look small at five percent of traffic; the same rate applied to
the full fleet is what actually matters.
