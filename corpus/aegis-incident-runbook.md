# Aegis Payments Incident Runbook

This runbook covers on-call response for aegis-payments, the service that authorizes and
captures card payments on behalf of checkout. It assumes you already have access to the
service dashboards and the deploy tooling. If you are paged for aegis-payments and this is
your first incident on the service, read the whole runbook once before acting.

## Severity Classification

Sev1: checkout cannot complete payments for a majority of traffic. Page immediately and
declare an incident channel.

Sev2: a meaningful minority of payment attempts are failing or timing out, or latency has
degraded enough that customers are abandoning checkout. Page during business hours,
escalate after hours only if the error rate keeps climbing.

Sev3: an isolated or low-volume issue with a clear workaround. Track it, do not page.

## Immediate Response

### Triage Steps

1. Check the aegis-payments dashboard for error rate and p99 latency over the last 30
   minutes.
2. Check whether a deployment happened in the last two hours. Recent deploys are the most
   common cause of a sudden change in error rate.
3. Check GET /health/ready on at least two instances behind the load balancer. If instances
   are reporting not ready, traffic is likely being routed to a shrinking healthy pool.
4. Check upstream card processor status. Aegis payments depends on an external gateway for
   authorization, and gateway degradation looks similar to an aegis-payments outage from the
   dashboards alone.

### Escalation

If triage points at a recent deployment, move directly to the rollback procedure in the
deployment guide rather than debugging the new code under pressure. Escalate to the payments
platform lead if the incident is Sev1 for more than fifteen minutes, or if the cause is not
narrowed down within thirty minutes.

## Common Incident Patterns

### Elevated 5xx Responses

A sustained rise in 5xx responses from aegis-payments after a deploy is the single most
common incident pattern. It is usually caused by a configuration change, a schema migration
that has not finished, or a bug in the new code path. Roll back first, investigate second,
unless the rollback itself is unsafe for a reason specific to that deploy.

A rise in 5xx responses without a recent deploy usually points at the upstream card
processor or at database connection exhaustion. Check the troubleshooting guide for the
diagnostic steps specific to each of those.

### Payment Gateway Timeouts

If the dashboards show a spike in PaymentGatewayTimeoutError, the service is waiting too
long for a response from the upstream card processor. This is not always aegis-payments'
fault: it can be genuine upstream latency. See the troubleshooting guide for how to tell the
difference between a slow upstream and a local configuration problem, and how MAX_RETRY_COUNT
interacts with a slow upstream to make an incident worse.

## Rollback During an Incident

If a deployment is implicated, follow the rollback procedure documented in the deployment
guide. Do not attempt an ad hoc rollback outside that procedure during an active incident;
the documented procedure exists specifically because an incorrect rollback has caused
secondary incidents before.

After a rollback, confirm recovery using the same rollback verification steps described in
the deployment guide, not just a glance at the dashboard. Error rate can look recovered for a
few minutes before a slower-moving symptom reappears.

## Postmortem Requirements

Every Sev1 and every Sev2 that lasted more than fifteen minutes requires a postmortem within
three business days. The postmortem should reference the specific triage steps that worked
or failed, not just a narrative summary, so this runbook can be improved from real incidents.
