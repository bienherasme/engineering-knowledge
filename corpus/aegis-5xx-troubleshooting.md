# Troubleshooting Aegis 5xx Errors

## Overview

This guide covers the recurring causes of 5xx responses from aegis-payments and how to tell
them apart quickly. It assumes you have already checked whether a deployment happened
recently; if so, start with the incident runbook's rollback guidance before working through
the diagnostics below.

## PaymentGatewayTimeoutError

### Symptoms

Requests to the payment authorization endpoint return a 502 with an error body naming
PaymentGatewayTimeoutError. The error rate usually rises sharply rather than gradually, and
p99 latency on the same endpoint rises alongside it, often past the configured upstream
timeout.

### Likely Causes

The most common cause is genuine latency or an outage on the upstream card processor.
Aegis payments enforces its own timeout on the call to the processor, and once that timeout
is reached it raises PaymentGatewayTimeoutError rather than waiting indefinitely.

A less common cause is a local network problem between aegis-payments and the processor,
such as a DNS resolution failure or a saturated outbound connection pool. This tends to
affect only a subset of instances rather than the whole fleet, which is the key signal to
distinguish it from a genuine upstream outage.

### Diagnostic Steps

Check the upstream card processor's own status page and their reported latency, if
available. If they are reporting a known issue, this is very likely the cause and the fix is
outside aegis-payments.

Compare error rate across instances. If only some instances are affected, suspect the local
network path from those instances rather than the upstream processor.

Check whether MAX_RETRY_COUNT retries against the processor are amplifying the problem. Each
retry after a timeout adds further load to an already slow upstream, which can turn a
partial degradation into a full outage. See Retry Storms below.

## Elevated Latency on Card Processing

A rise in p95 or p99 latency on the payment authorization endpoint, without an accompanying
rise in error rate, usually means the upstream card processor has slowed down without
failing outright. Requests are succeeding, just more slowly.

Check the same upstream status signals described above. If the upstream is confirmed slow
but not failing, consider whether checkout's own client-side timeout is shorter than aegis
payments' timeout; if so, checkout may be abandoning requests that would have succeeded,
which looks like a client-side failure even though aegis-payments itself never returned an
error.

## Retry Storms and MAX_RETRY_COUNT

MAX_RETRY_COUNT controls how many times aegis-payments retries a failed call to the upstream
card processor before giving up and returning PaymentGatewayTimeoutError to the caller. It
exists to smooth over brief, transient upstream blips without surfacing an error to checkout.

When the upstream is genuinely degraded rather than blipping, MAX_RETRY_COUNT retries per
request multiply the load aegis-payments sends to an already struggling processor, which can
turn a partial upstream slowdown into a hard outage. If you suspect this is happening,
confirm with the upstream status page before considering a temporary reduction to
MAX_RETRY_COUNT; do not change it reactively without that confirmation, since a lower value
also means genuinely transient blips will surface as errors more often.

## When to Escalate

Escalate to the payments platform lead if the cause is not identified within thirty minutes,
if the upstream card processor is confirmed degraded and there is no clear mitigation, or if
the symptoms do not match any pattern in this guide.
