# Aegis Payments API and Configuration Reference

## Endpoints

### POST /v1/payments

Authorizes and captures a card payment. Requires an `Idempotency-Key` header on every
request; requests without one are rejected. If the same `Idempotency-Key` is reused within
IDEMPOTENCY_TTL_SECONDS of the original request, aegis-payments returns the stored outcome
of that original request instead of processing a new charge.

On success, returns the payment identifier and its status. On upstream failure after
exhausting retries, returns a 502 with an error body naming PaymentGatewayTimeoutError.

### GET /health/ready

Returns 200 when the instance has finished startup and is ready to receive traffic, and a
non-200 status otherwise. Used by the load balancer for routing decisions and by the
deployment pipeline to confirm a rollout or rollback has completed. This endpoint does not
call the upstream card processor; it only reports the instance's own internal readiness.

### GET /v1/payments/{payment_id}

Returns the current status of a previously authorized payment. Does not mutate any state and
is safe to call repeatedly.

## Configuration Keys

`MAX_RETRY_COUNT`

Number of times aegis-payments retries a failed call to the upstream card processor before
giving up and returning PaymentGatewayTimeoutError. Applies per incoming request, not per
process lifetime. Default is 3.

`IDEMPOTENCY_TTL_SECONDS`

How long a stored idempotency key outcome is retained before the key is considered expired
and a reused key is treated as a new payment attempt. Default is 86400, one day.

`UPSTREAM_TIMEOUT_MS`

Per-attempt timeout for a single call to the upstream card processor, in milliseconds. This
is the timeout that, once exceeded, counts as a failed attempt against MAX_RETRY_COUNT.
Default is 4000.

`CONNECTION_POOL_SIZE`

Maximum number of concurrent outbound connections aegis-payments keeps open to the upstream
card processor per instance. Sized too small, this can itself become a source of latency
under load that looks similar to upstream degradation.

## Error Codes

`PaymentGatewayTimeoutError`

Returned when every attempt to reach the upstream card processor for a request failed or
timed out, after exhausting MAX_RETRY_COUNT retries. Surfaced to the caller as a 502.

`IdempotencyKeyMissingError`

Returned when a request to POST /v1/payments does not include an `Idempotency-Key` header.
Surfaced to the caller as a 400.

`PaymentDeclinedError`

Returned when the upstream card processor reaches a decision and that decision is a decline,
as opposed to a timeout or failure to reach the processor at all. This is a normal business
outcome, not an infrastructure failure, and does not count against MAX_RETRY_COUNT.
