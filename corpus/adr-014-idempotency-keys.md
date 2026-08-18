# ADR-014: Idempotency Keys for Payment Retries

## Status

Accepted. Implemented in aegis-payments as of the 2025 Q3 release.

## Context

Checkout clients retry POST requests to aegis-payments when a request times out or when the
client receives a connection error before a response arrives. From the client's point of
view, it cannot tell whether the original request reached the server and was processed, or
whether it never arrived at all.

Before this decision, a retried POST to the payment authorization endpoint could result in
the card being charged twice: once for the original request that actually succeeded but
whose response was lost, and once for the retry that the client sent believing the first
attempt had failed. This happened rarely, but every occurrence required a manual refund and
damaged trust with the merchant on whose behalf the charge was made.

We considered this a payments correctness issue, not just a reliability nuisance, because a
duplicate charge is a direct financial impact on a customer.

## Decision

Every payment authorization request must include a client-generated idempotency key. Aegis
payments stores the outcome of the first request seen for a given idempotency key and
returns that same outcome for any subsequent request with the same key, without repeating
the charge against the upstream card processor.

The stored outcome is retained for IDEMPOTENCY_TTL_SECONDS after the first request, after
which the key is considered expired and a request reusing it is treated as a new payment
attempt. This bounds the amount of state aegis-payments has to retain per key while still
covering the realistic retry window a checkout client would use.

Idempotency keys are scoped per merchant. Two different merchants using the same idempotency
key value are treated as entirely unrelated requests.

## Consequences

### Positive

A retried request with the same idempotency key is now guaranteed not to double-charge the
customer, regardless of how many times the client retries or how long the client waits
between retries, as long as it retries within IDEMPOTENCY_TTL_SECONDS.

Client implementations become simpler: a client can retry aggressively on any ambiguous
failure without needing its own logic to detect whether the original request actually
succeeded.

### Negative

Aegis payments must persist idempotency key state, which adds a storage dependency and a
cleanup job for expired keys. If that storage becomes unavailable, the service must decide
whether to fail closed and reject new payments, or fail open and risk a duplicate charge; the
current implementation fails closed.

Clients that generate a new idempotency key on every retry, instead of reusing the key from
the original attempt, defeat the protection entirely. This has happened in practice with one
integration and required a conversation with that team rather than a code fix.

## Alternatives Considered

Deduplicating by request body hash instead of an explicit key was rejected because two
legitimately different payment attempts for the same amount, from the same customer, in
quick succession, are not rare, and hashing the body would have merged them incorrectly.

Relying on the upstream card processor's own idempotency support was rejected as the sole
mechanism because aegis-payments talks to more than one processor depending on region, and
we did not want payment correctness to depend on every current and future processor
implementing idempotency consistently.
