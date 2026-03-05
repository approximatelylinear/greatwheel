use gw_core::{OrgId, RateLimitConfig, Task, UserId};

/// Task scheduler with per-org concurrency and rate limiting.
pub struct Scheduler {
    // TODO: task queue, concurrency limiter, sqlx pool
}

impl Scheduler {
    pub fn new() -> Self {
        Self {}
    }
}

/// Rate limiter enforcing soft + hard token budgets.
pub struct RateLimiter {
    // TODO: per-org/user counters
}

/// Result of a rate limit check.
pub enum RateLimitResult {
    Ok,
    SoftLimitExceeded { used: u64, soft_limit: u64 },
    HardLimitExceeded { used: u64, hard_limit: u64 },
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {}
    }
}
