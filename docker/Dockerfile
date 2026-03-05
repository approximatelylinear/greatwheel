FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/

RUN cargo build --release --bin greatwheel

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/greatwheel /usr/local/bin/greatwheel
COPY config/ /etc/greatwheel/
COPY migrations/ /app/migrations/

EXPOSE 8080 8081

ENTRYPOINT ["greatwheel"]
CMD ["--config", "/etc/greatwheel/greatwheel.toml"]
