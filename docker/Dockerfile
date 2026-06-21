FROM rust:1.91-bookworm AS builder

# `protoc` is required by lance-encoding's build script (and other crates that
# pull in prost-build). `libprotobuf-dev` ships the well-known proto files
# (google/protobuf/empty.proto etc.) that protobuf-compiler alone does not
# include. PROTOC_INCLUDE tells prost-build where to find them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PROTOC_INCLUDE=/usr/include

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/
# `bench/conv-loop` is a workspace member referenced from the root Cargo.toml,
# so its manifest must exist for cargo to load the workspace — even though the
# `greatwheel` binary doesn't depend on it.
COPY bench/conv-loop bench/conv-loop

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
