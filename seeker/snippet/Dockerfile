#date: 2023-02-27T16:46:21Z
#url: https://api.github.com/gists/0481f2e561098965e4ec63d2ae411888
#owner: https://api.github.com/users/seanwatters

## reference: https://andygrove.io/2020/05/why-musl-extremely-slow/

# Base image extends rust:nightly which extends debian:buster-slim
FROM ballistacompute/rust-cached-deps:0.2.3 as build

# Compile Ballista
RUN rm -rf /tmp/ballista/src/
COPY rust/Cargo.* /tmp/ballista/
COPY rust/build.rs /tmp/ballista/
COPY rust/src/ /tmp/ballista/src/
COPY proto/ballista.proto /tmp/ballista/proto/

# workaround for Arrow 0.17.0 build issue
COPY rust/format/Flight.proto /format

RUN cargo build --release

# Copy the binary into a new container for a smaller docker image
FROM debian:buster-slim

COPY --from=build /tmp/ballista/target/release/executor /
USER root

ENV RUST_LOG=info
ENV RUST_BACKTRACE=full

CMD ["/executor"]

