.PHONY: all run-dev release-web release
all: run-dev
run-dev:
	RUSTC_WRAPPER=sccache cargo run
release-web:
	wasm-pack build --target web
release:
	cargo build --release
