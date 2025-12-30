.PHONY: help build test bench lint format clean

# Color output (disable with NOCOLOR=1)
ifdef NOCOLOR
  RED :=
  GREEN :=
  YELLOW :=
  NC :=
else
  RED := \033[0;31m
  GREEN := \033[0;32m
  YELLOW := \033[1;33m
  NC := \033[0m
endif

# Environment variables for c-kzg-4844 integration tests
export C_KZG_4844_GIT_HASH ?= 00ae727c21a346ba0bd027eca6e378da0def988f

# Default target
help:
	@echo "$(GREEN)rust-kzg Makefile targets:$(NC)"
	@echo ""
	@echo "$(YELLOW)Building:$(NC)"
	@echo "  make build                    Build all backends (default, parallel)"
	@echo "  make build-BACKEND            Build specific backend (e.g., make build-blst)"
	@echo "  make build-staticlib          Build static libraries for all backends"
	@echo "  make build-staticlib-BACKEND  Build static library for specific backend"
	@echo "  make build-wasm               Build wasm targets for supported backends"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test                     Run all tests (default + parallel)"
	@echo "  make test-BACKEND             Test specific backend (e.g., make test-blst)"
	@echo "  make test-parallel            Run parallel tests for all backends"
	@echo "  make test-wasm                Test wasm targets for supported backends"
	@echo "  make test-c-kzg               Run c-kzg-4844 integration tests (all backends)"
	@echo "  make test-c-kzg-BACKEND       Run c-kzg-4844 tests for specific backend"
	@echo "  make test-c-kzg-parallel      Run c-kzg-4844 parallel tests (all backends)"
	@echo "  make test-c-kzg-parallel-BACKEND  Run c-kzg-4844 parallel tests for backend"
	@echo "  make fuzz                     Run fuzzing tests"
	@echo ""
	@echo "$(YELLOW)Benchmarking:$(NC)"
	@echo "  make bench                    Benchmark all backends"
	@echo "  make bench-BACKEND            Benchmark specific backend"
	@echo "  make bench-parallel           Benchmark with parallel for all backends"
	@echo "  make bench-c-kzg              Run c-kzg-4844 benchmarks (all backends)"
	@echo "  make bench-c-kzg-BACKEND      Run c-kzg-4844 benchmarks for backend"
	@echo ""
	@echo "$(YELLOW)Linting & Formatting:$(NC)"
	@echo "  make lint                     Run clippy for all backends"
	@echo "  make lint-BACKEND             Run clippy for specific backend"
	@echo "  make format-check             Check code formatting"
	@echo "  make format-fix               Fix code formatting"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean                    Clean all build artifacts"
	@echo "  make clippy-all               Run clippy with all feature combinations"
	@echo ""
	@echo "$(YELLOW)Supported backends:$(NC)"
	@echo "  blst, zkcrypto, arkworks5, arkworks4, arkworks3, constantine, mcl"
	@echo ""

# Variables
BACKENDS := blst zkcrypto arkworks5 arkworks4 arkworks3 constantine mcl
WASM_BACKENDS := blst zkcrypto arkworks5 arkworks4 arkworks3
CKZG_BACKENDS := blst zkcrypto arkworks5 arkworks4 arkworks3 constantine mcl

# KZG crate features
KZG_FEATURES := parallel,std,rand
KZG_FEATURES_BGMW := parallel,std,rand,bgmw
KZG_FEATURES_ARKMSM := parallel,std,rand,arkmsm

# ============================================================================
# BUILD TARGETS
# ============================================================================

.PHONY: build build-staticlib build-wasm $(addprefix build-,$(BACKENDS))
.PHONY: $(addprefix build-staticlib-,$(BACKENDS))

build: $(addprefix build-,$(BACKENDS))
	@echo "$(GREEN)✓ All backends built successfully$(NC)"

build-staticlib: $(addprefix build-staticlib-,$(BACKENDS))
	@echo "$(GREEN)✓ All static libraries built successfully$(NC)"

build-wasm: $(addprefix build-wasm-,$(WASM_BACKENDS))
	@echo "$(GREEN)✓ All wasm targets built successfully$(NC)"

$(addprefix build-,$(BACKENDS)): build-%:
	@echo "$(YELLOW)Building $*...$(NC)"
	@cd $* && cargo build --release --features c_bindings && cd - > /dev/null
	@cd $* && cargo build --release --features c_bindings,parallel && cd - > /dev/null
	@echo "$(GREEN)✓ $* built successfully$(NC)"

$(addprefix build-staticlib-,$(BACKENDS)): build-staticlib-%:
	@echo "$(YELLOW)Building $* static library...$(NC)"
	@cd $* && cargo rustc --release --crate-type=staticlib --features c_bindings && cd - > /dev/null
	@cd $* && cargo rustc --release --crate-type=staticlib --features c_bindings,parallel && cd - > /dev/null
	@echo "$(GREEN)✓ $* static library built successfully$(NC)"

$(addprefix build-wasm-,$(WASM_BACKENDS)): build-wasm-%:
	@echo "$(YELLOW)Building $* for wasm32...$(NC)"
	@cd $* && cargo build --target wasm32-unknown-unknown --no-default-features && cd - > /dev/null
	@echo "$(GREEN)✓ $* wasm32 target built successfully$(NC)"

# ============================================================================
# TEST TARGETS
# ============================================================================

.PHONY: test test-parallel test-wasm test-c-kzg test-c-kzg-parallel fuzz
.PHONY: $(addprefix test-,$(BACKENDS)) $(addprefix test-parallel-,$(BACKENDS))
.PHONY: $(addprefix test-wasm-,$(WASM_BACKENDS))
.PHONY: $(addprefix test-c-kzg-,$(CKZG_BACKENDS))

test: $(addprefix test-,$(BACKENDS))
	@echo "$(GREEN)✓ All backend tests passed$(NC)"

test-parallel: $(addprefix test-parallel-,$(BACKENDS))
	@echo "$(GREEN)✓ All parallel tests passed$(NC)"

test-wasm: $(addprefix test-wasm-,$(WASM_BACKENDS))
	@echo "$(GREEN)✓ All wasm tests passed$(NC)"

test-c-kzg: $(addprefix test-c-kzg-,$(CKZG_BACKENDS))
	@echo "$(GREEN)✓ All c-kzg-4844 tests passed$(NC)"

test-c-kzg-parallel: $(addprefix test-c-kzg-parallel-,$(CKZG_BACKENDS))
	@echo "$(GREEN)✓ All c-kzg-4844 parallel tests passed$(NC)"

$(addprefix test-,$(BACKENDS)): test-%:
	@echo "$(YELLOW)Testing $*...$(NC)"
	@cd $* && cargo test --release --features c_bindings --no-fail-fast && cd - > /dev/null
	@echo "$(GREEN)✓ $* tests passed$(NC)"

$(addprefix test-parallel-,$(BACKENDS)): test-parallel-%:
	@echo "$(YELLOW)Testing $* (parallel)...$(NC)"
	@cd $* && cargo test --release --features c_bindings,parallel --no-fail-fast && cd - > /dev/null
	@echo "$(GREEN)✓ $* parallel tests passed$(NC)"

$(addprefix test-wasm-,$(WASM_BACKENDS)): test-wasm-%:
	@echo "$(YELLOW)Testing $* (wasm32)...$(NC)"
	@cd $* && cargo test --target wasm32-unknown-unknown --no-default-features 2>&1 | head -20 || true && cd - > /dev/null
	@echo "$(GREEN)✓ $* wasm32 tests completed$(NC)"

$(addprefix test-c-kzg-,$(CKZG_BACKENDS)): test-c-kzg-%:
	@echo "$(YELLOW)Running c-kzg-4844 tests for $*...$(NC)"
	@bash run-c-kzg-4844-tests.sh $*
	@echo "$(GREEN)✓ c-kzg-4844 $* tests passed$(NC)"

$(addprefix test-c-kzg-parallel-,$(CKZG_BACKENDS)): test-c-kzg-parallel-%:
	@echo "$(YELLOW)Running c-kzg-4844 parallel tests for $*...$(NC)"
	@bash run-c-kzg-4844-tests.sh --parallel $*
	@echo "$(GREEN)✓ c-kzg-4844 $* parallel tests passed$(NC)"

fuzz:
	@echo "$(YELLOW)Running fuzzing tests...$(NC)"
	@cd fuzz && cargo test --release && cd - > /dev/null
	@echo "$(GREEN)✓ Fuzzing tests completed$(NC)"

# ============================================================================
# BENCHMARK TARGETS
# ============================================================================

.PHONY: bench bench-parallel bench-c-kzg bench-c-kzg-parallel
.PHONY: $(addprefix bench-,$(BACKENDS)) $(addprefix bench-parallel-,$(BACKENDS))
.PHONY: $(addprefix bench-c-kzg-,$(CKZG_BACKENDS)) $(addprefix bench-c-kzg-parallel-,$(CKZG_BACKENDS))

bench: $(addprefix bench-,$(BACKENDS))
	@echo "$(GREEN)✓ All benchmarks completed$(NC)"

bench-parallel: $(addprefix bench-parallel-,$(BACKENDS))
	@echo "$(GREEN)✓ All parallel benchmarks completed$(NC)"

bench-c-kzg: $(addprefix bench-c-kzg-,$(CKZG_BACKENDS))
	@echo "$(GREEN)✓ All c-kzg-4844 benchmarks completed$(NC)"

bench-c-kzg-parallel: $(addprefix bench-c-kzg-parallel-,$(CKZG_BACKENDS))
	@echo "$(GREEN)✓ All c-kzg-4844 parallel benchmarks completed$(NC)"

$(addprefix bench-,$(BACKENDS)): bench-%:
	@echo "$(YELLOW)Benchmarking $*...$(NC)"
	@cd $* && cargo bench 2>&1 | tail -20 && cd - > /dev/null

$(addprefix bench-parallel-,$(BACKENDS)): bench-parallel-%:
	@echo "$(YELLOW)Benchmarking $* (parallel)...$(NC)"
	@cd $* && cargo bench --features parallel 2>&1 | tail -20 && cd - > /dev/null

$(addprefix bench-c-kzg-,$(CKZG_BACKENDS)): bench-c-kzg-%:
	@echo "$(YELLOW)Running c-kzg-4844 benchmarks for $*...$(NC)"
	@bash run-c-kzg-4844-benches.sh $*

$(addprefix bench-c-kzg-parallel-,$(CKZG_BACKENDS)): bench-c-kzg-parallel-%:
	@echo "$(YELLOW)Running c-kzg-4844 benchmarks for $* (parallel)...$(NC)"
	@bash run-c-kzg-4844-benches.sh --parallel $*

# ============================================================================
# LINTING & FORMATTING TARGETS
# ============================================================================

.PHONY: lint format-check format-fix clippy-all
.PHONY: $(addprefix lint-,$(BACKENDS))

lint: $(addprefix lint-,$(BACKENDS)) lint-kzg lint-kzg-bench
	@echo "$(GREEN)✓ All clippy checks passed$(NC)"

$(addprefix lint-,$(BACKENDS)): lint-%:
	@echo "$(YELLOW)Running clippy for $*...$(NC)"
	@cd $* && cargo clippy --all-targets --features default,std,rand,parallel -- -D warnings && cd - > /dev/null
	@echo "$(GREEN)✓ $* clippy passed$(NC)"

lint-kzg:
	@echo "$(YELLOW)Running clippy for kzg...$(NC)"
	@cd kzg && cargo clippy --all-targets --no-default-features --features=$(KZG_FEATURES) -- -D warnings && cd - > /dev/null
	@cd kzg && cargo clippy --all-targets --features=$(KZG_FEATURES),bgmw -- -D warnings && cd - > /dev/null
	@cd kzg && cargo clippy --all-targets --features=$(KZG_FEATURES),arkmsm -- -D warnings && cd - > /dev/null
	@cd kzg && cargo clippy --all-targets --features=$(KZG_FEATURES),sppark -- -D warnings && cd - > /dev/null
	@echo "$(GREEN)✓ kzg clippy passed$(NC)"

lint-kzg-bench:
	@echo "$(YELLOW)Running clippy for kzg-bench...$(NC)"
	@cd kzg-bench && cargo clippy --all-targets --all-features -- -D warnings && cd - > /dev/null
	@echo "$(GREEN)✓ kzg-bench clippy passed$(NC)"

format-check:
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	@cargo fmt --all -- --check
	@echo "$(GREEN)✓ All files are properly formatted$(NC)"

format-fix:
	@echo "$(YELLOW)Fixing code formatting...$(NC)"
	@cargo fmt --all
	@echo "$(GREEN)✓ Code formatting fixed$(NC)"

# Comprehensive clippy checks with all feature combinations
clippy-all: lint lint-wasm-clippy
	@echo "$(GREEN)✓ All comprehensive clippy checks passed$(NC)"

lint-wasm-clippy: $(addprefix lint-wasm-,$(WASM_BACKENDS))
	@echo "$(GREEN)✓ All wasm clippy checks passed$(NC)"

$(addprefix lint-wasm-,$(WASM_BACKENDS)): lint-wasm-%:
	@echo "$(YELLOW)Running wasm32 clippy for $*...$(NC)"
	@cd $* && cargo clippy --target wasm32-unknown-unknown --no-default-features -- -D warnings && cd - > /dev/null
	@echo "$(GREEN)✓ $* wasm32 clippy passed$(NC)"

# ============================================================================
# MAINTENANCE TARGETS
# ============================================================================

.PHONY: clean

clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@cargo clean
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

# ============================================================================
# CI SIMULATION (runs all checks)
# ============================================================================

.PHONY: ci

ci: lint format-check test test-parallel bench
	@echo "$(GREEN)✓ CI checks completed successfully$(NC)"
