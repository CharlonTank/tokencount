# tokencount

`tokencount` is a blazing-fast CLI for counting GPT-style tokens across your project. It walks your tree in parallel, respects `.gitignore`, and reports per-file counts with totals and percentiles so you can plan prompts, costs, and chunking strategies.

## Features

- Parallel file scanning using Rayon
- Ignore handling powered by the `ignore` crate with `.gitignore` respected by default
- UTF-8 safe token counting via [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs)
- Flexible filtering: include extensions, exclude globs, follow symlinks, size limits
- Multiple output formats: table, JSON, NDJSON streaming
- Summary statistics with totals, averages, and P50/P90/P99 percentiles

## Installation

```bash
cargo install tokencount
```

Or install from source:

```bash
cargo install --path .
```

## Usage

```bash
# default: scan current directory, include only *.elm
 tokencount

# scan a project and include Elm + TypeScript files
 tokencount ./frontend --include-ext elm --include-ext ts

# return the top 10 files by token count
 tokencount --top 10

# emit JSON summary (great for CI)
 tokencount --format json > tokens.json

# stream NDJSON for downstream processing
 tokencount --format ndjson

# sort by token count descending
 tokencount --sort tokens
```

### CLI Options

Run `tokencount --help` for the full list of flags, including:

- `--include-ext` / `--exclude`
- `--max-bytes`
- `--encoding cl100k_base|o200k_base`
- `--format table|json|ndjson`
- `--top N`
- `--sort path|tokens`
- `--threads N`
- `--follow-symlinks`
- `--no-respect-gitignore`
- `-v/--verbose`, `-q/--quiet`

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

The project is dual-licensed under MIT or Apache-2.0.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in `tokencount` by you shall be dual licensed as above, without any additional terms or conditions.
