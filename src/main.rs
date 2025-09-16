//! tokencount is a fast token counting CLI for GPT-style models.
//!
//! ```sh
//! # default: scan cwd, only *.elm, table output
//! tokencount
//!
//! # scan a directory, include TS + Elm, respect .gitignore
//! tokencount ./frontend --include-ext elm --include-ext ts
//!
//! # top-10 largest by tokens
//! tokencount --top 10
//!
//! # JSON summary for CI
//! tokencount --format json > tokens.json
//!
//! # NDJSON streaming
//! tokencount --format ndjson
//!
//! # sort by tokens desc
//! tokencount --sort tokens
//! ```

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::Serialize;
use thiserror::Error;
use tiktoken_rs::{cl100k_base, o200k_base, CoreBPE};

#[derive(Debug, Parser)]
#[command(name = "tokencount", version, about = "Count GPT tokens across files.", long_about = None)]
struct Args {
    /// Paths to scan (defaults to current directory).
    #[arg(value_name = "PATH", default_value = ".")]
    paths: Vec<PathBuf>,

    /// File extensions to include (can repeat, default: elm).
    #[arg(long = "include-ext", value_name = "EXT", action = ArgAction::Append)]
    include_ext: Vec<String>,

    /// Glob patterns to exclude (e.g. node_modules/**).
    #[arg(long = "exclude", value_name = "GLOB", action = ArgAction::Append)]
    exclude: Vec<String>,

    /// Disable respecting .gitignore files.
    #[arg(long = "no-respect-gitignore", action = ArgAction::SetTrue)]
    no_respect_gitignore: bool,

    /// Follow symlinks when walking.
    #[arg(long = "follow-symlinks", action = ArgAction::SetTrue)]
    follow_symlinks: bool,

    /// Skip files larger than this many bytes.
    #[arg(long = "max-bytes", value_name = "BYTES")]
    max_bytes: Option<u64>,

    /// Encoding/model to use for tokenization.
    #[arg(long = "encoding", value_enum, default_value = "cl100k-base")]
    encoding: Encoding,

    /// Output format to use.
    #[arg(long = "format", value_enum, default_value = "table")]
    format: OutputFormat,

    /// Limit output to the top-N largest files by tokens.
    #[arg(long = "top", value_name = "N")]
    top: Option<usize>,

    /// Suppress warnings.
    #[arg(short = 'q', long = "quiet", action = ArgAction::SetTrue)]
    quiet: bool,

    /// Increase logging verbosity.
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    verbosity: u8,

    /// Sort order for output.
    #[arg(long = "sort", value_enum, default_value = "path")]
    sort: SortBy,

    /// Limit the number of Rayon worker threads.
    #[arg(long = "threads", value_name = "N")]
    threads: Option<usize>,

    /// Emit summary footer in ndjson mode.
    #[arg(long = "with-summary", action = ArgAction::SetTrue)]
    with_summary_flag: bool,

    /// Disable summary footer in ndjson mode.
    #[arg(long = "no-summary", action = ArgAction::SetTrue)]
    no_summary_flag: bool,
}

impl Args {
    fn include_extensions(&self) -> HashSet<String> {
        let mut exts = if self.include_ext.is_empty() {
            vec!["elm".to_string()]
        } else {
            self.include_ext.clone()
        };
        exts.iter_mut().for_each(|ext| {
            if ext.starts_with('.') {
                ext.remove(0);
            }
        });
        exts.into_iter().map(|ext| ext.to_lowercase()).collect()
    }

    fn respect_gitignore(&self) -> bool {
        !self.no_respect_gitignore
    }

    fn with_summary(&self) -> bool {
        if self.no_summary_flag {
            return false;
        }
        if self.with_summary_flag {
            return true;
        }
        true
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
    Ndjson,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SortBy {
    Path,
    Tokens,
}

#[derive(Clone, Debug, Serialize)]
struct FileStat {
    path: String,
    tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
struct Summary {
    files: u64,
    total: u64,
    average: f64,
    p50: u64,
    p90: u64,
    p99: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    top: Option<Vec<FileStat>>, // sorted by tokens desc
}

#[derive(Debug, Error)]
enum ProcessError {
    #[error("failed to read metadata for {path}")]
    Metadata {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("skipping {path}: file size {size} exceeds max {limit}")]
    TooLarge { path: String, size: u64, limit: u64 },
    #[error("skipping {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Clone, Debug, ValueEnum)]
enum Encoding {
    #[value(alias = "cl100k_base")]
    Cl100kBase,
    #[value(alias = "o200k_base")]
    O200kBase,
}

impl Encoding {
    fn load(&self) -> Result<Arc<CoreBPE>> {
        let bpe = match self {
            Encoding::Cl100kBase => cl100k_base()?,
            Encoding::O200kBase => o200k_base()?,
        };
        Ok(Arc::new(bpe))
    }
}

fn init_logging(quiet: bool, verbosity: u8) {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"));
    builder
        .format_module_path(false)
        .format_timestamp(None)
        .format_level(true);

    if quiet {
        builder.filter_level(log::LevelFilter::Off);
    } else {
        let level = match verbosity {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            _ => log::LevelFilter::Debug,
        };
        builder.filter_level(level);
    }

    let _ = builder.try_init();
}

fn main() {
    let args = Args::parse();
    init_logging(args.quiet, args.verbosity);
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to configure rayon thread pool")?;
    }

    let include_exts = args.include_extensions();
    let encoding = args.encoding.load().context("failed to load encoding")?;
    let paths = if args.paths.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        args.paths.clone()
    };

    let exclude_set = build_exclude_globset(args.exclude.clone())?;
    let mut files = Vec::new();

    for root in paths {
        collect_files(&root, &args, &exclude_set, &include_exts, &mut files)?;
    }

    debug!("collected {} candidate files", files.len());

    let stats = count_tokens(files, &args, encoding)?;
    output_results(&stats, &args);
    Ok(())
}

fn build_exclude_globset(mut patterns: Vec<String>) -> Result<Arc<GlobSet>> {
    let defaults = vec![
        "**/.git/**",
        "**/.git",
        "**/target/**",
        "**/target",
        "**/node_modules/**",
        "**/node_modules",
    ];
    for pattern in defaults {
        patterns.push(pattern.to_string());
    }

    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob =
            Glob::new(&pattern).with_context(|| format!("invalid glob pattern: {pattern}"))?;
        builder.add(glob);
    }
    let set = builder.build().context("failed to build glob set")?;
    Ok(Arc::new(set))
}

fn collect_files(
    root: &Path,
    args: &Args,
    excludes: &Arc<GlobSet>,
    include_exts: &HashSet<String>,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    let respect_gitignore = args.respect_gitignore();
    let excludes_for_filter = Arc::clone(excludes);
    let mut builder = WalkBuilder::new(root);
    builder.standard_filters(false);
    builder.follow_links(args.follow_symlinks);

    builder.git_ignore(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.git_exclude(respect_gitignore);

    builder.filter_entry(move |entry| {
        let excludes = &excludes_for_filter;
        if entry.depth() == 0 {
            return true;
        }
        let path = entry.path();
        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
        if excludes.is_match(path) {
            if is_dir {
                debug!("excluding directory {}", path.display());
            }
            return !is_dir;
        }
        true
    });

    for result in builder.build() {
        match result {
            Ok(entry) => {
                if excludes.is_match(entry.path()) {
                    continue;
                }
                if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    continue;
                }
                if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                    if !include_exts.contains(&ext.to_ascii_lowercase()) {
                        continue;
                    }
                } else {
                    continue;
                }
                files.push(entry.into_path());
            }
            Err(err) => {
                if !args.quiet {
                    warn!("walk error: {err}");
                }
            }
        }
    }
    Ok(())
}

fn count_tokens(files: Vec<PathBuf>, args: &Args, encoding: Arc<CoreBPE>) -> Result<Vec<FileStat>> {
    let max_bytes = args.max_bytes;
    let quiet = args.quiet;
    let stats: Vec<FileStat> = files
        .par_iter()
        .filter_map(|path| {
            let encoder = encoding.clone();
            match process_file(path, max_bytes, encoder.as_ref()) {
                Ok(stat) => Some(stat),
                Err(err @ ProcessError::TooLarge { .. }) => {
                    if !quiet {
                        info!("{}", err);
                    }
                    None
                }
                Err(err) => {
                    if !quiet {
                        warn!("{}", err);
                    }
                    None
                }
            }
        })
        .collect();
    Ok(stats)
}

fn process_file(
    path: &Path,
    max_bytes: Option<u64>,
    encoding: &CoreBPE,
) -> std::result::Result<FileStat, ProcessError> {
    let display_path = normalize_display_path(path);
    let metadata = fs::metadata(path).map_err(|source| ProcessError::Metadata {
        path: display_path.clone(),
        source,
    })?;

    if let Some(limit) = max_bytes {
        if metadata.len() > limit {
            return Err(ProcessError::TooLarge {
                path: display_path.clone(),
                size: metadata.len(),
                limit,
            });
        }
    }

    let contents = fs::read_to_string(path).map_err(|source| ProcessError::Read {
        path: display_path.clone(),
        source,
    })?;

    let tokens = encoding.encode_ordinary(&contents);
    Ok(FileStat {
        path: display_path,
        tokens: tokens.len() as u64,
    })
}

fn output_results(stats: &[FileStat], args: &Args) {
    let mut all = stats.to_owned();
    all.sort_by(|a, b| a.path.cmp(&b.path));

    let mut token_sorted = stats.to_owned();
    token_sorted.sort_by(|a, b| b.tokens.cmp(&a.tokens).then_with(|| a.path.cmp(&b.path)));

    let display_stats = if let Some(top) = args.top {
        token_sorted.iter().take(top).cloned().collect::<Vec<_>>()
    } else {
        match args.sort {
            SortBy::Path => all.clone(),
            SortBy::Tokens => token_sorted.clone(),
        }
    };

    let mut ordered = display_stats;
    if args.top.is_some() {
        match args.sort {
            SortBy::Path => ordered.sort_by(|a, b| a.path.cmp(&b.path)),
            SortBy::Tokens => {
                ordered.sort_by(|a, b| b.tokens.cmp(&a.tokens).then_with(|| a.path.cmp(&b.path)))
            }
        }
    }

    let summary = build_summary(
        stats,
        args.top
            .map(|n| token_sorted.iter().take(n).cloned().collect::<Vec<_>>()),
    );

    match args.format {
        OutputFormat::Table => print_table(&ordered, &summary),
        OutputFormat::Json => print_json(&ordered, &summary),
        OutputFormat::Ndjson => print_ndjson(&ordered, &summary, args.with_summary()),
    }
}

fn build_summary(all_stats: &[FileStat], top: Option<Vec<FileStat>>) -> Summary {
    let files = all_stats.len() as u64;
    let total: u64 = all_stats.iter().map(|s| s.tokens).sum();
    let average = if files > 0 {
        total as f64 / files as f64
    } else {
        0.0
    };
    let mut counts: Vec<u64> = all_stats.iter().map(|s| s.tokens).collect();
    counts.sort_unstable();

    Summary {
        files,
        total,
        average,
        p50: percentile(&counts, 0.50),
        p90: percentile(&counts, 0.90),
        p99: percentile(&counts, 0.99),
        top,
    }
}

fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percentile * (sorted.len() as f64)).ceil().max(1.0) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn print_table(stats: &[FileStat], summary: &Summary) {
    let width = stats
        .iter()
        .map(|s| num_digits(s.tokens))
        .max()
        .unwrap_or(1);

    for stat in stats {
        println!("{:>width$}  {}", stat.tokens, stat.path, width = width);
    }

    println!("\n---");
    println!("total files: {}", summary.files);
    println!("total tokens: {}", summary.total);
    println!("average/file: {:.2}", summary.average);
    println!("p50: {}", summary.p50);
    println!("p90: {}", summary.p90);
    println!("p99: {}", summary.p99);
    if let Some(top) = &summary.top {
        println!("top files:");
        for stat in top {
            println!("  {} ({})", stat.path, stat.tokens);
        }
    }
}

fn print_json(stats: &[FileStat], summary: &Summary) {
    let mut rows: Vec<serde_json::Value> = stats
        .iter()
        .map(|stat| {
            serde_json::json!({
                "path": stat.path,
                "tokens": stat.tokens,
            })
        })
        .collect();
    rows.push(serde_json::json!({ "summary": summary }));

    match serde_json::to_string_pretty(&rows) {
        Ok(json) => println!("{}", json),
        Err(err) => eprintln!("failed to serialize json: {err}"),
    }
}

fn print_ndjson(stats: &[FileStat], summary: &Summary, with_summary: bool) {
    for stat in stats {
        match serde_json::to_string(stat) {
            Ok(json) => println!("{}", json),
            Err(err) => eprintln!("failed to serialize ndjson row: {err}"),
        }
    }

    if with_summary {
        match serde_json::to_string(&serde_json::json!({ "summary": summary })) {
            Ok(json) => println!("{}", json),
            Err(err) => eprintln!("failed to serialize ndjson summary: {err}"),
        }
    }
}

fn num_digits(mut value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    let mut count = 0;
    while value > 0 {
        value /= 10;
        count += 1;
    }
    count
}

fn normalize_display_path(path: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(stripped) = path.strip_prefix(&cwd) {
            let display = stripped.to_string_lossy();
            return if display.is_empty() {
                String::from(".")
            } else {
                display.into_owned()
            };
        }
    }
    if let Ok(stripped) = path.strip_prefix(Path::new(".")) {
        let display = stripped.to_string_lossy();
        if display.is_empty() {
            return String::from(".");
        }
        return display.into_owned();
    }
    path.to_string_lossy().into_owned()
}
