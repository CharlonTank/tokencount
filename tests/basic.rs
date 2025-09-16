use std::fs;
use std::process::Command;

use anyhow::Result;
use assert_cmd::prelude::*;
use serde_json::Value;
use tempfile::TempDir;
use tiktoken_rs::cl100k_base;

#[test]
fn counts_tokens_for_known_input() -> Result<()> {
    let dir = TempDir::new()?;
    let file_path = dir.path().join("Main.elm");
    fs::write(&file_path, "hello world\n")?;

    let output = Command::cargo_bin("tokencount")?
        .current_dir(dir.path())
        .args(["--format", "json"])
        .output()?;

    assert!(output.status.success(), "CLI failed: {:?}", output);

    let rows: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    let file_row = rows
        .iter()
        .find(|row| row.get("path").is_some())
        .expect("expected file row");
    let tokens = file_row
        .get("tokens")
        .and_then(Value::as_u64)
        .expect("expected tokens field");

    let bpe = cl100k_base()?;
    let expected = bpe.encode_ordinary("hello world\n").len() as u64;
    assert_eq!(tokens, expected);

    Ok(())
}

#[test]
fn include_extension_filtering() -> Result<()> {
    let dir = TempDir::new()?;
    fs::write(dir.path().join("Only.elm"), "one")?;
    fs::write(dir.path().join("Extra.ts"), "two")?;

    let default_output = Command::cargo_bin("tokencount")?
        .current_dir(dir.path())
        .args(["--format", "json"])
        .output()?;
    assert!(
        default_output.status.success(),
        "default scan failed: {:?}",
        default_output
    );
    let rows: Vec<Value> = serde_json::from_slice(&default_output.stdout)?;
    let files: Vec<&str> = rows
        .iter()
        .filter_map(|row| row.get("path").and_then(Value::as_str))
        .collect();
    assert_eq!(files, vec!["Only.elm"]);

    let expanded_output = Command::cargo_bin("tokencount")?
        .current_dir(dir.path())
        .args([
            "--format",
            "json",
            "--include-ext",
            "elm",
            "--include-ext",
            "ts",
        ])
        .output()?;
    assert!(
        expanded_output.status.success(),
        "expanded scan failed: {:?}",
        expanded_output
    );
    let rows: Vec<Value> = serde_json::from_slice(&expanded_output.stdout)?;
    let mut files: Vec<&str> = rows
        .iter()
        .filter_map(|row| row.get("path").and_then(Value::as_str))
        .collect();
    files.sort();
    assert_eq!(files, vec!["Extra.ts", "Only.elm"]);

    Ok(())
}

#[test]
fn json_summary_contains_stats() -> Result<()> {
    let dir = TempDir::new()?;
    fs::write(dir.path().join("A.elm"), "alpha")?;
    fs::write(dir.path().join("B.elm"), "beta")?;

    let output = Command::cargo_bin("tokencount")?
        .current_dir(dir.path())
        .args(["--format", "json"])
        .output()?;
    assert!(output.status.success(), "json scan failed: {:?}", output);
    let rows: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    let summary = rows
        .last()
        .and_then(|row| row.get("summary"))
        .expect("summary row");

    for key in ["files", "total", "average", "p50", "p90", "p99"] {
        assert!(summary.get(key).is_some(), "missing {key}");
    }

    Ok(())
}
