use crate::cli::{print_json, print_table};
use crate::db::{create_artifact, list_artifacts, Database};
use crate::models::{generate_id, Artifact};
use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::{Args, Subcommand};
use std::fs;

#[derive(Args, Debug)]
#[command(about = "Attach, read, and list artifacts on tasks.\n\n\
              Artifacts are named outputs attached to tasks — code files, configs, reports, etc.\n\
              Downstream tasks can read artifacts from their dependencies via the handoff protocol.")]
pub struct ArtifactCommand {
    #[command(subcommand)]
    command: ArtifactSubcommand,
}

#[derive(Subcommand, Debug)]
enum ArtifactSubcommand {
    #[command(about = "Write an artifact to a task (from file or inline content)")]
    Write(WriteArtifactArgs),
    #[command(about = "Read an artifact by name from a task")]
    Read(ReadArtifactArgs),
    #[command(about = "List all artifacts attached to a task")]
    List(ListArtifactArgs),
}

#[derive(Args, Debug)]
struct WriteArtifactArgs {
    #[arg(long, help = "Task ID to attach artifact to")]
    task: String,
    #[arg(long, help = "Artifact name (used for retrieval)")]
    name: String,
    #[arg(long, help = "Read artifact content from this file path")]
    file: Option<String>,
    #[arg(long, help = "Inline artifact content (alternative to --file)")]
    content: Option<String>,
    #[arg(long, help = "Artifact kind (e.g. source, config, report)")]
    kind: Option<String>,
    #[arg(long, help = "MIME type (e.g. application/json, text/plain)")]
    mime: Option<String>,
}

#[derive(Args, Debug)]
struct ReadArtifactArgs {
    #[arg(long, help = "Task ID")]
    task: String,
    #[arg(long, help = "Artifact name to read")]
    name: String,
}

#[derive(Args, Debug)]
struct ListArtifactArgs {
    #[arg(long, help = "Task ID to list artifacts for")]
    task: String,
}

pub fn run(db: &Database, command: ArtifactCommand, json: bool) -> Result<()> {
    match command.command {
        ArtifactSubcommand::Write(args) => {
            if args.file.is_none() && args.content.is_none() {
                return Err(anyhow!("provide --file or --content"));
            }
            let content = match (&args.file, &args.content) {
                (_, Some(content)) => Some(content.clone()),
                (Some(path), None) => Some(fs::read_to_string(path)?),
                (None, None) => None,
            };
            let size_bytes = match &args.file {
                Some(path) => Some(fs::metadata(path)?.len() as i64),
                None => content.as_ref().map(|c| c.len() as i64),
            };

            let artifact = Artifact {
                id: generate_id("art"),
                task_id: args.task,
                name: args.name,
                kind: args.kind,
                content,
                path: args.file,
                size_bytes,
                mime_type: args.mime,
                metadata: None,
                created_at: Utc::now().naive_utc(),
            };
            let created = create_artifact(db, &artifact)?;
            if json {
                print_json(&created)?;
            } else {
                println!("artifact {} written", created.id);
            }
        }
        ArtifactSubcommand::Read(args) => {
            let mut artifacts = list_artifacts(db, &args.task)?;
            artifacts.retain(|a| a.name == args.name);
            let Some(artifact) = artifacts.pop() else {
                return Err(anyhow!(
                    "artifact not found: task={} name={}",
                    args.task,
                    args.name
                ));
            };
            if json {
                print_json(&artifact)?;
            } else if let Some(content) = artifact.content {
                println!("{content}");
            } else if let Some(path) = artifact.path {
                println!("artifact path: {path}");
            } else {
                println!("artifact has no content");
            }
        }
        ArtifactSubcommand::List(args) => {
            let artifacts = list_artifacts(db, &args.task)?;
            if json {
                print_json(&artifacts)?;
            } else {
                let rows = artifacts
                    .iter()
                    .map(|a| {
                        vec![
                            a.id.clone(),
                            a.name.clone(),
                            a.kind.clone().unwrap_or_default(),
                            a.mime_type.clone().unwrap_or_default(),
                            a.size_bytes.map(|v| v.to_string()).unwrap_or_default(),
                            a.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                        ]
                    })
                    .collect::<Vec<_>>();
                print_table(&["ID", "NAME", "KIND", "MIME", "SIZE", "CREATED"], &rows);
            }
        }
    }

    Ok(())
}
