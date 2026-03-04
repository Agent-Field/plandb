use crate::cli::{parse_event_type, print_json, print_table};
use crate::db::{list_events, Database, EventFilters};
use crate::models::EventType;
use anyhow::{anyhow, Result};
use chrono::NaiveDateTime;
use clap::{Args, Subcommand};
use std::thread;
use std::time::Duration;

#[derive(Args, Debug)]
#[command(about = "List or watch project events.\n\n\
              Events are emitted on every task state change (created, claimed, started, done, failed, etc.).\n\
              Useful for monitoring, building harnesses, and debugging agent coordination.")]
pub struct EventsCommand {
    #[command(subcommand)]
    command: EventsSubcommand,
}

#[derive(Subcommand, Debug)]
enum EventsSubcommand {
    #[command(about = "List past events with optional filters")]
    List(ListEventsArgs),
    #[command(about = "Watch for new events in real-time (polls every 1s)")]
    Watch(WatchEventsArgs),
}

#[derive(Args, Debug)]
struct ListEventsArgs {
    #[arg(long, help = "Project ID")]
    project: String,
    #[arg(long = "type", value_parser = parse_event_type, help = "Filter by event type")]
    event_type: Option<EventType>,
    #[arg(
        long,
        help = "Only show events after this datetime (YYYY-MM-DD HH:MM:SS)"
    )]
    since: Option<String>,
    #[arg(long, help = "Max number of events to return")]
    limit: Option<usize>,
}

#[derive(Args, Debug)]
struct WatchEventsArgs {
    #[arg(long, help = "Project ID")]
    project: String,
    #[arg(long = "type", value_parser = parse_event_type, help = "Filter by event type")]
    event_type: Option<EventType>,
}

pub fn run(db: &Database, command: EventsCommand, json: bool) -> Result<()> {
    match command.command {
        EventsSubcommand::List(args) => list_cmd(db, args, json)?,
        EventsSubcommand::Watch(args) => watch_cmd(db, args, json)?,
    }
    Ok(())
}

fn list_cmd(db: &Database, args: ListEventsArgs, json: bool) -> Result<()> {
    let since = args.since.as_deref().map(parse_datetime).transpose()?;
    let mut events = list_events(
        db,
        EventFilters {
            project_id: Some(args.project),
            task_id: None,
            event_type: args.event_type,
            since,
        },
    )?;

    if let Some(limit) = args.limit {
        if events.len() > limit {
            events = events[events.len() - limit..].to_vec();
        }
    }

    if json {
        print_json(&events)?;
    } else {
        let rows = events
            .iter()
            .map(|e| {
                vec![
                    e.id.to_string(),
                    e.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                    e.event_type.to_string(),
                    e.task_id.clone().unwrap_or_default(),
                    e.agent_id.clone().unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>();
        print_table(&["ID", "TIMESTAMP", "TYPE", "TASK", "AGENT"], &rows);
    }
    Ok(())
}

fn watch_cmd(db: &Database, args: WatchEventsArgs, json: bool) -> Result<()> {
    let mut last_timestamp: Option<NaiveDateTime> = None;
    let mut last_id: i64 = 0;
    loop {
        let events = list_events(
            db,
            EventFilters {
                project_id: Some(args.project.clone()),
                task_id: None,
                event_type: args.event_type.clone(),
                since: last_timestamp,
            },
        )?;

        for event in events {
            if let Some(ts) = last_timestamp {
                if event.timestamp < ts {
                    continue;
                }
                if event.timestamp == ts && event.id <= last_id {
                    continue;
                }
            }

            if json {
                print_json(&event)?;
            } else {
                println!(
                    "{} {} task={} agent={}",
                    event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    event.event_type,
                    event.task_id.clone().unwrap_or_else(|| "-".to_string()),
                    event.agent_id.clone().unwrap_or_else(|| "-".to_string())
                );
            }

            last_timestamp = Some(event.timestamp);
            last_id = event.id;
        }

        thread::sleep(Duration::from_secs(1));
    }
}

fn parse_datetime(input: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S"))
        .map_err(|_| anyhow!("invalid --since datetime, use 'YYYY-MM-DD HH:MM:SS'"))
}
