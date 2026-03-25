use crate::cli::{
    compact_task, minimal_task, print_json, print_table, resolve_project_id,
};
use crate::db::{
    add_dependency, add_note, add_task_files, amend_task_description, approve_task,
    batch_create_tasks, cancel_task, claim_next_task, claim_task, complete_task, compute_effects,
    create_task, fail_task, fuzzy_find_task, get_lookahead, get_running_task_for_agent, get_task,
    insert_task_between, list_dependencies, list_notes, list_tasks, pause_task,
    pivot_subtree, project_state, promote_ready_tasks, remove_dependency, snapshot_task_statuses,
    split_task, start_task, update_heartbeat, update_progress, update_task, Database, TaskListFilters,
};
use crate::models::{
    generate_id, DependencyCondition, DependencyKind, RetryBackoff, Task, TaskKind, TaskStatus,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

use super::helpers::{
    decompose_or_replan, enrich_transition_error, go_payload, parse_dep_arg, parse_files_arg,
    parse_new_subtasks, parse_simple_split, print_task_detail, resolve_agent,
};
use super::{
    BatchYaml, CreateBatchArgs, CreateTaskArgs, DoneArgs, GoArgs, ListTasksArgs, SplitTaskArgs,
    TaskSubcommand, WhatIfCommand, WhatIfSubcommand,
};

/// Execute a task lifecycle hook (pre_hook or post_hook).
/// Sets environment variables and runs the shell command.
/// Prints output to stderr. Never blocks the operation on failure.
fn execute_hook(hook: &str, task: &Task, agent_id: &str) {
    let mut cmd = std::process::Command::new("sh");
    cmd.arg("-c").arg(hook);
    cmd.env("PLANDB_TASK_ID", &task.id);
    cmd.env("PLANDB_PROJECT_ID", &task.project_id);
    cmd.env("PLANDB_TASK_TITLE", &task.title);
    cmd.env("PLANDB_AGENT_ID", agent_id);
    match cmd.output() {
        Ok(output) => {
            if !output.stdout.is_empty() {
                let text = String::from_utf8_lossy(&output.stdout);
                eprint!("{}", text);
            }
            if !output.stderr.is_empty() {
                let text = String::from_utf8_lossy(&output.stderr);
                eprint!("{}", text);
            }
            if !output.status.success() {
                eprintln!(
                    "warning: hook exited with status {} for task {}",
                    output.status, task.id
                );
            }
        }
        Err(e) => {
            eprintln!("warning: failed to execute hook for task {}: {}", task.id, e);
        }
    }
}

pub fn run_subcommand(
    db: &Database,
    subcommand: TaskSubcommand,
    global_json: bool,
    compact: bool,
) -> Result<()> {
    match subcommand {
        TaskSubcommand::Create(args) => create_task_cmd(db, args, global_json, compact)?,
        TaskSubcommand::CreateBatch(args) => create_batch_cmd(db, args, global_json)?,
        TaskSubcommand::List(args) => list_tasks_cmd(db, args, global_json, compact)?,
        TaskSubcommand::Get(args) => {
            let task = fuzzy_find_task(db, &args.task_id, None)?;
            if global_json || args.json {
                if compact {
                    print_json(&compact_task(&task))?;
                } else {
                    print_json(&task)?;
                }
            } else {
                print_task_detail(&task);
            }
        }
        TaskSubcommand::Next(args) => {
            let project_id = resolve_project_id(db, args.project.as_deref())?;
            if args.claim {
                let task = claim_next_task(db, &project_id, &args.agent)?;
                if global_json {
                    print_json(&task)?;
                } else if let Some(task) = task {
                    println!("claimed {} for {}", task.id, args.agent);
                } else {
                    println!("no ready task found");
                }
            } else {
                let tasks = list_tasks(
                    db,
                    TaskListFilters {
                        project_id: Some(project_id),
                        status: Some(TaskStatus::Ready),
                        ..Default::default()
                    },
                )?;
                let next = tasks.first().cloned();
                if global_json {
                    print_json(&next)?;
                } else if let Some(task) = next {
                    println!("next ready task: {} ({})", task.id, task.title);
                } else {
                    println!("no ready task found");
                }
            }
        }
        TaskSubcommand::Go(args) => go_cmd(db, &args, global_json)?,
        TaskSubcommand::Claim(args) => {
            let claimed = claim_task(db, &args.task_id, &args.agent)?;
            if global_json {
                if compact {
                    print_json(&claimed.as_ref().map(minimal_task))?;
                } else {
                    print_json(&claimed)?;
                }
            } else if let Some(task) = claimed {
                println!("claimed {} for {}", task.id, args.agent);
            } else {
                println!("task not claimable (must be ready)");
            }
        }
        TaskSubcommand::Start(args) => {
            let task = start_task(db, &args.task_id)
                .map_err(|err| enrich_transition_error(db, &args.task_id, "start", err))?;
            if global_json {
                if compact {
                    print_json(&minimal_task(&task))?;
                } else {
                    print_json(&task)?;
                }
            } else {
                println!("started {}", task.id);
            }
        }
        TaskSubcommand::Heartbeat(args) => {
            let changed = update_heartbeat(db, &args.task_id)?;
            if global_json {
                print_json(&serde_json::json!({ "updated": changed }))?;
            } else {
                println!("heartbeat updated rows={changed}");
            }
        }
        TaskSubcommand::Progress(args) => {
            if !(0..=100).contains(&args.percent) {
                return Err(anyhow!("--percent must be between 0 and 100"));
            }
            let changed = update_progress(db, &args.task_id, Some(args.percent), args.note)?;
            if global_json {
                print_json(&serde_json::json!({ "updated": changed }))?;
            } else {
                println!("progress updated rows={changed}");
            }
        }
        TaskSubcommand::Done(args) => done_cmd(db, args, global_json, compact)?,
        TaskSubcommand::Fail(args) => {
            let task = fail_task(db, &args.task_id, &args.error)
                .map_err(|err| enrich_transition_error(db, &args.task_id, "fail", err))?;
            if global_json {
                if compact {
                    print_json(&minimal_task(&task))?;
                } else {
                    print_json(&task)?;
                }
            } else {
                println!("failed {}", task.id);
            }
        }
        TaskSubcommand::Cancel(args) => {
            let cancelled = cancel_task(db, &args.task_id, args.cascade)?;
            if global_json {
                print_json(&serde_json::json!({ "cancelled": cancelled }))?;
            } else {
                println!("cancelled rows={cancelled}");
            }
        }
        TaskSubcommand::Approve(args) => {
            let changed = approve_task(db, &args.task_id, "approved", args.by, args.comment)?;
            if global_json {
                print_json(&serde_json::json!({ "updated": changed }))?;
            } else {
                println!("approved rows={changed}");
            }
        }
        TaskSubcommand::AddDep(args) => {
            add_dependency(
                db,
                &args.after,
                &args.to_task,
                args.kind.clone(),
                DependencyCondition::All,
                None,
            )?;
            let to_task = get_task(db, &args.to_task)?;
            if to_task.status == TaskStatus::Ready {
                let from_task = get_task(db, &args.after)?;
                if from_task.status != TaskStatus::Done
                    && from_task.status != TaskStatus::DonePartial
                {
                    let conn = db.lock()?;
                    conn.execute(
                        "UPDATE tasks SET status = 'pending', updated_at = datetime('now') WHERE id = ?1 AND status = 'ready'",
                        rusqlite::params![args.to_task],
                    )?;
                }
            }
            let _ = promote_ready_tasks(db)?;
            if global_json {
                print_json(
                    &serde_json::json!({ "added": true, "from": args.after, "to": args.to_task }),
                )?;
            } else {
                println!("added dependency {} -> {}", args.after, args.to_task);
            }
        }
        TaskSubcommand::RemoveDep(args) => {
            let removed = remove_dependency(db, &args.after, &args.to_task)?;
            let _ = promote_ready_tasks(db)?;
            if global_json {
                print_json(&serde_json::json!({ "removed": removed }))?;
            } else {
                println!(
                    "removed dependency {} -> {} (rows={})",
                    args.after, args.to_task, removed
                );
            }
        }
        TaskSubcommand::Update(args) => {
            let task = update_task(
                db,
                &args.task_id,
                args.title,
                args.description,
                args.kind,
                args.priority,
                None,
            )?;
            if global_json {
                print_json(&task)?;
            } else {
                println!("updated task {} ({})", task.id, task.title);
            }
        }
        TaskSubcommand::Insert(args) => {
            let project_id = resolve_project_id(db, args.project.as_deref())?;
            let before_snapshot = snapshot_task_statuses(db, &project_id)?;
            let created = insert_task_between(
                db,
                &project_id,
                &args.after,
                args.before.as_deref(),
                &args.title,
                args.description,
            )?;
            let after_snapshot = snapshot_task_statuses(db, &project_id)?;
            let effect = compute_effects(db, &project_id, &before_snapshot, &after_snapshot)?;
            let state = project_state(db, &project_id)?;
            if global_json {
                print_json(&serde_json::json!({
                    "id": created.id,
                    "title": created.title,
                    "status": created.status,
                    "effect": effect,
                    "project_state": state,
                }))?;
            } else {
                println!("inserted {}", created.id);
            }
        }
        TaskSubcommand::Amend(args) => {
            let task = amend_task_description(db, &args.task_id, &args.prepend)?;
            if global_json {
                if compact {
                    print_json(&minimal_task(&task))?;
                } else {
                    print_json(&task)?;
                }
            } else {
                println!("amended {}", task.id);
            }
        }
        TaskSubcommand::Pivot(args) => {
            let subtasks = parse_new_subtasks(args.subtasks, args.file)?;
            let parent = get_task(db, &args.parent_id)?;
            let before_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
            let result = pivot_subtree(db, &args.parent_id, args.keep_done, subtasks)?;
            let after_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
            let effect =
                compute_effects(db, &parent.project_id, &before_snapshot, &after_snapshot)?;
            if global_json {
                print_json(&serde_json::json!({
                    "kept": result.kept,
                    "cancelled": result.cancelled,
                    "created": result.created,
                    "effect": effect,
                    "project_state": project_state(db, &parent.project_id)?,
                }))?;
            } else {
                println!("pivoted {}", args.parent_id);
            }
        }
        TaskSubcommand::Split(args) => split_cmd(db, args, global_json)?,
        TaskSubcommand::Decompose(args) => {
            let title_to_id = decompose_or_replan(db, &args.task_id, &args.file, false)?;
            if global_json {
                print_json(&serde_json::json!({
                    "parent_task_id": args.task_id,
                    "subtasks_created": title_to_id.len(),
                    "title_to_id": title_to_id,
                }))?;
            } else {
                println!(
                    "decomposed {} into {} subtasks",
                    args.task_id,
                    title_to_id.len()
                );
                for (title, id) in &title_to_id {
                    println!("  {} -> {}", id, title);
                }
            }
        }
        TaskSubcommand::Replan(args) => {
            let title_to_id = decompose_or_replan(db, &args.task_id, &args.file, true)?;
            if global_json {
                print_json(&serde_json::json!({
                    "parent_task_id": args.task_id,
                    "subtasks_created": title_to_id.len(),
                    "title_to_id": title_to_id,
                }))?;
            } else {
                println!(
                    "replanned {} into {} subtasks",
                    args.task_id,
                    title_to_id.len()
                );
            }
        }
        TaskSubcommand::Pause(args) => {
            let task = pause_task(db, &args.task_id, args.progress, args.note)?;
            if global_json {
                if compact {
                    print_json(&minimal_task(&task))?;
                } else {
                    print_json(&task)?;
                }
            } else {
                println!("paused {}", task.id);
            }
        }
        TaskSubcommand::Note(args) => {
            let note = add_note(db, &args.task_id, args.agent, &args.content)?;
            if global_json {
                print_json(&note)?;
            } else {
                println!("added note {}", note.id);
            }
        }
        TaskSubcommand::Notes(args) => {
            let notes = list_notes(db, &args.task_id)?;
            if global_json {
                print_json(&notes)?;
            } else {
                for note in notes {
                    println!(
                        "{} {} {}",
                        note.created_at.format("%Y-%m-%d %H:%M:%S"),
                        note.agent_id.unwrap_or_else(|| "-".to_string()),
                        note.content
                    );
                }
            }
        }
        TaskSubcommand::Overview(args) => {
            let project_id = resolve_project_id(db, args.project.as_deref())?;
            let tasks = list_tasks(
                db,
                TaskListFilters {
                    project_id: Some(project_id.clone()),
                    ..Default::default()
                },
            )?;
            let deps = {
                let mut all_deps = Vec::new();
                for t in &tasks {
                    let task_deps = list_dependencies(db, &t.id)?;
                    for d in task_deps {
                        all_deps.push(d);
                    }
                }
                let mut seen = std::collections::HashSet::new();
                all_deps.retain(|d| seen.insert((d.from_task.clone(), d.to_task.clone())));
                all_deps
            };

            if global_json || args.json {
                if compact {
                    let mut pending = 0usize;
                    let mut ready = 0usize;
                    let mut claimed = 0usize;
                    let mut running = 0usize;
                    let mut done = 0usize;
                    let mut failed = 0usize;
                    let mut cancelled = 0usize;
                    let mut ready_ids = Vec::new();

                    let compact_tasks = tasks
                        .iter()
                        .map(|t| {
                            match t.status {
                                TaskStatus::Pending => pending += 1,
                                TaskStatus::Ready => {
                                    ready += 1;
                                    ready_ids.push(t.id.clone());
                                }
                                TaskStatus::Claimed => claimed += 1,
                                TaskStatus::Running => running += 1,
                                TaskStatus::Done | TaskStatus::DonePartial => done += 1,
                                TaskStatus::Failed => failed += 1,
                                TaskStatus::Cancelled => cancelled += 1,
                            }
                            serde_json::json!({
                                "id": t.id,
                                "title": t.title,
                                "status": t.status,
                            })
                        })
                        .collect::<Vec<_>>();

                    let compact_edges = deps
                        .iter()
                        .map(|d| serde_json::json!({ "from": d.from_task, "to": d.to_task }))
                        .collect::<Vec<_>>();

                    let total = tasks.len();
                    let progress_pct = if total == 0 {
                        0.0
                    } else {
                        (done as f64 / total as f64) * 100.0
                    };

                    print_json(&serde_json::json!({
                        "summary": {
                            "total": total,
                            "pending": pending,
                            "ready": ready,
                            "claimed": claimed,
                            "running": running,
                            "done": done,
                            "failed": failed,
                            "cancelled": cancelled,
                            "progress_pct": progress_pct,
                        },
                        "ready": ready_ids,
                        "tasks": compact_tasks,
                        "edges": compact_edges,
                    }))?;
                } else {
                    print_json(&serde_json::json!({
                        "tasks": tasks,
                        "dependencies": deps,
                        "total": tasks.len(),
                    }))?;
                }
            } else {
                println!("Project overview: {} tasks", tasks.len());
                for t in &tasks {
                    println!(
                        "  {} {} {} [{}] {}",
                        crate::cli::status_icon(&t.status),
                        t.id,
                        t.title,
                        t.status,
                        t.agent_id.as_deref().unwrap_or("")
                    );
                }
                if !deps.is_empty() {
                    println!("\nDependencies:");
                    for d in &deps {
                        println!("  {} -> {} ({})", d.from_task, d.to_task, d.kind);
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn create_task_cmd(
    db: &Database,
    args: CreateTaskArgs,
    json: bool,
    compact: bool,
) -> Result<()> {
    let now = Utc::now().naive_utc();
    let project_id = resolve_project_id(db, args.project.as_deref())?;
    let task_id = match args.custom_id {
        Some(ref custom) => {
            let name = custom.strip_prefix("t-").unwrap_or(custom);
            if name.is_empty() || name.len() > 64 {
                return Err(anyhow!("custom ID must be 1-64 characters"));
            }
            if !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(anyhow!(
                    "custom ID must contain only alphanumeric, hyphen, or underscore characters"
                ));
            }
            format!("t-{name}")
        }
        None => generate_id("task"),
    };
    let task = Task {
        id: task_id,
        project_id,
        parent_task_id: args.parent,
        is_composite: false,
        title: args.title,
        description: args.description,
        status: TaskStatus::Pending,
        kind: args.kind.unwrap_or(TaskKind::Generic),
        priority: args.priority,
        agent_id: None,
        claimed_at: None,
        started_at: None,
        completed_at: None,
        result: None,
        error: None,
        progress: None,
        progress_note: None,
        max_retries: args.max_retries,
        retry_count: 0,
        retry_backoff: RetryBackoff::Exponential,
        retry_delay_ms: 1000,
        timeout_seconds: args.timeout_seconds,
        heartbeat_interval: 30,
        last_heartbeat: None,
        requires_approval: args.requires_approval,
        approval_status: None,
        approved_by: None,
        approval_comment: None,
        pre_condition: args.pre_condition,
        post_condition: args.post_condition,
        pre_hook: args.pre_hook,
        post_hook: args.post_hook,
        metadata: None,
        created_at: now,
        updated_at: now,
    };

    // Validate all dependency targets exist before creating the task.
    // Without this check, the task would be created but dependency addition
    // would fail with a raw FOREIGN KEY error, leaving an orphaned task.
    let mut parsed_deps = Vec::new();
    for dep in &args.deps {
        let (from_task, kind) = parse_dep_arg(dep)?;
        match get_task(db, &from_task) {
            Ok(_) => parsed_deps.push((from_task, kind)),
            Err(_) => {
                // Try fuzzy match to give a helpful suggestion
                if let Ok(fuzzy) = fuzzy_find_task(db, &from_task, Some(&task.project_id)) {
                    return Err(anyhow!(
                        "dependency task '{}' not found. Did you mean '{}'? ({})",
                        from_task,
                        fuzzy.id,
                        fuzzy.title
                    ));
                }
                return Err(anyhow!(
                    "dependency task '{}' not found. Create it first, then add the dependency.",
                    from_task
                ));
            }
        }
    }

    let created = create_task(db, &task, &args.tags)?;
    for (from_task, kind) in &parsed_deps {
        add_dependency(
            db,
            from_task,
            &created.id,
            kind.clone(),
            DependencyCondition::All,
            None,
        )?;
    }
    let _ = promote_ready_tasks(db)?;

    if json {
        if compact {
            print_json(&serde_json::json!({
                "id": created.id,
                "title": created.title,
                "status": created.status,
            }))?;
        } else {
            print_json(&created)?;
        }
    } else {
        println!("created task {} ({})", created.id, created.title);
    }
    Ok(())
}

fn create_batch_cmd(db: &Database, args: CreateBatchArgs, json: bool) -> Result<()> {
    let content = fs::read_to_string(&args.file)?;
    let parsed: BatchYaml = serde_yaml::from_str(&content)?;
    if parsed.tasks.is_empty() {
        return Err(anyhow!("batch file has no tasks"));
    }
    let project_id = resolve_project_id(db, args.project.as_deref())?;

    let now = Utc::now().naive_utc();
    let mut task_specs = Vec::new();
    let mut id_aliases = HashMap::new();
    for spec in parsed.tasks {
        let id = spec.id.clone().unwrap_or_else(|| generate_id("task"));
        id_aliases.insert(id.clone(), id.clone());
        task_specs.push((id, spec));
    }

    let mut tasks = Vec::new();
    for (id, spec) in &task_specs {
        tasks.push(Task {
            id: id.clone(),
            project_id: project_id.clone(),
            parent_task_id: None,
            is_composite: false,
            title: spec.title.clone(),
            description: spec.description.clone(),
            status: TaskStatus::Pending,
            kind: spec.kind.clone().unwrap_or(TaskKind::Generic),
            priority: spec.priority.unwrap_or(0),
            agent_id: None,
            claimed_at: None,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            progress: None,
            progress_note: None,
            max_retries: 0,
            retry_count: 0,
            retry_backoff: RetryBackoff::Exponential,
            retry_delay_ms: 1000,
            timeout_seconds: None,
            heartbeat_interval: 30,
            last_heartbeat: None,
            requires_approval: false,
            approval_status: None,
            approved_by: None,
            approval_comment: None,
            pre_condition: None,
            post_condition: None,
            pre_hook: None,
            post_hook: None,
            metadata: None,
            created_at: now,
            updated_at: now,
        });
    }

    let inserted = batch_create_tasks(db, &tasks)?;

    for (task_id, spec) in &task_specs {
        if let Some(deps) = &spec.deps {
            for dep in deps {
                let from_task = id_aliases
                    .get(&dep.from)
                    .cloned()
                    .unwrap_or_else(|| dep.from.clone());
                add_dependency(
                    db,
                    &from_task,
                    task_id,
                    dep.kind.clone().unwrap_or(DependencyKind::FeedsInto),
                    DependencyCondition::All,
                    None,
                )?;
            }
        }
        if let Some(tags) = &spec.tags {
            let conn = db.lock()?;
            for tag in tags {
                conn.execute(
                    "INSERT OR IGNORE INTO task_tags(task_id, tag) VALUES (?1, ?2)",
                    rusqlite::params![task_id, tag],
                )?;
            }
        }
    }

    let _ = promote_ready_tasks(db)?;

    if json {
        #[derive(Serialize)]
        struct BatchResult {
            inserted: usize,
            task_ids: Vec<String>,
        }
        let task_ids = tasks.into_iter().map(|t| t.id).collect::<Vec<_>>();
        print_json(&BatchResult { inserted, task_ids })?;
    } else {
        println!("inserted {inserted} tasks from {}", args.file);
    }
    Ok(())
}

pub fn list_tasks_cmd(
    db: &Database,
    args: ListTasksArgs,
    global_json: bool,
    compact: bool,
) -> Result<()> {
    let project_id = resolve_project_id(db, args.project.as_deref())?;
    let mut filters = TaskListFilters {
        project_id: Some(project_id),
        status: args.status,
        kind: args.kind,
        parent_task_id: None,
        agent_id: args.agent,
        tags: Vec::new(),
    };
    if let Some(tag) = args.tag {
        filters.tags.push(tag);
    }
    let tasks = list_tasks(db, filters)?;
    if global_json || args.json {
        if compact {
            let compact_tasks = tasks
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "id": t.id,
                        "title": t.title,
                        "status": t.status,
                        "agent_id": t.agent_id,
                    })
                })
                .collect::<Vec<_>>();
            print_json(&compact_tasks)?;
        } else {
            print_json(&tasks)?;
        }
    } else {
        let rows = tasks
            .iter()
            .map(|t| {
                vec![
                    t.id.clone(),
                    t.title.clone(),
                    format!(
                        "{} {}",
                        crate::cli::status_icon(&t.status),
                        crate::cli::color_task_status(&t.status)
                    ),
                    t.kind.to_string(),
                    t.priority.to_string(),
                    t.agent_id.clone().unwrap_or_default(),
                    t.progress.map(|p| format!("{p}%")).unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>();
        print_table(
            &[
                "ID", "TITLE", "STATUS", "KIND", "PRIORITY", "AGENT", "PROGRESS",
            ],
            &rows,
        );
    }
    Ok(())
}

pub fn go_cmd(db: &Database, args: &GoArgs, json: bool) -> Result<()> {
    let agent = resolve_agent(&args.agent);
    let response = go_payload(db, args.project.as_deref(), &agent)?;

    // Execute pre_hook if the task has one
    if let Some(task_id) = response["task"]["id"].as_str() {
        if let Ok(task) = get_task(db, task_id) {
            if let Some(ref hook) = task.pre_hook {
                execute_hook(hook, &task, &agent);
            }
        }
    }

    if json {
        print_json(&response)?;
    } else if response["task"].is_null() {
        println!("no ready task found");
    } else {
        let task_id = response["task"]["id"].as_str().unwrap_or("");
        let title = response["task"]["title"].as_str().unwrap_or("");
        let remaining = &response["remaining"];
        let done = remaining["done"].as_u64().unwrap_or(0);
        let total = remaining["total"].as_u64().unwrap_or(0);
        let ready = remaining["ready"].as_u64().unwrap_or(0);
        let pending = remaining["pending"].as_u64().unwrap_or(0);

        println!(
            "→ {} \"{}\" [{}/{} · {} ready · {} blocked]",
            task_id, title, done, total, ready, pending
        );

        // Show template indicator if project was created from a template
        if let Ok(Some(template_name)) = crate::db::get_meta(db, "imported_template") {
            eprintln!("  (from template: {} — adapt as needed)", template_name);
        }

        if let Some(handoff) = response["handoff"].as_array() {
            if !handoff.is_empty() {
                eprintln!();
                eprintln!("upstream:");
                for h in handoff {
                    let from_id = h["from_task"].as_str().unwrap_or("");
                    let result = h["result"]
                        .as_str()
                        .or_else(|| {
                            if h["result"].is_null() {
                                None
                            } else {
                                Some("(structured)")
                            }
                        })
                        .unwrap_or("(no result)");
                    let result_display = if result.len() > 120 {
                        format!("{}…", &result[..117])
                    } else {
                        result.to_string()
                    };
                    eprintln!("  {} → {}", from_id, result_display);
                }
            }
        }

        if let Ok(deps) = list_dependencies(db, task_id) {
            let downstream: Vec<_> = deps.iter().filter(|d| d.from_task == task_id).collect();
            if !downstream.is_empty() {
                eprintln!();
                for d in &downstream {
                    let to_title = get_task(db, &d.to_task)
                        .map(|t| t.title)
                        .unwrap_or_default();
                    eprintln!(
                        "downstream: {} \"{}\" (receives YOUR result)",
                        d.to_task, to_title
                    );
                }
            }
        }

        // Show relevant context from project memory (lazy recall)
        if let Some(context) = response["relevant_context"].as_array() {
            if !context.is_empty() {
                eprintln!();
                eprintln!("relevant context:");
                for c in context {
                    let kind = c["kind"].as_str().unwrap_or("?");
                    let content = c["content"].as_str().unwrap_or("");
                    let display = if content.len() > 100 {
                        format!("{}…", &content[..97])
                    } else {
                        content.to_string()
                    };
                    eprintln!("  [{}] {}", kind, display);
                }
            }
        }

        if let Some(pre) = response["task"]["pre_condition"].as_str() {
            eprintln!();
            eprintln!("pre-condition: {}", pre);
        }

        // Contextual action hints — teach the agent what it can do while working
        // Hints are tailored to the task kind for maximum relevance
        if !json {
            let ready = response["remaining"]["ready"].as_u64().unwrap_or(0);
            let kind = response["task"]["kind"].as_str().unwrap_or("generic");
            eprintln!();
            eprintln!("while working:");
            match kind {
                "research" => {
                    eprintln!("  plandb context \"what you discovered\" --kind discovery   # record findings");
                    eprintln!("  plandb context \"decision: X because Y\" --kind decision  # record decisions");
                }
                "code" => {
                    eprintln!("  plandb context \"approach: ...\" --kind decision           # record design choices");
                    eprintln!("  plandb split --into \"A, B, C\"                           # decompose if complex");
                }
                "test" => {
                    eprintln!("  plandb context \"test gap: ...\" --kind bug               # record failures/gaps");
                    eprintln!("  plandb search \"query\"                                   # check known issues");
                }
                "review" => {
                    eprintln!("  plandb context \"finding: ...\" --kind finding             # record review findings");
                    eprintln!("  plandb task amend <id> --prepend \"NOTE: ...\"             # annotate tasks with feedback");
                }
                _ => {
                    eprintln!("  plandb context \"what you discovered\" --kind discovery   # record findings");
                    eprintln!("  plandb split --into \"A, B, C\"                           # decompose if complex");
                }
            }
            eprintln!("  plandb search \"query\"                                   # recall project knowledge");
            eprintln!("when done:");
            eprintln!("  plandb done --next                                       # complete + claim next");
            if ready > 0 {
                eprintln!("  ({} other task(s) ready — can parallelize with PLANDB_AGENT=worker-N plandb go)", ready);
            }
        }
    }
    Ok(())
}

pub fn split_cmd(db: &Database, args: SplitTaskArgs, json: bool) -> Result<()> {
    let parts = parse_simple_split(&args.into);
    if parts.is_empty() {
        return Err(anyhow!("--into produced no parts"));
    }
    // Resolve task_id: explicit or infer from agent's running task
    let split_task_id = match args.task_id {
        Some(id) => id,
        None => {
            let agent = resolve_agent("default");
            let running = get_running_task_for_agent(db, &agent)?;
            match running {
                Some(task) => task.id,
                None => {
                    return Err(anyhow!(
                        "no running task found. Specify task ID explicitly."
                    ))
                }
            }
        }
    };
    let parent = get_task(db, &split_task_id)?;
    let before_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let result = split_task(db, &split_task_id, parts)?;
    let after_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let effect = compute_effects(db, &parent.project_id, &before_snapshot, &after_snapshot)?;
    if json {
        print_json(&serde_json::json!({
            "parent_task_id": result.parent_task_id,
            "created": result.created,
            "done": result.done,
            "title_to_id": result.title_to_id,
            "effect": effect,
            "project_state": project_state(db, &parent.project_id)?,
        }))?;
    } else {
        println!("split {}", split_task_id);
    }
    Ok(())
}

pub fn done_cmd(db: &Database, args: DoneArgs, json: bool, compact: bool) -> Result<()> {
    let agent_id = resolve_agent(args.agent.as_deref().unwrap_or("default"));

    // Resolve task_id: explicit or infer from running task
    let task_id = match args.task_id {
        Some(id) => id,
        None => {
            let running = get_running_task_for_agent(db, &agent_id)?;
            match running {
                Some(task) => task.id,
                None => {
                    return Err(anyhow!(
                        "no running task found for agent '{}'. Specify task ID explicitly.",
                        agent_id
                    ))
                }
            }
        }
    };

    // Execute post_hook before completing the task
    if let Ok(running_task) = get_task(db, &task_id) {
        if let Some(ref hook) = running_task.post_hook {
            execute_hook(hook, &running_task, &agent_id);
        }
    }

    // Capture context atomically with completion (--context flag)
    if let Some(ref context_text) = args.context {
        let project_id = get_task(db, &task_id)?.project_id;
        let entry = crate::db::add_context(
            db,
            &project_id,
            Some(&task_id),
            Some(&agent_id),
            &args.context_kind,
            context_text,
            &[],
        )?;
        if !json {
            eprintln!("{} [{}]", entry.id, entry.kind);
        }
    }

    let result_provided = args.result.is_some();
    let result = match args.result {
        Some(text) => match serde_json::from_str(&text) {
            Ok(v) => Some(v),
            Err(_) => Some(serde_json::Value::String(text)),
        },
        None => None,
    };
    let task = complete_task(db, &task_id, result)
        .map_err(|err| enrich_transition_error(db, &task_id, "complete", err))?;
    if let Some(files) = args.files {
        let paths = parse_files_arg(&files);
        let _ = add_task_files(db, &task.id, &paths)?;
    }
    let promoted = promote_ready_tasks(db)?;

    let has_downstream = list_dependencies(db, &task.id)
        .map(|deps| deps.iter().any(|d| d.from_task == task.id))
        .unwrap_or(false);

    if !result_provided && has_downstream {
        eprintln!(
            "hint: this task has downstream dependents. Consider: plandb done {} --result '<your findings>'",
            task.id
        );
    }

    let state = project_state(db, &task.project_id)?;

    let next = if args.next {
        Some(go_payload(db, Some(&task.project_id), &agent_id)?)
    } else {
        None
    };

    if json {
        if args.next {
            print_json(&serde_json::json!({
                "completed": minimal_task(&task),
                "next": next,
            }))?;
        } else if compact {
            print_json(&minimal_task(&task))?;
        } else {
            print_json(&task)?;
        }
    } else if args.next && next.is_some() {
        let n = next.as_ref().unwrap();
        if n["task"].is_null() {
            println!(
                "✓ {} done [{}/{} · {} ready]",
                task.id, state.done, state.total, state.ready
            );
            println!("no ready task to claim next");
        } else {
            let next_id = n["task"]["id"].as_str().unwrap_or("");
            let next_title = n["task"]["title"].as_str().unwrap_or("");
            println!(
                "✓ {} done → claimed {} \"{}\" [{}/{} · {} ready]",
                task.id, next_id, next_title, state.done, state.total, state.ready
            );
            if let Some(handoff) = n["handoff"].as_array() {
                for h in handoff {
                    let from_id = h["from_task"].as_str().unwrap_or("");
                    let result_str = h["result"].as_str().unwrap_or("(no result)");
                    let display = if result_str.len() > 120 {
                        format!("{}…", &result_str[..117])
                    } else {
                        result_str.to_string()
                    };
                    eprintln!("  upstream: {} → {}", from_id, display);
                }
            }
        }
    } else {
        println!(
            "✓ {} done [{}/{} · {} ready · {} blocked]",
            task.id, state.done, state.total, state.ready, state.pending
        );
        if !promoted.is_empty() {
            eprintln!();
            eprintln!("unlocked:");
            for (id, title) in &promoted {
                eprintln!("  → {} \"{}\"  (now ready)", id, title);
            }
        }
        if !compact {
            eprintln!();
            if !has_downstream {
                eprintln!("hint: no downstream tasks. options:");
                eprintln!("  plandb add \"...\" --dep {}   # add dependent task", task.id);
            }
            eprintln!("next steps:");
            eprintln!("  plandb status --detail              # reassess — does the plan still make sense?");
            eprintln!("  plandb go                           # claim next ready task");
            if state.ready > 1 {
                eprintln!("  ({} tasks ready — parallelize with PLANDB_AGENT=worker-N plandb go)", state.ready);
            }
            if state.pending > 0 {
                eprintln!("  plandb task insert --after {} --before <id> --title \"...\"  # add missed step", task.id);
            }
            // When all tasks are done, suggest exporting as evolved template
            if state.done == state.total && state.total > 0 && state.ready == 0 && state.pending == 0 {
                eprintln!();
                eprintln!("all tasks complete!");
                eprintln!("  plandb export > evolved-template.yaml    # save as reusable template (carries all context)");
            }
        }
    }
    if let Some(post) = &task.post_condition {
        eprintln!("post-condition: {}", post);
        eprintln!("  (verify this is satisfied before moving on)");
    }
    Ok(())
}

pub fn run_what_if(
    db: &Database,
    command: WhatIfCommand,
    global_json: bool,
    _compact: bool,
) -> Result<()> {
    match command.command {
        WhatIfSubcommand::Cancel { task_id } => {
            let task = get_task(db, &task_id)?;
            let project_id = task.project_id.clone();
            let before_snapshot = snapshot_task_statuses(db, &project_id)?;
            {
                let mut conn = db.lock()?;
                let tx = conn.transaction()?;
                tx.execute(
                    "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE id = ?1 AND status NOT IN ('done', 'done_partial')",
                    rusqlite::params![task_id],
                )?;
                tx.rollback()?;
            }
            let mut simulated_after = before_snapshot.clone();
            if let Some(status) = simulated_after.get_mut(&task_id) {
                if !matches!(status, TaskStatus::Done | TaskStatus::DonePartial) {
                    *status = TaskStatus::Cancelled;
                }
            }
            let effect = compute_effects(db, &project_id, &before_snapshot, &simulated_after)?;
            let response = serde_json::json!({
                "action": "cancel",
                "effect": effect,
                "project_state": project_state(db, &project_id)?,
            });
            if global_json {
                print_json(&response)?;
            } else {
                println!("what-if cancel {}", task_id);
            }
        }
        WhatIfSubcommand::Insert {
            after,
            before,
            title,
            project,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let before_snapshot = snapshot_task_statuses(db, &project_id)?;
            {
                let mut conn = db.lock()?;
                let tx = conn.transaction()?;
                let simulated_id = "t-whatif-insert";
                tx.execute(
                    "INSERT INTO tasks (id, project_id, title, status, kind) VALUES (?1, ?2, ?3, 'pending', 'generic')",
                    rusqlite::params![simulated_id, project_id, title],
                )?;
                tx.execute(
                    "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES (?1, ?2, 'feeds_into', 'all', NULL)",
                    rusqlite::params![after, simulated_id],
                )?;
                if let Some(before_task) = before.as_deref() {
                    tx.execute(
                        "DELETE FROM dependencies WHERE from_task = ?1 AND to_task = ?2",
                        rusqlite::params![after, before_task],
                    )?;
                    tx.execute(
                        "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES (?1, ?2, 'feeds_into', 'all', NULL)",
                        rusqlite::params![simulated_id, before_task],
                    )?;
                }
                tx.rollback()?;
            }

            let mut simulated_after = before_snapshot.clone();
            simulated_after.insert("t-whatif-insert".to_string(), TaskStatus::Pending);
            let effect = compute_effects(db, &project_id, &before_snapshot, &simulated_after)?;
            let response = serde_json::json!({
                "action": "insert",
                "effect": effect,
                "project_state": project_state(db, &project_id)?,
            });
            if global_json {
                print_json(&response)?;
            } else {
                println!("what-if insert after {}", after);
            }
        }
    }
    Ok(())
}

pub fn ahead_cmd(
    db: &Database,
    project: Option<String>,
    depth: usize,
    global_json: bool,
    _compact: bool,
) -> Result<()> {
    let project_id = resolve_project_id(db, project.as_deref())?;
    let lookahead = get_lookahead(db, &project_id, depth)?;
    if global_json {
        print_json(&lookahead)?;
    } else {
        println!("current={}", lookahead.current.len());
    }
    Ok(())
}
