use crate::cli::{
    compact_task, minimal_task, parse_dependency_kind, parse_task_kind, parse_task_status,
    print_json, print_table, resolve_project_id,
};
use crate::db::{
    add_dependency, add_note, add_task_files, amend_task_description, approve_task,
    batch_create_tasks, cancel_task, check_file_conflicts, claim_next_task, claim_task,
    complete_task, compute_effects, create_task, fail_task, fuzzy_find_task, get_handoff_context,
    get_lookahead, get_task, insert_task_between, list_dependencies, list_notes, list_task_files,
    list_tasks, pause_task, pivot_subtree, project_state, promote_ready_tasks, remove_dependency,
    snapshot_task_statuses, split_task, start_task, update_heartbeat, update_progress, update_task,
    Database, NewSubtask, PlandbError, SplitPart, TaskListFilters,
};
use crate::models::{
    generate_id, DependencyCondition, DependencyKind, RetryBackoff, Task, TaskKind, TaskStatus,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::{Args, Subcommand};
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

#[derive(Args, Debug)]
#[command(about = "Manage tasks within a project.\n\n\
              LIFECYCLE: create → [ready] → go/claim → [running] → done/fail\n\
              Dependencies control when tasks become ready. Only tasks with all deps done can be claimed.\n\n\
              CORE LOOP (2 commands):\n\
              \x20 plandb go --agent NAME        Claim + start next ready task\n\
              \x20 plandb done ID --next --agent  Complete current + claim next\n\n\
              PLAN ADAPTATION (mid-flight changes):\n\
              \x20 plandb task insert    Add a step between existing tasks\n\
              \x20 plandb task amend     Prepend context to a future task's description\n\
              \x20 plandb task pivot     Replace a subtree with new tasks\n\
              \x20 plandb task split     Break one task into multiple sub-tasks\n\
              \x20 plandb task decompose Break a task into subtasks from a YAML file\n\
              \x20 plandb task replan    Cancel pending subtasks and create new ones from YAML")]
pub struct TaskCommand {
    #[command(subcommand)]
    command: TaskSubcommand,
}

#[derive(Subcommand, Debug)]
enum TaskSubcommand {
    #[command(about = "Create a new task in a project")]
    Create(CreateTaskArgs),
    #[command(
        name = "create-batch",
        about = "Create multiple tasks from a YAML file"
    )]
    CreateBatch(CreateBatchArgs),
    #[command(about = "List tasks with optional filters (status, kind, tag, agent)")]
    List(ListTasksArgs),
    #[command(about = "Get full details of a single task (supports fuzzy ID matching)")]
    Get(GetTaskArgs),
    #[command(about = "Show or claim the next ready task for an agent")]
    Next(NextTaskArgs),
    #[command(
        about = "Claim + start the next ready task in one command (preferred agent entry point)"
    )]
    Go(GoArgs),
    #[command(about = "Claim a specific task by ID for an agent")]
    Claim(ClaimTaskArgs),
    #[command(about = "Transition a claimed task to running status")]
    Start(TaskIdArg),
    #[command(about = "Update heartbeat timestamp (proves agent is still working)")]
    Heartbeat(TaskIdArg),
    #[command(about = "Report progress percentage (0-100) on a running task")]
    Progress(ProgressArgs),
    #[command(
        about = "Mark a task as complete, optionally with result data and --next to continue"
    )]
    Done(DoneArgs),
    #[command(about = "Mark a task as failed with an error message")]
    Fail(FailArgs),
    #[command(about = "Cancel a task (optionally cascade to dependent tasks)")]
    Cancel(CancelArgs),
    #[command(about = "Approve a task that requires human approval before completion")]
    Approve(ApproveArgs),
    #[command(name = "add-dep", about = "Add a dependency edge between two tasks")]
    AddDep(AddDepArgs),
    #[command(
        name = "remove-dep",
        about = "Remove a dependency edge between two tasks"
    )]
    RemoveDep(RemoveDepArgs),
    #[command(about = "Update task fields (title, description, kind, priority)")]
    Update(UpdateTaskArgs),
    #[command(about = "Insert a new task between two existing tasks, rewiring dependencies")]
    Insert(InsertTaskArgs),
    #[command(about = "Prepend context to a task's description (annotate future work)")]
    Amend(AmendTaskArgs),
    #[command(about = "Replace a task's pending subtree with new tasks from JSON/YAML")]
    Pivot(PivotTaskArgs),
    #[command(about = "Split one task into multiple sub-tasks from a JSON spec")]
    Split(SplitTaskArgs),
    #[command(about = "Decompose a task into subtasks defined in a YAML file")]
    Decompose(DecomposeArgs),
    #[command(about = "Cancel pending subtasks and recreate from a YAML file")]
    Replan(ReplanArgs),
    #[command(about = "Pause a running task, saving progress for later resumption")]
    Pause(PauseArgs),
    #[command(about = "Add a note to a task (inter-agent communication)")]
    Note(NoteArgs),
    #[command(about = "List all notes on a task")]
    Notes(NotesArgs),
    #[command(about = "Full project overview: all tasks, dependencies, and progress summary")]
    Overview(OverviewArgs),
}

#[derive(Args, Debug)]
#[command(about = "Preview effects of mutations without applying them.\n\n\
              Simulates a change and shows what would happen to the task graph:\n\
              which tasks get delayed, which become ready, how the critical path changes.\n\
              Nothing is modified — safe to run anytime.\n\n\
              EXAMPLES:\n\
              \x20 plandb what-if cancel t-a1b2c3\n\
              \x20 plandb what-if insert --after t-a1 --before t-b2 --title \"Add auth\"")]
pub struct WhatIfCommand {
    #[command(subcommand)]
    command: WhatIfSubcommand,
}

#[derive(Subcommand, Debug)]
enum WhatIfSubcommand {
    #[command(about = "Preview what happens if a task is cancelled")]
    Cancel {
        #[arg(help = "Task ID to simulate cancelling")]
        task_id: String,
    },
    #[command(about = "Preview what happens if a task is inserted between two existing tasks")]
    Insert {
        #[arg(long, help = "Task that the new task depends on")]
        after: String,
        #[arg(long, help = "Task that will depend on the new task")]
        before: Option<String>,
        #[arg(long, help = "Title of the simulated task")]
        title: String,
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
    },
}

#[derive(Args, Debug)]
pub struct CreateTaskArgs {
    #[arg(long, help = "Project ID (uses default if set via 'plandb use')")]
    pub project: Option<String>,
    #[arg(long, help = "Task title (concise, descriptive)")]
    pub title: String,
    #[arg(long, value_name = "KIND", value_parser = parse_task_kind, help = "Task kind: generic, code, research, review, test, shell")]
    pub kind: Option<TaskKind>,
    #[arg(long, help = "Detailed description of what the task involves")]
    pub description: Option<String>,
    #[arg(
        long,
        default_value_t = 0,
        help = "Priority (higher = more important, default: 0)"
    )]
    pub priority: i32,
    #[arg(
        long = "dep",
        help = "Dependency: TASK_ID (default: feeds_into) or TASK_ID:KIND where KIND is feeds_into|blocks|suggests"
    )]
    pub deps: Vec<String>,
    #[arg(long, help = "Parent task ID (for hierarchical decomposition)")]
    pub parent: Option<String>,
    #[arg(
        long = "max-retries",
        default_value_t = 0,
        help = "Max auto-retry attempts on failure"
    )]
    pub max_retries: i32,
    #[arg(
        long = "timeout",
        help = "Timeout in seconds (reclaims task if exceeded)"
    )]
    pub timeout_seconds: Option<i64>,
    #[arg(
        long = "requires-approval",
        default_value_t = false,
        help = "Require human approval before task completes"
    )]
    pub requires_approval: bool,
    #[arg(
        long = "tag",
        help = "Tags for filtering (repeatable: --tag api --tag auth)"
    )]
    pub tags: Vec<String>,
}

#[derive(Args, Debug)]
struct CreateBatchArgs {
    #[arg(long, help = "Project ID (uses default if not set)")]
    project: Option<String>,
    #[arg(long, help = "YAML file with task definitions (see docs for schema)")]
    file: String,
}

#[derive(Args, Debug)]
pub struct ListTasksArgs {
    #[arg(long)]
    pub project: Option<String>,
    #[arg(long, value_parser = parse_task_status)]
    pub status: Option<TaskStatus>,
    #[arg(long, value_parser = parse_task_kind)]
    pub kind: Option<TaskKind>,
    #[arg(long)]
    pub tag: Option<String>,
    #[arg(long)]
    pub agent: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct GetTaskArgs {
    pub task_id: String,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
struct NextTaskArgs {
    #[arg(long, help = "Project ID (uses default if not set)")]
    project: Option<String>,
    #[arg(long, help = "Agent identifier")]
    agent: String,
    #[arg(
        long,
        default_value_t = false,
        help = "Also claim the task atomically (prefer 'go' command instead)"
    )]
    claim: bool,
}

#[derive(Args, Debug)]
struct ClaimTaskArgs {
    #[arg(help = "Task ID to claim (must be in ready status)")]
    task_id: String,
    #[arg(long, help = "Agent identifier claiming the task")]
    agent: String,
}

#[derive(Args, Debug)]
struct ProgressArgs {
    task_id: String,
    #[arg(long)]
    percent: i32,
    #[arg(long)]
    note: Option<String>,
}

#[derive(Args, Debug)]
pub struct DoneArgs {
    #[arg(help = "ID of the task to complete")]
    pub task_id: String,
    #[arg(
        long,
        alias = "output",
        help = "Result data (JSON string or plain text, passed to downstream tasks via handoff)"
    )]
    pub result: Option<String>,
    #[arg(
        long,
        help = "Files modified by this task (comma-separated paths, enables conflict detection)"
    )]
    pub files: Option<String>,
    #[arg(
        long,
        help = "After completing, claim + start next ready task (requires --agent)"
    )]
    pub next: bool,
    #[arg(long, help = "Agent ID for --next (required when --next is used)")]
    pub agent: Option<String>,
}

#[derive(Args, Debug)]
struct FailArgs {
    #[arg(help = "Task ID to mark as failed")]
    task_id: String,
    #[arg(long, help = "Error message describing why the task failed")]
    error: String,
}

#[derive(Args, Debug)]
struct CancelArgs {
    #[arg(help = "Task ID to cancel")]
    task_id: String,
    #[arg(
        long,
        default_value_t = false,
        help = "Also cancel all downstream dependent tasks"
    )]
    cascade: bool,
}

#[derive(Args, Debug)]
struct ApproveArgs {
    #[arg(help = "Task ID to approve")]
    task_id: String,
    #[arg(long, help = "Who approved (human name or ID)")]
    by: Option<String>,
    #[arg(long, help = "Approval comment or feedback")]
    comment: Option<String>,
}

#[derive(Args, Debug)]
pub struct TaskIdArg {
    pub task_id: String,
}

#[derive(Args, Debug)]
struct AddDepArgs {
    to_task: String,
    #[arg(
        long,
        alias = "from",
        help = "Task that must complete before TO_TASK can start"
    )]
    after: String,
    #[arg(long, default_value = "feeds_into", value_parser = parse_dependency_kind)]
    kind: DependencyKind,
}

#[derive(Args, Debug)]
struct RemoveDepArgs {
    to_task: String,
    #[arg(
        long,
        alias = "from",
        help = "Task that must complete before TO_TASK can start"
    )]
    after: String,
}

#[derive(Args, Debug)]
struct UpdateTaskArgs {
    task_id: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    description: Option<String>,
    #[arg(long, value_parser = parse_task_kind)]
    kind: Option<TaskKind>,
    #[arg(long)]
    priority: Option<i32>,
}

#[derive(Args, Debug)]
struct InsertTaskArgs {
    #[arg(long, help = "Task that the new task depends on (upstream)")]
    after: String,
    #[arg(
        long,
        help = "Task that will depend on the new task (downstream). Rewires the after→before edge"
    )]
    before: Option<String>,
    #[arg(long, help = "Title of the new task to insert")]
    title: String,
    #[arg(long, help = "Description for the new task")]
    description: Option<String>,
    #[arg(long, help = "Project ID (uses default if not set)")]
    project: Option<String>,
}

#[derive(Args, Debug)]
struct AmendTaskArgs {
    #[arg(help = "Task ID to amend")]
    task_id: String,
    #[arg(
        long,
        help = "Text to prepend to the task's description (e.g. 'NOTE: use JWT not sessions')"
    )]
    prepend: String,
}

#[derive(Args, Debug)]
struct PivotTaskArgs {
    #[arg(help = "Parent task whose subtree will be replaced")]
    parent_id: String,
    #[arg(
        long,
        default_value_t = false,
        help = "Keep already-completed subtasks (only replace pending/ready ones)"
    )]
    keep_done: bool,
    #[arg(
        long,
        help = "New subtasks as JSON array: [{\"title\":\"...\",\"description\":\"...\"}]"
    )]
    subtasks: Option<String>,
    #[arg(long, help = "YAML file with new subtasks (alternative to --subtasks)")]
    file: Option<String>,
}

#[derive(Args, Debug)]
struct SplitTaskArgs {
    #[arg(help = "Task ID to split into sub-tasks")]
    task_id: String,
    #[arg(
        long,
        help = "JSON array of parts: [{\"title\":\"...\",\"description\":\"...\"}]"
    )]
    into: String,
}

#[derive(Args, Debug)]
struct DecomposeArgs {
    #[arg(help = "Task ID to decompose (becomes composite parent)")]
    task_id: String,
    #[arg(
        long,
        help = "YAML file defining subtasks with optional deps_on references"
    )]
    file: String,
}

#[derive(Args, Debug)]
struct ReplanArgs {
    #[arg(help = "Task ID whose pending subtasks will be cancelled and recreated")]
    task_id: String,
    #[arg(long, help = "YAML file defining the new subtask plan")]
    file: String,
}

#[derive(Args, Debug)]
struct PauseArgs {
    #[arg(help = "Task ID to pause")]
    task_id: String,
    #[arg(long, help = "Save progress percentage (0-100) before pausing")]
    progress: Option<i32>,
    #[arg(long, help = "Note explaining why the task was paused / what remains")]
    note: Option<String>,
}

#[derive(Args, Debug)]
struct NoteArgs {
    #[arg(help = "Task ID to attach note to")]
    task_id: String,
    #[arg(help = "Note content (visible to all agents working on related tasks)")]
    content: String,
    #[arg(long, help = "Agent ID who is leaving the note")]
    agent: Option<String>,
}

#[derive(Args, Debug)]
struct NotesArgs {
    #[arg(help = "Task ID to list notes for")]
    task_id: String,
}

#[derive(Args, Debug)]
pub struct GoArgs {
    #[arg(long, help = "Agent identifier (e.g. claude-1, agent-backend)")]
    pub agent: String,
    #[arg(long, help = "Project ID (uses default if not set)")]
    pub project: Option<String>,
}

#[derive(Args, Debug)]
struct OverviewArgs {
    #[arg(long, help = "Project ID (uses default if not set)")]
    project: Option<String>,
    #[arg(long, help = "Force JSON output")]
    json: bool,
}

#[derive(Deserialize)]
struct BatchYaml {
    tasks: Vec<BatchTaskSpec>,
}

#[derive(Deserialize)]
struct BatchTaskSpec {
    id: Option<String>,
    title: String,
    kind: Option<TaskKind>,
    description: Option<String>,
    priority: Option<i32>,
    deps: Option<Vec<BatchDepSpec>>,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct BatchDepSpec {
    from: String,
    kind: Option<DependencyKind>,
}

#[derive(Deserialize)]
struct DecomposeYaml {
    subtasks: Vec<DecomposeSubtaskSpec>,
}

#[derive(Deserialize)]
struct DecomposeSubtaskSpec {
    title: String,
    kind: Option<TaskKind>,
    description: Option<String>,
    priority: Option<i32>,
    deps_on: Option<Vec<String>>,
}

pub fn run(db: &Database, command: TaskCommand, global_json: bool, compact: bool) -> Result<()> {
    match command.command {
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
        TaskSubcommand::Split(args) => {
            let parts: Vec<SplitPart> = serde_json::from_str(&args.into)?;
            let parent = get_task(db, &args.task_id)?;
            let before_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
            let result = split_task(db, &args.task_id, parts)?;
            let after_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
            let effect =
                compute_effects(db, &parent.project_id, &before_snapshot, &after_snapshot)?;
            if global_json {
                print_json(&serde_json::json!({
                    "parent_task_id": result.parent_task_id,
                    "created": result.created,
                    "done": result.done,
                    "title_to_id": result.title_to_id,
                    "effect": effect,
                    "project_state": project_state(db, &parent.project_id)?,
                }))?;
            } else {
                println!("split {}", args.task_id);
            }
        }
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
    let task = Task {
        id: generate_id("task"),
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
        metadata: None,
        created_at: now,
        updated_at: now,
    };

    let created = create_task(db, &task, &args.tags)?;
    for dep in &args.deps {
        let (from_task, kind) = parse_dep_arg(dep)?;
        add_dependency(
            db,
            &from_task,
            &created.id,
            kind,
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

fn parse_dep_arg(dep: &str) -> Result<(String, DependencyKind)> {
    if let Some(idx) = dep.rfind(':') {
        let (from_task, kind_str) = dep.split_at(idx);
        let kind_str = &kind_str[1..];
        if let Ok(kind) = parse_dependency_kind(kind_str) {
            return Ok((from_task.to_string(), kind));
        }
    }
    Ok((dep.to_string(), DependencyKind::FeedsInto))
}

pub fn parse_files_arg(files: &str) -> Vec<String> {
    files
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub fn go_cmd(db: &Database, args: &GoArgs, json: bool) -> Result<()> {
    let response = go_payload(db, args.project.as_deref(), &args.agent)?;
    if json {
        print_json(&response)?;
    } else if response["task"].is_null() {
        println!("no ready task found");
    } else {
        println!("started {}", response["task"]["id"].as_str().unwrap_or(""));
    }
    Ok(())
}

pub fn done_cmd(db: &Database, args: DoneArgs, json: bool, compact: bool) -> Result<()> {
    let result_provided = args.result.is_some();
    let result = match args.result {
        Some(text) => match serde_json::from_str(&text) {
            Ok(v) => Some(v),
            Err(_) => Some(serde_json::Value::String(text)),
        },
        None => None,
    };
    let task = complete_task(db, &args.task_id, result)
        .map_err(|err| enrich_transition_error(db, &args.task_id, "complete", err))?;
    if let Some(files) = args.files {
        let paths = parse_files_arg(&files);
        let _ = add_task_files(db, &task.id, &paths)?;
    }
    let _ = promote_ready_tasks(db)?;
    // Handoff visibility warning
    if !result_provided {
        if let Ok(deps) = list_dependencies(db, &task.id) {
            let feeds_count = deps
                .iter()
                .filter(|d| {
                    d.from_task == task.id
                        && matches!(d.kind, crate::models::DependencyKind::FeedsInto)
                })
                .count();
            if feeds_count > 0 {
                eprintln!(
                    "hint: this task feeds into {feeds_count} downstream task(s). Consider: plandb done {} --result '{{\"key\": \"value\"}}'",
                    task.id
                );
            }
        }
    }
    let next = if args.next {
        let agent = args
            .agent
            .ok_or_else(|| anyhow!("--agent is required when using --next"))?;
        Some(go_payload(db, Some(&task.project_id), &agent)?)
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
    } else {
        println!("completed {}", task.id);
    }
    Ok(())
}

fn enrich_transition_error(
    db: &Database,
    task_id: &str,
    action: &str,
    err: anyhow::Error,
) -> anyhow::Error {
    let Some(PlandbError::InvalidTransition(_)) = err.downcast_ref::<PlandbError>() else {
        return err;
    };

    let Ok(task) = get_task(db, task_id) else {
        return anyhow!(
            "cannot {action} task {task_id}: task not found. Try `plandb show {task_id}` to verify the id"
        );
    };

    let suggestion = match (action, &task.status) {
        ("start", TaskStatus::Ready) => {
            format!(
                "`plandb task claim {} --agent <agent>` then `plandb task start {}`",
                task.id, task.id
            )
        }
        ("start", TaskStatus::Pending) => {
            format!(
                "task is blocked by dependencies; inspect with `plandb show {}`",
                task.id
            )
        }
        ("start", TaskStatus::Running) => {
            format!(
                "task is already running; continue work or run `plandb done {}`",
                task.id
            )
        }
        ("complete", TaskStatus::Claimed) => {
            format!(
                "run `plandb task start {}` first, then `plandb done {}`",
                task.id, task.id
            )
        }
        ("complete", TaskStatus::Ready) => {
            format!(
                "run `plandb task claim {} --agent <agent>` and `plandb task start {}` first",
                task.id, task.id
            )
        }
        ("complete", TaskStatus::Pending) => {
            format!(
                "task is blocked by dependencies; inspect with `plandb show {}`",
                task.id
            )
        }
        ("complete", TaskStatus::Done | TaskStatus::DonePartial) => {
            format!(
                "task is already completed; inspect with `plandb show {}`",
                task.id
            )
        }
        ("fail", TaskStatus::Claimed) => {
            format!(
                "run `plandb task start {}` first, then `plandb task fail {} --error <msg>`",
                task.id, task.id
            )
        }
        ("fail", TaskStatus::Ready) => {
            format!(
                "run `plandb task claim {} --agent <agent>` and `plandb task start {}` first",
                task.id, task.id
            )
        }
        _ => format!("inspect task state with `plandb show {}`", task.id),
    };

    anyhow!(
        "cannot {action} task {}: current state is '{}'. Suggestion: {}",
        task.id,
        task.status,
        suggestion
    )
}

pub fn go_payload(
    db: &Database,
    project: Option<&str>,
    agent_id: &str,
) -> Result<serde_json::Value> {
    let project_id = resolve_project_id(db, project)?;
    let claimed = claim_next_task(db, &project_id, agent_id)?;
    let task = match claimed {
        Some(t) => Some(start_task(db, &t.id)?),
        None => None,
    };

    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.clone()),
            ..Default::default()
        },
    )?;
    let total = tasks.len();
    let done = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::DonePartial))
        .count();
    let ready = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Ready)
        .count();
    let running = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Claimed))
        .count();
    let pending = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Pending)
        .count();
    let progress = if total == 0 {
        "0%".to_string()
    } else {
        format!("{}%", ((done as f64 / total as f64) * 100.0).round() as i32)
    };

    let (task_json, handoff_json, notes_json, conflicts_json) = if let Some(task) = &task {
        let handoff = get_handoff_context(db, &task.id)?;
        let notes = list_notes(db, &task.id)?;
        let task_files = list_task_files(db, &task.id)?;
        let mut conflicts = check_file_conflicts(db, &project_id, Some(&task.id))?;
        if !task_files.is_empty() {
            let path_set = task_files
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            conflicts.retain(|c| path_set.contains(&c.path));
        }
        (
            serde_json::json!({
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "description": task.description,
            }),
            handoff
                .into_iter()
                .map(|h| {
                    serde_json::json!({
                        "from_task": h.from_task_id,
                        "from_title": h.from_title,
                        "result": h.result,
                        "agent_id": h.agent_id,
                    })
                })
                .collect::<Vec<_>>(),
            notes
                .into_iter()
                .map(|n| {
                    serde_json::json!({
                        "content": n.content,
                        "agent_id": n.agent_id,
                        "created_at": n.created_at,
                    })
                })
                .collect::<Vec<_>>(),
            serde_json::to_value(conflicts)?,
        )
    } else {
        (Value::Null, Vec::new(), Vec::new(), serde_json::json!([]))
    };

    Ok(serde_json::json!({
        "task": task_json,
        "handoff": handoff_json,
        "notes": notes_json,
        "file_conflicts": conflicts_json,
        "remaining": {
            "total": total,
            "done": done,
            "ready": ready,
            "running": running,
            "pending": pending,
        },
        "progress": progress,
    }))
}

fn decompose_or_replan(
    db: &Database,
    task_id: &str,
    file: &str,
    cancel_remaining: bool,
) -> Result<HashMap<String, String>> {
    let content = fs::read_to_string(file)?;
    let parsed: DecomposeYaml = serde_yaml::from_str(&content)?;
    if parsed.subtasks.is_empty() {
        return Err(anyhow!("decompose file has no subtasks"));
    }

    {
        let conn = db.lock()?;
        conn.execute(
            "UPDATE tasks SET is_composite = 1, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![task_id],
        )?;
        if cancel_remaining {
            conn.execute(
                "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE parent_task_id = ?1 AND status NOT IN ('done', 'done_partial', 'running')",
                rusqlite::params![task_id],
            )?;
        }
    }
    let parent = get_task(db, task_id)?;

    let mut title_to_id: HashMap<String, String> = HashMap::new();
    let now = Utc::now().naive_utc();
    for sub in &parsed.subtasks {
        let has_deps = sub.deps_on.as_ref().map(|d| !d.is_empty()).unwrap_or(false);
        let task = Task {
            id: generate_id("task"),
            project_id: parent.project_id.clone(),
            parent_task_id: Some(task_id.to_string()),
            is_composite: false,
            title: sub.title.clone(),
            description: sub.description.clone(),
            status: if has_deps {
                TaskStatus::Pending
            } else {
                TaskStatus::Ready
            },
            kind: sub.kind.clone().unwrap_or(TaskKind::Generic),
            priority: sub.priority.unwrap_or(0),
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
            metadata: None,
            created_at: now,
            updated_at: now,
        };
        let created = create_task(db, &task, &[])?;
        title_to_id.insert(sub.title.clone(), created.id.clone());
    }

    for sub in &parsed.subtasks {
        if let Some(deps_on) = &sub.deps_on {
            let to_id = title_to_id
                .get(&sub.title)
                .ok_or_else(|| anyhow!("internal error: subtask title not found"))?;
            for dep_title in deps_on {
                let from_id = title_to_id
                    .get(dep_title)
                    .ok_or_else(|| anyhow!("deps_on references unknown subtask: {}", dep_title))?;
                add_dependency(
                    db,
                    from_id,
                    to_id,
                    DependencyKind::FeedsInto,
                    DependencyCondition::All,
                    None,
                )?;
            }
        }
    }
    let _ = promote_ready_tasks(db)?;
    Ok(title_to_id)
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

fn parse_new_subtasks(subtasks: Option<String>, file: Option<String>) -> Result<Vec<NewSubtask>> {
    if let Some(raw) = subtasks {
        let parsed: Vec<NewSubtask> = serde_json::from_str(&raw)?;
        return Ok(parsed);
    }
    if let Some(path) = file {
        let content = fs::read_to_string(path)?;
        #[derive(Deserialize)]
        struct PivotYaml {
            subtasks: Vec<NewSubtask>,
        }
        let parsed: PivotYaml = serde_yaml::from_str(&content)?;
        return Ok(parsed.subtasks);
    }
    Err(anyhow!("provide either --subtasks or --file"))
}

pub fn print_task_detail(task: &Task) {
    println!("id: {}", task.id);
    println!("project: {}", task.project_id);
    println!("title: {}", task.title);
    println!(
        "status: {} {}",
        crate::cli::status_icon(&task.status),
        task.status
    );
    println!("kind: {}", task.kind);
    println!("priority: {}", task.priority);
    if let Some(agent) = &task.agent_id {
        println!("agent: {agent}");
    }
    if let Some(progress) = task.progress {
        println!("progress: {progress}%");
    }
    if let Some(note) = &task.progress_note {
        println!("note: {note}");
    }
    if let Some(desc) = &task.description {
        println!("description: {desc}");
    }
}
