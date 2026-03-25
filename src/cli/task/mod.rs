mod commands;
pub(crate) mod helpers;

// Re-export all public items that were previously accessible from crate::cli::task::*
pub use commands::{
    ahead_cmd, create_task_cmd, done_cmd, go_cmd, list_tasks_cmd, run_what_if, split_cmd,
};
pub use helpers::{build_search_query as build_search_query_for_task, go_payload, parse_files_arg, print_task_detail};

use crate::cli::{parse_dependency_kind, parse_task_kind, parse_task_status};
use crate::db::Database;
use crate::models::{DependencyKind, TaskKind, TaskStatus};
use anyhow::Result;
use clap::{Args, Subcommand};
use serde::Deserialize;

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
    #[arg(help = "Task title")]
    pub title: String,
    #[arg(long = "as", help = "Custom short ID (e.g. --as api creates t-api)")]
    pub custom_id: Option<String>,
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
    #[arg(long = "pre", help = "Pre-condition: what must be true before this task can be worked on")]
    pub pre_condition: Option<String>,
    #[arg(long = "post", help = "Post-condition: what must be verified after completion")]
    pub post_condition: Option<String>,
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
    #[arg(help = "Task ID to complete (omit to complete your current running task)")]
    pub task_id: Option<String>,
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
        help = "After completing, claim + start next ready task"
    )]
    pub next: bool,
    #[arg(long, default_value = "default", help = "Agent ID (default: 'default' or PLANDB_AGENT env var)")]
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
#[command(
    about = "Add a dependency edge between two tasks.\n\n\
              Usage: plandb task add-dep --after t-upstream t-downstream\n\
              This means t-upstream must complete before t-downstream can start."
)]
struct AddDepArgs {
    #[arg(help = "Downstream task (the one that waits)")]
    to_task: String,
    #[arg(
        long,
        alias = "from",
        help = "Upstream task (must complete first). Example: --after t-design t-implement"
    )]
    after: String,
    #[arg(long, default_value = "feeds_into", value_parser = parse_dependency_kind,
        help = "Dependency type: feeds_into (default), blocks, suggests")]
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
pub struct SplitTaskArgs {
    #[arg(help = "Task ID to split (omit to split current running task)")]
    pub task_id: Option<String>,
    #[arg(
        long,
        help = "Parts: comma-separated titles (\"A, B, C\"), chain with > (\"A > B > C\"), or JSON array"
    )]
    pub into: String,
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
    #[arg(long, default_value = "default", help = "Agent identifier (default: 'default', or set PLANDB_AGENT env var)")]
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
pub(crate) struct DecomposeYaml {
    pub(crate) subtasks: Vec<DecomposeSubtaskSpec>,
}

#[derive(Deserialize)]
pub(crate) struct DecomposeSubtaskSpec {
    pub(crate) title: String,
    pub(crate) kind: Option<TaskKind>,
    pub(crate) description: Option<String>,
    pub(crate) priority: Option<i32>,
    pub(crate) deps_on: Option<Vec<String>>,
}

pub fn run(db: &Database, command: TaskCommand, global_json: bool, compact: bool) -> Result<()> {
    commands::run_subcommand(db, command.command, global_json, compact)
}
