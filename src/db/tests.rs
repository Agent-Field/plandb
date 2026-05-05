use crate::db::*;
use crate::models::*;
use chrono::{Duration, Utc};
use std::sync::{Arc, Barrier};
use std::thread;

fn test_db_path() -> String {
    let mut path = std::env::temp_dir();
    path.push(format!("plandb-test-{}.db", generate_id("tmp")));
    path.to_string_lossy().to_string()
}

fn now() -> chrono::NaiveDateTime {
    Utc::now().naive_utc()
}

fn make_task(project_id: &str, title: &str, status: TaskStatus) -> Task {
    let t = now();
    Task {
        id: generate_id("task"),
        project_id: project_id.to_owned(),
        parent_task_id: None,
        is_composite: false,
        title: title.to_owned(),
        description: None,
        status,
        kind: TaskKind::Generic,
        priority: 0,
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
        created_at: t,
        updated_at: t,
    }
}

#[test]
fn create_project_and_task() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();

    let project = create_project(&db, "Demo", Some("desc".to_string()), None).unwrap();
    assert_eq!(project.status, ProjectStatus::Active);

    let task = make_task(&project.id, "first", TaskStatus::Pending);
    let created_task = create_task(&db, &task, &[]).unwrap();
    assert_eq!(created_task.status, TaskStatus::Pending);

    let fetched = get_task(&db, &created_task.id).unwrap();
    assert_eq!(fetched.title, "first");
}

#[test]
fn claim_task_concurrency_single_winner() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "Claim", None, None).unwrap();

    let task = make_task(&project.id, "claim me", TaskStatus::Ready);
    let created_task = create_task(&db, &task, &[]).unwrap();

    let db1 = db.clone();
    let db2 = db.clone();
    let barrier = Arc::new(Barrier::new(2));
    let b1 = barrier.clone();
    let b2 = barrier.clone();
    let task_id_1 = created_task.id.clone();
    let task_id_2 = created_task.id.clone();

    let t1 = thread::spawn(move || {
        b1.wait();
        claim_task(&db1, &task_id_1, "agent-a").unwrap().is_some()
    });
    let t2 = thread::spawn(move || {
        b2.wait();
        claim_task(&db2, &task_id_2, "agent-b").unwrap().is_some()
    });

    let won_1 = t1.join().unwrap();
    let won_2 = t2.join().unwrap();
    assert!(won_1 ^ won_2);
}

#[test]
fn promote_sweep_moves_pending_to_ready() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "Promote", None, None).unwrap();

    let upstream = create_task(&db, &make_task(&project.id, "up", TaskStatus::Done), &[]).unwrap();
    let downstream = create_task(
        &db,
        &make_task(&project.id, "down", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    add_dependency(
        &db,
        &upstream.id,
        &downstream.id,
        DependencyKind::Blocks,
        DependencyCondition::All,
        None,
    )
    .unwrap();

    let sweep = run_sweep(&db).unwrap();
    assert_eq!(sweep.promoted, 1);
    let updated = get_task(&db, &downstream.id).unwrap();
    assert_eq!(updated.status, TaskStatus::Ready);
}

#[test]
fn test_child_of_composite_with_unmet_parent_deps_not_ready() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "CompositeBlocked", None, None).unwrap();

    let blocker = create_task(
        &db,
        &make_task(&project.id, "blocker", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    let mut parent = make_task(&project.id, "parent", TaskStatus::Pending);
    parent.is_composite = true;
    let parent = create_task(&db, &parent, &[]).unwrap();

    let mut child = make_task(&project.id, "child", TaskStatus::Pending);
    child.parent_task_id = Some(parent.id.clone());
    let child = create_task(&db, &child, &[]).unwrap();

    add_dependency(
        &db,
        &blocker.id,
        &parent.id,
        DependencyKind::FeedsInto,
        DependencyCondition::All,
        None,
    )
    .unwrap();

    promote_ready_tasks(&db).unwrap();

    assert_eq!(
        get_task(&db, &blocker.id).unwrap().status,
        TaskStatus::Ready
    );
    assert_eq!(
        get_task(&db, &parent.id).unwrap().status,
        TaskStatus::Pending
    );
    assert_eq!(
        get_task(&db, &child.id).unwrap().status,
        TaskStatus::Pending
    );
}

#[test]
fn test_child_promotes_when_parent_deps_satisfied() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "CompositeUnblocked", None, None).unwrap();

    let blocker = create_task(
        &db,
        &make_task(&project.id, "blocker", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    let mut parent = make_task(&project.id, "parent", TaskStatus::Pending);
    parent.is_composite = true;
    let parent = create_task(&db, &parent, &[]).unwrap();

    let mut child = make_task(&project.id, "child", TaskStatus::Pending);
    child.parent_task_id = Some(parent.id.clone());
    let child = create_task(&db, &child, &[]).unwrap();

    add_dependency(
        &db,
        &blocker.id,
        &parent.id,
        DependencyKind::FeedsInto,
        DependencyCondition::All,
        None,
    )
    .unwrap();

    promote_ready_tasks(&db).unwrap();
    complete_task(&db, &blocker.id, None).unwrap();
    promote_ready_tasks(&db).unwrap();

    assert_eq!(get_task(&db, &parent.id).unwrap().status, TaskStatus::Ready);
    assert_eq!(get_task(&db, &child.id).unwrap().status, TaskStatus::Ready);
}

#[test]
fn test_grandchild_inherits_through_two_levels() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "CompositeGrandchild", None, None).unwrap();

    let blocker = create_task(
        &db,
        &make_task(&project.id, "blocker", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    let mut parent = make_task(&project.id, "parent", TaskStatus::Pending);
    parent.is_composite = true;
    let parent = create_task(&db, &parent, &[]).unwrap();

    let mut child = make_task(&project.id, "child", TaskStatus::Pending);
    child.parent_task_id = Some(parent.id.clone());
    child.is_composite = true;
    let child = create_task(&db, &child, &[]).unwrap();

    let mut grandchild = make_task(&project.id, "grandchild", TaskStatus::Pending);
    grandchild.parent_task_id = Some(child.id.clone());
    let grandchild = create_task(&db, &grandchild, &[]).unwrap();

    add_dependency(
        &db,
        &blocker.id,
        &parent.id,
        DependencyKind::Blocks,
        DependencyCondition::All,
        None,
    )
    .unwrap();

    promote_ready_tasks(&db).unwrap();

    assert_eq!(
        get_task(&db, &blocker.id).unwrap().status,
        TaskStatus::Ready
    );
    assert_eq!(
        get_task(&db, &parent.id).unwrap().status,
        TaskStatus::Pending
    );
    assert_eq!(
        get_task(&db, &child.id).unwrap().status,
        TaskStatus::Pending
    );
    assert_eq!(
        get_task(&db, &grandchild.id).unwrap().status,
        TaskStatus::Pending
    );

    complete_task(&db, &blocker.id, None).unwrap();
    promote_ready_tasks(&db).unwrap();

    assert_eq!(get_task(&db, &parent.id).unwrap().status, TaskStatus::Ready);
    assert_eq!(get_task(&db, &child.id).unwrap().status, TaskStatus::Ready);
    assert_eq!(
        get_task(&db, &grandchild.id).unwrap().status,
        TaskStatus::Ready
    );
}

#[test]
fn test_no_parent_existing_behavior_preserved() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "FlatDeps", None, None).unwrap();

    let upstream = create_task(
        &db,
        &make_task(&project.id, "upstream", TaskStatus::Pending),
        &[],
    )
    .unwrap();
    let downstream = create_task(
        &db,
        &make_task(&project.id, "downstream", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    add_dependency(
        &db,
        &upstream.id,
        &downstream.id,
        DependencyKind::FeedsInto,
        DependencyCondition::All,
        None,
    )
    .unwrap();

    promote_ready_tasks(&db).unwrap();

    assert_eq!(
        get_task(&db, &upstream.id).unwrap().status,
        TaskStatus::Ready
    );
    assert_eq!(
        get_task(&db, &downstream.id).unwrap().status,
        TaskStatus::Pending
    );

    complete_task(&db, &upstream.id, None).unwrap();
    promote_ready_tasks(&db).unwrap();

    assert_eq!(
        get_task(&db, &downstream.id).unwrap().status,
        TaskStatus::Ready
    );
}

#[test]
fn sweeper_reclaims_retries_and_rolls_up_composites() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "Sweep", None, None).unwrap();
    let now = now();

    let mut heartbeat_stale = make_task(&project.id, "stale heartbeat", TaskStatus::Running);
    heartbeat_stale.agent_id = Some("agent-x".to_string());
    heartbeat_stale.started_at = Some(now - Duration::seconds(120));
    heartbeat_stale.last_heartbeat = Some(now - Duration::seconds(120));
    heartbeat_stale.heartbeat_interval = 10;
    heartbeat_stale.timeout_seconds = Some(600);
    create_task(&db, &heartbeat_stale, &[]).unwrap();

    let mut timed_out = make_task(&project.id, "timeout", TaskStatus::Running);
    timed_out.started_at = Some(now - Duration::seconds(90));
    timed_out.last_heartbeat = Some(now - Duration::seconds(1));
    timed_out.heartbeat_interval = 100_000;
    timed_out.timeout_seconds = Some(10);
    create_task(&db, &timed_out, &[]).unwrap();

    let mut retryable = make_task(&project.id, "retry", TaskStatus::Failed);
    retryable.max_retries = 3;
    retryable.retry_count = 0;
    retryable.retry_backoff = RetryBackoff::Fixed;
    retryable.retry_delay_ms = 1;
    retryable.completed_at = Some(now - Duration::seconds(10));
    create_task(&db, &retryable, &[]).unwrap();

    let mut parent = make_task(&project.id, "parent", TaskStatus::Pending);
    parent.is_composite = true;
    let parent = create_task(&db, &parent, &[]).unwrap();

    let mut child1 = make_task(&project.id, "child1", TaskStatus::Done);
    child1.parent_task_id = Some(parent.id.clone());
    create_task(&db, &child1, &[]).unwrap();

    let mut child2 = make_task(&project.id, "child2", TaskStatus::Done);
    child2.parent_task_id = Some(parent.id.clone());
    create_task(&db, &child2, &[]).unwrap();

    let sweep = run_sweep(&db).unwrap();
    assert!(sweep.reclaimed >= 1);
    assert!(sweep.timed_out >= 1);
    assert!(sweep.retried >= 1);
    assert!(sweep.composites_completed >= 1);
}

#[test]
fn lenient_done_from_ready() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "LenientReady", None, None).unwrap();

    let task = create_task(
        &db,
        &make_task(&project.id, "ready->done", TaskStatus::Ready),
        &[],
    )
    .unwrap();
    let completed = complete_task(&db, &task.id, None).unwrap();

    assert_eq!(completed.status, TaskStatus::Done);
    assert!(completed.claimed_at.is_some());
    assert!(completed.started_at.is_some());
    assert!(completed.completed_at.is_some());
}

#[test]
fn lenient_done_from_claimed() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "LenientClaimed", None, None).unwrap();

    let task = create_task(
        &db,
        &make_task(&project.id, "claimed->done", TaskStatus::Ready),
        &[],
    )
    .unwrap();
    let claimed = claim_task(&db, &task.id, "agent-a").unwrap().unwrap();
    assert_eq!(claimed.status, TaskStatus::Claimed);
    assert!(claimed.claimed_at.is_some());
    assert!(claimed.started_at.is_none());

    let completed = complete_task(&db, &task.id, None).unwrap();
    assert_eq!(completed.status, TaskStatus::Done);
    assert!(completed.claimed_at.is_some());
    assert!(completed.started_at.is_some());
    assert!(completed.completed_at.is_some());
}

#[test]
fn lenient_done_from_running() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "LenientRunning", None, None).unwrap();

    let task = create_task(
        &db,
        &make_task(&project.id, "running->done", TaskStatus::Ready),
        &[],
    )
    .unwrap();
    claim_task(&db, &task.id, "agent-a").unwrap().unwrap();
    let running = start_task(&db, &task.id).unwrap();
    assert_eq!(running.status, TaskStatus::Running);

    let completed = complete_task(&db, &task.id, None).unwrap();
    assert_eq!(completed.status, TaskStatus::Done);
    assert!(completed.claimed_at.is_some());
    assert!(completed.started_at.is_some());
    assert!(completed.completed_at.is_some());
}

#[test]
fn done_rejects_pending() {
    let db_path = test_db_path();
    let db = init_db(&db_path).unwrap();
    let project = create_project(&db, "LenientPending", None, None).unwrap();

    let task = create_task(
        &db,
        &make_task(&project.id, "pending cannot complete", TaskStatus::Pending),
        &[],
    )
    .unwrap();

    let err = complete_task(&db, &task.id, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("is 'pending'"));
    assert!(msg.contains("ready/claimed/running"));
}
