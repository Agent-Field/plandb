use crate::db::{list_tasks, Database, TaskListFilters};
use crate::models::{generate_id, DependencyCondition, DependencyKind, Task, TaskKind, TaskStatus};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphTemplate {
    pub name: String,
    pub description: Option<String>,
    pub tasks: Vec<TemplateTask>,
    pub dependencies: Vec<TemplateDep>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TemplateTask {
    pub ref_id: String,
    pub title: String,
    pub description: Option<String>,
    pub kind: String,
    pub priority: i32,
    pub parent_ref: Option<String>,
    pub pre_condition: Option<String>,
    pub post_condition: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TemplateDep {
    pub from_ref: String,
    pub to_ref: String,
    pub kind: String,
}

pub fn export_graph(
    db: &Database,
    project_id: &str,
    name: &str,
    description: Option<&str>,
) -> Result<GraphTemplate> {
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        },
    )?;

    // Build ID -> ref mapping (use custom short refs)
    let mut id_to_ref: HashMap<String, String> = HashMap::new();
    for (i, task) in tasks.iter().enumerate() {
        let ref_id = format!("task-{}", i + 1);
        id_to_ref.insert(task.id.clone(), ref_id);
    }

    let template_tasks: Vec<TemplateTask> = tasks
        .iter()
        .map(|t| TemplateTask {
            ref_id: id_to_ref.get(&t.id).cloned().unwrap_or_default(),
            title: t.title.clone(),
            description: t.description.clone(),
            kind: t.kind.to_string(),
            priority: t.priority,
            parent_ref: t
                .parent_task_id
                .as_ref()
                .and_then(|pid| id_to_ref.get(pid).cloned()),
            pre_condition: t.pre_condition.clone(),
            post_condition: t.post_condition.clone(),
        })
        .collect();

    // Collect all dependencies
    let mut template_deps = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for task in &tasks {
        let deps = crate::db::list_dependencies(db, &task.id)?;
        for dep in deps {
            let key = (dep.from_task.clone(), dep.to_task.clone());
            if seen.insert(key) {
                if let (Some(from_ref), Some(to_ref)) =
                    (id_to_ref.get(&dep.from_task), id_to_ref.get(&dep.to_task))
                {
                    template_deps.push(TemplateDep {
                        from_ref: from_ref.clone(),
                        to_ref: to_ref.clone(),
                        kind: dep.kind.to_string(),
                    });
                }
            }
        }
    }

    Ok(GraphTemplate {
        name: name.to_string(),
        description: description.map(|s| s.to_string()),
        tasks: template_tasks,
        dependencies: template_deps,
    })
}

pub fn import_graph(
    db: &Database,
    project_id: &str,
    template: &GraphTemplate,
) -> Result<HashMap<String, String>> {
    let mut ref_to_id: HashMap<String, String> = HashMap::new();
    let now = crate::db::now_utc_naive();

    // Create tasks in order (parents before children)
    // First pass: tasks without parents
    // Second pass: tasks with parents
    let mut remaining: Vec<&TemplateTask> = template.tasks.iter().collect();
    let mut max_iterations = remaining.len() + 1;

    while !remaining.is_empty() && max_iterations > 0 {
        max_iterations -= 1;
        let mut created_this_round = Vec::new();

        for tt in &remaining {
            let parent_resolved = match &tt.parent_ref {
                Some(pref) => {
                    if let Some(pid) = ref_to_id.get(pref) {
                        Some(pid.clone())
                    } else {
                        continue; // parent not created yet, skip
                    }
                }
                None => None,
            };

            let task_id = generate_id("task");
            let kind = tt.kind.parse::<TaskKind>().unwrap_or(TaskKind::Generic);
            let task = Task {
                id: task_id.clone(),
                project_id: project_id.to_string(),
                parent_task_id: parent_resolved,
                is_composite: false,
                title: tt.title.clone(),
                description: tt.description.clone(),
                status: TaskStatus::Pending,
                kind,
                priority: tt.priority,
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
                retry_backoff: crate::models::RetryBackoff::Exponential,
                retry_delay_ms: 1000,
                timeout_seconds: None,
                heartbeat_interval: 30,
                last_heartbeat: None,
                requires_approval: false,
                approval_status: None,
                approved_by: None,
                approval_comment: None,
                pre_condition: tt.pre_condition.clone(),
                post_condition: tt.post_condition.clone(),
                metadata: None,
                created_at: now,
                updated_at: now,
            };
            crate::db::create_task(db, &task, &[])?;
            ref_to_id.insert(tt.ref_id.clone(), task_id);
            created_this_round.push(tt.ref_id.clone());
        }

        remaining.retain(|tt| !created_this_round.contains(&tt.ref_id));
    }

    // Create dependencies
    for td in &template.dependencies {
        if let (Some(from_id), Some(to_id)) =
            (ref_to_id.get(&td.from_ref), ref_to_id.get(&td.to_ref))
        {
            let kind = td
                .kind
                .parse::<DependencyKind>()
                .unwrap_or(DependencyKind::FeedsInto);
            crate::db::add_dependency(db, from_id, to_id, kind, DependencyCondition::All, None)?;
        }
    }

    // Promote ready tasks
    crate::db::promote_ready_tasks(db)?;

    Ok(ref_to_id)
}
