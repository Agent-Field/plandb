mod artifact;
mod dependency;
mod event;
mod note;
mod project;
mod task;

pub use artifact::Artifact;
pub use dependency::{Dependency, DependencyCondition, DependencyKind};
pub use event::{Event, EventType};
pub use note::TaskNote;
pub use project::{Project, ProjectStatus};
pub use task::{RetryBackoff, Task, TaskKind, TaskStatus};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn short_prefix(prefix: &str) -> String {
    match prefix {
        "task" => "t".to_string(),
        "proj" | "project" => "p".to_string(),
        "artifact" | "art" => "a".to_string(),
        "dep" | "dependency" => "d".to_string(),
        "event" => "e".to_string(),
        _ => {
            let ch = prefix
                .chars()
                .next()
                .map(|c| c.to_ascii_lowercase())
                .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                .unwrap_or('x');
            ch.to_string()
        }
    }
}

fn next_entropy() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut x = nanos ^ counter.rotate_left(17) ^ 0x9e37_79b9_7f4a_7c15;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x
}

fn random_suffix() -> String {
    const ALPHABET: &[u8; 36] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut value = next_entropy();
    let mut out = String::with_capacity(4);
    for _ in 0..4 {
        let idx = (value % 36) as usize;
        out.push(ALPHABET[idx] as char);
        value /= 36;
    }
    out
}

pub fn generate_id(prefix: &str) -> String {
    format!("{}-{}", short_prefix(prefix), random_suffix())
}

pub fn generate_id_with_custom(prefix: &str, custom: Option<&str>) -> String {
    match custom {
        Some(name) => {
            let p = short_prefix(prefix);
            format!("{p}-{name}")
        }
        None => generate_id(prefix),
    }
}
