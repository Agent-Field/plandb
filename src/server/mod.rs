pub mod routes;
pub mod sse;

use crate::db::{init_db, run_sweep, Database};
use anyhow::Result;
use axum::Router;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::CorsLayer;

pub use routes::api_routes;

pub async fn run_server(db_path: &str, port: u16) -> Result<()> {
    let db = Arc::new(init_db(db_path)?);

    let sweep_db: Arc<Database> = db.clone();
    tokio::spawn(async move {
        loop {
            let _ = run_sweep(&sweep_db);
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    });

    let app = Router::new()
        .nest("/api", api_routes())
        .layer(CorsLayer::permissive())
        .with_state(db);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    println!("Plandb server listening on http://0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
