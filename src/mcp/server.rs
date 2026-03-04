use anyhow::Result;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

use crate::db::init_db;
use crate::mcp::protocol::{
    JsonRpcRequest, JsonRpcResponse, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST,
    JSONRPC_VERSION, METHOD_NOT_FOUND, PARSE_ERROR,
};
use crate::mcp::tools::{call_tool, parse_tool_call, tool_schemas, wrap_tool_result};

pub fn run_server(db_path: &str) -> Result<()> {
    let db = init_db(db_path)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(raw) => {
                let id = raw.get("id").cloned().unwrap_or(Value::Null);
                match serde_json::from_value::<JsonRpcRequest>(raw) {
                    Ok(req) => handle_request(&db, req),
                    Err(e) => {
                        JsonRpcResponse::error(id, INVALID_REQUEST, format!("invalid request: {e}"))
                    }
                }
            }
            Err(e) => JsonRpcResponse::error(Value::Null, PARSE_ERROR, format!("parse error: {e}")),
        };

        writeln!(out, "{}", serde_json::to_string(&response)?)?;
        out.flush()?;
    }

    Ok(())
}

fn handle_request(db: &crate::db::Database, req: JsonRpcRequest) -> JsonRpcResponse {
    if req.jsonrpc != JSONRPC_VERSION {
        return JsonRpcResponse::error(req.id, INVALID_REQUEST, "jsonrpc must be '2.0'");
    }

    match req.method.as_str() {
        "initialize" => JsonRpcResponse::success(
            req.id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "plandb", "version": "0.1.0" }
            }),
        ),
        "tools/list" => JsonRpcResponse::success(req.id, json!({ "tools": tool_schemas() })),
        "tools/call" => match parse_tool_call(req.params) {
            Ok((tool_name, arguments)) => match call_tool(db, &tool_name, arguments) {
                Ok(value) => JsonRpcResponse::success(req.id, wrap_tool_result(value)),
                Err(e) => JsonRpcResponse::error(req.id, INTERNAL_ERROR, e.to_string()),
            },
            Err(e) => JsonRpcResponse::error(req.id, INVALID_PARAMS, e.to_string()),
        },
        _ => JsonRpcResponse::error(req.id, METHOD_NOT_FOUND, "method not found"),
    }
}
