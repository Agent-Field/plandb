(() => {
  const outputEl = document.getElementById("terminal-output");
  const formEl = document.getElementById("terminal-form");
  const inputEl = document.getElementById("terminal-input");
  const tutorialList = document.getElementById("tutorial-steps");
  const tutorialStartButton = document.getElementById("tutorial-start");
  const tutorialHintButton = document.getElementById("tutorial-hint");
  const resetButton = document.getElementById("session-reset");

  if (!outputEl || !formEl || !inputEl || !tutorialList) {
    return;
  }

  const tutorialSteps = [
    {
      command: /^plandb init "demo-flow"$/i,
      hint: 'Run `plandb init "demo-flow"` to create the tutorial project.',
    },
    {
      command:
        /^plandb add "Map graph" --as map --kind research --description "Explain the compound graph model"$/i,
      hint:
        'Run `plandb add "Map graph" --as map --kind research --description "Explain the compound graph model"`.',
    },
    {
      command:
        /^plandb add "Build walkthrough" --as walkthrough --kind code --dep t-map --description "Create the guided terminal tutorial"$/i,
      hint:
        'Run `plandb add "Build walkthrough" --as walkthrough --kind code --dep t-map --description "Create the guided terminal tutorial"`.',
    },
    {
      command: /^plandb go$/i,
      hint: "Run `plandb go` to claim the first ready task.",
    },
    {
      command: /^plandb done --next$/i,
      hint: "Run `plandb done --next` to finish the current task and claim the next one.",
    },
    {
      command: /^plandb split --into "Draft content > Review examples > Publish page"$/i,
      hint:
        'Run `plandb split --into "Draft content > Review examples > Publish page"` to decompose the current task.',
    },
    {
      command: /^plandb status --detail$/i,
      hint: "Run `plandb status --detail` to inspect the graph.",
    },
  ];

  const initialState = () => ({
    projectId: null,
    projectName: null,
    tasks: [],
    currentTaskId: null,
    idCounter: 1,
    tutorialActive: false,
    tutorialStep: 0,
  });

  let state = initialState();

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function appendLine(text, kind = "output") {
    const line = document.createElement("p");
    line.className = `terminal-line ${kind}`;
    line.innerHTML = escapeHtml(text);
    outputEl.appendChild(line);
    outputEl.scrollTop = outputEl.scrollHeight;
  }

  function appendBlock(text, kind = "output") {
    text.split("\n").forEach((line) => appendLine(line, kind));
  }

  function resetTutorialVisuals() {
    tutorialList.querySelectorAll(".tutorial-step").forEach((stepNode, index) => {
      let status = "pending";
      if (state.tutorialActive) {
        if (index < state.tutorialStep) status = "done";
        if (index === state.tutorialStep) status = "active";
      }
      stepNode.dataset.state = status;
    });
  }

  function resetSession(announce = true) {
    state = initialState();
    outputEl.innerHTML = "";
    appendBlock(
      "PlanDB browser playground\nType `plandb help` for supported commands.\nUse `tutorial start` or the tutorial panel to begin the guided flow.",
      "meta"
    );
    if (announce) {
      appendLine("Session reset.", "meta");
    }
    resetTutorialVisuals();
  }

  function createTask(title, options = {}) {
    const baseId = options.customId ? `t-${options.customId}` : `t-${String(state.idCounter).padStart(3, "0")}`;
    if (!options.customId) {
      state.idCounter += 1;
    }
    const task = {
      id: baseId,
      title,
      description: options.description || "No description provided.",
      kind: options.kind || "generic",
      deps: options.dep ? [options.dep] : [],
      status: "pending",
      agent: null,
    };
    state.tasks.push(task);
    return task;
  }

  function findTask(id) {
    return state.tasks.find((task) => task.id === id);
  }

  function readyTasks() {
    return state.tasks.filter((task) => {
      if (task.status !== "pending") return false;
      return task.deps.every((depId) => findTask(depId)?.status === "done");
    });
  }

  function parseQuotedValue(input, flag) {
    const match = input.match(new RegExp(`${flag} "([^"]+)"`));
    return match ? match[1] : null;
  }

  function parseBareValue(input, flag) {
    const match = input.match(new RegExp(`${flag} ([^\\s]+)`));
    return match ? match[1] : null;
  }

  function renderStatus(detail) {
    if (!state.projectId) {
      return "error: No projects found. Run 'plandb init <name>' to create one.";
    }
    const doneCount = state.tasks.filter((task) => task.status === "done").length;
    const percent = state.tasks.length ? Math.round((doneCount / state.tasks.length) * 100) : 0;
    const summary = `${state.projectId} ${state.projectName}: ${doneCount}/${state.tasks.length} done (${percent}%)`;
    if (!detail) return summary;

    const lines = [summary];
    state.tasks.forEach((task) => {
      const icon = task.status === "done" ? "✓" : task.status === "running" ? "◉" : readyTasks().some((ready) => ready.id === task.id) ? "↺" : "○";
      const depText = task.deps.length ? ` [deps: ${task.deps.join(", ")}]` : "";
      lines.push(`${icon} ${task.id} ${task.title}${depText}`);
    });
    return lines.join("\n");
  }

  function renderTask(task) {
    return [
      `id: ${task.id}`,
      `title: ${task.title}`,
      `status: ${task.status}`,
      `kind: ${task.kind}`,
      `description: ${task.description}`,
      `deps: ${task.deps.length ? task.deps.join(", ") : "(none)"}`,
    ].join("\n");
  }

  function completeTutorialStep(command) {
    if (!state.tutorialActive) return;
    const step = tutorialSteps[state.tutorialStep];
    if (step && step.command.test(command)) {
      state.tutorialStep += 1;
      if (state.tutorialStep >= tutorialSteps.length) {
        appendLine("Tutorial complete. You walked the core PlanDB workflow.", "meta");
        state.tutorialActive = false;
      } else {
        appendLine(`Tutorial advanced. ${tutorialSteps[state.tutorialStep].hint}`, "meta");
      }
      resetTutorialVisuals();
    }
  }

  function handleInit(command) {
    const match = command.match(/^plandb init "([^"]+)"$/i);
    if (!match) {
      return 'usage: plandb init "project-name"';
    }
    const name = match[1];
    state.projectName = name;
    state.projectId = "p-demo";
    state.tasks = [];
    state.currentTaskId = null;
    state.idCounter = 1;
    return `created ${state.projectId} (${name})`;
  }

  function handleAdd(command) {
    const titleMatch = command.match(/^plandb add "([^"]+)"/i);
    if (!titleMatch) {
      return 'usage: plandb add "task" --description "..."';
    }
    if (!state.projectId) {
      return "error: initialize a project first with plandb init";
    }

    const description = parseQuotedValue(command, "--description");
    if (!description) {
      return "error: --description is required in this playground";
    }

    const customId = parseBareValue(command, "--as");
    const dep = parseBareValue(command, "--dep");
    const kind = parseBareValue(command, "--kind") || "generic";
    const validKinds = ["generic", "code", "research", "review", "test", "shell"];
    if (!validKinds.includes(kind)) {
      return `error: invalid --kind '${kind}'`;
    }
    if (dep && !findTask(dep)) {
      return `error: dependency ${dep} does not exist`;
    }

    const task = createTask(titleMatch[1], { description, customId, dep, kind });
    return `created task ${task.id} (${task.title})`;
  }

  function handleGo() {
    if (!state.projectId) {
      return "error: initialize a project first with plandb init";
    }
    if (state.currentTaskId) {
      return `already running ${state.currentTaskId}`;
    }
    const next = readyTasks()[0];
    if (!next) {
      return "No ready tasks.";
    }
    next.status = "running";
    next.agent = "demo";
    state.currentTaskId = next.id;
    const blockedCount = state.tasks.filter((task) => task.status === "pending" && !readyTasks().some((ready) => ready.id === task.id)).length;
    return `→ ${next.id} "${next.title}" [${state.tasks.filter((task) => task.status === "done").length}/${state.tasks.length} · ${readyTasks().length} ready · ${blockedCount} blocked]`;
  }

  function handleDone(command) {
    if (!state.currentTaskId) {
      return "error: no running task";
    }
    const current = findTask(state.currentTaskId);
    current.status = "done";
    current.agent = null;
    state.currentTaskId = null;
    if (!command.includes("--next")) {
      return `✓ ${current.id} done`;
    }

    const next = readyTasks()[0];
    if (!next) {
      return `✓ ${current.id} done`;
    }
    next.status = "running";
    next.agent = "demo";
    state.currentTaskId = next.id;
    return `✓ ${current.id} done → claimed ${next.id} "${next.title}"`;
  }

  function handleSplit(command) {
    if (!state.currentTaskId) {
      return "error: split requires a current running task in this playground";
    }
    const into = parseQuotedValue(command, "--into");
    if (!into) {
      return 'usage: plandb split --into "A, B"';
    }
    const current = findTask(state.currentTaskId);
    const chain = into.includes(">");
    const titles = into
      .split(chain ? ">" : ",")
      .map((part) => part.trim())
      .filter(Boolean);

    if (titles.length < 2) {
      return "error: split needs at least two parts";
    }

    current.status = "done";
    current.agent = null;
    state.currentTaskId = null;

    let previousId = null;
    const createdIds = titles.map((title) => {
      const task = createTask(title, {
        description: `Derived from ${current.id}: ${current.title}`,
        dep: chain && previousId ? previousId : null,
        kind: current.kind,
      });
      if (!chain) {
        task.deps.push(current.id);
      }
      previousId = task.id;
      return task.id;
    });

    return `split ${current.id} into ${createdIds.join(", ")}`;
  }

  function handleShow(command) {
    const match = command.match(/^plandb show ([^\s]+)$/i);
    if (!match) return "usage: plandb show t-task";
    const task = findTask(match[1]);
    if (!task) return `error: task ${match[1]} not found`;
    return renderTask(task);
  }

  function helpText() {
    return [
      "PlanDB playground commands:",
      '  plandb init "name"',
      '  plandb add "task" --description "..." [--as id] [--dep t-id] [--kind kind]',
      "  plandb go",
      "  plandb done --next",
      '  plandb split --into "A, B" or "A > B"',
      "  plandb status --detail",
      "  plandb show t-id",
      "  tutorial start | tutorial hint | clear",
    ].join("\n");
  }

  function execute(command) {
    const trimmed = command.trim();
    if (!trimmed) return;

    appendLine(`demo@plandb $ ${trimmed}`, "command");

    let result;
    if (/^(plandb help|plandb --help)$/i.test(trimmed)) {
      result = helpText();
    } else if (/^clear$/i.test(trimmed)) {
      outputEl.innerHTML = "";
      return;
    } else if (/^tutorial start$/i.test(trimmed)) {
      resetSession(false);
      state.tutorialActive = true;
      state.tutorialStep = 0;
      resetTutorialVisuals();
      result = `Tutorial started. ${tutorialSteps[0].hint}`;
    } else if (/^tutorial hint$/i.test(trimmed)) {
      result = state.tutorialActive ? tutorialSteps[state.tutorialStep].hint : "Start the tutorial first with `tutorial start`.";
    } else if (/^plandb init /i.test(trimmed)) {
      result = handleInit(trimmed);
    } else if (/^plandb add /i.test(trimmed)) {
      result = handleAdd(trimmed);
    } else if (/^plandb go$/i.test(trimmed)) {
      result = handleGo();
    } else if (/^plandb done(?:\s+--next)?$/i.test(trimmed)) {
      result = handleDone(trimmed);
    } else if (/^plandb split /i.test(trimmed)) {
      result = handleSplit(trimmed);
    } else if (/^plandb status(?:\s+--detail)?$/i.test(trimmed)) {
      result = renderStatus(trimmed.includes("--detail"));
    } else if (/^plandb show /i.test(trimmed)) {
      result = handleShow(trimmed);
    } else {
      result = `command not recognized: ${trimmed}`;
    }

    appendBlock(result, result.startsWith("error:") ? "meta" : "output");
    completeTutorialStep(trimmed);
  }

  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = inputEl.value;
    inputEl.value = "";
    execute(value);
  });

  tutorialStartButton.addEventListener("click", () => execute("tutorial start"));
  tutorialHintButton.addEventListener("click", () => execute("tutorial hint"));
  resetButton.addEventListener("click", () => resetSession());

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "ArrowUp") {
      event.preventDefault();
      if (state.tutorialActive) {
        inputEl.value = tutorialSteps[Math.min(state.tutorialStep, tutorialSteps.length - 1)].hint
          .replace(/^Run `/, "")
          .replace(/`\.$/, "")
          .replace(/` to .*$/, "");
      }
    }
  });

  resetSession(false);
})();
