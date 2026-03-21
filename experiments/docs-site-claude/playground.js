/* PlanDB Interactive Playground — Terminal Emulator + Simulated State */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function randomId() {
    var chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    var id = '';
    for (var i = 0; i < 4; i++) {
      id += chars[Math.floor(Math.random() * chars.length)];
    }
    return 't-' + id;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function span(cls, text) {
    return '<span class="' + cls + '">' + escapeHtml(text) + '</span>';
  }

  // Color helpers that return raw HTML strings
  function cyan(t) { return span('token-id', t); }
  function green(t) { return span('token-success', t); }
  function yellow(t) { return span('token-warning', t); }
  function red(t) { return span('token-error', t); }
  function dim(t) { return span('token-output', t); }
  function muted(t) { return span('token-comment', t); }
  function accent(t) { return span('token-accent', t); }

  function statusColor(status) {
    var map = { ready: 'token-accent', running: 'token-warning', done: 'token-success', failed: 'token-error', pending: 'token-comment' };
    return map[status] || 'token-output';
  }

  function statusIcon(status) {
    var map = { ready: '○', running: '◉', done: '✓', failed: '✗', pending: '·' };
    return map[status] || '?';
  }

  function progressBar(done, total, width) {
    width = width || 20;
    var filled = total > 0 ? Math.round((done / total) * width) : 0;
    var empty = width - filled;
    var bar = '';
    for (var i = 0; i < filled; i++) bar += '█';
    for (var j = 0; j < empty; j++) bar += '░';
    var pct = total > 0 ? Math.round((done / total) * 100) : 0;
    return bar + ' ' + pct + '%';
  }

  // ---------------------------------------------------------------------------
  // PlanDB State Machine
  // ---------------------------------------------------------------------------
  var state = {
    projectName: null,
    tasks: [],
    currentAgent: 'you',
    runningTaskId: null
  };

  function resetState() {
    state.projectName = null;
    state.tasks = [];
    state.runningTaskId = null;
  }

  function findTask(id) {
    for (var i = 0; i < state.tasks.length; i++) {
      if (state.tasks[i].id === id) return state.tasks[i];
    }
    return null;
  }

  function recalcReady() {
    state.tasks.forEach(function (task) {
      if (task.status !== 'pending') return;
      var allDone = task.deps.every(function (depId) {
        var dep = findTask(depId);
        return dep && dep.status === 'done';
      });
      if (allDone) task.status = 'ready';
    });
  }

  function getNextReady() {
    for (var i = 0; i < state.tasks.length; i++) {
      if (state.tasks[i].status === 'ready') return state.tasks[i];
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Command Parser
  // ---------------------------------------------------------------------------
  function parseArgs(raw) {
    // Split respecting quoted strings
    var args = [];
    var current = '';
    var inQuote = null;
    for (var i = 0; i < raw.length; i++) {
      var ch = raw[i];
      if (inQuote) {
        if (ch === inQuote) { inQuote = null; }
        else { current += ch; }
      } else if (ch === '"' || ch === "'") {
        inQuote = ch;
      } else if (ch === ' ') {
        if (current) { args.push(current); current = ''; }
      } else {
        current += ch;
      }
    }
    if (current) args.push(current);
    return args;
  }

  function getFlag(args, name) {
    var idx = args.indexOf(name);
    if (idx === -1) return null;
    if (idx + 1 < args.length) return args[idx + 1];
    return true;
  }

  function hasFlag(args, name) {
    return args.indexOf(name) !== -1;
  }

  // ---------------------------------------------------------------------------
  // Commands
  // ---------------------------------------------------------------------------
  function cmdInit(args) {
    var name = args[2];
    if (!name) return [red('Usage: plandb init <project-name>')];
    resetState();
    state.projectName = name;
    return [green('✓') + ' Initialized project ' + accent(name)];
  }

  function cmdAdd(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    // plandb add "title" [--dep t-xxx] [--kind xxx] [--as xxx] [--description "..."]
    var title = args[2];
    if (!title) return [red('Usage: plandb add "title" [--dep t-xxx] [--kind xxx]')];

    var depId = getFlag(args, '--dep');
    var kind = getFlag(args, '--kind') || 'generic';
    var agent = getFlag(args, '--as');
    var desc = getFlag(args, '--description') || '';

    var task = {
      id: randomId(),
      title: title,
      kind: kind,
      status: 'ready',
      deps: [],
      description: desc,
      agent: agent || null,
      notes: []
    };

    if (depId && typeof depId === 'string') {
      var dep = findTask(depId);
      if (!dep) return [red('Dependency ' + depId + ' not found')];
      task.deps.push(depId);
      task.status = dep.status === 'done' ? 'ready' : 'pending';
    }

    state.tasks.push(task);
    recalcReady();

    var lines = [green('✓') + ' Added ' + cyan(task.id) + ' ' + escapeHtml(task.title)];
    if (task.deps.length > 0) {
      lines.push(dim('  dep: ' + task.deps.join(', ') + '  status: ' + task.status));
    }
    return lines;
  }

  function cmdStatus(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    var detail = hasFlag(args, '--detail');

    var counts = { ready: 0, running: 0, done: 0, failed: 0, pending: 0 };
    state.tasks.forEach(function (t) { counts[t.status] = (counts[t.status] || 0) + 1; });
    var total = state.tasks.length;

    var lines = [];
    lines.push(accent('Project: ') + escapeHtml(state.projectName));
    lines.push('');

    if (total === 0) {
      lines.push(dim('No tasks yet. Run: plandb add "title"'));
      return lines;
    }

    // Progress bar
    lines.push(
      span(statusColor('done'), String(counts.done)) + dim('/') + escapeHtml(String(total)) +
      dim(' tasks done  ') +
      green(progressBar(counts.done, total))
    );
    lines.push('');

    // Summary counts
    var parts = [];
    if (counts.ready > 0) parts.push(span(statusColor('ready'), counts.ready + ' ready'));
    if (counts.running > 0) parts.push(span(statusColor('running'), counts.running + ' running'));
    if (counts.done > 0) parts.push(span(statusColor('done'), counts.done + ' done'));
    if (counts.pending > 0) parts.push(span(statusColor('pending'), counts.pending + ' pending'));
    if (counts.failed > 0) parts.push(span(statusColor('failed'), counts.failed + ' failed'));
    lines.push(parts.join(dim('  ·  ')));

    if (detail) {
      lines.push('');
      lines.push(dim('─'.repeat(60)));
      // Build tree
      var roots = state.tasks.filter(function (t) { return t.deps.length === 0; });
      var children = {};
      state.tasks.forEach(function (t) {
        t.deps.forEach(function (depId) {
          if (!children[depId]) children[depId] = [];
          children[depId].push(t);
        });
      });

      function renderTree(task, prefix, isLast) {
        var connector = prefix === '' ? '' : (isLast ? '└── ' : '├── ');
        var icon = statusIcon(task.status);
        lines.push(
          dim(prefix + connector) +
          span(statusColor(task.status), icon + ' ' + task.status.toUpperCase()) +
          '  ' + cyan(task.id) + '  ' + escapeHtml(task.title)
        );
        var kids = children[task.id] || [];
        for (var i = 0; i < kids.length; i++) {
          var childPrefix = prefix === '' ? '' : (prefix + (isLast ? '    ' : '│   '));
          if (prefix === '') childPrefix = '';
          renderTree(kids[i], prefix + (isLast ? '    ' : '│   '), i === kids.length - 1);
        }
      }

      for (var r = 0; r < roots.length; r++) {
        renderTree(roots[r], '', r === roots.length - 1);
      }
    }

    return lines;
  }

  function cmdList(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    var filterStatus = getFlag(args, '--status');
    var tasks = state.tasks;
    if (filterStatus) {
      tasks = tasks.filter(function (t) { return t.status === filterStatus; });
    }
    if (tasks.length === 0) {
      return [dim(filterStatus ? 'No ' + filterStatus + ' tasks.' : 'No tasks. Run: plandb add "title"')];
    }

    var lines = [];
    // Header
    lines.push(
      dim(pad('ID', 10) + pad('STATUS', 10) + pad('KIND', 10) + 'TITLE')
    );
    lines.push(dim('─'.repeat(56)));

    tasks.forEach(function (t) {
      lines.push(
        cyan(pad(t.id, 10)) +
        span(statusColor(t.status), pad(t.status, 10)) +
        dim(pad(t.kind, 10)) +
        escapeHtml(t.title)
      );
    });
    return lines;
  }

  function pad(str, len) {
    while (str.length < len) str += ' ';
    return str;
  }

  function cmdGo(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    if (state.runningTaskId) {
      var cur = findTask(state.runningTaskId);
      return [yellow('Already working on ' + state.runningTaskId + ' "' + (cur ? cur.title : '') + '"')];
    }
    var next = getNextReady();
    if (!next) return [dim('No ready tasks to claim.')];
    next.status = 'running';
    next.agent = state.currentAgent;
    state.runningTaskId = next.id;
    return [
      green('→') + ' ' + cyan(next.id) + ' ' + escapeHtml('"' + next.title + '"'),
      dim('  status: ') + span(statusColor('running'), 'running')
    ];
  }

  function cmdDone(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    if (!state.runningTaskId) return [yellow('Nothing running. Run: plandb go')];
    var task = findTask(state.runningTaskId);
    if (!task) return [red('Task not found.')];

    task.status = 'done';
    var lines = [green('✓') + ' Completed ' + cyan(task.id) + ' ' + escapeHtml('"' + task.title + '"')];
    state.runningTaskId = null;

    recalcReady();

    if (hasFlag(args, '--next')) {
      var next = getNextReady();
      if (next) {
        next.status = 'running';
        next.agent = state.currentAgent;
        state.runningTaskId = next.id;
        lines.push(green('→') + ' Next: ' + cyan(next.id) + ' ' + escapeHtml('"' + next.title + '"'));
      } else {
        lines.push(dim('  No more ready tasks.'));
      }
    }
    return lines;
  }

  function cmdShow(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    var id = args[2];
    if (!id) return [red('Usage: plandb show <task-id>')];
    var task = findTask(id);
    if (!task) return [red('Task ' + id + ' not found.')];

    var lines = [];
    lines.push(accent('id:      ') + cyan(task.id));
    lines.push(accent('title:   ') + escapeHtml(task.title));
    lines.push(accent('status:  ') + span(statusColor(task.status), statusIcon(task.status) + ' ' + task.status));
    lines.push(accent('kind:    ') + dim(task.kind));
    if (task.agent) lines.push(accent('agent:   ') + dim(task.agent));
    if (task.deps.length > 0) lines.push(accent('deps:    ') + cyan(task.deps.join(', ')));
    if (task.description) lines.push(accent('desc:    ') + dim(task.description));
    return lines;
  }

  function cmdSplit(args) {
    if (!state.projectName) return [red('No project. Run: plandb init <name>')];
    if (!state.runningTaskId) return [yellow('Nothing running to split. Run: plandb go')];

    var intoRaw = getFlag(args, '--into');
    if (!intoRaw) return [red('Usage: plandb split --into "A, B, C" or "A > B > C"')];

    var task = findTask(state.runningTaskId);
    if (!task) return [red('Current task not found.')];

    var isChain = intoRaw.indexOf('>') !== -1;
    var parts = isChain
      ? intoRaw.split('>').map(function (s) { return s.trim(); }).filter(Boolean)
      : intoRaw.split(',').map(function (s) { return s.trim(); }).filter(Boolean);

    if (parts.length < 2) return [red('Provide at least 2 sub-tasks.')];

    // Remove original from running
    task.status = 'done';
    state.runningTaskId = null;

    var lines = [green('✓') + ' Split ' + cyan(task.id) + ' into ' + parts.length + ' sub-tasks:'];
    var prevId = null;

    parts.forEach(function (title, idx) {
      var sub = {
        id: randomId(),
        title: title,
        kind: task.kind,
        status: 'ready',
        deps: [],
        description: '',
        agent: null,
        notes: []
      };

      // Inherit parent deps for first subtask
      if (idx === 0) {
        sub.deps = task.deps.slice();
      }

      // Chain mode: each depends on previous
      if (isChain && prevId) {
        sub.deps.push(prevId);
        sub.status = 'pending';
      }

      state.tasks.push(sub);
      lines.push('  ' + cyan(sub.id) + ' ' + escapeHtml(sub.title) + dim(isChain && prevId ? ' (dep: ' + prevId + ')' : ''));
      prevId = sub.id;
    });

    // Re-wire: anything that depended on original now depends on last subtask
    var lastSubId = prevId;
    state.tasks.forEach(function (t) {
      var depIdx = t.deps.indexOf(task.id);
      if (depIdx !== -1) {
        t.deps[depIdx] = lastSubId;
      }
    });

    recalcReady();
    return lines;
  }

  function cmdHelp() {
    var lines = [];
    lines.push(accent('PlanDB Playground — Available Commands'));
    lines.push(dim('─'.repeat(48)));
    lines.push('');
    var cmds = [
      ['plandb init <name>', 'Create a new project (resets state)'],
      ['plandb add "title"', 'Add a task'],
      ['  --dep t-xxx', 'Set dependency on another task'],
      ['  --kind <type>', 'Set kind (code, test, research...)'],
      ['  --description "..."', 'Add description'],
      ['plandb status', 'Show project overview + progress bar'],
      ['plandb status --detail', 'Show dependency tree'],
      ['plandb list', 'List all tasks'],
      ['plandb list --status <s>', 'Filter: ready, running, done, pending'],
      ['plandb go', 'Claim next ready task'],
      ['plandb done', 'Complete current task'],
      ['plandb done --next', 'Complete and claim next'],
      ['plandb show <id>', 'Show task details'],
      ['plandb split --into "A, B, C"', 'Split current into parallel tasks'],
      ['plandb split --into "A > B > C"', 'Split current into chained tasks'],
      ['help', 'Show this help'],
      ['clear', 'Clear terminal'],
      ['start tutorial', 'Start the guided tutorial'],
      ['exit tutorial', 'Exit the tutorial']
    ];
    cmds.forEach(function (c) {
      lines.push(cyan(pad(c[0], 32)) + dim(c[1]));
    });
    return lines;
  }

  function executeCommand(input) {
    var trimmed = input.trim();
    if (!trimmed) return [];

    var lower = trimmed.toLowerCase();
    if (lower === 'clear') return null; // special: handled by caller
    if (lower === 'help') return cmdHelp();
    if (lower === 'start tutorial') { startTutorial(); return [green('Tutorial started! Follow the steps in the info panel.')]; }
    if (lower === 'exit tutorial') { exitTutorial(); return [dim('Tutorial exited.')]; }

    var args = parseArgs(trimmed);
    if (args[0] !== 'plandb') return [red('Unknown command. Type "help" for available commands.')];

    var sub = args[1];
    switch (sub) {
      case 'init': return cmdInit(args);
      case 'add': return cmdAdd(args);
      case 'status': return cmdStatus(args);
      case 'list': return cmdList(args);
      case 'go': return cmdGo(args);
      case 'done': return cmdDone(args);
      case 'show': return cmdShow(args);
      case 'split': return cmdSplit(args);
      default: return [red('Unknown subcommand: ' + sub + '. Type "help" for available commands.')];
    }
  }

  // ---------------------------------------------------------------------------
  // Terminal UI
  // ---------------------------------------------------------------------------
  var terminalBody = document.getElementById('terminal-body');
  var hiddenInput = document.getElementById('terminal-hidden-input');
  var commandHistory = [];
  var historyIndex = -1;
  var currentInput = '';

  function createInputRow() {
    var row = document.createElement('div');
    row.className = 'terminal-input-row';
    row.innerHTML =
      '<span class="prompt">$ </span>' +
      '<span class="input-area"><span class="input-text"></span><span class="cursor"></span></span>';
    return row;
  }

  var inputRow = createInputRow();
  terminalBody.appendChild(inputRow);

  function getInputTextEl() {
    return inputRow.querySelector('.input-text');
  }

  function getCursorEl() {
    return inputRow.querySelector('.cursor');
  }

  function updateInputDisplay() {
    getInputTextEl().textContent = currentInput;
  }

  function scrollToBottom() {
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  function writeLine(html) {
    var line = document.createElement('div');
    line.className = 'terminal-output-line';
    line.innerHTML = html;
    terminalBody.insertBefore(line, inputRow);
  }

  function writeLines(lines) {
    lines.forEach(function (l) { writeLine(l); });
  }

  function writeEcho(cmdText) {
    writeLine(span('token-success', '$ ') + escapeHtml(cmdText));
  }

  function clearTerminal() {
    while (terminalBody.firstChild !== inputRow) {
      terminalBody.removeChild(terminalBody.firstChild);
    }
  }

  function processCommand() {
    var cmd = currentInput;
    currentInput = '';
    updateInputDisplay();

    writeEcho(cmd);

    if (cmd.trim()) {
      commandHistory.push(cmd);
      historyIndex = commandHistory.length;
    }

    if (cmd.trim().toLowerCase() === 'clear') {
      clearTerminal();
      scrollToBottom();
      if (tutorialActive) checkTutorialStep(cmd.trim());
      return;
    }

    var output = executeCommand(cmd);
    if (output && output.length > 0) {
      writeLines(output);
    }

    writeLine(''); // blank line after output
    scrollToBottom();

    // Tutorial validation
    if (tutorialActive) {
      checkTutorialStep(cmd.trim());
    }
  }

  // Focus management
  function focusTerminal() {
    hiddenInput.focus();
  }

  terminalBody.addEventListener('click', focusTerminal);

  hiddenInput.addEventListener('input', function () {
    currentInput = hiddenInput.value;
    updateInputDisplay();
    scrollToBottom();
  });

  hiddenInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      processCommand();
      hiddenInput.value = '';
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0 && historyIndex > 0) {
        historyIndex--;
        currentInput = commandHistory[historyIndex];
        hiddenInput.value = currentInput;
        updateInputDisplay();
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        historyIndex++;
        currentInput = commandHistory[historyIndex];
        hiddenInput.value = currentInput;
        updateInputDisplay();
      } else {
        historyIndex = commandHistory.length;
        currentInput = '';
        hiddenInput.value = '';
        updateInputDisplay();
      }
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      clearTerminal();
      scrollToBottom();
    }
  });

  // Focus/blur cursor animation
  hiddenInput.addEventListener('focus', function () {
    getCursorEl().classList.remove('no-blink');
  });

  hiddenInput.addEventListener('blur', function () {
    getCursorEl().classList.add('no-blink');
  });

  // ---------------------------------------------------------------------------
  // Tutorial System
  // ---------------------------------------------------------------------------
  var tutorialActive = false;
  var tutorialStep = 0;

  var tutorialSteps = [
    {
      instruction: 'Let\'s create a project! Initialize a new PlanDB project called "my-app".',
      hint: 'plandb init my-app',
      pattern: /^plandb\s+init\s+\S+/i,
      success: 'Project created! Now let\'s add some tasks.'
    },
    {
      instruction: 'Add your first task. Let\'s add a task for setting up the database.',
      hint: 'plandb add "Setup database"',
      pattern: /^plandb\s+add\s+.+/i,
      success: 'Great! You\'ve got your first task. Let\'s add another one with a dependency.'
    },
    {
      instruction: 'Add a task that depends on the first one. Use --dep with the task ID shown above.',
      hint: 'plandb add "Build API" --dep t-xxxx',
      pattern: /^plandb\s+add\s+.+--dep\s+t-/i,
      success: 'Notice how the dependent task starts as "pending" since its dependency isn\'t done yet.'
    },
    {
      instruction: 'Add one more independent task to see parallel work.',
      hint: 'plandb add "Write tests" --kind test',
      pattern: /^plandb\s+add\s+.+/i,
      success: 'Good. Now let\'s see the project overview.'
    },
    {
      instruction: 'Check your project status with the detail view to see the dependency tree.',
      hint: 'plandb status --detail',
      pattern: /^plandb\s+status/i,
      success: 'You can see the tree structure and progress. Time to start working!'
    },
    {
      instruction: 'Claim the next ready task to start working on it.',
      hint: 'plandb go',
      pattern: /^plandb\s+go$/i,
      success: 'You\'re now working on a task. Complete it when ready.'
    },
    {
      instruction: 'Complete the current task and automatically claim the next one.',
      hint: 'plandb done --next',
      pattern: /^plandb\s+done/i,
      success: 'Tasks completed and next one claimed. Dependencies are automatically resolved!'
    },
    {
      instruction: 'Let\'s say the current task is too big. Split it into sub-tasks.',
      hint: 'plandb split --into "Part A > Part B > Part C"',
      pattern: /^plandb\s+split\s+--into\s+.+/i,
      success: 'The task was split! Using ">" creates a chain where each depends on the previous.'
    },
    {
      instruction: 'Check the final status to see everything come together.',
      hint: 'plandb status --detail',
      pattern: /^plandb\s+status/i,
      success: 'Congratulations! You\'ve learned the core PlanDB workflow.'
    }
  ];

  var tutorialContentEl = document.getElementById('tutorial-content');

  function renderTutorialIdle() {
    tutorialContentEl.innerHTML =
      '<div class="tutorial-idle">' +
        '<p>Learn PlanDB step by step with a guided walkthrough.</p>' +
        '<button class="btn btn-primary" id="start-tutorial-btn">Start Tutorial</button>' +
      '</div>';
    document.getElementById('start-tutorial-btn').addEventListener('click', startTutorial);
  }

  function renderTutorialStep() {
    var step = tutorialSteps[tutorialStep];
    var total = tutorialSteps.length;
    var pct = Math.round((tutorialStep / total) * 100);

    tutorialContentEl.innerHTML =
      '<div class="tutorial-progress">Step ' + (tutorialStep + 1) + ' of ' + total + '</div>' +
      '<div class="tutorial-progress-bar"><div class="tutorial-progress-fill" style="width:' + pct + '%"></div></div>' +
      '<div class="tutorial-step-text">' + escapeHtml(step.instruction) + '</div>' +
      '<div class="tutorial-step-hint">' + escapeHtml(step.hint) + '</div>' +
      '<div style="font-size:var(--text-xs);color:var(--color-text-muted);">Type "exit tutorial" to quit anytime.</div>';
  }

  function renderTutorialComplete() {
    tutorialContentEl.innerHTML =
      '<div class="tutorial-progress-bar"><div class="tutorial-progress-fill" style="width:100%"></div></div>' +
      '<div class="tutorial-complete">' +
        '<p>You\'ve completed the tutorial!</p>' +
        '<p style="color:var(--color-text-dim);margin-top:var(--space-2);">Keep exploring — try "help" for all commands.</p>' +
        '<button class="btn btn-secondary mt-4" id="restart-tutorial-btn" style="margin-top:var(--space-4);">Restart Tutorial</button>' +
      '</div>';
    document.getElementById('restart-tutorial-btn').addEventListener('click', function () {
      resetState();
      clearTerminal();
      startTutorial();
    });
  }

  function startTutorial() {
    tutorialActive = true;
    tutorialStep = 0;
    resetState();
    clearTerminal();

    writeLines([
      accent('Welcome to the PlanDB Tutorial!'),
      dim('Follow the steps in the panel on the right.'),
      dim('Type commands below and press Enter.'),
      ''
    ]);
    scrollToBottom();
    renderTutorialStep();
    focusTerminal();
  }

  function exitTutorial() {
    tutorialActive = false;
    tutorialStep = 0;
    renderTutorialIdle();
  }

  function checkTutorialStep(cmd) {
    if (!tutorialActive || tutorialStep >= tutorialSteps.length) return;
    var step = tutorialSteps[tutorialStep];
    if (step.pattern.test(cmd)) {
      // Success — show message in terminal
      writeLine(green('  ✓ ' + step.success));
      writeLine('');
      scrollToBottom();

      tutorialStep++;
      if (tutorialStep >= tutorialSteps.length) {
        tutorialActive = false;
        renderTutorialComplete();
      } else {
        renderTutorialStep();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  renderTutorialIdle();

  // Welcome message
  writeLines([
    accent('PlanDB Playground') + dim(' — interactive terminal simulator'),
    dim('Type "help" for commands or "start tutorial" for a guided walkthrough.'),
    ''
  ]);
  scrollToBottom();
  focusTerminal();

})();
