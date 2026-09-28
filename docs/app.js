/**
 * LearnRL web sandbox — tabular RL engine in JS.
 * Mirrors the Python package's GridWorld / Q / SARSA / Value Iteration core.
 */
"use strict";

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

const ACTIONS = ["U", "D", "L", "R"];
const DELTA = { U: [-1, 0], D: [1, 0], L: [0, -1], R: [0, 1] };
const ARROW = { U: "↑", D: "↓", L: "←", R: "→" };

function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return {
    random() {
      s = (1664525 * s + 1013904223) >>> 0;
      return (s >>> 8) / 0x1000000;
    },
    pick(arr) {
      s = (1664525 * s + 1013904223) >>> 0;
      return arr[(s >>> 8) % arr.length];
    },
    get state() { return s; },
  };
}

function parseGrid(rows) {
  const walls = new Set();
  const starts = [];
  const goals = new Set();
  const pits = new Set();
  const free = [];
  rows.forEach((row, r) => {
    for (let c = 0; c < row.length; c++) {
      const ch = row[c];
      const key = `${r},${c}`;
      if (ch === "#") walls.add(key);
      else free.push(key);
      if (ch === "S") starts.push(key);
      if (ch === "G") goals.add(key);
      if (ch === "P") pits.add(key);
    }
  });
  return {
    rows,
    walls,
    starts: starts.length ? starts : [free[0] || "0,0"],
    goals,
    pits,
    free,
    h: rows.length,
    w: rows[0].length,
  };
}

class GridWorld {
  constructor(spec) {
    this.name = spec.name || "gridworld";
    this.map = parseGrid(spec.grid);
    this.gamma = spec.gamma ?? 0.9;
    this.slip = spec.slip ?? 0;
    this.rewards = {
      step: spec.rewards?.step ?? -1,
      goal: spec.rewards?.goal ?? 10,
      pit: spec.rewards?.pit ?? -10,
      wall: spec.rewards?.wall ?? -1,
    };
    this.rng = lcg(spec.seed ?? 1);
    this.state = null;
    this.done = true;
  }

  reset(seed) {
    this.rng = lcg(seed ?? this.rng.state);
    this.state = this.rng.pick(this.map.starts);
    this.done = false;
    return this.state;
  }

  cellKind(key) {
    if (this.map.walls.has(key)) return "wall";
    if (this.map.goals.has(key)) return "goal";
    if (this.map.pits.has(key)) return "pit";
    if (this.map.starts.includes(key)) return "start";
    return "empty";
  }

  isTerminal(key) {
    return this.map.goals.has(key) || this.map.pits.has(key);
  }

  move(from, action) {
    const [dr, dc] = DELTA[action];
    const [r, c] = from.split(",").map(Number);
    const nr = r + dr;
    const nc = c + dc;
    const next = `${nr},${nc}`;
    if (nr < 0 || nc < 0 || nr >= this.map.h || nc >= this.map.w || this.map.walls.has(next)) {
      return { next: from, hitWall: true };
    }
    return { next, hitWall: false };
  }

  step(action) {
    if (this.done) throw new Error("episode done — reset first");
    let act = action;
    if (this.slip > 0 && this.rng.random() < this.slip) {
      const perp = ACTIONS.filter((a) => {
        const d1 = DELTA[a], d2 = DELTA[act];
        return d1[0] * d2[0] + d1[1] * d2[1] === 0;
      });
      act = this.rng.pick(perp);
    }
    const prev = this.state;
    const { next, hitWall } = this.move(prev, act);
    this.state = next;
    let reward, done;
    if (hitWall) {
      reward = this.rewards.wall; done = false;
    } else if (this.map.goals.has(next)) {
      reward = this.rewards.goal; done = true;
    } else if (this.map.pits.has(next)) {
      reward = this.rewards.pit; done = true;
    } else {
      reward = this.rewards.step; done = false;
    }
    this.done = done;
    return { prev, action: act, reward, next, done };
  }
}

/** Q-learning / SARSA / Expected SARSA tabular agent. */
class QAgent {
  constructor(actions, opts = {}) {
    this.actions = actions.slice();
    this.mode = opts.mode || "q";
    this.alpha = opts.alpha ?? 0.25;
    this.gamma = opts.gamma ?? 0.9;
    this.epsilon = opts.epsilon ?? 0.15;
    this.rng = lcg(opts.seed ?? 7);
    this.q = new Map();
    this.episode = [];
  }

  key(s, a) { return `${s}|${a}`; }

  get(s, a) { return this.q.get(this.key(s, a)) ?? 0; }

  set(s, a, v) { this.q.set(this.key(s, a), v); }

  greedy(s) {
    let best = this.actions[0], bestV = this.get(s, best);
    for (const a of this.actions.slice(1)) {
      const v = this.get(s, a);
      if (v > bestV + 1e-9) { best = a; bestV = v; }
    }
    return best;
  }

  act(s, explore = true) {
    if (explore && this.rng.random() < this.epsilon) return this.rng.pick(this.actions);
    return this.greedy(s);
  }

  values(s) {
    const out = {};
    for (const a of this.actions) out[a] = this.get(s, a);
    return out;
  }

  learn(tr) {
    const [s, a, r, s2, done] = tr;
    const q = this.get(s, a);
    let target;
    if (done) target = r;
    else if (this.mode === "q") target = r + this.gamma * Math.max(...this.actions.map((a) => this.get(s2, a)));
    else if (this.mode === "sarsa") target = r + this.gamma * this.get(s2, this.greedy(s2));
    else {
      const n = this.actions.length;
      const greedy = this.greedy(s2);
      let exp = 0;
      for (const a of this.actions) {
        const p = this.epsilon / n + (a === greedy ? 1 - this.epsilon : 0);
        exp += p * this.get(s2, a);
      }
      target = r + this.gamma * exp;
    }
    const td = target - q;
    this.set(s, a, q + this.alpha * td);
    return td;
  }
}

/** Synchronous value iteration (known model). */
function valueIteration(env, gamma, iters = 50) {
  const V = new Map();
  for (const s of env.map.free) V.set(s, 0);
  for (let it = 0; it < iters; it++) {
    let delta = 0;
    for (const s of env.map.free) {
      if (env.isTerminal(s)) continue;
      let best = -Infinity;
      for (const a of ACTIONS) {
        const { next, hitWall } = env.move(s, a);
        let r, done;
        if (hitWall) { r = env.rewards.wall; done = false; }
        else if (env.map.goals.has(next)) { r = env.rewards.goal; done = true; }
        else if (env.map.pits.has(next)) { r = env.rewards.pit; done = true; }
        else { r = env.rewards.step; done = false; }
        const boot = done ? 0 : V.get(next) ?? 0;
        best = Math.max(best, r + gamma * boot);
      }
      const nv = best === -Infinity ? 0 : best;
      delta = Math.max(delta, Math.abs(nv - (V.get(s) ?? 0)));
      V.set(s, nv);
    }
    if (delta < 1e-4) break;
  }
  const policy = new Map();
  for (const s of env.map.free) {
    if (env.isTerminal(s)) continue;
    let bestA = "U", bestQ = -Infinity;
    for (const a of ACTIONS) {
      const { next, hitWall } = env.move(s, a);
      let r, done;
      if (hitWall) { r = env.rewards.wall; done = false; }
      else if (env.map.goals.has(next)) { r = env.rewards.goal; done = true; }
      else if (env.map.pits.has(next)) { r = env.rewards.pit; done = true; }
      else { r = env.rewards.step; done = false; }
      const boot = done ? 0 : V.get(next) ?? 0;
      const q = r + gamma * boot;
      if (q > bestQ + 1e-9) { bestQ = q; bestA = a; }
    }
    policy.set(s, bestA);
  }
  return { V, policy };
}

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

const LEVELS = [
  {
    id: "intro.goal",
    name: "Reach the Goal",
    desc: "Step until you land on G. The agent starts random — keep going.",
    goal: { episodeDone: true, minReturn: 0 },
    env: { name: "intro", grid: ["S.", ".G"], gamma: 0.9, seed: 2, rewards: { step: -1, goal: 10 } },
  },
  {
    id: "intro.pit",
    name: "Mind the Pit",
    desc: "P is a pit (−50). Reach G with return ≥ −5.",
    goal: { episodeDone: true, minReturn: -5 },
    env: { name: "pit", grid: ["S.P", "..P", "..G"], gamma: 0.9, seed: 3, rewards: { step: -1, goal: 5, pit: -50 } },
  },
  {
    id: "dp.values",
    name: "Value Iteration",
    desc: "Train with value iteration and inspect V(s). V(start) should be near +8.",
    goal: { valueClose: { "0,0": 8, tol: 1.5 } },
    env: { name: "corridor", grid: ["S.G"], gamma: 0.9, seed: 9, rewards: { step: -1, goal: 10 } },
  },
  {
    id: "td.qlearning",
    name: "Q-Learning",
    desc: "Train Q-learning on this map. Eval mean return should be ≥ 5.",
    goal: { minEvalReturn: 5, evalEpisodes: 15 },
    env: { name: "tiny", grid: ["S.", ".G"], gamma: 0.9, seed: 13, rewards: { step: -1, goal: 10 } },
  },
  {
    id: "advanced.cliff",
    name: "The Cliff",
    desc: "Sutton & Barto's cliff. Q-learning finds the risky optimal path along the top.",
    goal: { minEvalReturn: -15, evalEpisodes: 15 },
    env: { name: "cliff", grid: ["S.......G", ".........", "PPPPPPPPP"], gamma: 1, seed: 15, rewards: { step: -1, goal: -1, pit: -100 } },
  },
];

const SEQUENCES = [
  {
    title: "intro — Introduction",
    about: "States, actions, rewards, episodes.",
    items: ["Take a step", "Reach the goal", "Mind the pit", "Undo / reset"],
  },
  {
    title: "prediction — MC & TD",
    about: "Estimate V^π before you control it.",
    items: ["Monte Carlo prediction", "TD(0)", "n-step returns"],
  },
  {
    title: "dp — Dynamic Programming",
    about: "Policy evaluation, improvement, value iteration.",
    items: ["Policy evaluation", "Policy improvement", "VI vs PI"],
  },
  {
    title: "td — Control",
    about: "Learn from interaction.",
    items: ["SARSA", "Q-learning", "Expected SARSA"],
  },
  {
    title: "advanced — Trade-offs",
    about: "Exploration pitfalls and on/off-policy cliffs.",
    items: ["Cliff walk (Q)", "Cliff walk (SARSA safe)", "Double Q", "Command golf"],
  },
  {
    title: "CLI package",
    about: "Python engine: 20+ levels, MC/TD/n-step, goal language, golf.",
    items: ["pip install learnrl", "python -m learnrl", "build level / import"],
  },
];

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

const state = {
  env: null,
  agent: null,
  level: LEVELS[0],
  episode: { steps: 0, reward: 0, done: false, history: [] },
  commands: 0,
  view: "map",
  history: [],
  learningCurve: [],
  lastEval: null,
};

function snapshot() {
  return {
    envState: state.env.state,
    envDone: state.env.done,
    envSeed: state.env.rng.state,
    q: new Map(state.agent.q),
    episode: JSON.parse(JSON.stringify(state.episode)),
    commands: state.commands,
    curve: state.learningCurve.slice(),
    agentAlpha: state.agent.alpha,
    agentGamma: state.agent.gamma,
    agentEps: state.agent.epsilon,
  };
}

function restore(snap) {
  state.env.state = snap.envState;
  state.env.done = snap.envDone;
  state.env.rng = lcg(snap.envSeed);
  state.agent.q = new Map(snap.q);
  state.episode = snap.episode;
  state.commands = snap.commands;
  state.learningCurve = snap.curve.slice();
  state.agent.alpha = snap.agentAlpha;
  state.agent.gamma = snap.agentGamma;
  state.agent.epsilon = snap.agentEps;
}

function pushUndo() {
  state.history.push(snapshot());
  if (state.history.length > 40) state.history.shift();
}

function loadLevel(level) {
  state.level = level;
  state.env = new GridWorld(level.env);
  state.agent = new QAgent(ACTIONS, {
    mode: "q",
    alpha: +document.getElementById("alpha").value,
    gamma: +document.getElementById("gamma").value,
    epsilon: +document.getElementById("eps").value,
    seed: (level.env.seed || 1) + 99,
  });
  state.env.reset(level.env.seed || 1);
  state.episode = { steps: 0, reward: 0, done: false, history: [] };
  state.commands = 0;
  state.history = [];
  state.learningCurve = [];
  state.lastEval = null;
  document.getElementById("levelDesc").textContent = level.desc;
  renderGoal(false);
  render();
}

function checkGoal() {
  const g = state.level.goal;
  const parts = [];
  let ok = true;
  const fail = (msg) => { parts.push("✗ " + msg); ok = false; };
  const pass = (msg) => parts.push("✓ " + msg);

  if (g.episodeDone) {
    state.episode.done ? pass("episode finished") : fail("episode not finished");
  }
  if (g.minReturn !== undefined) {
    (state.episode.reward >= g.minReturn ? pass : fail)(`return ≥ ${g.minReturn} (got ${state.episode.reward.toFixed(2)})`);
  }
  if (g.valueClose) {
    const { tol = 1.5, ...targets } = g.valueClose;
    for (const [key, target] of Object.entries(targets)) {
      const v = valueOfState(key);
      if (v === null) fail(`V(${key}) missing`);
      else if (Math.abs(v - target) <= tol) pass(`V(${key}) ≈ ${target} (${v.toFixed(2)})`);
      else fail(`V(${key}) ≈ ${target} (got ${v.toFixed(2)})`);
    }
  }
  if (g.minEvalReturn !== undefined) {
    const ev = evaluate(g.evalEpisodes || 15);
    if (ev.mean >= g.minEvalReturn) pass(`mean eval ≥ ${g.minEvalReturn} (${ev.mean.toFixed(2)})`);
    else fail(`mean eval ≥ ${g.minEvalReturn} (got ${ev.mean.toFixed(2)})`);
  }
  return { ok, parts };
}

function valueOfState(key) {
  const q = state.agent.values(key);
  const vals = Object.values(q);
  if (!vals.length) return null;
  return Math.max(...vals);
}

function evaluate(episodes = 15) {
  const returns = [];
  for (let e = 0; e < episodes; e++) {
    const env = new GridWorld(state.level.env);
    let s = env.reset(10000 + e);
    let total = 0;
    for (let t = 0; t < 200; t++) {
      if (env.isTerminal(s)) break;
      const a = state.agent.greedy(s);
      const r = env.step(a);
      total += r.reward;
      s = r.next;
      if (r.done) break;
    }
    returns.push(total);
  }
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  state.lastEval = { mean, returns };
  return { mean, returns };
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

function runCommand(raw) {
  const text = raw.trim();
  if (!text) return;
  log(`> ${text}`);
  const parts = text.split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const args = parts.slice(1);

  try {
    switch (cmd) {
      case "help":
        log("help · show/map · values · policy · q · step [n] · reset · undo\ntrain q|sarsa|vi|expected [n] · evaluate [n] · goal · levels · clear");
        break;
      case "clear":
        document.getElementById("output").textContent = "";
        break;
      case "show":
      case "map":
      case "values":
      case "policy":
      case "q":
        state.view = cmd === "map" || cmd === "show" ? "map" : cmd;
        syncTabs();
        render();
        break;
      case "step": {
        pushUndo();
        const n = parseInt(args[0] || "1", 10) || 1;
        let last = null;
        for (let i = 0; i < n; i++) {
          if (state.episode.done) break;
          const a = state.agent.act(state.env.state, true);
          const r = state.env.step(a);
          state.episode.history.push(r);
          state.episode.steps += 1;
          state.episode.reward += r.reward;
          state.episode.done = r.done;
          state.agent.learn([r.prev, r.action, r.reward, r.next, r.done]);
          last = r;
          if (r.done) break;
        }
        state.commands += 1;
        if (last) log(`step → ${last.next}  r=${last.reward.toFixed(2)}  return=${state.episode.reward.toFixed(2)}`);
        if (state.episode.done) {
          log(`episode done · return=${state.episode.reward.toFixed(2)}`);
          const g = checkGoal();
          g.parts.forEach((p) => log(p));
          if (g.ok) log("LEVEL GOAL MET");
          renderGoal(g.ok, g.parts);
        }
        render();
        break;
      }
      case "reset":
        pushUndo();
        state.env.reset(state.level.env.seed || 1);
        state.episode = { steps: 0, reward: 0, done: false, history: [] };
        state.commands += 1;
        log("episode reset");
        render();
        break;
      case "undo":
        if (state.history.length) {
          restore(state.history.pop());
          log("undid last command");
          render();
        } else log("nothing to undo");
        break;
      case "train": {
        pushUndo();
        const algo = (args[0] || "q").toLowerCase();
        const n = parseInt(args[1] || "50", 10) || 50;
        state.commands += 1;
        if (algo === "vi") {
          const { V, policy } = valueIteration(state.env, state.agent.gamma, n);
          // Store VI values into Q as greedy-implied for display
          state.agent.q.clear();
          for (const [s, a] of policy.entries()) {
            state.agent.set(s, a, V.get(s) ?? 0);
            for (const other of ACTIONS) if (other !== a) state.agent.set(s, other, (V.get(s) ?? 0) - 0.01);
          }
          log(`value iteration ×${n} · V(start)≈${(V.get(state.env.map.starts[0]) ?? 0).toFixed(2)}`);
        } else {
          state.agent.mode = algo === "sarsa" ? "sarsa" : algo === "expected" ? "expected" : "q";
          let sum = 0;
          for (let e = 0; e < n; e++) {
            const env = state.env;
            // fresh episode on a cloned env for training isolation
            const tmp = new GridWorld(state.level.env);
            let s = tmp.reset(2000 + e);
            let total = 0;
            for (let t = 0; t < 200; t++) {
              const a = state.agent.act(s, true);
              const r = tmp.step(a);
              state.agent.learn([r.prev, r.action, r.reward, r.next, r.done]);
              total += r.reward;
              s = r.next;
              if (r.done) break;
            }
            sum += total;
            state.learningCurve.push(total);
          }
          log(`train ${state.agent.mode} ×${n} · mean_return=${(sum / n).toFixed(2)}`);
        }
        const g = checkGoal();
        g.parts.forEach((p) => log(p));
        if (g.ok) log("LEVEL GOAL MET");
        renderGoal(g.ok, g.parts);
        render();
        break;
      }
      case "evaluate": {
        pushUndo();
        const n = parseInt(args[0] || "15", 10) || 15;
        state.commands += 1;
        const ev = evaluate(n);
        log(`eval ×${n} · mean=${ev.mean.toFixed(2)}`);
        const g = checkGoal();
        g.parts.forEach((p) => log(p));
        if (g.ok) log("LEVEL GOAL MET");
        renderGoal(g.ok, g.parts);
        render();
        break;
      }
      case "goal": {
        const g = checkGoal();
        g.parts.forEach((p) => log(p));
        renderGoal(g.ok, g.parts);
        break;
      }
      case "levels":
        log(LEVELS.map((l) => `${l.id.padEnd(18)} ${l.name}`).join("\n"));
        break;
      default:
        log(`unknown command: ${cmd} — try help`);
    }
  } catch (err) {
    log(String(err.message || err));
  }
}

function log(msg) {
  const el = document.getElementById("output");
  el.textContent += (el.textContent ? "\n" : "") + msg;
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render() {
  const vp = document.getElementById("viewport");
  const env = state.env;
  if (!env) return;

  if (state.view === "map") {
    vp.innerHTML = renderMap(env);
    return;
  }
  if (state.view === "values") {
    vp.innerHTML = renderValues(env);
    return;
  }
  if (state.view === "policy") {
    vp.innerHTML = renderPolicy(env);
    return;
  }
  vp.innerHTML = renderQ(env);
}

function renderMap(env) {
  const cells = [];
  for (let r = 0; r < env.map.h; r++) {
    for (let c = 0; c < env.map.w; c++) {
      const key = `${r},${c}`;
      const kind = env.cellKind(key);
      const isAgent = env.state === key && kind !== "wall";
      const cls = ["cell", kind === "empty" ? "" : kind, isAgent ? "agent" : ""]
        .filter(Boolean).join(" ");
      let inner = isAgent ? "A" : kind === "goal" ? "G" : kind === "pit" ? "P" : kind === "start" ? "S" : "·";
      cells.push(`<div class="${cls}">${inner}</div>`);
    }
  }
  return `<div class="grid" style="grid-template-columns:repeat(${env.map.w},52px)">${cells.join("")}</div>`;
}

function renderValues(env) {
  const cells = [];
  for (let r = 0; r < env.map.h; r++) {
    for (let c = 0; c < env.map.w; c++) {
      const key = `${r},${c}`;
      const kind = env.cellKind(key);
      const cls = ["cell", kind === "empty" ? "" : kind].filter(Boolean).join(" ");
      if (kind === "wall") { cells.push(`<div class="${cls}"></div>`); continue; }
      const v = valueOfState(key) ?? 0;
      const isAgent = env.state === key;
      cells.push(
        `<div class="${cls}${isAgent ? " agent" : ""}"><span class="val">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span></div>`
      );
    }
  }
  return `<div class="grid" style="grid-template-columns:repeat(${env.map.w},52px)">${cells.join("")}</div>`;
}

function renderPolicy(env) {
  const cells = [];
  for (let r = 0; r < env.map.h; r++) {
    for (let c = 0; c < env.map.w; c++) {
      const key = `${r},${c}`;
      const kind = env.cellKind(key);
      const cls = ["cell", kind === "empty" ? "" : kind].filter(Boolean).join(" ");
      if (kind === "wall") { cells.push(`<div class="${cls}"></div>`); continue; }
      if (kind === "goal") { cells.push(`<div class="${cls}">G</div>`); continue; }
      if (kind === "pit") { cells.push(`<div class="${cls}">P</div>`); continue; }
      const a = state.agent.greedy(key);
      const isAgent = env.state === key;
      cells.push(
        `<div class="${cls}${isAgent ? " agent" : ""}"><span class="arrow">${ARROW[a]}</span></div>`
      );
    }
  }
  return `<div class="grid" style="grid-template-columns:repeat(${env.map.w},52px)">${cells.join("")}</div>`;
}

function renderQ(env) {
  const rows = [];
  for (const s of env.map.free) {
    if (env.isTerminal(s)) continue;
    const q = state.agent.values(s);
    const best = state.agent.greedy(s);
    const parts = ACTIONS.map((a) => {
      const v = q[a] ?? 0;
      const mark = a === best ? "*" : " ";
      return `${ARROW[a]}${mark}${v.toFixed(2)}`;
    });
    rows.push(`<div style="font-family:var(--mono);font-size:0.78rem;color:var(--muted);margin:0.25rem 0">
      <b style="color:var(--ink)">${s}</b>  ${parts.join("  ")}
    </div>`);
  }
  return `<div style="width:100%;padding:0.5rem">${rows.join("")}
    <div style="margin-top:0.75rem;font-size:0.72rem;color:var(--muted)">format: action*best value</div></div>`;
}

function renderGoal(ok, parts) {
  const el = document.getElementById("goalBox");
  if (!parts || !parts.length) {
    el.className = "goal";
    el.textContent = describeGoal(state.level.goal);
    return;
  }
  el.className = "goal" + (ok ? " ok" : "");
  el.textContent = parts.join("\n") + (ok ? "\n\nLEVEL GOAL MET" : "");
}

function describeGoal(g) {
  const lines = [];
  if (g.episodeDone) lines.push("• finish the episode");
  if (g.minReturn !== undefined) lines.push(`• return ≥ ${g.minReturn}`);
  if (g.valueClose) {
    const { tol, ...t } = g.valueClose;
    for (const [k, v] of Object.entries(t)) lines.push(`• V(${k}) ≈ ${v} ± ${tol ?? 1.5}`);
  }
  if (g.minEvalReturn !== undefined) lines.push(`• mean eval return ≥ ${g.minEvalReturn}`);
  return lines.join("\n") || "sandbox";
}

function syncTabs() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.view === state.view);
  });
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------

function init() {
  const sel = document.getElementById("levelSelect");
  LEVELS.forEach((lv, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${lv.id} — ${lv.name}`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    loadLevel(LEVELS[+sel.value]);
    log(`loaded ${LEVELS[+sel.value].id}`);
  });

  const grid = document.getElementById("seqGrid");
  grid.innerHTML = SEQUENCES.map((s) => `
    <article class="seq">
      <h3>${s.title}</h3>
      <p class="about">${s.about}</p>
      <ul>${s.items.map((it) => `<li><b>${it}</b><span></span></li>`).join("")}</ul>
    </article>
  `).join("");

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.view = tab.dataset.view;
      syncTabs();
      render();
    });
  });

  document.querySelectorAll("[data-cmd]").forEach((btn) => {
    btn.addEventListener("click", () => runCommand(btn.dataset.cmd));
  });

  document.getElementById("cmdForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("cmdInput");
    runCommand(input.value);
    input.value = "";
  });

  const bind = (id, out, fmt = (v) => v) => {
    const el = document.getElementById(id);
    const o = document.getElementById(out);
    el.addEventListener("input", () => {
      o.textContent = fmt(+el.value);
      if (state.agent) {
        if (id === "eps") state.agent.epsilon = +el.value;
        if (id === "alpha") state.agent.alpha = +el.value;
        if (id === "gamma") state.agent.gamma = +el.value;
      }
    });
  };
  bind("eps", "epsOut", (v) => v.toFixed(2));
  bind("alpha", "alphaOut", (v) => v.toFixed(2));
  bind("gamma", "gammaOut", (v) => v.toFixed(2));

  loadLevel(LEVELS[0]);
  log("LearnRL web sandbox ready. Type help.");
}

document.addEventListener("DOMContentLoaded", init);
