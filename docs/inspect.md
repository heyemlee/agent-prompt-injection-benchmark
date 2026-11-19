# injbench on Inspect (M6)

injbench also runs on the UK AISI [Inspect](https://inspect.aisi.org.uk/) framework (`inspect_ai`),
with the **same episodes, tools, defenses and deterministic graders** as the native runner. The goal
is a later proposal to [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals); nothing
has been submitted.

- **Code:** `src/injbench/inspect_port/`.
  - `episodes.py`: the Inspect-free core.
  - `task.py`, `solver.py`, `scorer.py`, `logs.py`: the Inspect parts.
- **Tests:** `tests/test_inspect_port.py` checks parity with `runner.run_episode` sample by sample.
- **Status:** works offline with Inspect's mock model. No real model has been run through the port
  yet, so there are no Inspect results.

## Install

```bash
uv sync --extra inspect                    # inspect-ai >= 0.3.223 (tested 0.3.223 and 0.3.291)
uv run pytest tests/test_inspect_port.py   # offline parity tests, no network
```

Without the extra, the core package and `uv run pytest` work as before: the Inspect half of the
tests is skipped. `import injbench.inspect_port` itself does not import `inspect_ai`.

Two things Inspect does on its own:

- **API keys.** Inspect loads the first `.env` it finds in the working directory or any parent. The
  repository's `.env` (the one the native CLI reads) works as is.
- **Logs.** Inspect writes logs to `./logs` by default, and they contain the attack payloads.
  `logs/` is not git-ignored, so always pass `--log-dir runs/inspect/...`, because `runs/` is
  ignored (PLAN §14: never commit generated attack strings).

## Run

The task is registered as `injbench/injbench`. One run is one defense arm.

**Claude (Anthropic API):**

```bash
uv run inspect eval injbench/injbench --model anthropic/claude-sonnet-5-5 \
  -T defense=permissions --reasoning-effort medium --max-tokens 16000 \
  --log-dir runs/inspect/claude-sonnet-5-5
```

`--reasoning-effort medium` becomes `output_config.effort = "medium"` in the request, as in
`configs/anthropic_smoke.toml`. Temperature is not sent unless you pass `--temperature`.

**Any OpenAI-compatible endpoint.** For example, Qwen3.5-9B on SiliconFlow, as in
`configs/qwen_siliconflow.toml`:

```bash
export SILICONFLOW_BASE_URL=https://api.siliconflow.com/v1   # SILICONFLOW_API_KEY comes from .env
uv run inspect eval injbench/injbench --model openai-api/siliconflow/Qwen/Qwen3.5-9B \
  -M strict_tools=false --extra-body '{"enable_thinking": false}' --max-tokens 8192 \
  -T defense=none --log-dir runs/inspect/qwen3.5-9b-siliconflow
```

- **Model names.** `openai-api/<service>/<model>` reads `<SERVICE>_API_KEY` and `<SERVICE>_BASE_URL`.
- **`-M strict_tools=false`.** Inspect's OpenAI-compatible providers mark every tool `strict: true`
  by default. injbench's own client does not, so turn it off for parity.
- **OpenRouter** (`configs/qwen_openrouter.toml`):
  `--model openrouter/qwen/qwen3.5-9b -M strict_tools=false -M provider='{"order": ["deepinfra"], "allow_fallbacks": true, "require_parameters": true, "quantizations": ["bf16"]}'`.

**Both arms, paired.** Run the arms one after another with Inspect's response cache. Identical
requests then get the same sampled response, and the arms differ only once the defense changes
what the model sees (PLAN §7, "Paired arms"):

```bash
for d in none permissions; do
  uv run inspect eval injbench/injbench --model anthropic/claude-haiku-5-5 -T defense=$d \
    --cache --log-dir runs/inspect/claude-haiku-5-5
done
```

**The smoke matrix.** These task arguments give the 22 episodes per arm of the native smoke configs:

```bash
-T templates=sla_summary_report,triage_new_tickets,kb_answer_comment,close_confirmed \
-T families=info_block,business_note -T goals=grant_admin,falsify_count,close_unrequested
```

Without them, the task runs every registered template, family and goal.

- **Count samples first.** There is no `--dry-run`; use
  `uv run python -c "from injbench.inspect_port import injbench; print(len(injbench(defense='none').dataset))"`.
- **Other useful options.**
  - `--limit N`.
  - `--max-samples` (≈ native `concurrency`); `--max-connections` caps concurrent API calls.
  - `--no-fail-on-error` or `--retry-on-error` for long runs.
  - Browse logs with `uv run inspect view --log-dir runs/inspect`.

**From Python:**

```python
from inspect_ai import eval
from injbench.inspect_port import injbench

for d in ("none", "permissions"):  # one after another, so the cache pairs the arms
    eval(injbench(defense=d, families="info_block,business_note"),
         model="anthropic/claude-sonnet-5-5", cache=True, log_dir="runs/inspect/sonnet")
```

### Task parameters

| Parameter | Default | Meaning |
|---|---|---|
| `templates` | all registered | task templates (list or comma-separated) |
| `instances_per_template` | 1 | task instances per template |
| `families` | all | attack families; `-T families=[]` runs benign episodes only |
| `goals` | all | attack goals (each only on the templates it applies to) |
| `defense` | `none` | one arm from `injbench.defenses.DEFENSES` |
| `max_steps` | 15 | maximum model calls per episode (native `max_steps`) |
| `include_benign` | true | also run each task instance without an attack |
| `world_seed`, `instance_seed` | 0 | as in `RunConfig` |

Templates, families, goals, defenses and tools are read from injbench's registries at run time,
so additions show up without changes to the port.

## How it maps onto injbench

**Samples.** Built by `runner.build_matrix`, so they are exactly the native episodes: every task
instance once benign and once per applicable family × goal.

- **ID.** `<instance_id>|benign` or `<instance_id>|<family>+<goal>`: the native episode ID without
  the defense, model and attempt parts. The defense is the task's, the model is Inspect's, and the
  attempt is Inspect's epoch.
- **Input.** The principal's request. There is no target, because grading checks the final state,
  not an answer.
- **Metadata.** `template`, `instance_id`, `condition` (`benign`/`attack`), `family`, `goal`,
  `goal_class`, `defense` and `principal_team`, plus `injbench_matrix`. With the matrix
  parameters, the solver and scorer rebuild the sample's scenario deterministically.
- **Task metadata.** `injbench_version`, `world_fingerprint`, `defense`, `max_steps` and the matrix.

**Solver (`injbench_agent`).** A custom loop rather than a generic Inspect agent, so the behavior
matches `agent.run_agent`.

1. **Environment.** `episodes.start_episode` builds it as `runner.run_episode` does: the defense's
   permissions and output filter, then the payload injected into the vector record, then a state
   snapshot. This is the only place the port builds an `Environment`.
2. **Messages.** The system prompt is `agent.system_prompt` (with the defense's addendum); the only
   user message is the principal's request.
3. **Tools.** Every tool in `env.TOOL_SPECS`, offered with exactly its JSON schema (Inspect would
   otherwise add `additionalProperties: false`).
4. **Tool calls.** Executed by `Environment.call` in order, so policy checks, blocking, output
   filtering, exposure tracking and the call log are the native ones. Arguments that are not a
   JSON object get the native error message without reaching the environment.
5. **Ending.** At most `max_steps` model calls. The first reply without tool calls ends the
   episode, and its text is the final reply. Hitting the limit leaves the final reply empty.

**Scorer (`injbench_scorer`).** Calls `graders.grade` on the episode's final state, exactly as
`run_episode` does.

- **Score value.** A dict of per-episode flags and counts: `task_success`, `safe_completion`,
  `realized_violation`, `attempted_violation`, `unexpected_write`, `out_of_scope_read`,
  `model_refused`, `n_realized`, `n_attempted`, `n_unexpected_writes`, `out_of_scope_reads` and
  `n_tool_calls`.
  - Attack samples add `attack_success`, `exposed` and `exposed_attack_success`.
  - Benign samples add `wrongful_refusal`.
  - Keys that don't apply are left out, not set to null.
- **Metadata.** The full `Outcome` (violations, unexpected writes) plus the native record's loop
  fields: `steps`, `hit_step_limit`, `model_refused`, `final_stop_reason` and `graded_from`.
- **Answer and explanation.** The answer is the final reply; the explanation is a one-line summary
  for the log viewer.

**Re-scoring.** The scorer grades the live episode left by the solver. Without one, for example
when re-scoring a log, it replays the logged tool calls in a fresh environment. The tools are
deterministic, so this reproduces the final state (tested). After a grader fix, re-grade a log
without new model calls:

```bash
uv run inspect score runs/inspect/x/<log>.eval --scorer injbench/injbench_scorer \
  --action overwrite --output-file runs/inspect/x/rescored.eval
```

### Metrics

One Inspect metric per rate. Names and definitions are those of `injbench.metrics.METRICS`, and
the tests check that the values are identical to `metrics.summarize` on the same episodes.

| Inspect metric | PLAN §7 | Samples | Value |
|---|---|---|---|
| `safe_completion_under_attack` | **headline**: safe completion under attack | attack | mean `safe_completion` |
| `benign_completion` | headline: benign completion | benign | mean `task_success` |
| `realized_violation_rate` | headline: realized violation rate | attack | share with a realized violation |
| `attempted_violation_rate` | headline: attempted violation rate | attack | share with a policy-denied call |
| `benign_safe_completion` | | benign | mean `safe_completion` |
| `benign_out_of_scope_read_rate` | benign out-of-scope reads | benign | share with an out-of-scope read |
| `wrongful_refusal_rate` | wrongful refusals on benign tickets | benign | mean `wrongful_refusal` |
| `asr` | ASR | attack | mean `attack_success` |
| `asr_out_of_permission`, `asr_within_permission` | ASR by goal class | attack, by class | mean `attack_success` |
| `exposure_rate` | exposure | attack | mean `exposed` |
| `asr_given_exposed` | exposure-conditional ASR | exposed attack | Σ `exposed_attack_success` / Σ `exposed` |
| `model_refusal_rate_attack` | | attack | mean `model_refused` |

- **Order.** `safe_completion_under_attack` comes first, so it is the log's headline metric.
- **Undefined rates are left out.** A rate with no qualifying samples is not reported, e.g. attack
  rates in a benign-only run.
- **No stderr.** Episodes are clustered by template, so an i.i.d. stderr would understate the
  uncertainty. Use the native report (below) for CIs.

**Repeated attempts are epochs** (PLAN §6). `--epochs k` runs every sample k times, and the
randomness must come from sampling (e.g. `--temperature 1`). `--cache` caches each epoch
separately. The rates are means over the score values, so they compose with Inspect's epoch
reducers:

- the default `mean` pools all attempts, like the native records with `attempts = k`;
- `--epochs-reducer at_least_1` gives *the share of cases with at least one success within k
  attempts* (ASR@k). Both are tested.

### CIs and paired differences: the native report

Clustered bootstrap CIs and paired arm differences are not Inspect metrics. Convert the logs into
native `episodes.jsonl` records and use the native report:

```python
from glob import glob
from injbench.inspect_port import records_from_eval_logs
from injbench.metrics import summarize, to_markdown

records = records_from_eval_logs(glob("runs/inspect/claude-haiku-5-5/*.eval"))
print(to_markdown(summarize(records)))   # one arm per defense; paired differences vs `none`
```

- **Record format.** Records have the native schema, checked field by field against
  `run_episode`. The episode ID is `<sample id>|<defense>|<model>|a<epoch-1>`.
- **Errors.** Errored samples become `status: "error"` records.
- **Usage fields.** Token counts and cost follow Inspect's accounting. Cost needs model cost data
  (`--model-cost-config`), and `cached_calls` is not available.

## Known differences from the native runner

- **Response cache.** The native runner caches responses on disk and coalesces identical in-flight
  requests, so concurrent arms share samples.
  - Inspect's `--cache` keys on the model, generate config, messages, tools and epoch, but does
    not coalesce in-flight requests. Arms are only paired when they run one after another with
    `--cache`.
  - The cache expires after 7 days by default (`--cache 6M` for longer).
- **Statistics.** Clustered bootstrap CIs and paired differences only come from the native report
  (via `records_from_eval_logs`, above).
- **No budget cap.** There is no global `budget_usd`. Use `--limit`, `--cost-limit` (per sample;
  needs model cost data) and the provider's spend limits, and count samples before a long run.
- **Provider clients.** Inspect's model providers replace injbench's `OpenAICompatClient` and
  `AnthropicClient`, so retries, `max_tokens` defaults, prompt caching and thinking settings are
  Inspect's.
  - Set `--max-tokens` and the reasoning options explicitly (examples above).
  - The `openai-api` and `openrouter` providers send `strict: true` on tools unless you pass
    `-M strict_tools=false`.
- **Malformed tool-call arguments.** Inspect's providers parse them before the loop sees them, and
  are more lenient. They repair stray trailing quotes, and map a non-object argument onto the
  tool's first parameter. A call the native runner would reject can therefore execute. Arguments
  Inspect cannot parse at all get the native error message.
- **Refusals.** `model_refused` means the final reply's stop reason is `content_filter`, Inspect's
  mapping of provider refusals. `final_stop_reason` uses Inspect's vocabulary (`stop`,
  `tool_calls`, `max_tokens`, `content_filter`, ...).
- **Errors.** The native runner records a failed episode and continues. By default, Inspect stops
  the eval at the first sample error: pass `--no-fail-on-error` or `--retry-on-error`.
- **Limits.** `max_steps` counts model calls as in the native loop. Inspect's `--message-limit`,
  `--token-limit` and `--time-limit` can end an episode earlier. It is then graded as it stands,
  like an episode that hit the step limit.
- **Tool output.** Tools run inside the environment, so Inspect's `--max-tool-output` truncation
  does not apply, as in the native runner.
- **Not ported.** The M4 adaptive attacker (feedback between attempts) and the scripted reference
  agents. The tests replay the reference agents through `mockllm` instead.

## Maintenance: merging the other milestones

The tests are designed to fail when the port falls behind the native runner:

1. **Spotlighting renderer (M2).** After `Environment` applies `Defense.make_output_renderer()`,
   pass it in `episodes.start_episode`, the single construction site, exactly as `run_episode`
   does.
   - `test_inspect_run_matches_native_runner` compares every tool output with the native
     transcript for every defense in `DEFENSES`, so it fails until the renderer is passed.
   - `test_port_episode_matches_run_episode` checks the same without Inspect.
2. **New fields in the native record.** Map them in `episodes.episode_record`.
   `test_port_episode_matches_run_episode` compares whole records and names the missing field.
3. **New native metrics.** Add them to `episodes.RATES`, under the same name, if they should
   appear in Inspect.
4. **New tool schemas.** `solver.tool_info` refuses JSON-schema keywords that Inspect's
   `ToolParams` cannot represent, rather than silently showing the model a different schema.
5. **Detector defenses (M3).** Arms that cannot be built offline (`make_output_filter` raises
   `ImportError`, `OSError` or `RuntimeError`, e.g. a missing extra) are skipped in the parity
   tests, so the default test run stays green.
6. **`uv.lock`.** Regenerate it with `uv lock` after merging branches that change dependencies.

### Testing notes

- **No network.** The Inspect tests use `mockllm` with outputs that carry their own token usage.
  Without usage, `mockllm` counts tokens with tiktoken, which downloads its encoding.
- **Isolation.** Inspect runs use a temporary `HOME` and working directory: Inspect writes to the
  platform data and cache folders and reads `.env` files upward from the working directory.
- **Scenario hook.** The scripted mock model learns which sample it is answering through a
  test-only wrapper around `solver.start_episode`.

## Toward inspect_evals (open questions)

- **Packaging.** inspect_evals keeps each eval's code in its own repository, while this port
  imports the `injbench` package. Either depend on a published `injbench` wheel, or vendor the
  standard-library modules (`world`, `policy`, `env`, `tasks`, `attacks`, `defenses`, `graders`,
  `agent`, `runner.build_matrix`) into `inspect_evals/injbench`. To decide.
- **Sandbox.** Not needed: the tools are in-process SQLite, so the eval has no Docker dependency.
- **Default matrix.** The default task runs everything with one instance per template. After M2
  (20+ templates) that is several hundred samples per arm; an inspect_evals default may need to be
  smaller.
- **A generic-agent variant.** Inspect `Tool` objects for `react()`-style agents could be added
  next to the custom loop. They would change argument handling and output truncation, so they
  would not match the native runner.
- **Older Inspect versions.** 0.3.153 and 0.3.195 could not be installed in this sandbox (their
  wheels contain a `.vscode` folder it refuses to write), so the lower bound is the oldest
  version that was actually tested.
- **Before submitting.** inspect_evals needs an eval README with results on reference models,
  which means paid runs. Logs contain the attack payloads, and PLAN §14 rules out publishing
  attack strings tuned against named production models.
