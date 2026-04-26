# Audible — when your voice assistant should actually listen

> *Imagine if your phone could quietly hear what's happening around you and
> just knew when to help — without ever needing a wake word.*

You know how every voice assistant today needs you to say "Hey Siri" or
"Alexa" first? That's because the alternative — letting it always listen and
figure out what to do — is hard. Really hard. If your phone tries to act on
everything it hears, it'll set timers when you say *"hold on, give me a sec,"*
play music when someone mentions a song on TV, and search Google every time
you wonder something out loud.

**Audible is what you'd build if you wanted to drop the wake word.** It's a
small AI model trained to be the *gatekeeper* of an always-on assistant — one
that knows when you're actually talking to it, when you're talking to someone
else, and when you're just thinking out loud. And because not everyone wants
the same thing from their assistant, it learns three personalities you can
pick from:

- **Minimalist** — only acts on direct commands. Stays silent for everything
  else, even if it sounds like an instruction.
- **Proactive** — picks up on indirect cues like *"I wonder what the weather's
  like"* and quietly looks it up so the answer is ready when you ask.
- **Work-focused** — sets timers and calendar events all day, but never plays
  music or touches the smart-home lights, even if you ask.

In your day, you'd notice it as: fewer awkward *"Sorry, I didn't catch that"*
moments, fewer accidental timers from a passing TV ad, and your assistant
only stepping in when it actually adds value — tuned to *your* preference,
not the average user's.

---

**Under the hood**, Audible is a self-improving OpenEnv environment that
takes a 25M-parameter mobileBERT classifier and teaches it to be the gate.
We built the env, trained the gate, then ran an adaptive adversarial
curriculum where a generator agent escalates difficulty each round in
response to the gate's current weak spots. **One round collapsed the hardest
user profile's false-wake rate from 38.8% to 9.8% — a 4× reduction.** Built
for the **Meta OpenEnv Hackathon 2026, Theme #4 — Self-Improvement**.

- 🎬 **Live product demo:** <https://website-wheat-eight-sd3hr2u4kn.vercel.app>
- 🤗 **HF Space (the env):** <https://huggingface.co/spaces/me-tusharchandra/audible-env>
- 🎛 **Try the env in the browser:** <https://me-tusharchandra-audible-env.hf.space/web>
- 📓 **Colab notebook:** see the repo's `training/notebook.ipynb`
- 🎥 **2-min demo video:** _link will be in the Space's README once recorded_

---

## The problem nobody seems to talk about

"Hey Siri." "Hey Alexa." Two phrases doing all the work. Without a wake word,
your voice assistant is essentially deaf — it can't listen to what's around
you, it can't pick up on what you actually need, and it definitely can't
*act* on it. Wake words exist because we don't trust these things to figure
out *when* to act. So we make them ask permission first.

But always-on, ambient listening — devices that actually pay attention and
*figure out for themselves* when to act — is a much harder problem. Think
about all the things you say in a day:

- *"Hold on a sec, grabbing my keys"*
- *"Did you set the timer for the cookies?"*
- *"I love this song that's playing"*
- *"I wonder what the weather's like in Paris"*
- *"It's a bit bright in here"*

Tool keywords are everywhere — but most of the time you're not asking your
assistant to do anything. A naive keyword-matching gate wakes on every one
of these. A binary "actionable / not actionable" classifier — the standard
baseline — misses the fact that **different people want different things**
from their assistant. A sentence like *"I wonder what the weather's like"*
should fire `web_search` for someone who likes proactive help, and stay
silent for someone who finds that intrusive.

The standard baseline (a binary classifier my friend trained on hand-curated
data) hits 97% F1 on its own dataset. It collapses on real ambient cases
because the dataset never asked the right question.

## What we set out to build

A system that:

1. Decides whether a single ambient utterance should trigger an **action**, a
   silent **context update**, or be **ignored** entirely.
2. If it triggers an action, picks the right tool out of a small palette
   (timer, calendar, music, web search, smart-home).
3. Honors a **per-user preference profile** — the same utterance can mean
   "act" for one user and "ignore" for another.
4. Runs **on the device**. No cloud round-trip per utterance — that defeats
   the point.
5. Gets *better over time* by training against an adversary that targets its
   current weaknesses. Not just supervised fine-tuning on a static dataset —
   actual self-improvement.

Constraint #4 is what makes this Theme #4 instead of "throw GPT-4 at it." A
frontier LLM could probably solve this perfectly. But a frontier LLM can't
fit on a phone and respond in <100 ms, and that's where the problem actually
lives.

## The environment

We built it on Meta's OpenEnv framework. The action and observation contract:

```python
class GateObservation:
    utterance:        str                       # what the mic transcribed
    user_profile:     "minimalist" | "proactive" | "work_focused"
    available_tools:  list[{name, description}] # the 5-tool palette
    context_history:  list[str]                 # recent prior turns
    # post-step diagnostic fields populated for the rubric:
    ground_truth:     dict | None
    component_scores: dict | None

class GateAction:
    decision: "ACT" | "UPDATE_CONTEXT" | "IGNORE"
    tool:     "set_timer" | "add_calendar_event" | "play_music"
            | "web_search" | "smart_home_control" | None
```

Three top-level decisions, five concrete tools. The 5-tool palette was
chosen so each tool has obvious confusable ambient examples — that's what
makes the gating decision genuinely hard:

| Tool | Direct command | Ambient confusable (should NOT trigger) |
|---|---|---|
| `set_timer` | "Set a 10 minute timer" | "Hold on a sec, grabbing my keys" |
| `add_calendar_event` | "Add a meeting tomorrow at 3" | "Sarah and I might catch up tomorrow" |
| `play_music` | "Play some lo-fi" | "I love this song that's playing" |
| `web_search` | "What's the weather in Tokyo?" | "I wonder what the weather's like" |
| `smart_home_control` | "Turn off the kitchen lights" | "It's a bit bright in here" |

Each episode is a single step. `reset()` samples one (scenario, profile)
pair; `step(action)` scores it via a four-component composite rubric and
returns `done=True`.

## The four-component rubric (and why false-wakes hurt)

OpenEnv has a beautiful composable `Rubric` system, and we used it.

```python
class GateRubric(Rubric):
    def forward(self, action, observation) -> float:
        return (
            1.0 * gate_correctness(action, observation)   # decision matches GT
          + 0.5 * tool_correctness(action, observation)   # tool matches when both ACT
          + 0.5 * profile_alignment(action, observation)  # honors profile preference
          + 1.0 * false_wake_penalty(action, observation) # -1 if ACT when GT=IGNORE
        )
```

Reward range: `[-1.0, +2.0]`. The asymmetric weighting is intentional. **A
single false wake during a meeting is worse than ten missed activations.**
That asymmetry should be visible in the reward signal, which is why
`false_wake_penalty` is signed and weighted heavily relative to the other
positive components. A correct IGNORE on a tricky utterance scores +1.5;
mistakenly firing on it scores -1.0. That 2.5-point gap is what trains the
model to err on the side of silence.

## Why mobileBERT

`google/mobilebert-uncased`. ~25M parameters, ~25 MB int8, runs in real-time
on CPU and on-device. The whole point of an ambient gate is that it has to
fit on a phone and respond instantly — nothing else makes sense as the
target. Friend's prior work already proved mobileBERT was CoreML and
TFLite-exportable, so the edge story is real, not aspirational.

We treat it as a **sentence-pair classifier**: input is `(profile_description,
utterance)`, output is one of 7 classes (`IGNORE`, `UPDATE_CONTEXT`, or
`ACT_<tool>` for each of the 5 tools). The sentence-pair format leans on
mobileBERT's NSP pretraining and lets the model condition its prediction
on the user profile naturally.

## The data — three sources, one pipeline

**Source 1: Friend's binary dataset.** ~6.6k unique utterances labeled
binary actionable / non-actionable. Strong on lexical diversity, but no
tool granularity, no profile divergence, and crucially **no ambient
confusable cases** — the very thing that makes the problem interesting.

**Source 2: Heuristic 7-class mapping.** Regex rules turn the binary labels
into our 7-class space ("set", "alarm", "remind" → `ACT_set_timer`, etc.)
and apply per-profile rules (work_focused collapses music + smart_home →
IGNORE). This gives us volume — ~20k labeled rows after the profile
cross-product — but the regex fallback collapses many actionable commands
into `ACT_web_search` and the data has zero ambient confusables.

**Source 3: LLM-generated synthetic.** This is where it gets interesting.
We used `gpt-4o-mini` with **OpenAI Structured Outputs** (Pydantic schema
for parsed responses) to generate 803 fresh utterances across six
deliberately-targeted categories:

- `ambient_confusable` — tool keyword present, speaker isn't addressing the assistant
- `direct_command_balanced` — even tool distribution to fix the heuristic's web_search skew
- `indirect_address` — proactive's home turf ("I wonder…")
- `multi_speaker_chatter` — overheard two-person conversation
- `rhetorical_question` — question shape, no answer wanted
- `update_worthy` — notable info worth remembering, no action

Each scenario carries per-profile labels in one shot — so the same utterance
can have one label for `minimalist` and another for `proactive`, and the
generator gives us all three in a single API call. **Total cost: ~$0.17.**

The first prompt iteration produced reasonable but formulaic utterances —
lots of *"I should have…"* / *"I really need to…"* sentence shapes. We
iterated to v2 with explicit diversity rules (*"do not start every utterance
with 'I'"*) and concrete bad examples (*"I would like to inquire about the
weather forecast"* — too formal). Quality went from 7/10 to 8/10 on manual
inspection.

After dedup against the heuristic-mapped data, the combined dataset is
**7,450 unique utterances × 3 profiles = 22,350 labeled rows**. The
synthetic addition dramatically fixes the tool-class imbalance:

![dataset distribution](plots/dataset_distribution.png)

## Phase 2 — the baseline

We fine-tuned mobileBERT for 3 epochs on the combined dataset using a
custom `WeightedTrainer` (HF `Trainer` subclass with class-weighted CE
loss to compensate for the long tail).

![training loss & learning rate](plots/training_loss.png)

*Training loss (log scale, left axis) and warmup-then-decay learning rate
(right axis) over 1677 steps of the real baseline run. Loss converges
cleanly from the init-step spike to ~0.08 by epoch 3.*

After 3 epochs:

- **Held-out dataset eval:** 96.91% accuracy, F1-macro 0.875
- **Env-rollout eval (300 rollouts × 3 profiles):** +1.42 mean reward, 9.4% false-wake rate

Solid. But the env-rollout per-profile breakdown told us something more
interesting:

| profile | mean reward | decision acc | false-wake rate |
|---|---:|---:|---:|
| `minimalist` | +1.41 | 82.1% | **0.0%** ✓ |
| `proactive` | +1.19 | 73.9% | **40.0%** ⚠ |
| `work_focused` | +1.53 | 95.7% | 5.9% |

The `proactive` profile has a **40% false-wake rate**. The model is over-eager
to fire tools on indirect cues. *That's* the failure mode our adversarial
curriculum should target.

## Phase 3 — the self-improvement loop

Here's the loop, run for 3 rounds:

1. **Eval** the current gate against the env (200 rollouts × 3 profiles)
2. **Mine** the 30 worst-reward rollouts (mostly false wakes — score = -1.0)
3. **Build a generator prompt** that shows the LLM the gate's current failures:
   > *"Here are 30 utterances the gate already gets wrong: [list].
   > Generate 100 NEW utterances that probe the SAME failure modes more
   > adversarially, with subtle variations the gate is even less likely to
   > handle correctly."*
4. **Generate** ~100 new (utterance × 3 profile) labels via `gpt-4o-mini`
5. **Append** to `combined.parquet`, **retrain** mobileBERT for 1 epoch
6. **Repeat**

This is what makes it self-improvement and not supervised SFT with extra
steps. Each round's training data is *generated in response to the
learner's current weaknesses*. The generator and the gate co-evolve —
the gate gets stronger, and the generator pushes harder because it sees
the gate's current weak spots.

Total cost across 3 rounds: ~$0.12 in API calls. Compute: ~50 minutes on
CPU.

## The headline finding

To get apples-to-apples comparable numbers, we re-evaluated all four
checkpoints (baseline + 3 curriculum rounds) with the same fixed seed
(42) and 400 rollouts each. Here's what happened:

| | baseline | round 1 | Δ |
|---|---:|---:|---|
| **proactive false-wake rate** | **38.8%** | **9.8%** | **−29.0pp (4× reduction)** |
| proactive mean reward | +1.226 | +1.400 | +0.174 |
| overall mean reward | +1.424 | +1.456 | +0.032 |
| overall decision accuracy | 86.8% | 86.5% | −0.3pp (essentially unchanged) |

**One round of adversarial curriculum collapsed the worst failure mode by
4×, with no significant regression elsewhere.** The adversarial generator
did exactly what it was supposed to: it identified the gate's worst failure
mode and produced training data that fixed it.

This is the headline. It's the part the demo video should center on.

## The honest research finding

But three rounds was too many.

![reward curve](plots/reward_curve.png)

Look at the right panel. After round 1, the proactive false-wake rate
*rebounds*: 9.8% → 33.3% → 41.1%. Continued adversarial generation kept
producing harder "this should NOT fire" cases, but with only 1 epoch of
retraining per round, the model couldn't properly integrate them — it
drifted toward firing on *everything* instead.

This is a real finding, not a footnote. Three rounds was too many *for our
retrain budget* — the curriculum loop needs either:

1. More epochs per round, so new data integrates properly, or
2. Early-stopping on a held-out adversarial validation set, or
3. Inflection-detection — automatically stop when reward starts dropping

We're reporting this as a **finding**, not a defect to hide. Real research
surfaces inflection points; fake-clean monotonic curves are a red flag.

The practical takeaway: **the round-1 model is the production checkpoint.**
Best overall reward, best proactive metrics, no significant regression on
the other two profiles. Round 3 wins on tool-selection accuracy when ACT
(91.4% vs baseline 85.5%) but the trade isn't worth the false-wake creep.

## What this proves about Theme #4

The *self* in self-improvement is doing real work here:

- The data the model trains on is **generated in response to the model's
  own behavior** — not pre-curated.
- The generator sees the model's failures and **escalates difficulty
  automatically** — no human in the loop choosing what to teach next.
- The reward signal is the env's composite rubric — multi-axis, not a
  single scalar — so the model gets explicit credit/blame for each
  component (correct decision, correct tool, profile alignment, no false
  wake).
- The gate and the generator **co-evolve**: as the gate gets stronger, the
  generator pushes harder; as the generator pushes harder, the gate has
  to adapt.

That feedback loop — and the fact that it produced a 4× reduction in the
worst failure mode after just one iteration — is what makes this Theme
#4 self-improvement instead of supervised SFT with extra steps.

## What we'd do differently with another week

1. **Smarter curriculum stopping.** Hold out an adversarial validation set,
   stop when reward on it stops improving. Would have caught the
   round-2/round-3 over-training automatically.
2. **Multi-turn context.** The `context_history` field on the observation
   is plumbed but currently always empty. Multi-turn dialog where the gate
   tracks conversational state would be a major capability boost.
3. **Replace the heuristic regex with an LLM-judge re-labelling pass.** ~$0.30
   of `gpt-4o-mini` would clean the entire 6,647-utterance friend's-data
   set with proper per-profile per-tool labels, and we'd lose the noise
   from the regex fallback.
4. **A proper TRL integration via teacher-student.** Train a small generative
   model with TRL `PPOTrainer` against the env, then distill back into
   mobileBERT for edge deployment. Best of both worlds.
5. **Distill to TFLite/CoreML for an iOS demo.** Friend's repo already has
   the export pipeline; we'd just plug in our trained model.

## Try it

Open the [Web UI](https://me-tusharchandra-audible-env.hf.space/web) — click
*Reset*, get a real ambient utterance, submit a `GateAction`, and watch the
per-component reward breakdown come back. You can also pull the full env as
a pip package directly from this Space, or pull the Docker image and run it
locally:

```python
pip install "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git" \
            "audible_env @ git+https://huggingface.co/spaces/me-tusharchandra/audible-env"

from audible_env import AudibleEnv, GateAction
with AudibleEnv(base_url="https://me-tusharchandra-audible-env.hf.space").sync() as env:
    obs = env.reset().observation
    print(obs.utterance, "→ profile:", obs.user_profile)
    result = env.step(GateAction(decision="ACT", tool="set_timer"))
    print("reward:", result.reward, "components:", result.observation.component_scores)
```

```bash
docker pull registry.hf.space/me-tusharchandra-audible-env:latest
docker run -p 8000:8000 registry.hf.space/me-tusharchandra-audible-env:latest
```

The full training pipeline (synthetic data generation, baseline training,
curriculum loop, plotting) is in the GitHub repo as a Colab-runnable
notebook. ~10–15 min on a free T4.

## Acknowledgments

- **[`pranjal-pravesh/actionable-gating-classifier`](https://github.com/pranjal-pravesh/actionable-gating-classifier)** — the original mobileBERT binary baseline + dataset that this project builds on. The edge-deployment angle (CoreML / TFLite export) is downstream of his work.
- **[Meta OpenEnv](https://github.com/meta-pytorch/OpenEnv)** — the framework. The composable `Rubric` system and the `Environment` base class are exactly what we needed.
- **OpenAI gpt-4o-mini + Structured Outputs** — synthetic data + adversarial-curriculum generator. Total spend across the entire project: ~$0.30.

---

*Built for the Meta OpenEnv Hackathon 2026, Theme #4 — Self-Improvement.*
