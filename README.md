# Small / Local AI: Simple Playground

A small local research playground for one question:
**What changes when people alter the data or priorities used to train a model?**

The first screen puts the community question and training choice beside the main action.
Detailed data is optional; the comparison explains error counts, rates, targets, and tradeoffs.
No experiment naming, JSON, model selection, or training setup is required.
The full research laboratory remains in **Advanced**.

## Run

Python 3.11-3.13. Windows setup:

```powershell
git clone https://github.com/amstringer0303/small-local-ai-simple.git
cd small-local-ai-simple
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:7861**. Use `app.py --port 7863` if occupied.
On macOS/Linux, substitute `.venv/bin/python` for the Windows executable.
The model runs on CPU. No GPU, model download, API key, or cloud AI service is needed.
Installation needs internet; installed training and inference work offline.
Gradio telemetry is disabled and external asset tags are filtered.

## First experiment

1. Open **Playground**. The source is explicitly labeled **synthetic data**.
2. Keep **Missed reviews** selected, with **Alert-error importance** at **5**.
3. Click **Train and compare**.
4. Read the saved comparison's tradeoff sentence, original/changed counts,
   and **Met / Not met** status for each prediction target.
5. Change one choice and run again. **Saved runs** keeps every comparison.

The baseline learns from the unchanged original data with a 1x alert loss weight.
Your intervention is trained with the same seed, architecture, inference threshold,
and 144 test readings. The original baseline is reused after its first training;
the saved status identifies reuse. Comparisons are to the original, not automatically
to the previous intervention.

An **alert** is a simulated label for a reading needing review, not a health warning.
A **missed alert** is an alert example predicted as normal. A **false alarm** is a normal
example predicted as alert. Accuracy counts all correct predictions, so it can conceal
the errors that matter most to the question. Missed-alert and false-alarm rates have
different denominators: alert examples and normal examples, respectively.

Training importance changes what errors count most during learning. **Recorded success
targets** are separate evaluation limits; they are not training settings. A target marked
**Met** refers to one run's observed prediction rate, not proven reliability or every
resource/governance requirement. The original goals and underlying records remain accessible.

## Four consequential choices

- **Give missed alerts more importance:** multiply the training loss for alert examples.
- **Leave out flagged readings:** exclude flagged training rows, keeping test rows fixed.
- **Edit labels or include readings:** select **Community corrections**, choose a reading,
  then change its normal/alert label or training-inclusion checkbox. Corrections immediately
  update the draft and its counts; **Train and compare** saves an immutable data variant.
  The optional **Training readings** table also supports edits.
  Labels must be `normal` or `alert` here; Advanced supports additional categories.
- **Leave out location information:** exclude coordinates, site, and sensor identifiers
  from model inputs. Older raw data snapshots remain on disk; this is not data erasure.

The **Question** selector expresses these choices as community questions, not model settings.
The questions and targets are proposals for discussion, not community-approved requirements.

## Worked community example

We selected: **Can we catch more readings needing review without too many false alarms?**
The test records criteria before training, compares a non-AI demonstration rule and a simpler
linear classifier, then repeats the original/5x-weighted neural pair with five fixed seeds.
Python socket operations are blocked during training, inference, and checkpoint restoration.
Four of five altered runs met every target; one missed the 10% missed-alert limit.
That is not consistently passing, and it is not evidence of real-world health protection.
See the [recorded walkthrough and results](docs/community-study.md).

```powershell
python -m scripts.community_study --workspace outputs/new-community-study
python app.py --workspace outputs/new-community-study
```

Fresh-workspace runs save `protocol.json` before training and produce `study-results.json`,
`community-study.html`, and `community-study.md`. **Download worked example** downloads the
self-contained HTML report when the app uses that study workspace. Existing studies are
never overwritten by the script. Raw synthetic checkpoints/results remain ignored by Git.

Editing a label or Use reading checkbox selects the data-edit choice automatically.
**Train and compare** is disabled until a correction exists for the correction question.
Pending corrections keep other questions disabled until **Reset readings** or a return
to the correction question, so edits cannot be accidentally ignored.
Reading IDs and measurements are read-only on the simple screen. Advanced supports
editing measurements and adding examples. To switch to a different intervention
after editing, reset the table first; edits are never silently ignored.

The source contains 720 invented readings, with 576 training candidates and 144
original test readings. An additional validation split is used within the neural
training candidates. Labels are generated by a noisy, site-dependent synthetic rule;
they are not official thresholds. See [data provenance](data/PROVENANCE.md).
The model, training, checkpoints, learned weights, and measured outcomes are real.

**Learned weights** shows initialization, learned tensors, distributions, and individual
connections for both models. **Advanced** preserves the original revisitable lab:
goals, dataset variants, all four model families, actual weight editing/freezing,
threshold experiments, evaluation, comparison, and lineage.

## Goals and saved work

Before the first simple comparison, criteria are recorded: accuracy >=80%,
missed-alert rate <=10%, false-alarm rate <=20%, local/offline operation,
inspectable parameters, and sampled process RAM <=1 GB for new studies. Older studies
keep their original recorded budgets. Geographic removal is
**optional**, not a default ethical requirement. Advanced can revise criteria before
the first simple comparison. Subsequent simple comparisons retain the original
goal revision even if Advanced later saves revised goals.

No goal automatically changes training or guarantees an outcome. Text fields
in Advanced record the study; they are not prompts interpreted by an LLM.

Records, data snapshots, and local checkpoints persist under `outputs/simple-lab/`,
which is ignored by Git. Start an independent study with:

```powershell
python app.py --workspace outputs/another-study --port 7863
```

The experiment ZIP includes metrics, configurations, source/version hashes,
learned parameters, and original-goal results. It omits raw datasets and executable
checkpoint files. Reproduction needs the saved local workspace and matching dependencies.

## Tests and architecture

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m scripts.demo
python -m scripts.replay EXPERIMENT_ID --workspace outputs/simple-lab
```

The numerical engine is reused from the private `small-local-ai-playground` prototype.
`playground/service.py` adds paired, automatically named experiments; `playground/views.py`
provides concise comparison views; `app.py` presents Simple, Saved runs, and Advanced.
`advanced_lab.py` retains the original lab without changing the original repository.
See [design decisions](docs/design.md), [landscape comparison](docs/landscape.md),
and [OEDP alignment](docs/oedp-alignment.md).

Screenshots: [simple playground](docs/screenshots/playground.png),
[label editing](docs/screenshots/label-edit.png), [learned weights](docs/screenshots/weights.png),
and [390px layout](docs/screenshots/mobile-390.png).

## Scope and licensing

Single-user, local prototype. Synthetic data is not evidence about a real community.
The test set is visible and repeated experimentation can overfit it. RAM is sampled,
not enforced. Learned parameters are inspectable, but that does not establish causal
explanations or fairness. The ZIP is not a complete portable reproduction bundle.
Local joblib checkpoints are trusted files; do not replace them with unknown files.

No pretrained language models, LoRA, quantization, remote API, collaboration,
or live environmental feed is implemented. These remain future extensions.
Models use existing open-source architectures with locally learned weights;
this is not a claim that a pretrained model is being imported or that checkpoints
are a fully open licensed release.

**This repository is private. Licensing is intentionally undecided during the
research/prototype stage. No project open-source license has been selected.**
Dependency licenses remain their respective authors' licenses. This is a proposed
OEDP-aligned research prototype, not an official or endorsed OEDP product.
