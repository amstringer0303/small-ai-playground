# Small AI Playground

A playground for trying a small AI model and seeing how different choices affect
its predictions. The example is air quality: which sensor readings need a review?

The neural network trains on your computer. The training and results are real,
but the readings are dummy data. This is a research prototype. 

## Try it

You'll need Python 3.11-3.13. On Windows, run:

```powershell
git clone https://github.com/amstringer0303/small-ai-playground.git
cd small-ai-playground
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open [localhost:7861](http://127.0.0.1:7861) in your browser. On macOS or Linux,
use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.
If the port is busy, add `--port 7863` to the last command.

Installation needs internet. After that, training runs on CPU without a GPU,
model download, or API key. This repo contains the code, not a hosted website.

For a first run, leave **Missed reviews** selected and **Alert-error importance**
at **5**, then click **Train and compare**.

## What you can change

- **Missed reviews:** give alert examples more weight during training. This may
  catch more alerts, but it can also produce more false alarms.
- **Suspect readings:** test whether leaving out readings already marked as
  unreliable improves review predictions. The model isn't finding sensor faults.
- **Community corrections:** choose a reading and change its normal/alert label
  or whether it is used in training. The full table is under **Training readings**.
- **Location inputs:** train without coordinates, site names, or sensor IDs.
  This doesn't delete those details from the saved data.

For corrections, the training button becomes available once you've made a change.
Use **Reset readings** before switching to another question with unapplied edits.

These questions are starting points for discussion. No community has approved
the questions or targets used here.

## Reading the results

Every run compares your choice with the original model, not the previous run.
Both use the same model setup, random seed, and 144 test readings. The original
model is reused after its first training.

An **alert** means a fictional reading needs review; it isn't a health warning.
A **missed alert** is an alert that the model calls normal. A **false alarm** is a
normal reading that the model calls an alert. **Accuracy** is the share of all
readings classified correctly.

A **sensor-problem flag** is different: it's an existing data-quality tag,
not an alert predicted by the model.

The comparison shows the counts and the tradeoff. Fewer missed alerts can come
with more false alarms, even if overall accuracy falls.

The default prediction targets are at least 80% accuracy, no more than 10% missed
alerts, and no more than 20% false alarms. These are separate from training
importance. **Met** means a target was met in that run, not that the model is
reliable in the real world. The original targets are kept with each comparison.

**Learned weights** shows the numbers the network learned. **Advanced** has more
model choices, data editing, goals, and experiments. Writing a project description
there doesn't change training; those fields aren't prompts for a language model.

## Data and limits

There are 720 simulated readings: 576 training candidates and 144 test readings.
The network also sets aside some training candidates for validation. The test
readings and their original labels stay unchanged when you edit training data.
See [where the data comes from](data/PROVENANCE.md).

Fictional data, not health guidance. Reusing the same test readings doesn't prove
real-world reliability; local processing doesn't guarantee privacy. Resource
checks apply to this setup. Don't enter sensitive data.

## Saved work

**Saved runs** keeps your comparisons. By default, data versions and model files
are stored in `outputs/simple-lab/`, which isn't committed to GitHub.
To start a separate study on Windows:

```powershell
.\.venv\Scripts\python.exe app.py --workspace outputs/another-study --port 7863
```

**Download experiment record** exports the settings, metrics, learned parameters,
and data-version references. It doesn't include the raw datasets or checkpoint
files, so keep the local workspace to reproduce a run. Only load checkpoint files
you trust.

## Worked example

The goal was to catch more readings needing review without creating too many
unnecessary reviews. I gave alert examples five times the training importance;
I didn't remove suspect readings in this example. Missed alerts fell from 8 to 3,
but false alarms rose from 5 to 13. Four of five runs met the targets, not all five.

The study also compares a simple rule and a linear model. The
[write-up](docs/community-study.md) explains the goal, change, and result.
To repeat it in a new workspace:

```powershell
.\.venv\Scripts\python.exe -m scripts.community_study --workspace outputs/new-community-study
.\.venv\Scripts\python.exe app.py --workspace outputs/new-community-study
```

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

The interface uses Gradio, with PyTorch and scikit-learn for the models. There are
no pretrained language models or live sensor feeds.

