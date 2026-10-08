# Small AI Playground: worked example

Can we catch more readings needing review without too many false alarms?

## What this tests

Imagine reviewing sensor readings by hand. The model suggests which ones need a
closer look. The goal is fewer missed reviews without too many unnecessary reviews.

This gives the small local model a concrete task: change how it learns, compare
its errors, and check whether it runs on this computer without an AI API.

## Where flags fit

A sensor-problem flag is an existing tag for an unreliable reading, not an alert
predicted by AI. The separate **Suspect readings** experiment tests whether
leaving those readings out improves review predictions. This worked example
keeps them and changes training importance instead.

## What counts as success?
- Miss at most 10% of readings needing review.
- Send at most 20% of normal readings for unnecessary review.
- Classify at least 80% of all readings correctly.
- Local CPU; training <= 60 seconds; mean inference <= 50 ms; observed RAM <= 1 GB; saved model <= 1 MiB.
- All five changed-model runs must meet these targets; no cherry-picking a seed.

## What changed?

Train two versions on the same data: one treats both kinds of error equally;
the other gives alert examples 5x importance. Test both on the same 144 readings
(37 needing review, 107 not needing review).

The study also checks a simple PM2.5 rule and logistic regression, so AI isn't
assumed to be the best option.

## What happened?

Missed reviews fell from 8 to 3; unnecessary reviews rose from 5 to 13.
The model caught more alerts, but asked for more reviews.

```text
                     Candidate Missed alerts False alarms Accuracy Performance targets
     Non-AI demonstration rule       11 / 37      3 / 107    90.3%                Fail
           Logistic regression        7 / 37      6 / 107    91.0%                Fail
     Neural network / original        8 / 37      5 / 107    91.0%                Fail
Neural network / 5x importance        3 / 37     13 / 107    88.9%                Pass
```

## Did it work more than once?

Repeat training five times with different starting weights, using the same test
readings. The goal is to meet every target in all five runs.

```text
 Seed  Original missed  5x missed  Original false alarms  5x false alarms All targets
   42                8          3                      5               13        Pass
    7                7          3                      8               12        Pass
   23                7          3                      6               11        Pass
  101                8          2                      7               12        Pass
  202                6          4                      7               13        Fail
```

## Did it run locally?

Seed 42 changed model: 194 learned parameters; 5.0 KiB checkpoint; 0.05 s training; 2.18 ms mean inference; 605 MiB sampled process RAM. Python network operations blocked: zero attempts. Saved checkpoints reproduced their predictions.

## Did it meet the goal?

Only 4 of 5 changed-model runs passed every prototype target. The change was not
consistently successful: the goal was to pass all five.

## Background and limits

Fictional data and unapproved targets, not health guidance. Repeated runs reuse
just 37 alert test readings, so they do not show real-world reliability. Local
processing does not guarantee privacy; resource and offline checks cover only
this setup.
- [EPA Air Sensor Toolbox](https://www.epa.gov/air-sensor-toolbox)

Regenerate with `python -m scripts.community_study --workspace outputs/new-community-study`.
