# Sensor review: worked example

Can we catch more readings needing review without too many false alarms?

Synthetic study; proposed criteria, not community-approved or health guidance.

## Potential questions
- Can we catch more readings needing review without too many false alarms?
- Should we exclude suspect sensor readings?
- What changes when residents correct training labels?
- Can we remove location inputs without losing useful predictions?

## Criteria recorded before training
- Miss <= 10% of alert examples.
- Falsely flag <= 20% of normal examples.
- Accuracy >= 80%.
- Local CPU; training <= 60 seconds; mean inference <= 50 ms; observed RAM <= 1 GB; saved model <= 1 MiB.
- All five changed-model runs must meet these targets; no cherry-picking a seed.

## Results (seed 42)
```text
                     Candidate Missed alerts False alarms Accuracy Performance targets
     Non-AI demonstration rule       11 / 37      3 / 107    90.3%                Fail
           Logistic regression        7 / 37      6 / 107    91.0%                Fail
     Neural network / original        8 / 37      5 / 107    91.0%                Fail
Neural network / 5x importance        3 / 37     13 / 107    88.9%                Pass
```

## Five fixed seeds
```text
 Seed  Original missed  5x missed  Original false alarms  5x false alarms All targets
   42                8          3                      5               13        Pass
    7                7          3                      8               12        Pass
   23                7          3                      6               11        Pass
  101                8          2                      7               12        Pass
  202                6          4                      7               13        Fail
```

Seed 42 changed model: 194 learned parameters; 5.0 KiB checkpoint; 0.05 s training; 2.18 ms mean inference; 605 MiB sampled process RAM. Python network operations blocked: zero attempts. Saved checkpoints reproduced their predictions.

## Decision
Only 4 of 5 changed-model runs passed every prototype target. The intervention is not consistently successful under this test; do not loosen criteria after seeing the result.

## Limits
- Invented data and labels; no real residents participated or approved these targets.
- The development evaluation set was already visible in this prototype. Repeated seeds are not new test populations.
- There are only 37 alert examples. These checks use point estimates, not statistical guarantees about an underlying population.
- No claim of health protection, regulatory suitability, subgroup fairness, or deployment readiness.
- A small saved model is not a small installation. Python, PyTorch and the app have additional storage/RAM overhead.
- RAM is sampled process RSS; latency is a mean, not a worst-case guarantee. Results apply to this computer.
- The offline guard covers Python socket operations in this process, not every possible native-library network path or the whole device.
- Location inputs remain in this case. Local computation is not anonymization or a complete privacy/governance assessment.

## Background
Example questions and targets for this playground, not health standards.
- [EPA Air Sensor Toolbox](https://www.epa.gov/air-sensor-toolbox)

Regenerate with `python -m scripts.community_study --workspace outputs/new-community-study`.
