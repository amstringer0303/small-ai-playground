# Proposed alignment with OEDP

This is our design interpretation, not a claim of OEDP endorsement or a definitive
statement of an unpublished small/local AI program.

OEDP's [public mission](https://www.openenvironmentaldata.org/about) emphasizes usable
environmental information and collaborative, participatory governance. Its
[Community Data Hub background](https://www.openenvironmentaldata.org/pilots/background-and-concept)
emphasizes community authority over data, local values, and context-sensitive sharing.

The proposed small/local AI experience turns those ideas into testable choices:

| Design intent | Implemented interaction | Important limit |
| --- | --- | --- |
| Data usability | Source and simulated labels visible beside editable readings | Synthetic data, not an environmental monitoring deployment |
| Consequential participation | Change labels/inclusion, exclude unreliable readings, or alter loss weighting | No collaborative UI or claim that one user's choices represent a community |
| Local control | Train and retain data/weights/checkpoints locally on CPU | Local operation does not automatically establish good governance |
| Context-sensitive privacy | Optional geographic-input experiment | Older raw snapshots remain; location can be useful and is not categorically prohibited |
| Accountability | Compare original/changed predictions and fixed pre-training goals | A visible synthetic holdout is insufficient for deployment decisions |
| Shared learning | Save each run and export a transparent design record | Actual outcomes can conflict; no single ethical score |

The central question remains: how do participant decisions materially change what
a model learns, and which resulting tradeoffs do those participants consider acceptable?
