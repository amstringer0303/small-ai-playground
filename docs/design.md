# A smaller experimental interface

The earlier laboratory exposed seven views and technical controls before people
could understand the experimental question. This version adds a smaller default
interface while retaining that lab under Advanced.

## Decisions

- Present source data and simulated-label provenance before asking for an intervention.
- Use one deliberately small PyTorch network by default: 8 hidden units, 60 epochs,
  learning rate 0.02, model seed 42, threshold 0.5, CPU only.
- Separate each data/training intervention into an explicit choice. Never interpret
  descriptive text as training instructions or secretly combine priority ratings.
- Train an original model automatically on the first comparison. Reuse that immutable
  baseline for subsequent comparisons with the same configuration and original goals.
- Compare each intervention to the original, using the same 144 held-out examples.
  Never edit the test labels through the simple editor.
- Freeze the first simple comparison's goal revision. Preserve later Advanced revisions,
  but do not silently replace the acceptance criteria for the simple study.
- Save both learned models and a comparison manifest with every completed run.
  Dataset edits retain parent IDs, hashes, label authorship, and a change summary.
- Retain actual initialization/trained/delta tensor images and connection values.
  Different input schemas are shown in separate model views, not misleadingly subtracted.
- Numerical preprocessing is fitted separately on each model's training partition.
  A data intervention can therefore change both normalization and learned parameters.
- Keep advanced model/goal/data/parameter experiments available rather than replacing
  the research model with an educational animation or static explanation.
- Share a training concurrency group across Simple and Advanced callbacks so that
  CPU neural training/seed operations cannot overlap through the UI.
- Bind the server to loopback, disable telemetry, and filter external optional assets.

## Validation

Core tests cover data protection, actual objective effects, learned-weight edits/freezes,
replay, and exports. Simple-specific tests cover loss-induced parameter/output changes,
a deterministic no-change control, baseline reuse, fixed goal history, all four choices,
guarded editing, and export content. Browser checks exercise the public interface,
saved-run reloads, downloads, the retained Advanced lab, and desktop/mobile framing.

## Remaining research work

1. Test this interface with participants; keep distinct interpretations instead of
   combining them into one community score.
2. Add multiple seeds, uncertainty estimates, and a locked final test set before
   drawing substantive conclusions.
3. Add a clearly licensed real dataset with a community-defined task and governance
   agreement; later add one verified-license small pretrained model with a compatible adapter.
