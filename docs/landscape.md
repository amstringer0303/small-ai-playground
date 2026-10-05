# Landscape review

Reviewed October 5, 2026, before implementation. This is a focused review of
documented capabilities, not a claim that no other project addresses participation.

| Existing tool | Capabilities already available | What this prototype adds to that overlap |
| --- | --- | --- |
| [TensorFlow Playground](https://playground.tensorflow.org/) | Browser experiments with features, architecture, learning rate, editable weights, and training/test loss on toy problems. | Persistent versions of community data and goals; branches linking a specific data or objective decision to weights, predictions, and acceptance tests. Editable weights alone are not new. |
| [Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/main/en/tasks/llm_finetuning) | Dataset preparation, local/cloud training, training configurations, and language-model fine-tuning methods. | A deliberately small experimental lab that operationalizes separately recorded priorities and preserves original acceptance criteria across interventions. |
| [Edge Impulse Data Explorer](https://docs.edgeimpulse.com/studio/projects/data-acquisition/data-explorer) | Data inspection, outlier/mislabel detection, labeling, and visualization in an edge-model development workflow. | Dataset editing is not new; the focus here is reproducible comparisons of competing label definitions, sensitive-feature removal, loss penalties, and goal tradeoffs. |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | Fine-tuning through configuration and a web interface, including adapter and quantization workflows for language/vision models. | It is a future backend candidate. This prototype records the hypothesis, data lineage, parameter intervention, and goal evidence around training rather than reimplementing broad fine-tuning support. |
| [LM Studio](https://lmstudio.ai/docs/app/offline) | Local/offline inference after models are available locally. | Local inference is infrastructure, not the contribution; users here edit data and learned parameters, then compare resulting behavior. |
| [TensorSpace](https://github.com/tensorspace-team/tensorspace) | Interactive browser visualization of neural networks and pretrained models. | Visualization is paired with saved weight edits/freezes and measurable behavioral changes, not presented as a standalone network viewer. |

The product hypothesis is that making participatory decisions executable and
traceable changes what people can investigate. The MVP implements this hypothesis;
it does not establish research novelty or prove beneficial community outcomes.

## Scope decisions

- Reuse pandas, scikit-learn, PyTorch, and Gradio. Do not write another ML engine.
- Use four curated, trainable local architectures. Do not build a generic Hub browser.
- Make data versions, frozen goal revisions, checkpoint branches, and comparisons
  the primary objects, rather than chat sessions or a global leaderboard.
- Keep accuracy, false negatives, privacy, compute, and other criteria separate.
- Show disagreements through named value profiles and label authorship, rather
  than averaging priorities into one number.
