# STG-NF Epistemic Ensemble Utilities

This module mirrors the mixture-based ensemble aggregation strategy from the
`nflows_epistemic` reference while reusing the masked STG-NF implementation that
already exists in the project.

## Key Components

- `masked_layers.py`, `masked_model.py`, and `masked_trainer.py` host the
  masked ensemble STG-NF components that were previously at the repository
  root. They provide the base model wrapper plus the standard trainer.
- `mixture_utils.py` implements helpers to convert per-member negative
  log-likelihoods (expressed in bits per dimension) into natural-log
  probabilities and aggregate them via log-mean-exp. The resulting mixture NLL
  provides a closer analogue to the `nflows_epistemic` ensemble likelihood.
- `trainer.py` defines `EpistemicMaskedEnsembleTrainer`, a drop-in replacement
  for `MaskedEnsembleTrainer` that evaluates every ensemble member on each batch
  and optimises the log-mean-exp mixture loss.
- `create_epistemic_trainer` is a convenience factory mirroring
  `create_ensemble_trainer`, returning the epistemic trainer wired to a
  `MaskedEnsembleSTGNF` model.

## Usage Sketch

```python
from stg_nf_epistemic import create_epistemic_trainer

trainer = create_epistemic_trainer(
    model_args=stg_nf_args,
    ensemble_size=5,
    mask_type="random",
    training_args=args,
    train_loader=train_loader,
    test_loader=test_loader,
    test_metadata=test_metadata,
    device=args.device,
)

history = trainer.train(epochs=args.epochs, save_dir=args.ckpt_dir)
```

The resulting `history` entries report the mixture loss and reuse the existing
evaluation pipeline to compute ROC-AUC metrics.
