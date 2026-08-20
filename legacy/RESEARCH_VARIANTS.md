# Archived research variants

These are research directions explored during development but **not used by the released DiR experiment**. They are retained only to show the main alternatives considered around dictionary reuse, coefficient adaptation, sparsity, and training control.

## 1. Learnable dictionary variants

- **Free dictionary:** learn dictionary atoms directly instead of preserving a fixed reusable dictionary.
- **Dictionary delta / anchor:** start from a reference dictionary and learn a small residual update while penalizing movement away from the reference.
- **Double-sparse variant:** combine dictionary adaptation with explicit sparsity constraints so both the reusable structure and its usage can be restricted.

Core question: how much of the dictionary should remain reusable and stable, versus being allowed to adapt to the target task?

## 2. Explicit sparse projection

Periodically keep only the largest coefficient entries (top-k / hard support) and zero the rest. Competitive-reset variants additionally allowed discarded entries to compete again at selected stages.

Core question: whether explicit discrete sparsification gives a clearer reusable computational support than the softer sparsity used in the final method.

## 3. Coefficient noise

Perturb coefficients during training while keeping the dictionary structure intact.

Core question: whether forcing robustness to small coefficient changes improves stability of dictionary reuse and functional correspondence.

## 4. Coefficient-only refit

Temporarily optimize coefficients while holding the rest of the relevant representation fixed.

Core question: how much target adaptation can be recovered by remapping dictionary usage alone, without changing the reusable dictionary itself.

## 5. Blockwise curriculum

Adapt selected transformer blocks in stages instead of optimizing all eligible blocks with the same schedule.

Core question: whether correspondence is easier to preserve when adaptation proceeds progressively through the network.

## 6. Path intervention

Intervene on selected internal paths or components to isolate their functional contribution rather than relying only on representational similarity.

Core question: whether reused components retain not only similar activations but also similar causal/functional effects on the model output.

## 7. Plateau-based stopping

Use stabilization of a monitored training quantity to decide when an adaptation stage should stop instead of relying only on a fixed epoch schedule.

Core question: whether adaptive stage lengths can reduce unnecessary optimization while preserving the intended reuse behavior.

---

These variants are **concept records, not maintained implementations**. Their presence does not imply that they outperform the released method or that they can be enabled through the current configuration. Reproducing one requires a separate implementation or port.
