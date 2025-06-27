# MESS+ — Energy-Optimal LLM Inference 🔋🤖

NeurIPS 2024 AFM workshop project that routes each request to the cheapest LLM that still meets a user-defined service-level guarantee (accuracy ≥ $\alpha$).

## Key Contributions

- **Adaptive model selection**: Predicts performance (accuracy, latency) per query using lightweight FastText.
- **Energy minimization**: Solves a mixed-integer program to choose the most energy-efficient model under SLAs.
- **Operational support**: Handles batch inference, power monitoring (Jetson/x86), and reproducible notebooks.

## Cite
```
@inproceedings{zhang2024messplus,
  title = {MESS+: Energy‑Optimal Inference in Language Model Zoos with Service Level Guarantees},
  author = {Zhang, Ryan and Woisetschläger, Herbert and Wang, Shiqiang and Jacobsen, Hans Arno},
  booktitle = {Adaptive Foundation Models Workshop @ NeurIPS 2024},
  year = {2024}
}
```
