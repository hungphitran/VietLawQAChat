# LLM Response Generation

The final application layer is intended to turn retrieved evidence into a readable answer with grounding. In a legal QA setting, this means the model should not answer from memory alone. It should use the retrieved passages as context and ideally cite or expose the source passages used to support the answer.

If the current implementation is still under active development, the report should describe the intended prompting strategy, answer formatting, and grounding rules without overclaiming completion. The important technical point is that generation must remain evidence-aware, because unsupported legal answers are not acceptable for the target use case.
