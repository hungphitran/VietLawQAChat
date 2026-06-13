# Experimental Setup

The experiments are designed to compare methods under a shared evaluation protocol. The same dataset split, corpus, and scoring rules are used across baselines so that changes in result can be attributed to the model or pipeline choice rather than to a different test setting.

The implementation supports config-driven experiments, which makes it easier to reproduce runs and compare settings such as segmentation, model choice, retrieval depth, and fusion strategy. When reporting results, the paper should keep the hardware and hyperparameter details concise, but still explicit enough for reproducibility.
