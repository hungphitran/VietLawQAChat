# System Overview

The full application is a legal question-answering system built on top of the retrieval core. The user enters a question, the system retrieves candidate legal passages, reranks them, optionally expands the evidence through graph links, and then presents a grounded answer in the interface. The design aims to make the final answer readable without losing traceability.

This section should explain the complete user flow at a high level so that the reader can understand how the retrieval experiments connect to the final application. The important point is that the system is modular: each component can be tested separately, but the user only sees the integrated result.
