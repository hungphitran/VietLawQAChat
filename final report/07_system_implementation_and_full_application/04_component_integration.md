# Component Integration

Component integration is where the project becomes a full system rather than a set of separate experiments. The retriever produces candidates, the reranker refines them, the graph module can add related evidence, and the generator turns the final context into an answer. Each step depends on the previous one, so the integration order matters.

The report should explain this flow clearly and should also note which parts are already stable and which parts are still being integrated. That distinction is important because it keeps the system description honest while still showing the project direction.
