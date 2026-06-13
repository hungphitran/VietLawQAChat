# Backend Architecture

The backend architecture should describe how the main modules are connected in code and at runtime. In the current project structure, the retrieval and reranking components are separated into dedicated modules, and the pipeline layer combines them into a configurable workflow. This modular layout makes it easier to add new retrievers or rerankers without rewriting the whole system.

If the backend already includes an API or service layer, that should be documented here as well. If not, the report can still describe the intended service boundaries: retrieval service, graph service, generation service, and UI-facing response service.
