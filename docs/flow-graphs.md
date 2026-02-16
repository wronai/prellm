# preLLM Flow Graphs

Mermaid diagrams showing the preLLM preprocessing pipeline for each use case.

## Configuration & Diagnostics

```bash
make config         # interactive wizard + diagnostics
source .env
prellm doctor --live
make examples       # real-time demo scripts
```

## Core Pipeline

```mermaid
graph TD
    A["preprocess_and_execute(query, small_llm, large_llm)"] --> B[ContextEngine]
    B --> C["QueryDecomposer (Small LLM ≤3B)"]
    C --> D{Strategy?}
    D -->|classify| E[Classify Intent + Domain]
    D -->|structure| F[Extract Action/Target/Params]
    D -->|split| G[Break into Sub-queries]
    D -->|enrich| H[Add Missing Context]
    D -->|passthrough| I[Forward As-Is]
    E --> J[Compose Prompt]
    F --> J
    G --> J
    H --> J
    I --> K["LLMProvider (Large LLM)"]
    J --> K
    K --> L[PreLLMResponse]
    L --> M[Pydantic Validation]

    style C fill:#e1f5fe
    style K fill:#f3e5f5
    style M fill:#e8f5e9
```

## Use Case 1: Code Refactoring

```mermaid
graph TD
    A["USER: 'Popraw hardcode w projekcie'"] --> B["Qwen2.5:3b (local)"]
    B --> C["Context: Gdańsk / Python / Docker / K8s"]
    B --> D["Classify: intent=refactor, confidence=0.92"]
    B --> E["Structure: action=refactor, target=hardcoded_strings"]
    E --> F["Missing: file_paths, language_version"]
    D --> G[Compose Meta-Prompt]
    E --> G
    F --> G
    C --> G
    G --> H["Claude 4.6 (cloud)"]
    H --> I["Refactored Code + Tests"]
    I --> J[Pydantic Validated YAML]

    style B fill:#e1f5fe,stroke:#0288d1
    style H fill:#f3e5f5,stroke:#7b1fa2
    style J fill:#e8f5e9,stroke:#388e3c
```

## Use Case 2: Kubernetes Log Analysis

```mermaid
graph LR
    A["Logi 10k linii"] --> B["Qwen2.5:3b Parser"]
    B --> C["Root cause: OOM"]
    B --> D["Env: RPi cluster, 16GB RAM"]
    B --> E["Intent: diagnose, confidence=0.95"]
    C --> F["Enriched Meta-Prompt"]
    D --> F
    E --> F
    F --> G["Claude 4.6: Fix + Monitoring"]
    G --> H["YAML: K8s manifests"]
    G --> I["YAML: Prometheus rules"]
    G --> J["YAML: Ansible playbook"]

    style B fill:#e1f5fe,stroke:#0288d1
    style G fill:#f3e5f5,stroke:#7b1fa2
```

## Use Case 3: Business Automation (Leasing)

```mermaid
graph TD
    A["USER: 'Kalkulacja leasingu camper van'"] --> B["Qwen2.5:3b"]
    B --> C["Domain: automotive, locale=PL"]
    B --> D["Required: VAT, WIBOR, okres"]
    B --> E["Intent: calculate, confidence=0.88"]
    C --> F[Enriched Prompt]
    D --> F
    E --> F
    F --> G["Claude 4.6"]
    G --> H["Python Calculator"]
    G --> I["Excel Generator"]
    G --> J["PDF Templates"]
    G --> K["Email Automation"]

    style B fill:#e1f5fe
    style G fill:#f3e5f5
```

## API Server Flow

```mermaid
graph TD
    A[Client] -->|"POST /v1/chat/completions"| B[FastAPI Server]
    B --> C{Parse Model Pair}
    C -->|"prellm:qwen→claude"| D["Small: qwen, Large: claude"]
    C -->|"prellm:default"| E["Small: env.SMALL_MODEL, Large: env.LARGE_MODEL"]
    D --> F["preprocess_and_execute()"]
    E --> F
    F --> G{Stream?}
    G -->|No| H[JSON Response]
    G -->|Yes| I[SSE Stream]
    H --> J["OpenAI-compatible JSON + prellm_meta"]
    I --> K["Stage Progress + Content Chunks + [DONE]"]

    style B fill:#fff3e0,stroke:#e65100
    style F fill:#e1f5fe,stroke:#0288d1
```

## Streaming Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant S as preLLM Server
    participant Q as Qwen2.5 (Small)
    participant L as Claude (Large)

    C->>S: POST /v1/chat/completions {stream: true}
    S->>Q: Classify + Structure
    S-->>C: data: {stage: "preprocessing", progress: 0}
    Q-->>S: Classification + Structure
    S-->>C: data: {stage: "preprocessing", progress: 100}
    S->>L: Composed Prompt
    S-->>C: data: {stage: "execution", progress: 0}
    L-->>S: Response Content
    S-->>C: data: {stage: "execution", progress: 100}
    S-->>C: data: {choices: [{delta: {content: "chunk1"}}]}
    S-->>C: data: {choices: [{delta: {content: "chunk2"}}]}
    S-->>C: data: {choices: [{delta: {}}, finish_reason: "stop"]}
    S-->>C: data: [DONE]
```

## Batch Processing Flow

```mermaid
graph TD
    A["POST /v1/batch"] --> B[Parse Items]
    B --> C["Task 1: refactor"]
    B --> D["Task 2: k8s logs"]
    B --> E["Task 3: leasing"]
    C --> F["preprocess_and_execute()"]
    D --> G["preprocess_and_execute()"]
    E --> H["preprocess_and_execute()"]
    F --> I["asyncio.gather()"]
    G --> I
    H --> I
    I --> J["Batch Response JSON"]

    style I fill:#fff3e0,stroke:#e65100
```

## Docker Deployment

```mermaid
graph LR
    A[docker-compose up] --> B[Ollama Container]
    A --> C[preLLM Container]
    B -->|"ollama/qwen2.5:3b"| C
    C -->|"ANTHROPIC_API_KEY"| D[Claude API]
    C -->|"OPENAI_API_KEY"| E[OpenAI API]
    C -->|":8080"| F[Client]

    style B fill:#e1f5fe
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#f3e5f5
```

## Strategy Decision Tree

```mermaid
graph TD
    A[User Query] --> B{Query Type?}
    B -->|Simple/Direct| C["passthrough"]
    B -->|Needs Classification| D["classify"]
    B -->|DevOps Command| E["structure"]
    B -->|Complex Multi-part| F["split"]
    B -->|Missing Context| G["enrich"]

    C --> H["→ Large LLM directly"]
    D --> I["→ intent + domain → compose → Large LLM"]
    E --> J["→ action + target + params → compose → Large LLM"]
    F --> K["→ sub-queries → compose → Large LLM"]
    G --> L["→ missing fields + context → compose → Large LLM"]

    style C fill:#e8f5e9
    style D fill:#e1f5fe
    style E fill:#fff3e0
    style F fill:#fce4ec
    style G fill:#f3e5f5
```
