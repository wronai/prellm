# preLLM — Zadania wdrożeniowe: Pełne zbieranie kontekstu środowiska

> Na podstawie zaindeksowanej struktury projektu `project_functions.toon` (53 moduły, 685 funkcji)
> Data: 2026-02-16 | Wersja: 0.3.13

---

## Stan obecny vs. cel

### Co już jest (moduły z .toon):

| Moduł | Funkcji | Obecna odpowiedzialność |
|---|---|---|
| `prellm/analyzers/context_engine.py` | 8 | env, git, system — ale tylko wybrane zmienne |
| `prellm/context/codebase_indexer.py` | 12 | AST regex/tree-sitter — indeksowanie symboli |
| `prellm/context/user_memory.py` | 9 | SQLite historia interakcji |
| `prellm/env_config.py` | 12 | Konfiguracja .env, klucze API |
| `prellm/agents/preprocessor.py` | 6 | Orkiestracja: pipeline + kontekst → executor_input |
| `prellm/query_decomposer.py` | 9 | classify/structure/split/enrich/compose |
| `prellm/validators.py` | 8 | Walidacja schematów YAML |
| `prellm/pipeline.py` | 12 | PromptPipeline z krokami LLM + algo |

### Czego brakuje:

1. **Pełny shell context collector** — aktualnie `_gather_env` bierze tylko listę kluczy, nie cały env
2. **Filtr danych sensytywnych** — brak klasyfikatora co jest bezpieczne do largeLLM
3. **Kompresja folderu via code2logic** — `codebase_indexer` indeksuje symbole, ale nie kompresuje logiki
4. **Dynamiczne generowanie schematu** — brak auto-schema z kontekstu dla small-LLM
5. **Auto-wybór strategii** — small-LLM nie dostaje pełnego kontekstu do decyzji
6. **Pipeline sanityzacji** — brak etapu czyszczenia przed wysyłką do largeLLM

---

## TASK 1: Shell Context Collector

**Plik:** `prellm/context/shell_collector.py` (NOWY)
**Zależność:** rozszerza `prellm/analyzers/context_engine.py` → `_gather_env`, `_gather_system`

### 1.1 Zbieranie pełnego środowiska shell

```python
# Cel: klasa ShellContextCollector
# Metody do zaimplementowania:

class ShellContextCollector:
    def collect_env_vars(self, safe_only: bool = True) -> dict[str, str]
        """Pobierz wszystkie zmienne env, z filtrem sensytywnych."""

    def collect_process_info(self) -> dict[str, Any]
        """PID, CWD, user, uptime, parent process, TTY."""

    def collect_locale_info(self) -> dict[str, str]
        """LANG, LC_ALL, timezone, encoding."""

    def collect_shell_info(self) -> dict[str, str]
        """SHELL, TERM, COLUMNS, LINES, PS1 hash."""

    def collect_network_context(self) -> dict[str, str]
        """hostname, IP (local), DNS suffix — BEZ publicznego IP."""

    def collect_all(self) -> ShellContext
        """Pełny snapshot → Pydantic model."""
```

### 1.2 Integracja z istniejącym ContextEngine

**Plik do modyfikacji:** `prellm/analyzers/context_engine.py`
- Dodaj nowe `context_sources` type: `shell`
- Rozszerz `ContextEngine.gather()` o wywołanie `ShellContextCollector`
- Zachowaj kompatybilność z istniejącymi source'ami: `env`, `git`, `system`

### 1.3 Model danych

**Plik do modyfikacji:** `prellm/models.py`

```python
class ShellContext(BaseModel):
    env_vars: dict[str, str]          # przefiltrowane
    process: ProcessInfo
    locale: LocaleInfo
    shell: ShellInfo
    network: NetworkContext
    collected_at: datetime
    collection_duration_ms: float
```

### 1.4 Testy

**Plik:** `tests/test_shell_collector.py` (NOWY)
- `test_collect_env_vars_filters_secrets` — API_KEY, TOKEN, PASSWORD nie przechodzą
- `test_collect_process_info` — PID, CWD zwracane
- `test_collect_locale_info` — timezone poprawny
- `test_collect_all_returns_pydantic` — ShellContext waliduje się
- `test_collect_with_missing_env` — graceful handling brakujących zmiennych
- `test_safe_only_flag` — domyślnie filtruje, z `safe_only=False` zwraca wszystko

---

## TASK 2: Filtr danych sensytywnych (Sensitive Data Gate)

**Plik:** `prellm/context/sensitive_filter.py` (NOWY)
**Cel:** Zapewnić, że żadne klucze API, tokeny, hasła nie trafią do largeLLM

### 2.1 Klasyfikator sensytywności

```python
class SensitiveDataFilter:
    # Wbudowane wzorce
    SENSITIVE_PATTERNS: list[re.Pattern]  # API_KEY, TOKEN, SECRET, PASSWORD, PRIVATE_KEY, etc.
    SENSITIVE_VALUE_PATTERNS: list[re.Pattern]  # sk-..., ghp_..., gsk_..., sk-ant-...

    def classify_key(self, key: str) -> SensitivityLevel
        """SAFE | MASKED | BLOCKED na podstawie nazwy klucza."""

    def classify_value(self, value: str) -> SensitivityLevel
        """Wykryj tokeny/klucze po formacie wartości."""

    def filter_dict(self, data: dict, level: SensitivityLevel = MASKED) -> dict
        """Przefiltruj dict — zamaskuj lub usuń sensytywne."""

    def filter_context_for_large_llm(self, context: dict) -> dict
        """Specjalny filtr: usuwa WSZYSTKO co może być wrażliwe przed largeLLM."""

    def get_filter_report(self) -> FilterReport
        """Raport: co zostało zablokowane/zamaskowane (dla audytu)."""
```

### 2.2 Integracja z pipeline

**Pliki do modyfikacji:**
- `prellm/agents/preprocessor.py` → `PreprocessorAgent.preprocess()` — dodaj filtrowanie kontekstu
- `prellm/core.py` → `_execute_v3_pipeline()` — filtr przed `ExecutorAgent.execute()`
- `prellm/pipeline.py` → nowy algo step type: `sensitive_filter`

### 2.3 Konfiguracja YAML

```yaml
# configs/sensitive_rules.yaml
sensitive_keys:
  blocked: ["API_KEY", "SECRET", "TOKEN", "PASSWORD", "PRIVATE_KEY"]
  masked: ["DATABASE_URL", "REDIS_URL", "SMTP_"]
  safe: ["LANG", "TERM", "SHELL", "HOME", "USER", "PWD"]

sensitive_value_patterns:
  - "sk-[a-zA-Z0-9]{20,}"      # OpenAI
  - "sk-ant-[a-zA-Z0-9]{20,}"  # Anthropic
  - "ghp_[a-zA-Z0-9]{36}"      # GitHub
  - "gsk_[a-zA-Z0-9]{20,}"     # Groq
```

### 2.4 Testy

**Plik:** `tests/test_sensitive_filter.py` (NOWY)
- `test_blocks_api_keys`
- `test_masks_database_urls`
- `test_passes_safe_vars`
- `test_detects_token_by_value_pattern`
- `test_filter_report_tracks_blocked`
- `test_custom_rules_from_yaml`
- `test_filter_context_for_large_llm_is_strict`

---

## TASK 3: Kompresja folderu via code2logic

**Plik:** `prellm/context/folder_compressor.py` (NOWY)
**Zależność:** rozszerza `prellm/context/codebase_indexer.py`

### 3.1 Kompresor logiki projektu

Obecny `CodebaseIndexer` (12 funkcji) indeksuje symbole (funkcje, klasy, importy).
Trzeba dodać **kompresję logiki** — nie tylko listę symboli, ale:
- Relacje między modułami (kto importuje kogo)
- Schemat zależności (dependency graph skompresowany)
- Streszczenie odpowiedzialności każdego modułu (1 linia)

```python
class FolderCompressor:
    def __init__(self, indexer: CodebaseIndexer = None):
        """Użyj istniejącego CodebaseIndexer lub stwórz nowy."""

    def compress(self, root: Path, format: str = "toon") -> CompressedFolder
        """Skompresuj folder do lekkiej reprezentacji."""

    def to_toon(self, index: CodebaseIndex) -> str
        """Format .toon — jak w project_functions.toon."""

    def to_dependency_graph(self, index: CodebaseIndex) -> dict[str, list[str]]
        """Moduł → lista importowanych modułów (wewnętrznych)."""

    def to_summary(self, index: CodebaseIndex) -> str
        """1-liniowe streszczenie każdego modułu (z docstringów/nazw)."""

    def estimate_token_count(self, compressed: str) -> int
        """Oszacuj tokeny — musi się zmieścić w context window small-LLM."""
```

### 3.2 Integracja z PreprocessorAgent

**Plik do modyfikacji:** `prellm/agents/preprocessor.py`
- `PreprocessorAgent.__init__` — dodaj opcjonalny `folder_compressor`
- `PreprocessorAgent.preprocess()` → jeśli `codebase_path` podany, dołącz skompresowany folder do kontekstu
- Zachowaj kompatybilność z obecnym `codebase_indexer` (linia 69-93 w .toon)

### 3.3 Format .toon jako domyślny

Projekt już używa `project_functions.toon` — ustandaryzuj ten format:

```
project: {name}
generated: {ISO timestamp}
modules[{count}]{path,lang,items}:
  {path},{lang},{function_count}
  ...
function_details:
  {path}:
    functions[{count}]{name,kind,sig,loc,async,lines,cc,does}:
      {name},{kind},{sig},{loc},{async},{lines},{cc},{does}
```

### 3.4 Testy

**Plik:** `tests/test_folder_compressor.py` (NOWY)
- `test_compress_returns_toon_format`
- `test_dependency_graph_finds_internal_imports`
- `test_summary_uses_docstrings`
- `test_respects_gitignore`
- `test_token_estimate_reasonable`
- `test_large_project_fits_small_llm_context` — < 4096 tokenów

---

## TASK 4: Dynamiczne generowanie schematu kontekstu

**Plik:** `prellm/context/schema_generator.py` (NOWY)
**Cel:** Z zebranego kontekstu (shell + folder + memory) wygeneruj schema dla small-LLM

### 4.1 Generator schematu

```python
class ContextSchemaGenerator:
    def generate(self,
                 shell_context: ShellContext = None,
                 folder_compressed: CompressedFolder = None,
                 user_memory: list[dict] = None,
                 domain_rules: list[DomainRule] = None) -> ContextSchema
        """Złóż schema z dostępnych źródeł kontekstu."""

    def to_prompt_section(self, schema: ContextSchema) -> str
        """Sformatuj schema jako sekcja promptu dla small-LLM."""

    def estimate_relevance(self, schema: ContextSchema, query: str) -> dict[str, float]
        """Oceń które części kontekstu są istotne dla zapytania (0-1)."""
```

### 4.2 Schemat kontekstu (model Pydantic)

**Plik do modyfikacji:** `prellm/models.py`

```python
class ContextSchema(BaseModel):
    execution_env: str             # "shell" | "api" | "cli" | "server"
    platform: str                  # "linux" | "macos" | "windows"
    project_type: str | None       # "python" | "node" | "rust" | None
    project_summary: str | None    # skompresowany opis z .toon
    available_tools: list[str]     # ["git", "docker", "kubectl", ...]
    locale: str                    # "pl_PL.UTF-8"
    timezone: str                  # "Europe/Warsaw"
    user_history_summary: str | None  # z UserMemory
    sensitive_fields_blocked: int  # ile pól zablokowano
    schema_token_cost: int         # ile tokenów zajmuje schema
```

### 4.3 Integracja z pipeline

**Pliki do modyfikacji:**
- `prellm/pipeline.py` → nowy algo step: `context_schema_generator`
- `configs/pipelines.yaml` → dodaj krok `gather_schema` do pipeline'ów

### 4.4 Testy

**Plik:** `tests/test_schema_generator.py` (NOWY)
- `test_generate_from_shell_context`
- `test_generate_from_folder_only`
- `test_to_prompt_section_fits_token_budget`
- `test_relevance_scoring`
- `test_empty_sources_returns_minimal_schema`

---

## TASK 5: Auto-wybór strategii przez small-LLM

**Plik do modyfikacji:** `prellm/query_decomposer.py`
**Dotyczy:** `QueryDecomposer.decompose()` (linia 55-111) i `_classify()` (linia 113-127)

### 5.1 Rozszerzenie klasyfikacji o kontekst środowiska

Aktualnie `_classify` wysyła tylko query do small-LLM.
Powinien wysyłać: **query + ContextSchema** → small-LLM decyduje o strategii.

```python
# Modyfikacja w QueryDecomposer:

async def decompose(self, query, strategy=AUTO, context=None, env_schema=None):
    if strategy == DecompositionStrategy.AUTO:
        # NOWE: small-LLM dostaje schema i sam wybiera strategię
        strategy = await self._auto_select_strategy(query, env_schema)
    # ... reszta jak dotychczas
    
async def _auto_select_strategy(self, query: str, schema: ContextSchema) -> DecompositionStrategy:
    """Small-LLM wybiera strategię na podstawie query + kontekstu."""
```

### 5.2 Nowa strategia: AUTO

**Plik do modyfikacji:** `prellm/models.py`
- Dodaj `AUTO = "auto"` do `DecompositionStrategy` enum
- Auto mapuje się na jedną z 5 istniejących strategii po decyzji small-LLM

### 5.3 Prompt dla auto-strategii

**Plik do modyfikacji:** `configs/prompts.yaml`

```yaml
prompts:
  auto_strategy:
    system: |
      You are a strategy selector. Given a query and execution context,
      choose the best preprocessing strategy.
      
      Available strategies:
      - classify: Quick intent routing (simple queries)
      - structure: Extract action/target/params (DevOps, API calls)
      - split: Break into sub-queries (complex multi-part)
      - enrich: Add missing context (incomplete prompts)
      - passthrough: No preprocessing (direct queries)
      
      Context schema: {{ context_schema }}
      
      Respond ONLY with JSON: {"strategy": "...", "reason": "..."}
    max_tokens: 128
    temperature: 0.1
```

### 5.4 Testy

**Plik do modyfikacji:** `tests/test_decomposer.py` — dodaj:
- `test_auto_strategy_selects_classify_for_simple`
- `test_auto_strategy_selects_structure_for_devops`
- `test_auto_strategy_selects_enrich_for_incomplete`
- `test_auto_strategy_uses_context_schema`
- `test_auto_strategy_fallback_to_classify`

---

## TASK 6: Pipeline sanityzacji przed largeLLM

**Plik do modyfikacji:** `prellm/agents/executor.py`
**Dotyczy:** `ExecutorAgent.execute()` (linia 58-102)

### 6.1 Warstwa sanityzacji w ExecutorAgent

```python
# Modyfikacja ExecutorAgent:

class ExecutorAgent:
    def __init__(self, large_llm, response_validator=None, 
                 sensitive_filter=None,  # NOWE
                 response_schema_name=None):
        self.sensitive_filter = sensitive_filter

    async def execute(self, executor_input: str, system_prompt: str = '') -> ExecutorResult:
        # NOWE: sanityzacja przed wysyłką
        if self.sensitive_filter:
            executor_input = self.sensitive_filter.sanitize_text(executor_input)
            system_prompt = self.sensitive_filter.sanitize_text(system_prompt)
        # ... reszta jak dotychczas
```

### 6.2 Tekstowy sanitizer

**Plik do modyfikacji:** `prellm/context/sensitive_filter.py` (z Task 2)

```python
def sanitize_text(self, text: str) -> str:
    """Znajdź i zamaskuj tokeny/klucze w wolnym tekście."""
    # Regex na wzorce tokenów w środku tekstu
    # np. "use key sk-abc123def456" → "use key [REDACTED:openai_key]"
```

### 6.3 Testy

**Plik do modyfikacji:** `tests/test_executor_agent.py` — dodaj:
- `test_executor_sanitizes_input_before_llm_call`
- `test_executor_sanitizes_system_prompt`
- `test_executor_without_filter_passes_raw`

---

## TASK 7: Integracja pełnego pipeline'u (end-to-end)

**Plik do modyfikacji:** `prellm/core.py`
**Dotyczy:** `preprocess_and_execute()` (linia 59-156) i `_execute_v3_pipeline()` (linia 202-357)

### 7.1 Nowy przepływ w preprocess_and_execute

```python
async def preprocess_and_execute(
    query, small_llm, large_llm,
    strategy="auto",              # ZMIANA: domyślnie auto
    collect_env=True,             # NOWE: zbierz kontekst shell
    compress_folder=True,         # NOWE: kompresja via code2logic
    sanitize=True,                # NOWE: filtr sensytywnych
    ...
):
    # 1. Zbierz kontekst shell (NOWE)
    if collect_env:
        shell_ctx = ShellContextCollector().collect_all()
    
    # 2. Skompresuj folder (NOWE)
    if compress_folder and codebase_path:
        compressed = FolderCompressor().compress(codebase_path)
    
    # 3. Wygeneruj schema (NOWE)
    schema = ContextSchemaGenerator().generate(shell_ctx, compressed, ...)
    
    # 4. Small-LLM: decompose z pełnym kontekstem
    #    (strategy=auto → small-LLM sam wybierze)
    
    # 5. Filtr sensytywnych przed largeLLM (NOWE)
    if sanitize:
        executor_input = SensitiveDataFilter().filter_context_for_large_llm(...)
    
    # 6. Large-LLM: execute
```

### 7.2 Nowy pipeline YAML

**Plik:** `configs/pipelines.yaml` — dodaj:

```yaml
pipelines:
  full_context_aware:
    description: "Full context collection + auto strategy + sanitized execution"
    steps:
      - name: gather_shell
        type: algo
        handler: shell_context_collector
        output: shell_context
      - name: compress_folder
        type: algo
        handler: folder_compressor
        output: compressed_folder
      - name: generate_schema
        type: algo
        handler: context_schema_generator
        input: [shell_context, compressed_folder]
        output: context_schema
      - name: auto_classify
        prompt: auto_strategy
        input: [query, context_schema]
        output: selected_strategy
      - name: decompose
        prompt: "{{ selected_strategy.strategy }}"
        input: [query, context_schema]
        output: decomposition
      - name: compose
        prompt: compose
        input: [query, decomposition, context_schema]
        output: composed_prompt
      - name: sanitize
        type: algo
        handler: sensitive_filter
        input: [composed_prompt]
        output: sanitized_prompt
```

### 7.3 Testy integracyjne

**Plik:** `tests/test_full_context_pipeline.py` (NOWY)
- `test_end_to_end_with_shell_context`
- `test_sensitive_data_never_reaches_large_llm`
- `test_auto_strategy_with_full_context`
- `test_folder_compression_included_in_schema`
- `test_performance_under_200ms` — zbieranie kontekstu < 200ms
- `test_graceful_degradation` — brak shell/folderu → pipeline dalej działa

---

## TASK 8: CLI i konfiguracja

**Plik do modyfikacji:** `prellm/cli.py`

### 8.1 Nowe flagi CLI

```bash
# Nowe opcje w `prellm query`:
prellm query "Deploy app" \
  --collect-env          # zbierz pełny kontekst shell
  --compress-folder .    # skompresuj folder  
  --no-sanitize          # wyłącz filtr (dev only)
  --strategy auto        # auto-wybór strategii
  --show-schema          # pokaż schema (debug)
  --show-blocked         # pokaż co zostało zablokowane
```

### 8.2 Nowa komenda diagnostyczna

```bash
prellm context          # pokaż zebrany kontekst
prellm context --json   # jako JSON
prellm context --schema # pokaż wygenerowany schema
prellm context --blocked # pokaż zablokowane dane
```

### 8.3 Testy

**Plik do modyfikacji:** `tests/test_server.py` i nowy `tests/test_cli_context.py`

---

## Kolejność wdrożenia (zależności)

```
TASK 1: ShellContextCollector     ← brak zależności, start tutaj
  │
TASK 2: SensitiveDataFilter       ← potrzebuje wzorców z env_config
  │
TASK 3: FolderCompressor          ← zależy od CodebaseIndexer (istnieje)
  │
TASK 4: ContextSchemaGenerator    ← zależy od 1, 2, 3
  │
TASK 5: Auto-strategia            ← zależy od 4 (schema)
  │
TASK 6: Sanityzacja ExecutorAgent ← zależy od 2 (filter)
  │
TASK 7: Integracja end-to-end     ← zależy od 1-6
  │
TASK 8: CLI + konfiguracja        ← zależy od 7
```

---

## Podsumowanie zmian w istniejących plikach

| Plik (z .toon) | Obecne fn | Nowe fn | Typ zmiany |
|---|---|---|---|
| `prellm/models.py` | 0 | +4 | Nowe modele: ShellContext, ContextSchema, SensitivityLevel, AUTO |
| `prellm/analyzers/context_engine.py` | 8 | +2 | Nowy source `shell`, integracja z ShellCollector |
| `prellm/agents/preprocessor.py` | 6 | +2 | folder_compressor, schema w preprocess() |
| `prellm/agents/executor.py` | 3 | +1 | sensitive_filter w execute() |
| `prellm/query_decomposer.py` | 9 | +2 | _auto_select_strategy, AUTO enum |
| `prellm/pipeline.py` | 12 | +3 | Nowe algo handlery |
| `prellm/core.py` | 10 | +3 | collect_env, compress, sanitize |
| `prellm/cli.py` | 17 | +3 | context cmd, nowe flagi |

### Nowe pliki

| Plik | Szacowane fn | Odpowiedzialność |
|---|---|---|
| `prellm/context/shell_collector.py` | 7 | Pełne zbieranie env/shell/proces |
| `prellm/context/sensitive_filter.py` | 6 | Klasyfikacja + filtrowanie sensytywnych |
| `prellm/context/folder_compressor.py` | 6 | Kompresja folderu → .toon/graph/summary |
| `prellm/context/schema_generator.py` | 4 | Dynamiczny schema z kontekstu |
| `tests/test_shell_collector.py` | 6 | Testy shell collectora |
| `tests/test_sensitive_filter.py` | 7 | Testy filtra |
| `tests/test_folder_compressor.py` | 6 | Testy kompresji |
| `tests/test_schema_generator.py` | 5 | Testy generatora schematów |
| `tests/test_full_context_pipeline.py` | 6 | Testy E2E |

**Łącznie:** ~53 nowe funkcje, 9 nowych plików, 8 zmodyfikowanych plików.
