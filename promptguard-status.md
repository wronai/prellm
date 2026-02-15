---
title: "PromptGuard — Warstwa pośrednia do standaryzacji i walidacji promptów LLM"
slug: promptguard-status
date: 2026-02-15
author: Softreck
categories:
  - Projekty
  - AI
  - Developer Tools
tags:
  - promptguard
  - llm
  - bias-detection
  - litellm
  - prompt-engineering
  - devops
  - process-automation
excerpt: "PromptGuard to lekki framework Python wykrywający bias i nieprecyzję w zapytaniach do LLM, standaryzujący prompty via YAML config i działający jako proxy dla 100+ modeli. Wersja DevOps rozszerza możliwości o planowanie i wykonywanie procesów firmowych."
---

# PromptGuard — Status projektu

**Status:** 🚧 MVP in Development | **Typ:** Python Library / CLI Tool / DevOps Middleware  
**Repozytorium:** `softreck/promptguard` | **Licencja:** MIT  
**Wersja docelowa:** v0.1.0

---

## O projekcie

PromptGuard to warstwa pośrednia (middleware) między użytkownikiem a modelem językowym, która automatycznie wykrywa bias i nieprecyzję w zapytaniach, standaryzuje prompty przez deklaratywny config YAML i obsługuje 100+ modeli LLM przez LiteLLM. Wersja rozszerzona o moduł **DevOps Process Engine** umożliwia planowanie i wykonywanie procesów firmowych z walidacją na każdym etapie.

## Geneza

Inspiracją jest klasyczny test z myjnią samochodową — pytanie „człowiek jedzie na myjnię" jest interpretowane przez AI jako „jedzie umyć samochód", podczas gdy kontekst może być dowolny (odebrać auto po serwisie, zapytać o cenę, szukać pracy). W środowisku DevOps ten sam problem dotyczy poleceń typu „zdeployuj aplikację" — bez kontekstu (staging vs production, blue-green vs rolling, z migracją DB czy bez) LLM może wygenerować niebezpieczne instrukcje.

PromptGuard automatycznie wykrywa takie pułapki, wymusza doprecyzowanie i waliduje każdy krok procesu przed wykonaniem.

## Kluczowe założenia

**Architektura rdzenia:**

- Deklaratywny config YAML definiujący reguły biasu, szablony doprecyzowania i polityki bezpieczeństwa.
- Zero-downtime retry: niejasny input → enriched prompt → ponowne wywołanie LLM z kontekstem.
- Lekkość — poniżej 50MB, async-first, 5 głównych zależności (litellm, pydantic, pyyaml, nltk, typer).
- Kontrola w stylu MCP — użytkownik decyduje o polityce (strict / lenient / devops).

**Rozszerzenie DevOps:**

- Process Chains — definiowanie wieloetapowych procesów (CI/CD, deployment, audit) jako łańcuchów kroków z walidacją.
- Context Injection — automatyczne wstrzykiwanie kontekstu środowiska (env vars, git branch, cluster info) do promptów.
- Approval Gates — wymaganie zatwierdzenia przed wykonaniem krytycznych operacji.
- Audit Trail — pełny log każdego zapytania, decyzji i wyniku w formacie JSON/YAML.

## Architektura

```
┌─────────────────────────────────────────────────────────┐
│                    PromptGuard Core                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ QueryAnalyzer│  │ BiasDetector │  │ ContextEngine │  │
│  │  (regex+NLTK)│  │  (patterns)  │  │  (env+git)    │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                   │          │
│         ▼                 ▼                   ▼          │
│  ┌─────────────────────────────────────────────────┐     │
│  │              Enrichment Pipeline                 │     │
│  │   detect → clarify → enrich → validate          │     │
│  └─────────────────────┬───────────────────────────┘     │
│                        │                                 │
│  ┌─────────────────────▼───────────────────────────┐     │
│  │           AsyncGuardClient (LiteLLM)            │     │
│  │   fallback: gpt-4o-mini → llama3 → mistral      │     │
│  └─────────────────────┬───────────────────────────┘     │
│                        │                                 │
│  ┌─────────────────────▼───────────────────────────┐     │
│  │         Output Validator (Pydantic v2)           │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─────────────────── DevOps Extension ────────────┐     │
│  │  ProcessChain │ ApprovalGate │ AuditTrail       │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Moduł DevOps — Planowanie i Wykonywanie Procesów

Kluczowe rozszerzenie PromptGuard dla środowisk firmowych to moduł zarządzania procesami. Pozwala definiować wieloetapowe workflow w YAML, gdzie każdy krok jest walidowany przez PromptGuard przed wykonaniem.

**Przykładowy proces deployment:**

```yaml
process: deploy-production
context_sources:
  - env: [CLUSTER, NAMESPACE, GIT_SHA]
  - git: [branch, last_commit]
steps:
  - name: pre-check
    prompt: "Sprawdź gotowość {CLUSTER} do deployu {GIT_SHA}"
    policy: strict
    approval: auto
  - name: migration
    prompt: "Wygeneruj i zwaliduj migrację DB dla {NAMESPACE}"
    policy: strict
    approval: manual
  - name: deploy
    prompt: "Wykonaj rolling deploy na {CLUSTER}/{NAMESPACE}"
    policy: strict
    approval: manual
    rollback: true
  - name: verify
    prompt: "Zweryfikuj health-check po deployu"
    policy: lenient
    approval: auto
```

## Przykład użycia

**CLI — prosty prompt:**
```bash
promptguard run --config rules.yaml "Jedzie na myjnię?" --model llama3
# → Wykrywa brak kontekstu, dodaje szablon doprecyzowania
# → Odpowiedź z flagą clarified: true
```

**CLI — proces DevOps:**
```bash
promptguard process run --config deploy.yaml --env production
# → Krok 1: pre-check (auto-approved) ✓
# → Krok 2: migration (czeka na manual approval)
# → Krok 3: deploy (po zatwierdzeniu, z rollback ready)
# → Krok 4: verify (auto-approved) ✓
```

**Python API:**
```python
from promptguard import PromptGuard, ProcessChain

guard = PromptGuard("rules.yaml")
result = await guard("Zdeployuj na produkcję", model="gpt-4o-mini")
# → GuardResponse(clarified=True, content="Potrzebuję kontekstu: ...")

chain = ProcessChain("deploy.yaml")
await chain.execute(env="production", approval_callback=slack_notify)
```

## Przewaga nad istniejącymi rozwiązaniami

| Cecha | PromptGuard | LangChain | Guardrails AI | NeMo |
|-------|-------------|-----------|---------------|------|
| Footprint | <50MB, 5 deps | 100+ deps | Sztywne schemas | Enterprise |
| Proxy LLM | ✅ LiteLLM 100+ | Partial | ❌ | ❌ |
| Bias detection | ✅ NLTK+regex | ❌ | ❌ | ✅ |
| DevOps chains | ✅ YAML workflows | ❌ | ❌ | ❌ |
| Approval gates | ✅ Manual/auto | ❌ | ❌ | ❌ |
| Edge deploy | ✅ Docker <100MB | ❌ | ❌ | ❌ |
| Audit trail | ✅ JSON/YAML | Partial | Partial | ✅ |

## Stack techniczny

LiteLLM (proxy 100+ modeli), Pydantic v2 (type-safe output i walidacja procesów), PyYAML (deklaratywny config), NLTK/textstat (detekcja biasu i ambiguity), Typer (CLI z subcommands). Deploy via Docker poniżej 100MB, Nix flake dla RPi/edge, lub jako PyPI package.

## Plan rozwoju

**Faza 1 — Core MVP (v0.1.0):**
Setup repo (Poetry), core QueryAnalyzer, YAML loader, LiteLLM wrapper, retry chain, CLI tool, testy pytest z mock.

**Faza 2 — DevOps Extension (v0.2.0):**
ProcessChain engine, approval gates (Slack/Teams webhook), audit trail, context injection z env/git.

**Faza 3 — Enterprise (v0.3.0):**
Dashboard web (FastAPI + HTMX), metryki Prometheus, integracja z Kubernetes operators, RBAC dla approval flow.

**Faza 4 — Edge & Distribution (v1.0.0):**
Docker <100MB, Nix flake, publikacja PyPI, MicroPython port dla ESP32, dokumentacja i kursy online.

## Podsumowanie

PromptGuard wypełnia lukę między lekkimi bibliotekami do prompt engineering a ciężkimi frameworkami enterprise. Rozszerzenie DevOps Process Engine czyni z niego narzędzie nie tylko do walidacji promptów, ale do pełnego zarządzania procesami firmowymi z AI w pętli — od planowania przez wykonanie po audyt.

---

*Ostatnia aktualizacja: 2026-02-15 | Softreck*
