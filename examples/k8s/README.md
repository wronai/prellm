# preLLM Example — Kubernetes Debugging

Diagnose K8s cluster issues with domain-specific preprocessing.

## Quick Start

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Pod backend-api w namespace production restartuje się z CrashLoopBackOff",
    config_path="configs/domains/devops_k8s.yaml",
    strategy="structure",
)
print(result.content)
```

## With Context

```python
result = await preprocess_and_execute(
    query="Kubernetes pods killed by OOM on RPi cluster",
    config_path="configs/domains/devops_k8s.yaml",
    strategy="enrich",
    user_context={
        "cluster": "rpi-k3s-prod",
        "namespace": "backend",
        "node_ram": "4GB",
        "k8s_version": "1.28",
    },
)
```

## Run Example

```bash
# With real LLMs (requires API keys or Ollama)
python examples/k8s/main.py

# Run tests (no LLM needed, fully mocked)
pytest examples/k8s/test_k8s.py -v
```

## Domain Config

Uses `configs/domains/devops_k8s.yaml` with rules for:
- **CrashLoopBackOff** — pod restart diagnosis
- **OOM** — memory limit analysis
- **Scaling** — HPA/VPA configuration
- **Networking** — DNS, ingress, service mesh
- **Monitoring** — Prometheus, Grafana, alerting
