# preLLM Example — Polish Leasing Calculator

Generate leasing calculations with Polish financial domain rules.

## Quick Start

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Oblicz rate leasingu operacyjnego camper van za 250000 PLN netto, 48 miesiecy",
    config_path="configs/domains/polish_finance.yaml",
    strategy="structure",
)
print(result.content)
```

## Run Example

```bash
# With real LLMs
python examples/leasing/main.py

# Run tests (no LLM needed)
pytest examples/leasing/test_leasing.py -v
```

## Domain Config

Uses `configs/domains/polish_finance.yaml` with rules for:
- **Leasing** — rata, WIBOR, VAT, harmonogram
- **Faktura VAT** — NIP, KSeF, JPK
- **Kredyt** — hipoteka, RRSO, zdolnosc
- **Podatek** — PIT, CIT, ZUS
- **Inwestycja** — lokata, obligacja, fundusz
