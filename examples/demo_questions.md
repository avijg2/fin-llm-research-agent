# Demo questions

Try these against the bundled sample corpus (all fictional companies):

1. What was Northwind Cloud Systems revenue in Q3 2026?
   - Expected: $412.8 million, cited to the Northwind earnings call.

2. What is Northwind's revenue guidance for Q4 2026?
   - Expected: $430 million to $445 million.

3. How much did Bluepeak spend on research and development in fiscal 2026?
   - Expected: $212.3 million, cited to the Bluepeak 10-K excerpt.

4. What was Harborline's same-store sales change in Q3 2026?
   - Expected: declined 1.8%.

5. What was Tesla's revenue in Q3 2026?
   - Expected: refusal — "Insufficient evidence in the provided documents."
     Tesla is not in the corpus, so the agent must not invent a number.

Run them with:

```bash
python scripts/demo.py
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was Northwind Cloud Systems revenue in Q3 2026?"}'
```
