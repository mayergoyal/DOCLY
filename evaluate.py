import evaluate

# 1. Your generated summary (the "prediction")
machine_summary = "The fox jumped quickly over the lazy dog."

# 2. Your human-written "perfect" summary (the "reference")
human_summary = "A quick brown fox jumps over the lazy dog."

# 3. Load the ROUGE metric
rouge = evaluate.load('rouge')

# 4. Compute the scores
# Note: The predictions and references must be in lists.
results = rouge.compute(
    predictions=[machine_summary],
    references=[human_summary]
)

print(results)
# Output might look like:
# {'rouge1': 0.8235, 'rouge2': 0.5882, 'rougeL': 0.8235, ...}