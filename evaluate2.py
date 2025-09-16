import evaluate

machine_summary = "The car was big."
human_summary = "The automobile was large."

# 1. Load the BERTScore metric
bertscore = evaluate.load("bertscore")

# 2. Compute the scores
results = bertscore.compute(
    predictions=[machine_summary],
    references=[human_summary],
    lang="en"  # Specify the language
)

# The 'f1' score is what you'll look at most
print(f"BERTScore F1: {results['f1'][0]:.4f}")
# Output will be a high score (e.g., 0.95+),
# because it knows "car" is like "automobile".