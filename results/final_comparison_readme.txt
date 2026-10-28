FINAL COMPARISON FILES

final_comparison_summary.csv: language-level means and available sample counts.
final_comparison_per_article.csv: merged per-article metrics and preliminary AI-screening fields.
final_comparison_figures/: plots of preliminary factuality and coverage screening scores.

IMPORTANT INTERPRETATION NOTE
The factuality and coverage fields in human_eval_all100_ai_assisted_screening.csv are preliminary AI-assisted screening results, not verified human annotations. Do not describe them as ground truth or use them as definitive evidence that one model is better. Validate a subset using the article text as the only factual evidence. ROUGE/BERTScore values are taken from the existing per-example metrics file and are not recalculated here.

This script reads existing files and creates new outputs only. It does not run model inference or modify generation checkpoints.
