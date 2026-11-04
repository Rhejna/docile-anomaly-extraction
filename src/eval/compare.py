def build_comparison_table(baseline_scores: dict, model_scores: dict) -> str:
    """Build a Markdown comparison table between baseline and LayoutLMv3 model scores.
    
    Args:
        baseline_scores: Dictionary of scores from the baseline regex approach.
        model_scores: Dictionary of scores from the LayoutLMv3 model.
        
    Returns:
        A string containing the formatted Markdown table.
    """
    fields = ["company", "date", "address", "total", "global"]
    
    # Markdown table header
    table_lines = [
        "| Field | Baseline (%) | LayoutLMv3 (%) | Gap (%) |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    for field in fields:
        base_val = baseline_scores.get(field, 0.0)
        model_val = model_scores.get(field, 0.0)
        gap = model_val - base_val
        
        # Format field name nicely (capitalize)
        field_display = field.capitalize()
        
        # Add row with formatted values and explicit sign for the gap (+/-)
        row = f"| {field_display} | {base_val:.2f}% | {model_val:.2f}% | {gap:+.2f}% |"
        table_lines.append(row)
        
    return "\n".join(table_lines)