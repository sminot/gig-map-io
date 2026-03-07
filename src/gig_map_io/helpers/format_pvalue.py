def format_pvalue(p: float) -> str:
    """Format a p-value as a string."""
    if p >= 0.01:
        return f"{p:.2f}"
    elif p >= 0.001:
        return f"{p:.3f}"
    elif p >= 0.0001:
        return f"{p:.4f}"
    else:
        return f"{p:.2E}"