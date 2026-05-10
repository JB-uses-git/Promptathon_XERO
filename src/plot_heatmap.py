import argparse
import os
import sys

def main():
    try:
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Missing plotting dependencies. Install with: pip install pandas numpy matplotlib seaborn", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser(description="Create and save a readable correlation heatmap for a CSV dataset.")
    parser.add_argument("--input", "-i", default="synthetic_customer_churn_100k.csv", help="Input CSV file (workspace-relative).")
    parser.add_argument("--output", "-o", default="artifacts/plots/heatmap_correlation.png", help="Output image path to save the heatmap.")
    parser.add_argument("--annot", action="store_true", help="Annotate cells with correlation values (may be crowded for large matrices).")
    parser.add_argument("--max-cols", type=int, default=50, help="If more than this many numeric columns, annotation is disabled automatically.")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        # try workspace root
        alt = os.path.join(os.getcwd(), args.input)
        if os.path.exists(alt):
            input_path = alt
        else:
            print(f"Input file not found: {args.input}", file=sys.stderr)
            sys.exit(2)

    df = pd.read_csv(input_path)
    df_num = df.select_dtypes(include=[np.number]).copy()
    if df_num.shape[1] == 0:
        print("No numeric columns found to compute correlations.", file=sys.stderr)
        sys.exit(3)

    corr = df_num.corr()

    # Mask upper triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Figure size tuned to number of columns
    ncols = corr.shape[0]
    width = max(8, ncols * 0.45)
    height = max(6, ncols * 0.35)

    sns.set(style="white")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    plt.figure(figsize=(width, height))
    annot = args.annot and (ncols <= args.max_cols)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=annot,
        fmt=".2f",
        square=False,
        linewidths=.5,
        cbar_kws={"shrink": .5},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    out_dir = os.path.dirname(args.output) or "artifacts/plots"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved heatmap to: {args.output}")


if __name__ == "__main__":
    main()
