# pylint: disable=line-too-long, duplicate-code
from pathlib import Path

if __name__ == "__main__":
    matches_dir = Path(__file__).parent.parent.parent / "assets" / "matches"

    # LaTeX header rows
    signs_header = ["&"]
    metrics_header = ["Rank"]

    # LaTeX body rows
    rows = [[str(i + 1)] for i in range(10)]  # 10 rows

    for sign_dir in matches_dir.iterdir():
        # pylint: disable=invalid-name
        colspan = 0
        for metric_dir in sign_dir.iterdir():
            if metric_dir.is_dir():
                colspan += 1
                metrics_header.append(f"\\texttt{{{metric_dir.name}}}")
                for i in range(10):
                    rows[i].append(
                        f"\\includegraphics[width=0.07\\textwidth]{{assets/matches/{sign_dir.name}/{metric_dir.name}/{i}.png}}"
                    )
        signs_header.append(
            f"\\multicolumn{{{colspan}}}{{c|}}{{\\includegraphics[width=0.1\\textwidth]{{assets/matches/{sign_dir.name}/ref.png}}}}"
        )

    # Create LaTeX table
    print("\\begin{table*}[ht]")
    print("    \\centering")
    print("    \\begin{tabular}{c|" + "c" * len(metrics_header) + "}")
    print("        \\toprule")
    print(f"        {' & '.join(signs_header)} \\\\")
    print("        \\cmidrule{2-" + str(len(metrics_header) + 1) + "}")
    print(f"        {' & '.join(metrics_header)} \\\\")
    print("        \\midrule")
    for row in rows:
        print(f"        {' & '.join(row)} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{Top 10 nearest neighbors for selected signs using different evaluation metrics. The reference signs are shown at the top, and the retrieved signs are displayed in order of decreasing similarity score from left to right.}")
    print("    \\label{tab:nearest_neighbors}")
    print("\\end{table*}")
