from pathlib import Path

METRIC_ORDER = ["SymbolsDistancesV2", "SymbolsDistances", "TokenizedBLEU", "CHRF", "CLIPScore"]


def ordered_metric_dirs(sign_dir):
    dirs = [d for d in sign_dir.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda d: METRIC_ORDER.index(d.name) if d.name in METRIC_ORDER else len(METRIC_ORDER))


if __name__ == "__main__":
    matches_dir = Path(__file__).parent.parent.parent / "assets" / "matches"

    signs_header = ["<td></td>"]
    metrics_header = ["<td></td>"]

    rows = [[f"<td>{i + 1}</td>"] for i in range(10)]  # 10 rows

    for sign_dir in matches_dir.iterdir():
        # pylint: disable=invalid-name
        colspan = 0
        for metric_dir in ordered_metric_dirs(sign_dir):
            if metric_dir.is_dir():
                colspan += 1
                metrics_header.append(f"<td>{metric_dir.name}</td>")
                for i in range(10):
                    rows[i].append(f"<td><img alt='{metric_dir.name} rank {i + 1}' "
                                   f"src='assets/matches/{sign_dir.name}/{metric_dir.name}/{i}.png' /></td>")
        signs_header.append(f"<td colspan='{colspan}'>"
                            f"<img alt='reference sign' src='assets/matches/{sign_dir.name}/ref.png' /></td>")

    print("<table style=\"text-align: center\">")
    print("<thead>")
    print(f"<tr>{''.join(signs_header)}</tr>")
    print(f"<tr>{''.join(metrics_header)}</tr>")
    print("</thead>")
    print("<tbody>")
    for row in rows:
        print(f"<tr>{''.join(row)}</tr>")
    print("</tbody>")
    print("</table>")
