from pathlib import Path

if __name__ == "__main__":
    matches_dir = Path(__file__).parent.parent.parent / "assets" / "matches"

    signs_header = ["<td></td>"]
    metrics_header = ["<td></td>"]

    rows = [[f"<td>{i + 1}</td>"] for i in range(10)]  # 10 rows

    for sign_dir in matches_dir.iterdir():
        # pylint: disable=invalid-name
        colspan = 0
        for metric_dir in sign_dir.iterdir():
            if metric_dir.is_dir():
                colspan += 1
                metrics_header.append(f"<td>{metric_dir.name}</td>")
                for i in range(10):
                    rows[i].append(f"<td><img src='assets/matches/{sign_dir.name}/{metric_dir.name}/{i}.png' /></td>")
        signs_header.append(f"<td colspan='{colspan}'><img src='assets/matches/{sign_dir.name}/ref.png' /></td>")

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
