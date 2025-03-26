# generate_index_page.py

import os

def generate_index(feature_ids, output_dir="reports/features", index_path="reports/index.html"):
    rows = []

    for fid in feature_ids:
        fname = f"feature_{fid}.html"
        iframe = f'<iframe src="{fname}" width="100%" height="200"></iframe>'
        link = f'<a href="{fname}" target="_blank">Full View</a>'
        rows.append(f"<tr><td>{fid}</td><td>{iframe}</td><td>{link}</td></tr>")

    html = f"""
    <html>
    <head><title>SAE Feature Index</title></head>
    <body>
    <h1>Reasoning Feature Explorer</h1>
    <table border="1" cellpadding="10" cellspacing="0">
        <tr><th>Feature ID</th><th>Preview</th><th>Full Report</th></tr>
        {''.join(rows)}
    </table>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        f.write(html)

    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    feature_list = [17456, 1000, 42, 99]  # example
    generate_index(feature_list)
