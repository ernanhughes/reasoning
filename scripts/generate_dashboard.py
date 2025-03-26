import json
import os

def generate_dashboard(meta_path="features_meta.json", output_path="reports/dashboard.html"):
    with open(meta_path, "r") as f:
        features = json.load(f)

    categories = sorted(set(f["category"] for f in features))

    sidebar = "<ul>"
    for cat in categories:
        sidebar += f'<li><a href="#cat-{cat}">{cat.capitalize()}</a></li>'
    sidebar += "</ul>"

    sections = []
    for cat in categories:
        sections.append(f'<h2 id="cat-{cat}">{cat.capitalize()}</h2>')
        for f in [x for x in features if x["category"] == cat]:
            iframe = f'<iframe src="features/feature_{f["id"]}.html" width="100%" height="200" style="border:1px solid #ccc;"></iframe>'
            tokens = ", ".join(f['tokens'])
            sections.append(f"""
                <div style="margin-bottom:30px;">
                    <h3>Feature {f['id']}: {f['label']}</h3>
                    <p><strong>Top Tokens:</strong> {tokens}</p>
                    {iframe}
                    <p><a href="features/feature_{f["id"]}.html" target="_blank">Full Report</a></p>
                </div>
            """)

    html = f"""
    <html>
    <head>
        <title>Reasoning Feature Dashboard</title>
        <style>
            body {{ font-family: sans-serif; display: flex; }}
            #sidebar {{ width: 200px; padding: 20px; background: #f5f5f5; }}
            #main {{ flex: 1; padding: 20px; }}
            iframe {{ background: white; }}
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Categories</h2>
            {sidebar}
        </div>
        <div id="main">
            <h1>Reasoning Feature Dashboard</h1>
            {''.join(sections)}
        </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard written to {output_path}")

if __name__ == "__main__":
    generate_dashboard()
