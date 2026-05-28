import json
from pathlib import Path
import networkx as nx
from pyvis.network import Network


def load_triplets(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_nx_graph(data: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    triplets = data.get("triplets", [])

    for item in triplets:
        head = item["head"]
        relation = item["relation"]
        tail = item["tail"]
        proposition = item.get("proposition", "")
        source_post_id = item.get("source_post_id", "")

        G.add_node(head, label=head, title=f"Entity: {head}")
        G.add_node(tail, label=tail, title=f"Entity: {tail}")

        edge_title = (
            f"relation: {relation}<br>"
            f"post_id: {source_post_id}<br>"
            f"proposition: {proposition}"
        )
        G.add_edge(head, tail, label=relation, title=edge_title)

    return G


def export_pyvis_html(G: nx.DiGraph, output_path: str):
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line"
    )

    net.from_nx(G)

    # 美化：開啟物理效果，方便拖曳展示
    net.force_atlas_2based()

    # 先產生 HTML 字串，再手動用 UTF-8 寫檔
    html = net.generate_html(notebook=False)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "graph_data.json"
    output_file = base_dir / "kg_visualization.html"

    print(f"[Debug] Loading graph from: {input_file}")
    data = load_triplets(str(input_file))
    G = build_nx_graph(data)

    print(f"[Debug] Nodes: {len(G.nodes)}")
    print(f"[Debug] Edges: {len(G.edges)}")

    export_pyvis_html(G, str(output_file))
    print(f"[Done] Visualization exported to: {output_file}")


if __name__ == "__main__":
    main()