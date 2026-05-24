from config import DATA_FILE, TOP_M, TOP_K, N_HOP
from graph_loader import load_graph, build_indices
from retriever import (
    retrieve_top_m,
    extract_query_entities,
    build_subgraph,
    n_hop_traversal,
    select_top_k_chunks,
)
from reranker import rerank

def main():
    print("=== Proposition-Entity Graph Retriever Demo ===")

    data = load_graph(DATA_FILE)
    propositions, entity_to_props, prop_to_entities = build_indices(data)

    # 中間成果 1：資料是否成功載入
    print(f"\n[Debug] Total propositions loaded: {len(propositions)}")
    print(f"[Debug] Total unique entities loaded: {len(entity_to_props)}")

    query = input("\nPlease enter your query: ").strip()

    # Step 1: retrieve top-M propositions
    top_props = retrieve_top_m(query, propositions, TOP_M)

    # 中間成果 2：top-M propositions
    print("\n=== [Debug] Top-M Retrieved Propositions ===")
    for idx, prop in enumerate(top_props, start=1):
        print(f"[{idx}] {prop['prop_id']} | {prop['proposition']}")

    # 抓 query entities
    all_entities = list(entity_to_props.keys())
    query_entities = extract_query_entities(query, all_entities)

    # 中間成果 3：query entities
    print("\n=== [Debug] Query Entities ===")
    if query_entities:
        print(query_entities)
    else:
        print("No query entities matched.")

    # Step 2: build subgraph
    subgraph = build_subgraph(top_props, prop_to_entities, entity_to_props)

    # 中間成果 4：subgraph 基本資訊
    print("\n=== [Debug] Subgraph Summary ===")
    print(f"Subgraph propositions: {len(subgraph['props'])}")
    print(f"Subgraph entities: {len(subgraph['entities'])}")

    # Step 3: N-hop traversal
    if query_entities:
        candidate_prop_ids = n_hop_traversal(query_entities, subgraph, prop_to_entities, N_HOP)

        print("\n=== [Debug] Candidate Proposition IDs after N-hop Traversal ===")
        print(candidate_prop_ids)

        candidates = select_top_k_chunks(candidate_prop_ids, subgraph, query, TOP_K)
    else:
        candidate_prop_ids = []
        candidates = top_props[:TOP_K]

    # 中間成果 5：reranking 前的 candidates
    print("\n=== [Debug] Candidates Before Reranking ===")
    for idx, item in enumerate(candidates, start=1):
        print(f"[{idx}] {item['prop_id']} | {item['proposition']}")

    # Step 4: rerank
    final_results = rerank(query, candidates, TOP_K)

    print("\n=== Final Results ===")
    for idx, item in enumerate(final_results, start=1):
        print(f"\n[{idx}] Prop ID: {item['prop_id']}")
        print(f"Post ID: {item['source_post_id']}")
        print("Triplets:")
        for t in item["triplets"]:
            print(f"  - ({t['head']}, {t['relation']}, {t['tail']})")
        print(f"Proposition: {item['proposition']}")


if __name__ == "__main__":
    main()