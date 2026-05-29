import json


def load_graph(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_indices(data: dict) -> tuple[list[dict], dict[str, set], dict[str, set]]:
    """
    將 B 的 triplets 格式整理成 proposition-level graph
    去重規則：
    同 source_post_id + 同 proposition text 視為同一個 proposition
    """
    triplets = data.get("triplets", [])

    proposition_map = {}
    entity_to_props = {}
    prop_to_entities = {}

    prop_counter = 1

    for item in triplets:
        source_post_id = item["source_post_id"]
        proposition_text = item["proposition"]
        key = (source_post_id, proposition_text)

        head = item["head"]
        relation = item["relation"]
        tail = item["tail"]

        if key not in proposition_map:
            prop_id = f"P{prop_counter}"
            prop_counter += 1

            proposition_map[key] = {
                "prop_id": prop_id,
                "source_post_id": source_post_id,
                "proposition": proposition_text,
                "triplets": [],
            }

        proposition_map[key]["triplets"].append({
            "head": head,
            "relation": relation,
            "tail": tail,
        })

    propositions = []

    for record in proposition_map.values():
        prop_id = record["prop_id"]

        linked_entities = set()
        for t in record["triplets"]:
            linked_entities.add(t["head"])
            linked_entities.add(t["tail"])

        prop_to_entities[prop_id] = linked_entities

        for entity in linked_entities:
            if entity not in entity_to_props:
                entity_to_props[entity] = set()
            entity_to_props[entity].add(prop_id)

        propositions.append(record)

    return propositions, entity_to_props, prop_to_entities