"""
04_network_construction.py — Multi-Echelon Supply Chain Network Builder.

Constructs a spanning-tree supply chain network from flat retail data.

FIX: Added a virtual root node ("SUPPLY") above all DCs to ensure
the graph is a proper tree (single root). Multiple disconnected roots
made nx.is_tree() return False, violating the GSM assumption.

Network topology:

  [SUPPLY] -> [DC_R1] -> [WH_R1_A] -> [Store-Dept nodes]
           -> [DC_R2] -> [WH_R2_B] -> [Store-Dept nodes]
           -> [DC_R3] -> [WH_R3_C] -> [Store-Dept nodes]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field

import config


@dataclass
class Node:
    """Represents a node in the multi-echelon supply chain network."""
    node_id: str
    echelon: int
    node_type: str
    lead_time: float
    holding_cost: float
    max_service_time: int
    mu_demand: float = 0.0
    sigma_demand: float = 0.0
    parent_id: str = ""
    children: list = field(default_factory=list)


@dataclass
class SupplyChainNetwork:
    """Multi-echelon supply chain network container."""
    nodes: dict
    graph: nx.DiGraph
    leaf_nodes: list
    root_nodes: list    # Now contains just ["SUPPLY"]
    n_echelons: int = 4  # SUPPLY -> DC -> WH -> Store-Dept


def build_network(residual_stats, df, sigma_type="sigma_residual"):
    """
    Build the multi-echelon supply chain network.

    FIXES applied:
      1. Added virtual SUPPLY root node above all DCs -> proper tree
      2. Ensured all warehouses have children (skip empty ones)
      3. Added proper node counting and validation
    """
    print("\n" + "=" * 70)
    print("STAGE 4: MULTI-ECHELON NETWORK CONSTRUCTION")
    print("=" * 70)

    nodes = {}
    G = nx.DiGraph()
    leaf_ids = []

    store_info = df[["Store", "Type", "Region"]].drop_duplicates()

    # --- Virtual root: External Supply ---
    supply_node = Node(
        node_id="SUPPLY", echelon=0, node_type="supply",
        lead_time=0, holding_cost=0, max_service_time=0,
    )
    nodes["SUPPLY"] = supply_node
    G.add_node("SUPPLY", echelon=0, type="supply")

    # --- Echelon 1: Distribution Centers ---
    regions = sorted(df["Region"].unique())
    dc_ids = []
    for region in regions:
        dc_id = f"DC_{region}"
        node = Node(
            node_id=dc_id, echelon=1, node_type="dc",
            lead_time=config.LEAD_TIMES["dc"],
            holding_cost=config.HOLDING_COST_RATES["dc"],
            max_service_time=config.MAX_SERVICE_TIMES["dc"],
            parent_id="SUPPLY",
        )
        nodes[dc_id] = node
        supply_node.children.append(dc_id)
        dc_ids.append(dc_id)
        G.add_node(dc_id, echelon=1, type="dc")
        G.add_edge("SUPPLY", dc_id)

    print(f"  Echelon 1 (DCs): {len(dc_ids)} nodes")

    # --- Echelon 2: Regional Warehouses ---
    wh_count = 0
    region_type_pairs = store_info[["Region", "Type"]].drop_duplicates()
    for _, row in region_type_pairs.iterrows():
        region, stype = row["Region"], row["Type"]
        wh_id = f"WH_{region}_{stype}"
        dc_id = f"DC_{region}"

        node = Node(
            node_id=wh_id, echelon=2, node_type="warehouse",
            lead_time=config.LEAD_TIMES["warehouse"],
            holding_cost=config.HOLDING_COST_RATES["warehouse"],
            max_service_time=config.MAX_SERVICE_TIMES["warehouse"],
            parent_id=dc_id,
        )
        nodes[wh_id] = node
        nodes[dc_id].children.append(wh_id)
        G.add_node(wh_id, echelon=2, type="warehouse")
        G.add_edge(dc_id, wh_id)
        wh_count += 1

    print(f"  Echelon 2 (Warehouses): {wh_count} nodes")

    # --- Echelon 3: Store-Department demand nodes ---
    sd_count = 0
    for _, row in residual_stats.iterrows():
        store = int(row["Store"])
        dept = int(row["Dept"])
        sd_id = f"S{store}_D{dept}"

        sinfo = store_info[store_info["Store"] == store]
        if sinfo.empty:
            continue
        region = sinfo["Region"].iloc[0]
        stype = sinfo["Type"].iloc[0]
        wh_id = f"WH_{region}_{stype}"

        mu = row["mu_demand"]
        sigma = row[sigma_type] if sigma_type in row.index else row["sigma_historical"]

        node = Node(
            node_id=sd_id, echelon=3, node_type="store_dept",
            lead_time=config.LEAD_TIMES["store"],
            holding_cost=config.HOLDING_COST_RATES["store"],
            max_service_time=config.MAX_SERVICE_TIMES["store"],
            mu_demand=mu, sigma_demand=sigma, parent_id=wh_id,
        )
        nodes[sd_id] = node
        nodes[wh_id].children.append(sd_id)
        leaf_ids.append(sd_id)
        G.add_node(sd_id, echelon=3, type="store_dept")
        G.add_edge(wh_id, sd_id)
        sd_count += 1

    print(f"  Echelon 3 (Store-Depts): {sd_count} nodes")

    # Remove warehouses with no children (empty nodes break the DP)
    empty_whs = [nid for nid, n in nodes.items()
                 if n.node_type == "warehouse" and len(n.children) == 0]
    for wh_id in empty_whs:
        parent_id = nodes[wh_id].parent_id
        nodes[parent_id].children.remove(wh_id)
        del nodes[wh_id]
        G.remove_node(wh_id)
        print(f"  Removed empty warehouse: {wh_id}")

    # Propagate demand bottom-up
    _propagate_demand(nodes)

    total_nodes = len(nodes)
    total_edges = G.number_of_edges()
    is_tree = nx.is_tree(G)
    print(f"\n  Network Summary:")
    print(f"    Total nodes: {total_nodes}")
    print(f"    Total edges: {total_edges}")
    print(f"    Is tree: {is_tree}")
    print(f"    Leaf nodes: {len(leaf_ids)}")

    if not is_tree:
        print("    WARNING: Network is not a valid tree! Check for cycles or "
              "disconnected components.")

    network = SupplyChainNetwork(
        nodes=nodes, graph=G,
        leaf_nodes=leaf_ids,
        root_nodes=["SUPPLY"],
    )
    return network


def _propagate_demand(nodes):
    """Bottom-up demand aggregation with risk pooling."""
    # Warehouses
    for nid, node in nodes.items():
        if node.node_type == "warehouse" and node.children:
            node.mu_demand = sum(nodes[c].mu_demand for c in node.children)
            node.sigma_demand = np.sqrt(
                sum(nodes[c].sigma_demand ** 2 for c in node.children))

    # DCs
    for nid, node in nodes.items():
        if node.node_type == "dc" and node.children:
            node.mu_demand = sum(nodes[c].mu_demand for c in node.children)
            node.sigma_demand = np.sqrt(
                sum(nodes[c].sigma_demand ** 2 for c in node.children))

    # Supply root
    if "SUPPLY" in nodes and nodes["SUPPLY"].children:
        s = nodes["SUPPLY"]
        s.mu_demand = sum(nodes[c].mu_demand for c in s.children)
        s.sigma_demand = np.sqrt(
            sum(nodes[c].sigma_demand ** 2 for c in s.children))


def network_to_dataframe(network):
    """Convert network nodes to a summary DataFrame."""
    records = []
    for nid, node in network.nodes.items():
        if node.node_type == "supply":
            continue  # Skip virtual root in reports
        records.append({
            "node_id": nid, "echelon": node.echelon, "type": node.node_type,
            "lead_time": node.lead_time, "holding_cost": node.holding_cost,
            "max_service_time": node.max_service_time,
            "mu_demand": node.mu_demand, "sigma_demand": node.sigma_demand,
            "cv": (node.sigma_demand / node.mu_demand if node.mu_demand > 0 else 0),
            "n_children": len(node.children), "parent": node.parent_id,
        })
    return pd.DataFrame(records)


def run_network_construction(residual_stats, df, sigma_type="sigma_residual"):
    """Build and save the network."""
    network = build_network(residual_stats, df, sigma_type)
    net_df = network_to_dataframe(network)
    net_df.to_csv(config.RESULTS_DIR / f"network_summary_{sigma_type}.csv", index=False)

    print("\n  Echelon Statistics:")
    for echelon in [1, 2, 3]:
        echelon_nodes = [n for n in network.nodes.values() if n.echelon == echelon]
        if echelon_nodes:
            mus = [n.mu_demand for n in echelon_nodes]
            sigmas = [n.sigma_demand for n in echelon_nodes]
            cvs = [s / m if m > 0 else 0 for m, s in zip(mus, sigmas)]
            print(f"    Echelon {echelon}: {len(echelon_nodes):>4d} nodes | "
                  f"Avg mu = {np.mean(mus):>12,.1f} | "
                  f"Avg sigma = {np.mean(sigmas):>10,.1f} | "
                  f"Avg CV = {np.mean(cvs):.3f}")

    return network