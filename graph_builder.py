# =====================================================
# STEP 1 : IMPORT LIBRARIES
# =====================================================

import pandas as pd
import networkx as nx


# =====================================================
# STEP 2 : LOAD PAYSIM DATASET
# =====================================================

df = pd.read_csv("data/paysim.csv")

print("Dataset Loaded Successfully")
print()

print("Columns:")
print(df.columns)
print()

print("First 5 Rows:")
print(df.head())
print()


# =====================================================
# STEP 3 : CREATE DIRECTED TRANSACTION GRAPH
# =====================================================

# Directed graph because:
#
# Account A ---> Account B
#
# Direction matters in transactions

G = nx.DiGraph()


# =====================================================
# STEP 4 : ADD TRANSACTIONS AS EDGES
# =====================================================

# Every row in PaySim becomes:
#
# Sender -----> Receiver
#
# Example:
#
# C123 -----> C456
#
# Edge stores:
# amount
# fraud label

for row in df.itertuples(index=False):

    G.add_edge(
        row.nameOrig,
        row.nameDest,
        amount=row.amount,
        fraud=row.isFraud
    )


print("Graph Created Successfully")
print()

print("Number of Nodes:", G.number_of_nodes())
print("Number of Edges:", G.number_of_edges())
print()


# =====================================================
# STEP 5 : SHOW SAMPLE EDGES
# =====================================================

print("Sample Edges")
print()

for edge in list(G.edges(data=True))[:5]:
    print(edge)

print()


# =====================================================
# STEP 6 : GENERATE GRAPH FEATURES
# =====================================================

# Degree:
# Total connections

degree = dict(G.degree())

# Incoming transactions

in_degree = dict(G.in_degree())

# Outgoing transactions

out_degree = dict(G.out_degree())


# =====================================================
# STEP 7 : PAGERANK
# =====================================================

# Measures importance of account
# Useful for fraud detection

print("Calculating PageRank...")

pagerank = nx.pagerank(G)

print("PageRank Completed")
print()


# =====================================================
# STEP 8 : CREATE NODE FEATURE TABLE
# =====================================================

features = []

for node in G.nodes():

    features.append({

        "account": node,

        "degree":
        degree[node],

        "in_degree":
        in_degree[node],

        "out_degree":
        out_degree[node],

        "pagerank":
        pagerank[node]

    })


features_df = pd.DataFrame(features)


# =====================================================
# STEP 9 : DISPLAY FEATURES
# =====================================================

print("Node Features")
print()

print(features_df.head())

print()


# =====================================================
# STEP 10 : SAVE FEATURES
# =====================================================

features_df.to_csv(
    "node_features.csv",
    index=False
)

print("node_features.csv saved successfully")


# =====================================================
# STEP 11 : SAVE GRAPH
# =====================================================

nx.write_gml( G, "transaction_graph.gml" )

print("transaction_graph.gml saved successfully")