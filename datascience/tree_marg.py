import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, name, probability):
        self.name = name
        self.probability = probability
        self.children = []
        self.parent = None
        self.message_to_parent = None

def create_tree():
    # Create nodes
    root = Node("A", np.array([0.6, 0.4]))  # P(A)
    b = Node("B", np.array([[0.7, 0.3], [0.2, 0.8]]))  # P(B|A)
    c = Node("C", np.array([[0.5, 0.5], [0.1, 0.9]]))  # P(C|A)
    d = Node("D", np.array([[0.8, 0.2], [0.3, 0.7]]))  # P(D|B)

    # Connect nodes
    root.children = [b, c]
    b.parent = root
    c.parent = root
    b.children = [d]
    d.parent = b

    return root

def marginalize(node):
    if not node.children:
        node.message_to_parent = node.probability
        return node.probability

    child_messages = [marginalize(child) for child in node.children]
    if node.parent:
        node.message_to_parent = node.probability @ np.prod(child_messages, axis=0)
        return node.message_to_parent
    else:
        return node.probability * np.prod(child_messages, axis=0)

def visualize_tree(root):
    G = nx.Graph()
    node_labels = {}
    edge_labels = {}

    def add_nodes_edges(node):
        G.add_node(node.name)
        node_labels[node.name] = f"{node.name}\n{node.message_to_parent}"
        for child in node.children:
            G.add_edge(node.name, child.name)
            edge_labels[(node.name, child.name)] = f"P({child.name}|{node.name})"
            add_nodes_edges(child)

    add_nodes_edges(root)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold')
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Tree Marginalization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main execution
root = create_tree()
result = marginalize(root)
print(f"Marginalized probability: {result}")
visualize_tree(root)
