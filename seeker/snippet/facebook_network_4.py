#date: 2022-10-19T17:23:23Z
#url: https://api.github.com/gists/cbd00af2e562d1b1cc97d968a4e0a487
#owner: https://api.github.com/users/higor-gomes93

def draw_graph_highlight(a, b, dataset):
    plt.figure(figsize = (25, 15))
    
    dataset['source_name'] = dataset['source_name'].apply(lambda x: int(x))
    dataset['target_name'] = dataset['target_name'].apply(lambda x: int(x))
    
    graph_highlight = nx.from_pandas_edgelist(dataset, "source_name", "target_name")
    
    pos_highlight = nx.spring_layout(graph_highlight, iterations = 15, seed = 1721)
    
    node_size_neighbor_highlight = []
    for i in nodes:
        node_size_neighbor_highlight.append(len(list(nx.all_neighbors(graph_highlight, i))))
    
    edge_weight_highlight = []
    for i, j in zip(dataset['source_name'], dataset['target_name']):
        edge_weight_highlight.append(graph_highlight.number_of_edges(i, j))
        
    nx.draw_networkx(graph_highlight, pos = pos_highlight, node_size = node_size_neighbor, node_color = "grey", with_labels = False, width = edge_weight_highlight, edge_color = '#272727')
    nx.draw_networkx_nodes(graph_highlight, pos = pos_highlight, node_size = node_size_neighbor_highlight[a], node_color = "red", nodelist = [a])
    nx.draw_networkx_nodes(graph_highlight, pos = pos_highlight, node_size = node_size_neighbor_highlight[b], node_color = "red", nodelist = [b])
    
    if graph_highlight.has_edge(a, b):
        nx.draw_networkx_edges(graph_highlight, pos = pos_highlight, edgelist = [(a, b)], edge_color = "green", width = 3)
        
    limits = plt.axis("off")