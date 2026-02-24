def SubNetGrep(regulon_object,adjacency_file, gene:str):
    '''
    Parse GRN to find second-order subnetwork, includes TFs/regulons directly targetting gene, and important adjacencies.
    By default will filter adjacencies based on shared transcription factors. If no direct TF relationship is found, skips pruning.
    '''
    # inititalize dictionary for storing all results
    target_info = {}
    # regulons which target query gene are captured, and their weights attached.
    target_info['TF'] = {}
    # Target info for direct targets of query gene are stored. 
    target_info['reg'] = {}
    for regulon in regulon_object:
        if gene in regulon.gene2weight.keys():
            target_info['TF'][regulon.name] = regulon.gene2weight[gene]
            target_info['reg'][regulon.name] = {}
            for target_gene , weight in regulon.gene2weight.items():
                target_info['reg'][regulon.name][target_gene] = weight
    regulon_targets_flattened = [(label, value, sub_id) for sub_id, sub_dict in target_info['reg'].items() for label, value in sub_dict.items()]
    print(f"{gene} is a direct target of {list(target_info['TF'].keys())} regulon(s).")
    # Query adjacency file, prune based on connectivity to regulons directly targetting query. Otherwise, we can only prune based on adjacency weights. 
    target_info['adj'] = []
    with open(adjacency_file) as fh:
        for line in fh:
            line = str(line).strip().split(',')
            if line[0]== gene:
                # If we have identified direct targets, use shared regulons for pruning. (len() == 0 if there are no direct regulon connections)
                if len(target_info['reg']) > 0:
                    if any(line[1] == entry[0] for entry in regulon_targets_flattened):
                        target_info['adj'].append((line[1],line[2]))
                else:
                    target_info['adj'].append((line[1],line[2]))
    # Build the network edge table
    build_subnetwork(target_info, query_gene=gene, filename=f"/~/{gene}_subnetwork.gexf")

def build_subnetwork(target_info, query_gene, filename="subnetwork.gexf"):
    '''
    Builds a network from NetGrep result dictionary. Last step of NetGrep function. 
    '''
    G = nx.DiGraph() 

    # TF:Query gene edges
    for tf, weight in target_info['TF'].items():
        G.add_edge(tf, query_gene, weight=float(weight), type='direct_target')

    # TF:Other targets
    for tf, targets in target_info['reg'].items():
        for target_gene, weight in targets.items():
            if target_gene != query_gene: # Avoid duplicating the direct link
                G.add_edge(tf, target_gene, weight=float(weight), type='regulon_member')

    # Query gene: adjacency edges
    for adj_gene, weight in target_info['adj']:
        G.add_edge(query_gene, adj_gene, weight=float(weight), type='adjacency_validated')

    nx.write_gexf(G, filename)
    print(f"Network exported to {filename}")
