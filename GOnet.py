def GONet(deg_results_dict, species='mmusculus', out_dir='',lfc_up=0.5,lfc_down=-0.05,remove_sex_genes=True):
  '''
  With a DEG analysis result dictionary where keys are group labels, and values are pandas dataframes, associate GO terms with differentially expressed genes. 
  Optionally remove sex-linked genes from the DEG pool, and fine-tune bidirectional log2foldchange. Current default significance threshold is adjusted pvalue of 0.05
  Significance is treated as a node-attribute, and output is a .gexf-converted network edge table of connected terms to be visualized in Gephi.
  '''
    test_dict = {}
    # Map species names to GSEAPY organism names
    org_map = {'mmusculus': 'Mouse'}
    for group, result in deg_results_dict.items():
        test_dict[group] = {} 
        # Genes are stored as index values per DEG results dataframe
        gene_list = list(result.index)
        
        if remove_sex_genes == True:
            annot = sc.queries.biomart_annotations(
                species,
                ["ensembl_gene_id", "external_gene_name", "start_position", "end_position", "chromosome_name"]
            ).set_index("external_gene_name")
            
            y_genes = set(gene_list).intersection(annot.index[annot.chromosome_name == "Y"])
            x_genes = set(gene_list).intersection(annot.index[annot.chromosome_name == "X"])
            sex_genes = y_genes.union(x_genes)
            clean_genes = [x for x in gene_list if x not in sex_genes]
            print(f"Processing {group}: {len(clean_genes)} genes (after sex-gene removal)")
        else:
            clean_genes = gene_list
            
        result = result.loc[clean_genes] 
        glist_up = list(result[result['avg_log2FC'] > lfc_up].index)
        glist_down = list(result[result['avg_log2FC'] < lfc_down].index)

        # Make network table for each direction at each group. 
        for direction in ['up', 'down']:
            if direction == 'up':
                glist = glist_up
            else:
                glist = glist_down 
            
            # GO enrichment step
            if len(glist) < 30:
                print(f"Skipping {group} {direction}: list too small ({len(glist)} genes)")
                pass
            else:
                #print(glist)
                print(f'Making GOnetwork table for {len(glist)} genes.')
                enr_res = gseapy.enrichr(
                    gene_list = list(glist), # Ensure we pass the gene names/index
                    organism = org_map[species],
                    gene_sets = 'GO_Biological_Process_2023',
                    cutoff = 0.05
                )
                test_dict[group][direction] = enr_res.res2d                
                # Pval added as a node attribute for gephi aesthetics downstream. (i.e., node size)
                nodes, edges = gseapy.enrichment_map(enr_res.res2d, top_term=300, column='Adjusted P-value')
                
                network_table = edges[['src_name', 'targ_name', 'jaccard_coef', 'overlap_coef', 'overlap_genes']]
                network_table.columns = ['source', 'target', 'jaccard', 'overlap', 'intersection']
                
                G = nx.from_pandas_edgelist(
                    network_table, 
                    source='source', target='target', 
                    edge_attr=['jaccard', 'overlap', 'intersection']
                )
                node_attributes = {node: pval for node, pval in zip(nodes['Term'], nodes['Adjusted P-value'])}
                nx.set_node_attributes(G, node_attributes, "pval")
                print(f'Saving network table to {out_dir}/GOnet_{group}_{direction}.gexf')
                nx.write_gexf(G, f"{out_dir}/GOnet_{group}_{direction}.gexf")

    return test_dict
