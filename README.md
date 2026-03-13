# NetGrep
Collection of functions for large-scale downstream gene regulatory network analysis. Functions are currently adapted for use with pySCENIC.

Checklist:
- [x] **GoNet**: Visualize significant gene ontology as a network.
- [x] **SubNetGrep**: Examine a sub-network surrounding any gene in the network.
- [x] **InflectionFilter**: Filter regulon object based on target gene weight distribution.
- [x] **FindRegulon**: Subset network to a regulon of interest (Currently only highlights in app)
- [ ] **FindTF**: Query a gene by symbol and list what TFs target it in the network.
- [ ] GUI Adaptation for interactive network analysis. (Dash/Cytoscape)

## Usage

- For using pre-existing functions in python to generate .gexf files to visualize in Gephi. [See Functions](https://github.com/Tripfantasy/NetGrep/tree/main/functions)
- For using the standalone app with built in network visualizer.
    - Clone the github repository
    - Use the environment .yml to to create a conda environment/equivalent with the required packages. Activate environment.
    - launch the standalone_app.py via the following terminal command:
      
      ```
      python standalone_app.py
      ```
    - Load testing .csv and/or .pkl. Or supply your own. (More documentation to come)

<img width="750" alt="Screenshot 2026-03-04 at 2 48 35 PM" src="https://github.com/user-attachments/assets/36835da9-e461-4391-902c-1f5649b920c8" />
<p align="center">Visualizer App is in Early Development</p>
