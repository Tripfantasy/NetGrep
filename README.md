# NetGrep
Collection of functions for large-scale downstream gene regulatory network analysis. Functions are currently adapted for use with pySCENIC. 

## App Changelog
**03.19.2026:** Standalone NetGrep app is now available for testing! Features include:
- Weight-based edge filtering
- Custom layouts for GRNs (Spiral and GRN-Clusters) 
- Aesthetic customization of network graphs
- Regulon query and subnetwork generation.
      - Regulon component export as .csv
- Gene of interest query
      - Subnetwork of adjacencies, direct/indirect TF regulators. 



## Usage

- For using pre-existing functions in python to generate .gexf files to visualize in Gephi. [See Functions](https://github.com/Tripfantasy/NetGrep/tree/main/functions)
- For using the standalone app with built in network visualizer.
    - Clone the github repository
    - Use the netgrep .yml to to create a conda environment/equivalent with the required packages. Activate environment.
    - launch the standalone_app.py via the following terminal command:
      
      ```
      python standalone_app.py
      ```
    - Load testing .csv and/or regulon_object.pkl. Or supply your own. (More documentation to come)

## Previews 
**Note: App is in early development and subject to change**
<p align ="center">
    <img width="460" height="300" alt="Screenshot 2026-03-19 at 4 50 10 PM" src="https://github.com/user-attachments/assets/891e7a89-c8de-4cda-8fad-75c9e877864a" />
</p>
<p align="center">Explore and adjust whole network diagrams.</p>
<p align ="center">
<img width="460" height="300" alt="Screenshot 2026-03-19 at 4 52 47 PM" src="https://github.com/user-attachments/assets/59a629c5-6d87-4e9f-a5d8-d2ad53c39734" />
</p>
<p align="center">Visualize subnetworks of regulons of interest.</p>
<p align ="center">
<img width="460" height="300" alt="Screenshot 2026-03-19 at 4 53 10 PM" src="https://github.com/user-attachments/assets/60c281f6-2ba0-443c-bdcf-e9d19c2130f0" />
</p>
<p align="center">Query genes of interest and visualize meaningful regulatory connections.</p>





