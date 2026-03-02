import base64
import io
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_cytoscape as cyto

app = Dash(__name__)

# Color theme
theme = {
    'bg': '#0a0b0d',         
    'panel': '#14171a',      
    'border': '#2d333b',     
    'text': '#adbac7',       
    'accent': '#44d7b6',     
    'edge': '#3e444d'
}

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>NetGrep // Terminal</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&family=Inter:wght@300;400;600&display=swap');
            
            body {{ margin: 0; background-color: {theme['bg']}; color: {theme['text']}; font-family: 'Inter', sans-serif; overflow: hidden; }}
            
            .terminal-text {{ font-family: 'Fira Code', monospace; color: {theme['accent']}; text-transform: lowercase; }}

            /* Dropdown Styling */
            .Select-control {{ background-color: {theme['panel']} !important; border: 1px solid {theme['border']} !important; }}
            .Select-value-label, .Select-input {{ color: #ffffff !important; font-size: 13px; }}
            .Select-menu-outer {{ background-color: {theme['panel']} !important; border: 1px solid {theme['accent']} !important; }}
            .VirtualizedSelectOption {{ background-color: {theme['panel']} !important; color: #ffffff !important; }}
            .VirtualizedSelectFocusedOption {{ background-color: {theme['border']} !important; color: {theme['accent']} !important; }}
            
            @keyframes blink {{ 50% {{ opacity: 0; }} }}
            .cursor {{ animation: blink 1s step-start infinite; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}} {{%scripts%}} {{%renderer%}}</footer>
    </body>
</html>
'''

app.layout = html.Div(style={'display': 'flex', 'height': '100vh'}, children=[
    
    # LEFT: Sidebar Controls
    html.Div(style={'width': '320px', 'backgroundColor': theme['panel'], 'padding': '25px', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px', 'borderRight': f'1px solid {theme["border"]}'}, children=[
        
        html.Div([
            html.Span("> ", className="terminal-text", style={'fontSize': '18px'}),
            html.Span("netgrep", className="terminal-text", style={'fontSize': '22px', 'fontWeight': '500'}),
            html.Span("_", className="terminal-text cursor", style={'fontSize': '22px'})
        ], style={'marginBottom': '20px'}),

        dcc.Upload(
            id='upload-data',
            children=html.Div(['[ Upload GRN CSV ]'], style={'fontSize': '12px', 'fontFamily': 'Fira Code'}),
            style={'border': f'1px dashed {theme["border"]}', 'padding': '12px', 'textAlign': 'center', 'cursor': 'pointer', 'borderRadius': '4px'}
        ),

        html.Div(id='mapping-container', style={'display': 'none'}, children=[
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '18px'}, children=[
                
                html.H4("Parameters", className="terminal-text", style={'fontSize': '11px', 'margin': '0', 'opacity': '0.8'}),
                
                html.Div([
                    html.Label("SOURCE FIELD", style={'fontSize': '10px', 'display': 'block', 'marginBottom': '5px', 'color': '#888'}),
                    dcc.Dropdown(id='source-dropdown')
                ]),
                html.Div([
                    html.Label("TARGET FIELD", style={'fontSize': '10px', 'display': 'block', 'marginBottom': '5px', 'color': '#888'}),
                    dcc.Dropdown(id='target-dropdown')
                ]),
                html.Div([
                    html.Label("WEIGHT FIELD", style={'fontSize': '10px', 'display': 'block', 'marginBottom': '5px', 'color': '#888'}),
                    dcc.Dropdown(id='weight-dropdown')
                ]),

                html.Button("Build Network", id='build-btn', n_clicks=0, style={
                    'backgroundColor': 'transparent', 'border': f'1px solid {theme["accent"]}', 
                    'color': theme['accent'], 'padding': '12px', 'fontFamily': 'Fira Code', 'cursor': 'pointer', 'marginTop': '10px', 'borderRadius': '4px'
                }),

                html.Hr(style={'border': 'none', 'borderTop': f'1px solid {theme["border"]}', 'margin': '10px 0'}),

                html.H4("FILTER_ENGINE", className="terminal-text", style={'fontSize': '11px', 'margin': '0', 'opacity': '0.8'}),
                
                dcc.Graph(id='weight-histogram', config={'displayModeBar': False}, style={'height': '60px'}),
                
                dcc.RangeSlider(
                    id='weight-slider', 
                    min=0, max=1, step=0.01, value=[0, 1],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),

                html.Div([
                    html.Label("ALGORITHM", style={'fontSize': '10px', 'display': 'block', 'marginBottom': '5px', 'color': '#888'}),
                    dcc.Dropdown(
                        id='layout-dropdown', 
                        options=[{'label': 'Cose (Force)', 'value': 'cose'}, {'label': 'Circle (Radial)', 'value': 'circle'}], 
                        value='cose'
                    )
                ]),
            ])
        ])
    ]),

    # RIGHT: Main Network Section
    html.Div(style={'flex': '1', 'position': 'relative', 'backgroundColor': theme['bg']}, children=[
        cyto.Cytoscape(
            id='cytoscape-grn',
            layout={'name': 'cose', 'refresh': 20},
            style={'width': '100%', 'height': '100%'},
            elements=[],
            stylesheet=[
                {'selector': 'node', 'style': {
                    'label': 'data(label)', 'font-size': '10px', 'width': '12px', 'height': '12px', 
                    'background-color': theme['accent'], 'color': '#adbac7', 'text-margin-y': -8, 'font-family': 'Fira Code'
                }},
                {'selector': 'edge', 'style': {
                    'curve-style': 'haystack', 'line-color': theme['edge'], 'width': 'mapData(weight, 0, 1, 0.5, 4)', 'opacity': 0.4
                }}
            ]
        ),
        
        html.Div(id='node-edge-count', style={
            'position': 'absolute', 'bottom': '20px', 'right': '20px', 
            'fontSize': '11px', 'fontFamily': 'Fira Code', 'color': theme['accent'], 'opacity': '0.7'
        })
    ]),

    dcc.Store(id='stored-data')
])

# Callbacks / Functions

@app.callback(
    [Output('mapping-container', 'style'), Output('source-dropdown', 'options'), Output('target-dropdown', 'options'), Output('weight-dropdown', 'options'), Output('stored-data', 'data')],
    Input('upload-data', 'contents')
)
def handle_upload(contents):
    if not contents: return {'display': 'none'}, [], [], [], None
    _, content_string = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')), header=None)
    options = [{'label': f"col_{i} ({str(df.iloc[0, i])[:10]})", 'value': i} for i in range(len(df.columns))]
    return {'display': 'block'}, options, options, options, df.to_json(orient='records')

@app.callback(
    [Output('weight-histogram', 'figure'), Output('weight-slider', 'min'), Output('weight-slider', 'max'), Output('weight-slider', 'value')],
    Input('build-btn', 'n_clicks'), State('weight-dropdown', 'value'), State('stored-data', 'data'), prevent_initial_call=True
)
def update_histogram(n_clicks, weight_col, json_data):
    if not json_data or weight_col is None: return no_update, 0, 1, [0, 1]
    df = pd.read_json(io.StringIO(json_data), orient='records')
    weights = df[weight_col].dropna()
    fig = px.histogram(weights, nbins=25)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        showlegend=False, yaxis={'visible': False}, xaxis={'visible': False}
    )
    fig.update_traces(marker_color=theme['accent'], marker_line_width=0)
    return fig, float(weights.min()), float(weights.max()), [float(weights.min()), float(weights.max())]

@app.callback(
    [Output('cytoscape-grn', 'elements'), Output('cytoscape-grn', 'layout'), Output('node-edge-count', 'children')],
    [Input('weight-slider', 'value'), Input('layout-dropdown', 'value')],
    [State('source-dropdown', 'value'), State('target-dropdown', 'value'), State('weight-dropdown', 'value'), State('stored-data', 'data')],
    prevent_initial_call=True
)
def render_filtered_network(thresholds, layout_name, src_col, tgt_col, weight_col, json_data):
    if not json_data or src_col is None or tgt_col is None: return [], no_update, ""
    df = pd.read_json(io.StringIO(json_data), orient='records')
    if weight_col is not None:
        df = df[(df[weight_col] >= thresholds[0]) & (df[weight_col] <= thresholds[1])]
    
    src_series, tgt_series = df[src_col].astype(str), df[tgt_col].astype(str)
    nodes = [{'data': {'id': n, 'label': n}} for n in pd.concat([src_series, tgt_series]).unique()]
    edges = [{'data': {'source': s, 'target': t, 'weight': w}} 
             for s, t, w in zip(src_series, tgt_series, df[weight_col] if weight_col is not None else [1]*len(df))]
    
    return nodes + edges, {'name': layout_name, 'animate': True}, f"Network Size: {len(nodes)} NODES // {len(edges)} EDGES"

if __name__ == '__main__':
    app.run(debug=False)
