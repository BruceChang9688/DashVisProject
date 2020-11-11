import json
import dash
import dash_core_components as dcc
import dash_html_components as html
# import dash_daq as daq
from dash.dependencies import Input, Output,State

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Number of Top Group
NUMTOPGROUP = 6
#Number of Top Casuality
NUMCASUALITIES = 30

#dash tutorial original code
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Data Pre-Process
# Read data from original path
df=pd.read_csv(r"/Users/bruce/Documents/2020Fall/ECS289H/HW2/globalterrorismdb_0718dist.csv",
    encoding='ISO-8859-1',low_memory=False)
df = df[[
    'iyear',
    'country_txt',
    'gname',
    'longitude',
    'latitude',
    'nkill',
    'nwound',
    'region_txt',
    'imonth',
    'iday',
    'success', 'weaptype1_txt', 
            'nhostkid', 'nreleased', 'ransomamt', 'ransompaid', 
            'nperps', 'attacktype1_txt', 'summary']]

# Choose data after 2000
df = df.loc[df['iyear']>=2000]

# Remove data with missing value
df = df.loc[df['gname'] != "Unknown"]
df = df.loc[~df['nkill'].isna()]
df = df.loc[~df['nwound'].isna()]
df = df.loc[~df['nhostkid'].isna()]
df = df.loc[~df['nreleased'].isna()]
df['nkillwound'] = df['nkill']+df['nwound']
df['marker_size'] = 10*df['nkillwound'].pow(1./2)
df = df.reset_index()


#Theme Setting
theme_setting = {
    "bg_color": "rgba(255, 255, 255, 255)",
    "legend_bg_color": "rgba(255, 255, 255, 0.8)",
    "text_color": "black",
    "bar": {
        "colors": [
            '#636EFA',
            '#EF553B',
            '#00CC96',
            '#AB63FA',
            '#FFA15A',
            '#95def4'
        ]
    },
    "land_color":"rgb(255,206,116)",
    "ocean_color":"rgb(0,152,191)"
}

dark_theme_setting = {
    "bg_color": "rgba(64, 64, 64, 255)",
    "legend_bg_color": "rgba(64, 64, 64, 0)",
    "text_color": "white",
    "bar": {
        "colors": [
            '#636EFA',
            '#EF553B',
            '#00CC96',
            '#AB63FA',
            '#FFA15A',
            '#95def4'
        ]
    },
    "land_color":"rgb(255,206,116)",
    "ocean_color":"rgb(0,152,191)"
}

#Trending Line
def get_trending_line(df, y_label, theme_setting, **kwargs):
    fig = px.line(
        df, x='iyear', y='total', 
        color=df['gname'], 
        labels={
            'gname': 'Group',
            'iyear': 'Year',
            'total': y_label
        },
        hover_data={
            'iyear': False
        },
        height=300
    )
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(
        title="", 
        range=[df['iyear'].min(), df['iyear'].max()],
        showticklabels=True, 
        showgrid=False,
        visible=True, 
        matches=None
    )
    fig.update_yaxes(
        showgrid=False,
    )
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        plot_bgcolor=theme_setting["bg_color"],
        paper_bgcolor=theme_setting["bg_color"],
        font={"color": theme_setting["text_color"]},
        showlegend=False,  
        hovermode="x",
        hoverlabel={'font_size': 12},
        legend={
            'title': "",
            'bgcolor': theme_setting['legend_bg_color'],
            'yanchor': "top",
            'y': 0.99,
            'xanchor': "left",
            'x': 0.01
        }
    )

    if "selection_record" in kwargs.keys():
        group_list = kwargs["top_group_list"]
        hidden_group = []
        for idx, selection in enumerate(kwargs["selection_record"]):
            if selection == "legendonly":
                hidden_group.append(group_list[idx])
                
        fig.for_each_trace(
            lambda trace: trace.update(visible="legendonly") if trace.name in hidden_group else ()
        )

    return fig

#Horizontal Bar
def get_pie_graph(df, theme_setting, **kwargs):
    fig = px.pie(
    	df,
    	values = 'total',
    	names = 'gname',
    	color=df['gname'], 
    	color_discrete_sequence=theme_setting['bar']['colors'],      
    	height = 300,
    	)
    fig.update_traces(hovertemplate='%{label}: <br>Number Of Kill&Wound: %{value}')
    fig.update_layout(
        margin={'l': 10, 'r': 0, 't': 0, 'b': 52},
        plot_bgcolor=theme_setting["bg_color"],
        paper_bgcolor=theme_setting["bg_color"],
        font={
            "size": 12,
            "color": theme_setting["text_color"]
        },
        showlegend=False
    )
    
    if "selection_record" in kwargs.keys():
        group_list = df['gname']
        hidden_group = []
        for idx, selection in enumerate(kwargs["selection_record"]):
            if selection == "legendonly":
                hidden_group.append(group_list[idx])
                
        fig.for_each_trace(
            lambda trace: trace.update(visible="legendonly") if trace.name in hidden_group else ()
        )

    return fig

# PCA GRAPH
def get_PCA_graph(df,theme_setting, **kwargs):
	df_pca = df[['nwound','nkill','nhostkid','nreleased']]
	X = scale(df_pca)
	pca = PCA(n_components=2)
	Y_pca = pca.fit_transform(X)
	fig = px.scatter(
		x = Y_pca[:, 0],
		y = Y_pca[:, 1],
		color = df['gname'],
		width = 450,
		)
	fig.update_layout(
		showlegend = False
		)

	return fig





#Layout

main_col_1_layout = [
	html.Div(children=[
    html.H1(children='Global Terrorism Map After 2000',
    		style={
                    'background-color':theme_setting['bg_color'], 
                    'color': theme_setting['text_color']})
    ]),

	html.Div(
        children=[
            dcc.Graph(id='global_map'),
        ]
    ),
	html.Div(
    	id="year_slider_container",
    	children = dcc.RangeSlider(
        	id='year_slider',
        	min=df['iyear'].min(),
        	max=df['iyear'].max(),
        	value=[df['iyear'].min(),df['iyear'].max()],
        	marks={str(year): str(year) for year in df['iyear'].unique()},
        	step=None
    	)
	),
	html.Div(
        id='buttom_row',
        style={
            'margin-top': 20,
        },
        children=[
            html.Div(
                id='trending_line_container',
                children=[
                    dcc.Graph(
                        id='trending_line',
                        config={ 'displayModeBar': False }
                    )
                ],
                style={
                    'display': 'inline-block', 
                    'width': '70%',
                }
            ),
            html.Div(
                id='bar_container',
                children=[
                    dcc.Graph(
                        id='pie_graph',
                        config={ 'displayModeBar': False }
                    )
                ], 
                style={
                    'display': 'inline-block', 
                    'width': '30%',
                },
            ),
        ]
    )
]
main_col_2_layout = [

	# html.Div(
	# 		id = "Dark_mode",
	# 		children =[
	# 		html.H5(
 #                children=["Dark Mode"],
 #                style={
 #                    'background-color':theme_setting['bg_color'], 
 #                    'color': theme_setting['text_color'],
 #                    'font-size':"15px",
 #                    'margin-right': "0px"}
 #                    ),
 #            dcc.RadioItems(
 #                id='theme_type',
 #                options=[{'label': i, 'value': i} for i in ['on', 'off']],
 #                value='Linear',
 #                labelStyle={'display': 'inline-block'}
 #            )
 #        	],style={
 #                    'background-color':theme_setting['bg_color'], 
 #                    'color': theme_setting['text_color'],
 #                    'font-size':"15px",
 #                    'margin-right': "0px"}),
    html.Div(
        id="Dimentional_Reduction",
        children=[
            html.H5(
                children=["PCA"],
                style={
                    'background-color':theme_setting['bg_color'], 
                    'color': theme_setting['text_color'],
                    'font-size':"15px",
                    'margin-right': "0px"}
                    ),
        ],
        style={
            "padding": "30px",
            'background-color': theme_setting['bg_color'], 
            'color': theme_setting['text_color']
        }
    ),

    html.Div(
        id='PCA_graph_container',
        children=[
            dcc.Graph(
            id='PCA_graph',
            config={ 'displayModeBar': False }
            )
            ], 
        style={
        'display': 'inline-block', 
        },
    ),
]

app.layout = html.Div(
    id="app",
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    id="main_col_1",
                    className="nine columns",
                    children=main_col_1_layout,
                    style={
                        'display': 'inline-block',
                        'margin-top': 10,
                        'margin-right': 10,
                        'margin-buttom': 10,
                        'margin-left': 10,
                    }
                ),
                html.Div(
                    id="main_col_2",
                    className="three columns",
                    children=main_col_2_layout,
                    style={
                        "height": "100%",
                        'display': 'inline-block', 
                        'margin-top': 10,
                        'margin-right': 10,
                        'margin-buttom': 10,
                        'margin-left': 10,
                    }
                ),
            ]
        ),
        html.Br(),
        html.Div(
            id='state_df_top_gname', 
            style={ 'display': 'none' }
        ),
        html.Div(
            id='state_list_top_gname', 
            style={ 'display': 'none' }
        ),
        html.Div(
            id='state_top_gname', 
            style={ 'display': 'none' }
        ),
        html.Div(
            id='state_selection_record', 
            style={ 'display': 'none' }
        )
    ]
)

@app.callback(
    [
        Output('state_df_top_gname', 'children'),
        Output('state_list_top_gname', 'children'),
        Output('state_top_gname', 'children')
    ],
    [
        Input('year_slider', 'value')
    ]
)
def update_state(year):
    df_ = df.loc[(year[0] <= df['iyear'])]
    df_ = df_.loc[(df_['iyear'] <= year[1])]
    top_group = df_[['gname', 'nkillwound']].groupby(by='gname').sum().sort_values(by=['nkillwound'], ascending=False)[: NUMTOPGROUP]
    top_group_list = top_group.index
    df_ = df_[df_['gname'].isin(top_group_list)]
    sortIdx = dict(zip(top_group_list, range(len(top_group_list))))
    df_['top_group_idx'] = df_['gname'].map(sortIdx)
    top_group = pd.DataFrame(
        data={
            "gname": top_group_list,
            "total": top_group['nkillwound'],
        }
    )

    nkill = []
    nwound = []
    for i in range(NUMTOPGROUP):
        group_data = df_.loc[df_['gname'] == top_group_list[i]]
        nkill.append(group_data['nkill'].sum())
        nwound.append(group_data['nwound'].sum())

    top_group['nkill'] = nkill
    top_group['nwound'] = nwound

    return [
        df_.to_json(date_format='iso', orient='split'), 
        json.dumps({ 'top_gname': list(top_group_list) }, indent=4),
        top_group.to_json(date_format='iso', orient='split')
    ]

@app.callback(
    Output('global_map', 'figure'),
    [Input('state_df_top_gname', 'children')])
def update_global_map(json_df):
    df = pd.read_json(json_df, orient='split')
    df = df[df['nkillwound'] > NUMCASUALITIES]
    df = df.sort_values(by = 'top_group_idx') 

    fig = px.scatter_geo(
        df, 
        lat=df['latitude'], 
        lon=df['longitude'],
        color=df['gname'],
        size=df['marker_size'],
        hover_name="gname",
        hover_data={
            'latitude': False,
            'longitude': False,
            'gname': False,
            'marker_size': False,
            'nkillwound': True,
            'country_txt': True,
            'weaptype1_txt': True,
            'attacktype1_txt': True,
        },
        labels={
            "gname": "Group Name",
            "nkillwound": "Casuality",
            "country_txt": "Country",
            "weaptype1_txt": "Weapon",
            "attacktype1_txt": "Attack Type",
        },
        custom_data=['summary'],
        scope="world",
        height=500,
        title=None
    )
    fig.update_layout(
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        plot_bgcolor=theme_setting['bg_color'],
        paper_bgcolor=theme_setting['bg_color'],
        font={ "color": theme_setting["text_color"] },
        hoverlabel={
            'bgcolor': theme_setting['bg_color']
        },
        legend={
            'title': "",
            'bgcolor': theme_setting['legend_bg_color'],
            'traceorder': "normal",    
            'yanchor': "top",
            'y': 0.98,
            'xanchor': "left",
            'x': 0.0
        },
        geo={
            "showocean": True,
            "showcountries": True,
            "landcolor": theme_setting['land_color'],
            "oceancolor": theme_setting['ocean_color'],
            "countrycolor": 'rgb(255,255,255)',
            "fitbounds": "locations"
        }
    )

    return fig

@app.callback(
    Output('trending_line', 'figure'),
    [
        Input('state_df_top_gname', 'children'),
        Input('state_top_gname', 'children'),
        Input('state_selection_record', 'children'),
    ],
)

def update_trending_line(json_df, json_top_group, json_selection_record):
    isCasualty=False
    df = pd.read_json(json_df, orient='split')
    top_group = pd.read_json(json_top_group, orient='split')

    year_sum_df = pd.DataFrame([])
    year_min = df['iyear'].min()
    year_max = df['iyear'].max()
    x_axis = [year for year in range(year_min, year_max + 1)]
    for group in top_group['gname']:
        group_data = df[df['gname'] == group]
        if isCasualty:
            year_count = group_data[['iyear', 'nkillwound']].groupby(['iyear']).sum()
            year_count = pd.Series(year_count['nkillwound'], index=year_count.index)
        else: 
            year_count = group_data['iyear'].value_counts()
        
        for year in x_axis:
            if not year in year_count.index:
                year_count = year_count.append(pd.Series([0], index=[year]))
        tmp = pd.DataFrame({
             'gname': [group for i in range(year_count.shape[0])], 
             'iyear': year_count.index, 
             'total': year_count.values})
        tmp = tmp.sort_values(by=['iyear'])
        year_sum_df = pd.concat([year_sum_df, tmp])

    if isCasualty:
        y_label = "Casuality"
    else:
        y_label = "Number of Attacks"

    
    if json_selection_record is None:
        return get_trending_line(year_sum_df, y_label, theme_setting)
    else:
        selection_record = json.loads(json_selection_record)["record"]
        return get_trending_line(
            year_sum_df, 
            y_label, 
            theme_setting,
            selection_record=selection_record,
            top_group_list=top_group['gname'] 
    )


@app.callback(
    Output('pie_graph', 'figure'),
    [
        Input('state_top_gname', 'children'),
        Input('state_selection_record', 'children')
    ],
)
def update_pie_graph(json_top_group, json_selection_record):
    top_group = pd.read_json(json_top_group, orient='split')

    if json_selection_record is None:
        return get_pie_graph(top_group, theme_setting)
    else:
        selection_record = json.loads(json_selection_record)["record"]
        return get_pie_graph(
            top_group,
            theme_setting,
            selection_record=selection_record,
        )

@app.callback(
	Output('PCA_graph','figure'),
	[
		Input('state_df_top_gname','children'),
		Input('state_top_gname', 'children'),
		Input('global_map','selectedData')
	],
)
def update_selected_PCA_graph(json_df,json_top_group,json_selection_record):
	df = pd.read_json(json_df, orient='split')
	top_group = pd.read_json(json_top_group, orient='split')

	# df_pca = df[['nwound','nkill','nhostkid','nreleased']]
	# gname = df['gname'].unique()
	# df_pca_original = []
	# for group in gname:
 #  		num_wound = df.loc[df['gname']==group]['nwound']
 #  		num_kill = df.loc[df['gname']==group]['nkill']
 #  		num_hostkid = df.loc[df['gname']==group]['nhostkid']
 #  		num_release = df.loc[df['gname']==group]['nreleased']
 #  		#df_kw[group] = [num_wound,num_kill,num_wound+num_kill]
 #  		df_pca_original.append(
 #  			{
 #  				'num_wound': num_wound,
 #  				'num_kill': num_kill,
 #  				'num_hostkid': num_hostkid,
 #  				'num_release': num_release,
 #      		}
 #    	)

	# labels = ['num_wound','num_kill','num_hostkid','num_release']
	# df_pca = pd.DataFrame(data = df_pca_original,columns=labels)

	if json_selection_record is None:
		return get_PCA_graph(df,theme_setting)
	# else:
	# 	selection_record = json.loads(json_selection_record)["record"]
 #        return get_PCA_graph(
 #        	df,
 #        	theme_setting,
 #        	selection_record=selection_record,
 #        	)






if __name__ == '__main__':
    app.run_server(debug=True)
