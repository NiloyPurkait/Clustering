import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_3d(df, xyzk, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=df[xyzk[0]],
        y=df[xyzk[1]],
        z=df[xyzk[2]],
        mode='markers',
        marker=dict(
            size=5,
            color=df[xyzk[3]],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df[xyzk[3]]
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title=xyzk[0],
            yaxis_title=xyzk[1],
            zaxis_title=xyzk[2]
        ),
        title=title,
        width=800,
        height=600
    )
    
    fig.show()






def d_types_report(df):
    columns=[]
    d_types=[]
    uniques=[]
    n_uniques=[]
    null_values=[]
    null_values_percentage=[]
    rows = df.shape[0]
    
    for column_name in df.columns:
        columns.append(column_name)
        d_types.append(df[column_name].dtypes)
        uniques.append(df[column_name].unique()[:5])
        n_uniques.append(df[column_name].nunique())
        null_values.append(df[column_name].isna().sum())
        null_values_percentage.append(null_values[-1] * 100 / rows)

    return pd.DataFrame({"Columns": columns,
                         "Data_Types": d_types,
                         "Unique_values": uniques,
                         "N_Uniques": n_uniques,
                         "Null_Values": null_values,
                         "Null_Values_percentage": null_values_percentage})







def correlation_analysis(data, threshold = 0.33): # Adjust threshold as needed
    corr = data.corr()
    high_corr_features = corr['Cluster'][(corr['Cluster'] > threshold) | (corr['Cluster'] < -threshold)]
    print("Features with high correlation to 'Clusters':")
    return high_corr_features.sort_values()




def correlation_plot(data, ):
    corr = data.corr() # returns dataframe with pearson correlation btw features, other presets are kendall & spearman

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.index,
        y=corr.columns,
        colorscale='Greens',
        zmin=-1,  # Set the color scale range
        zmax=1,   
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        width=600,
        height=600
    )

    fig.show()



def bar_graph(data):
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    count_df = data["Cluster"].value_counts().reset_index()
    count_df.columns = ["Cluster", "Count"]

    bar_traces = []
    for i, row in count_df.iterrows():
        cluster_label = row["Cluster"]
        count = row["Count"]
        color = custom_colors[cluster_label]
        bar_trace = go.Bar(x=[cluster_label], y=[count], marker_color=color)
        bar_traces.append(bar_trace)

    # Create the layout for the bar chart
    layout = go.Layout(
        title="Distribution Of The Clusters",
        xaxis=dict(title="Cluster"),
        yaxis=dict(title="Count"),
        width=600,  # Set the width of the graph
        height=500  # Set the height of the graph
    )

    fig = go.Figure(data=bar_traces, layout=layout)
    fig.show()




def scatter_plot_(data, xyc):
    cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}

    data['Cluster Names'] = data[xyc[2]].apply(lambda x: f'cluster_{x}')

    fig = px.scatter(data, x=xyc[0], y=xyc[1], color="Cluster Names",
                     title="Cluster's Profile Based On Income And Spending")

    fig.update_layout(
        width=800,  # Set the width of the graph
        height=500,  # Set the height of the graph
        legend_title_text="Cluster Number"  # Rename the legend title
    )

    fig.show()



def is_discrete_integer(series):
    """Check if a pandas Series consists of discrete integers."""
    return all(series.apply(lambda x: (isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)))))

def add_jitter(arr, amount=0.5):
    """Add jitter to an array."""
    return arr + np.random.uniform(-amount, amount, size=arr.shape)

def scatter_plot(data, xyc):
    cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}

    data['Cluster Names'] = data[xyc[2]].apply(lambda x: f'cluster_{x}')
    
    # Check if x column is discrete integer and add jitter if true
    if is_discrete_integer(data[xyc[0]]):
        data[xyc[0]] = add_jitter(data[xyc[0]])
    
    # Check if y column is discrete integer and add jitter if true
    if is_discrete_integer(data[xyc[1]]):
        data[xyc[1]] = add_jitter(data[xyc[1]])

    fig = px.scatter(data, x=xyc[0], y=xyc[1], color="Cluster Names",
                     title=f"Cluster's Profile Based On {xyc[0]} And {xyc[1]}")

    fig.update_layout(
        width=800,  # Set the width of the graph
        height=500,  # Set the height of the graph
        legend_title_text="Cluster Number"  # Rename the legend title
    )

    fig.show()




def pca_plot(cumulative_explained_variance, df):
    fig = px.line(x=np.arange(1, df.shape[1] + 1), y=cumulative_explained_variance, markers=True)

    # Customize the figure
    fig.update_layout(
        title='Explained Variance by Number of Principal Components',
        xaxis_title='Number of Principal Components',
        yaxis_title='Cumulative Explained Variance',
        showlegend=False,
        width=800,
        height=500,
    )

    # Show the Plotly figure
    fig.show()