import plotly.express as px
import plotly.graph_objects as go

def bar_chart(df, x, y, title=""):
    fig = px.bar(df, x=x, y=y, title=title)
    return fig

def hist_chart(df, x, title=""):
    fig = px.histogram(df, x=x, title=title)
    return fig

def line_chart(df, x, y, title=""):
    fig = px.line(df, x=x, y=y, title=title)
    return fig

def scatter_chart(df, x, y, title=""):
    fig = px.scatter(df, x=x, y=y, title=title)
    return fig

def heatmap(df, title=""):
    fig = px.imshow(df, text_auto=True, title=title)
    return fig
