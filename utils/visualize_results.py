import pandas as pd
import plotly.express as px


def visualize_true_pred(true, pred):
    df = pd.DataFrame({"True": true, "Pred": pred})
    df["index"] = df.index

    fig = px.scatter(df, x='True', y='Pred', height=700,
                     template='plotly_dark', hover_data=["index"])
    fig.update_traces(marker_size=7)

    # Add line y=x
    min_ = df['True'].min()
    max_ = df['True'].max()
    fig.add_shape(type='line', x0=min_, y0=min_, x1=max_, y1=max_,
                  line=dict(color='Yellow', width=3))
    
    fig.show()
