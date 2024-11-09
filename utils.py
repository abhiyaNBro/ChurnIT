import plotly.graph_objects as go
import streamlit as st

def set_page_style():
    style = '''
    <style>
    .stApp {
        background-color: #1E1E2E;  /* Dark purplish background */
    }
    /* Main text color */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p {
        color: #FFFFFF !important;
    }
    /* Input fields */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select, 
    .stNumberInput > div > div > input {
        color: #FFFFFF !important;
        background-color: #2A2A3E !important;
    }
    /* Radio buttons and checkboxes */
    .stRadio > div, .stCheckbox > label > div {
        color: #FFFFFF !important;
    }
    /* Slider */
    .stSlider > div > div > div {
        color: #FFFFFF !important;
    }
    /* Button */
    .stButton > button {
        color: #FFFFFF !important;
        background-color: #3E3E5E !important;
        border: 1px solid #5E5E7E !important;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #2A2A3E;
    }
    </style>
    '''
    st.markdown(style, unsafe_allow_html=True)

def create_gauge_chart(probability):
  #Determine coloe based on churn probability
  if probability < 0.3:
    color = "green"
  elif probability < 0.6:
    color = "yellow"
  else:
    color = "red"

  #create a gauge chart
  fig = go.Figure(
    go.Indicator(mode="gauge+number",
                value=probability * 100,
                domain={
                  'x': [0,1],
                  'y': [0,1]
                },
                title={
                  'text': "Churn Probability",
                  'font': {
                    'size': 24,
                    'color': 'white'
                  }
                },
                number={'font': {
                  'size': 40,
                  'color': 'white'
                }},
                gauge={
                   'axis': {
                     'range': [0, 100],
                     'tickwidth': 1,
                     'tickcolor': 'white',
                   },
                  'bar': {
                    'color': color
                  },
                  'bgcolor': "rgba(0,0,0,0)",
                  'borderwidth': 2,
                  'bordercolor': "white",
                  'steps': [{
                    'range': [0, 30],
                    'color': "rgba(0, 255, 0, 0.3)"
                  }, {
                    'range': [30, 60],
                    'color': "rgba(255, 255, 0, 0.3)"
                  }, {
                    'range': [60, 100],
                    'color': "rgba(255, 0, 0, 0.3)"
                  }],
                  'threshold': {
                    'line': {
                      'color': "white",
                      'width': 4
                    },
                    'thickness': 0.75,
                    'value': 100
                  }
                }))

  fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",
                  font={'color': "white"},
                  width=400,
                  height=300,
                  margin=dict(l=20, r=20, t=50, b=20))

  return fig

def create_model_probability_chart(probabilities):
  models = list(probabilities.keys())
  probs = list(probabilities.values())

  fig = go.Figure(data=[
    go.Bar(y=models, x=probs, orientation='h', text=[f'{p:.2%}' for p in probs], textposition='auto')
  ])

  fig.update_layout(
        title='Churn Probability by Model',
        title_font_color="white",  # Set title color to white
        yaxis_title='Models',
        yaxis_title_font_color="white",  # Set y-axis title color to white
        xaxis_title='Probability',
        xaxis_title_font_color="white",  # Set x-axis title color to white
        xaxis=dict(
            tickformat='.0%', 
            range=[0,1], 
            color="white",
            tickfont=dict(color="white")  # Set x-axis tick label color to white
        ),
        yaxis=dict(
            color="white",
            tickfont=dict(color="white")  # Set y-axis tick label color to white
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#1E1E2E",  # Set paper background to match page background
        plot_bgcolor="#1E1E2E",  # Set plot background to match page background
        font=dict(color="white")  # Set overall font color to white
    )

  return fig

def create_percentile_chart(percentiles):
    metrics = list(percentiles.keys())
    values = list(percentiles.values())
    
    fig = go.Figure(data=[
        go.Bar(y=metrics, x=values, orientation='h', text=[f'{v:.1f}%' for v in values], textposition='auto')
    ])
    
    fig.update_layout(
        xaxis_title='Percentile',
        xaxis_title_font_color="white",
        yaxis_title='Metric',
        yaxis_title_font_color="white",
        xaxis=dict(
            range=[0, 100], 
            tickvals=[0, 20, 40, 60, 80, 100],  # increments of 20
            ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],  # Display as percentage
            color="white",  # Set axis color
            tickfont=dict(color="white")  # Set tick label color
        ),
        yaxis=dict(color="white", tickfont=dict(color="white")),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="white"),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
