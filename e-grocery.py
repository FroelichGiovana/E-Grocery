# Importing Liraries
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Importing data

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "E-Grocery Dashboard"

data = pd.read_csv("filtered_dataset.csv")
filtered_data = data.copy()

if 'product_name' in data.columns and 'department' in data.columns:
    data['combined_features'] = (
        data['product_name'].astype(str) + " " + data['department'].astype(str)
    )
# preparing the data for the contente-based part
data['combined_features'] = data['combined_features'].fillna("")

aggregated_data = data.groupby('product_id', as_index=False).agg({
    'combined_features': ' '.join
})

tfidf = TfidfVectorizer()
item_features_matrix = tfidf.fit_transform(aggregated_data['combined_features'])

product_to_index = {row['product_id']: idx for idx, row in aggregated_data.iterrows()}

user_product_matrix = filtered_data.pivot_table(
    index='user_id',
    columns='product_id',
    values='order_id',
    aggfunc='count',
    fill_value=0)

unique_users = filtered_data['user_id'].unique()

user_similarity = pd.DataFrame(
    cosine_similarity(user_product_matrix),
    index=user_product_matrix.index,
    columns=user_product_matrix.index)

item_similarity = pd.DataFrame(
    cosine_similarity(user_product_matrix.T),
    index=user_product_matrix.columns,
    columns=user_product_matrix.columns)


def create_dow_graph():
    day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    dow_counts = filtered_data['order_dow'].value_counts().sort_index()
    dow_data = pd.DataFrame({'Day of Week': dow_counts.index.map(day_map), 'Number of Orders': dow_counts.values})
    fig = px.bar(
        dow_data,
        x='Day of Week',
        y='Number of Orders',
        color='Number of Orders',
        title='Popularity by Day of the Week',
        color_continuous_scale='Blues',
        template='plotly_dark')
    return fig

def create_heatmap():
    heatmap_data = filtered_data.pivot_table(
        index='order_dow', columns='order_hour_of_day', values='order_id', aggfunc='count', fill_value=0 )
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Order Count"),
        title="Popularity by Day of the Week and Hours of the Day",  
        color_continuous_scale="Blues",
        template='plotly_dark'
    )
    return fig

def create_product_comparison():
    product_counts = filtered_data['product_name'].value_counts()
    lower_percentile = np.percentile(product_counts, 40)
    upper_percentile = np.percentile(product_counts, 60)
    most_popular_products = product_counts.head(15)
    mid_popular_products = product_counts[(product_counts >= lower_percentile) & (product_counts <= upper_percentile)].head(15)
    most_popular_df = pd.DataFrame({'Product Name': most_popular_products.index, 'Order Count': most_popular_products.values})
    mid_popular_df = pd.DataFrame({'Product Name': mid_popular_products.index, 'Order Count': mid_popular_products.values})
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Most Sold Products", "Mid-Level Products"))
    fig.add_trace(go.Bar(
        x=most_popular_df['Order Count'],
        y=most_popular_df['Product Name'],
        name='Most Sold Products',
        orientation='h',
        marker=dict(color='green')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=mid_popular_df['Order Count'],
        y=mid_popular_df['Product Name'],
        name='Mid-Level Products',
        orientation='h',
        marker=dict(color='red')
    ), row=1, col=2)
    fig.update_layout(
        height=500,
        title_text="Comparison: Most Sold vs Mid-Level Products",
        template='plotly_dark',
        showlegend=False
    )
    return fig

def create_reorder_scatter():
    reorder_rate = filtered_data.groupby('product_name')['reordered'].mean().reset_index()
    reorder_rate = reorder_rate.sort_values(by='reordered', ascending=False).head(15)
    fig = px.scatter(
        reorder_rate,
        x='reordered',
        y=reorder_rate['product_name'],
        size='reordered',
        color='reordered',
        color_continuous_scale='Reds',
        title='Top 15 Products with Highest Reorder Rate',
        labels={'reordered': 'Reorder Rate', 'product_name': 'Product Name'},
        template='plotly_dark')
    fig.update_layout(
        xaxis_title='Reorder Rate',
        yaxis_title='Product Name',
        xaxis=dict(tickformat=".0%"),
        coloraxis_colorbar=dict(
            title="Reorder Rate",
            tickformat=".0%"))
    return fig

def create_category_bar():
    category_counts = filtered_data.groupby('department')['product_name'].count().reset_index()
    category_counts.columns = ['Department', 'Total Products']
    fig = px.bar(
        category_counts,
        x='Department',
        y='Total Products',
        color='Total Products',
        title='Products by Category (Department)',
        color_continuous_scale='Reds',
        template='plotly_dark')
    return fig
def create_interactive_category():
    category_product_counts = filtered_data.groupby(['department', 'product_name'])['order_id'].count().reset_index()
    category_product_counts.columns = ['Category', 'Product Name', 'Order Count']
    categories = category_product_counts['Category'].unique()

    initial_category = categories[0]
    initial_data = category_product_counts[category_product_counts['Category'] == initial_category]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=initial_data['Order Count'],
        y=initial_data['Product Name'],
        orientation='h',
        marker=dict(color='red') ))

    fig.update_layout(
        title=f'Products in Category: {initial_category}',
        xaxis_title='Order Count',
        yaxis_title='Product Name',
        yaxis=dict(autorange='reversed'),
        template='plotly_dark',
        updatemenus=[{
            'buttons': [
                dict(
                    label=category,
                    method="update",
                    args=[
                        {'x': [category_product_counts[category_product_counts['Category'] == category]['Order Count']],
                         'y': [category_product_counts[category_product_counts['Category'] == category]['Product Name']],
                         'type': 'bar'},
                        {'title': f'Products in Category: {category}'}
                    ]
                ) for category in categories
            ],
            'direction': 'down',
            'showactive': True,
            'x': 1.15,
            'y': 1.15
        }]
    )
    return fig

def create_user_pie_chart():
    user_purchase_counts = data.groupby('user_id')['product_id'].count().reset_index()
    user_purchase_counts.columns = ['user_id', 'purchase_count']
    most_active_users = user_purchase_counts.sort_values(by='purchase_count', ascending=False).head(10)
    total_top_10_purchases = most_active_users['purchase_count'].sum()

    most_active_users['percentage'] = (most_active_users['purchase_count'] / total_top_10_purchases) * 100

    fig = px.pie(
        most_active_users,
        values='percentage',
        names='user_id',
        title="Top 10 Users Contribution to Total Purchases (%)",
        template='plotly_dark',
        color_discrete_sequence=px.colors.sequential.Greens)

    fig.update_traces(
        textinfo='percent',
        hovertemplate='User ID: %{label}<br>Percentage: %{value:.2f}%')
    return fig
def create_reorder_pie():
    """
    Create a pie chart for the proportion of reorders.
    """
    reorders_count = filtered_data['reordered'].value_counts().reset_index()
    reorders_count.columns = ['Type', 'Count']
    reorders_count['Type'] = reorders_count['Type'].replace({1: 'Reorders', 0: 'First-Time Orders'})

    fig = px.pie(
        reorders_count,
        values='Count',
        names='Type',
        title="Proportion of Reorders",
        color='Type',
        color_discrete_map={'Reorders': '#0057D9', 'First-Time Orders': '#66B2FF'},  
        template='plotly_dark'  
    )

    fig.update_traces(
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=1.5))  
    )

    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20, color="white")),
        font=dict(size=14, color="white"),
        legend=dict(
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    return fig


# Recommendation Systens
def recommend_user_user(user_id, user_product_matrix, user_similarity):
    """
    Recommends products for a user based on User-User Collaborative Filtering.

    Args:
        user_id (int): User ID for whom recommendations are being generated.
        user_product_matrix (DataFrame): User-product interaction matrix.
        user_similarity (DataFrame): Precomputed user-user similarity matrix.

    Returns:
        DataFrame: Recommended products with their scores.
    """
    if user_id not in user_product_matrix.index:
        return pd.DataFrame(columns=['product_id', 'count'])

    user_similarities = user_similarity[user_id]
    similar_users = user_similarities.sort_values(ascending=False)[1:11]  

    similar_users_products = user_product_matrix.loc[similar_users.index]
    product_scores = similar_users_products.sum(axis=0)

    user_products = user_product_matrix.loc[user_id]
    product_scores = product_scores[user_products == 0]

    return product_scores.sort_values(ascending=False).head(10)

def recommend_item_item(user_id, user_product_matrix, item_similarity):
    """
    Recommends products for a user based on Item-Item Collaborative Filtering.

    Args:
        user_id (int): User ID for whom recommendations are being generated.
        user_product_matrix (DataFrame): User-product interaction matrix.
        item_similarity (DataFrame): Precomputed item-item similarity matrix.

    Returns:
        DataFrame: Recommended products with their scores.
    """
    if user_id not in user_product_matrix.index:
        return pd.DataFrame(columns=['product_id', 'count'])

    user_products = user_product_matrix.loc[user_id]
    purchased_items = user_products[user_products > 0].index

    scores = pd.Series(dtype=float)
    for item in purchased_items:
        if item in item_similarity.columns:
            similar_items = item_similarity[item]
            scores = scores.add(similar_items, fill_value=0)

    scores = scores[~scores.index.isin(purchased_items)]

    return scores.sort_values(ascending=False).head(10)


def recommend_content_based(user_id, data, item_features_matrix, product_to_index, top_k=10):
    """
    Generate content-based recommendations for a specific user.
    """
    user_purchases = data[data['user_id'] == user_id]['product_id'].unique()

    if len(user_purchases) == 0:
        print(f"No purchases found for User ID {user_id}. Cannot generate Content-Based recommendations.")
        return pd.DataFrame(columns=['product_id', 'count'])

    content_recommendations = []
    for product_id in user_purchases:
        if product_id not in product_to_index:
            print(f"Product ID {product_id} not found in product_to_index mapping. Skipping.")
            continue

        product_idx = product_to_index[product_id]
        if product_idx >= item_features_matrix.shape[0]:
            print(f"Product index {product_idx} for product {product_id} is out of range. Skipping.")
            continue

        similarity_scores = cosine_similarity(item_features_matrix[product_idx], item_features_matrix).flatten()
        similar_items = similarity_scores.argsort()[-top_k-1:-1][::-1]
        content_recommendations.extend(aggregated_data.iloc[similar_items]['product_id'])

    recommendations_df = pd.Series(content_recommendations).value_counts().reset_index()
    recommendations_df.columns = ['product_id', 'count']
    return recommendations_df.head(top_k)

def plot_recommendations(selected_user, recommendation_type):
    """
    Plot recommendations for a user based on the selected recommendation type.

    Args:
        selected_user (int): User ID for which recommendations are plotted.
        recommendation_type (str): Recommendation type ('Content-Based').

    Returns:
        A Plotly bar chart of recommendations.
    """
    if recommendation_type == 'Content-Based':
        recommendations = recommend_content_based(selected_user, data, item_features_matrix, product_to_index)
    else:
        print(f"Recommendation type '{recommendation_type}' is not supported.")
        return go.Figure()

    recommendations_with_names = recommendations.merge(
        data[['product_id', 'product_name']].drop_duplicates(), 
        on='product_id', 
        how='left' )

    if recommendations_with_names.empty:
        print(f"No recommendations available for User ID {selected_user} with {recommendation_type}.")
        return go.Figure()

    fig = px.bar(
        recommendations_with_names,
        x='product_name',
        y='count',
        color='count',
        title=f"{recommendation_type} Recommendations for User {selected_user}",
        labels={'product_name': 'Product', 'count': 'Recommendation Count'},
        template='plotly_white',
        color_continuous_scale='Blues' )
    fig.update_layout(
        xaxis=dict(tickangle=45),
        xaxis_title='Recommended Product',
        yaxis_title='Recommendation Count' )
    return fig

product_names = data[['product_id', 'product_name']].drop_duplicates()

def generate_mba_rules(data):
    """
    Generate Market Basket Analysis rules using Apriori algorithm.
    Handles compatibility issues with `association_rules` in different versions of mlxtend.

    Args:
        data (DataFrame): Input dataset with 'order_id' and 'product_name'.

    Returns:
        DataFrame: Association rules with lift, confidence, and support.
    """
    transactions = data.groupby("order_id")["product_name"].apply(list)

    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_data, columns=te.columns_)

    frequent_itemsets = apriori(te_df, min_support=0.02, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try reducing the minimum support.")
        return pd.DataFrame()

    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    except TypeError as e:
        if "num_itemsets" in str(e):
            rules = association_rules(
                frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets)
            )
        else:
            raise

    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    return rules

rules = generate_mba_rules(data)

def plot_mba(selected_product):
    if not selected_product:
        return go.Figure()
    filtered_rules = rules[rules['antecedents_str'] == selected_product].head(10)
    if filtered_rules.empty:
        return go.Figure()
    fig = px.bar(
        filtered_rules,
        x='confidence',
        y='consequents_str',
        orientation='h',
        title=f"Top 10 Recommendations for '{selected_product}' (MBA)",
        labels={'confidence': 'Confidence', 'consequents_str': 'Recommended Products'},
        template='plotly_white',
        color='lift',
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Recommended Products",
        yaxis=dict(autorange="reversed")
    )
    return fig

# Layout Dashboard

app.layout = html.Div(
    style={
        "backgroundColor": "#000000",  
        "color": "#FFFFFF",  
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1(
            "E-Grocery Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "30px",
                "color": "#FFFFFF",  
                "fontWeight": "bold",
            },
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            style={
                "backgroundColor": "#000000",  
                "color": "#FFFFFF",
                "borderRadius": "8px",
                "boxShadow": "0px 2px 5px rgba(255, 255, 255, 0.4)",
            },
            children=[
                # Overview Tab
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    style={
                        "backgroundColor": "#000000",
                        "color": "#FFFFFF",
                        "padding": "10px",
                        "fontSize": "18px",
                        "borderRadius": "8px",
                    },
                    selected_style={
                        "backgroundColor": "#333333",  
                        "color": "#FFFFFF",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3(
                            "Overview Options",
                            style={"textAlign": "center", "color": "#FFFFFF"},  
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Orders by Day and Time",
                                    id="btn-orders",
                                    n_clicks=0,
                                    style={
                                        "marginRight": "10px",
                                        "padding": "10px 20px",
                                        "fontSize": "14px",
                                        "backgroundColor": "#1E90FF",  
                                        "border": "none",
                                        "color": "white",
                                        "borderRadius": "8px",
                                        "cursor": "pointer",
                                        "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)",
                                    },
                                ),
                                html.Button(
                                    "Product Information",
                                    id="btn-products",
                                    n_clicks=0,
                                    style={
                                        "marginRight": "10px",
                                        "padding": "10px 20px",
                                        "fontSize": "14px",
                                        "backgroundColor": "#1E90FF",  
                                        "border": "none",
                                        "color": "white",
                                        "borderRadius": "8px",
                                        "cursor": "pointer",
                                        "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)",
                                    },
                                ),
                                html.Button(
                                    "User Information",
                                    id="btn-users",
                                    n_clicks=0,
                                    style={
                                        "padding": "10px 20px",
                                        "fontSize": "14px",
                                        "backgroundColor": "#1E90FF",  
                                        "border": "none",
                                        "color": "white",
                                        "borderRadius": "8px",
                                        "cursor": "pointer",
                                        "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)",
                                    },
                                ),
                            ],
                            style={"textAlign": "center", "marginBottom": "30px"},
                        ),
                        # Dynamic content for the Overview tab
                        html.Div(
                            id="overview-content",
                            style={
                                "padding": "20px",
                                "backgroundColor": "#2B2B2B",  
                                "borderRadius": "8px",
                                "boxShadow": "0px 2px 5px rgba(255, 255, 255, 0.4)",
                                "margin": "20px auto",
                                "width": "90%",
                            },
                        ),
                    ],
                ),
                # Recommendation System Tab
                dcc.Tab(
                    label="Recommendation System",
                    value="recommendation",
                    style={
                        "backgroundColor": "#000000",
                        "color": "#FFFFFF",
                        "padding": "10px",
                        "fontSize": "18px",
                        "borderRadius": "8px",
                    },
                    selected_style={
                        "backgroundColor": "#333333",
                        "color": "#000000",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3(
                            "User Recommendation System",
                            style={"textAlign": "center", "color": "#FFFFFF"},  
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Select User ID", style={"color": "#FFFFFF"}
                                ),
                                dcc.Dropdown(
                                    id="user-dropdown",
                                    options=[
                                        {"label": f"User {user}", "value": user}
                                        for user in unique_users
                                    ],
                                    placeholder="Select a User ID",
                                    style={
                                        "width": "40%",
                                        "margin": "0 auto 20px auto",
                                        "padding": "5px",
                                        "backgroundColor": "#000000",  
                                        "color": "#000000",  
                                        "border": "1px solid #66CCFF", 
                                        "borderRadius": "8px",
                                        "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)",  
                                    },
                                ),
                                html.Label(
                                    "Select Recommendation Type",
                                    style={"color": "#FFFFFF"},
                                ),
                                dcc.Dropdown(
                                    id="recommendation-type-dropdown",
                                    options=[
                                        {"label": "User-User", "value": "User-User"},
                                        {"label": "Item-Item", "value": "Item-Item"},
                                        {"label": "Content-Based", "value": "Content-Based"},
                                    ],
                                    placeholder="Choose Recommendation Type",
                                    style={
                                        "width": "40%",
                                        "margin": "0 auto 20px auto",
                                        "padding": "5px",
                                        "backgroundColor": "#000000",  
                                        "color": "#000000", 
                                        "border": "1px solid #66CCFF",  
                                        "borderRadius": "8px",
                                        "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)",  
                                    },
                                ),
                                dcc.Graph(id="recommendation-graph"),
                            ]
                        ),
                    ],
                ),
                # Market Basket Analysis Tab
                dcc.Tab(
                    label="Market Basket Analysis",
                    value="mba",
                    style={
                        "backgroundColor": "#000000",
                        "color": "#FFFFFF",
                        "padding": "10px",
                        "fontSize": "18px",
                        "borderRadius": "8px",
                    },
                    selected_style={
                        "backgroundColor": "#000000",
                        "color": "#FFFFFF",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3(
                            "Market Basket Analysis",
                            style={"textAlign": "center", "color": "#FFFFFF"},  
                        ),
                        html.Label(
                            "Select Product (for MBA):", style={"color": "#FFFFFF"}
                        ),
                        dcc.Dropdown(
                            id="mba-product-dropdown",
                            options=[
                                {"label": product, "value": product}
                                for product in rules["antecedents_str"].unique()
                            ],
                            placeholder="Choose a product",
                            style={
                                "width": "40%",
                                "margin": "10px auto",
                                "backgroundColor": "#000000", 
                                "color": "#000000",  
                                "border": "1px solid #66CCFF",  
                                "borderRadius": "8px",
                                "boxShadow": "2px 4px 6px rgba(255, 255, 255, 0.3)", 
                            },
                        ),
                        dcc.Graph(id="mba-graph"),
                    ],
                ),
            ],
        ),
    ],
)

# Callback"Overview"
@app.callback(
    Output("overview-content", "children"),
    [Input("btn-orders", "n_clicks"),
     Input("btn-products", "n_clicks"),
     Input("btn-users", "n_clicks")]
)
def update_overview(btn_orders, btn_products, btn_users):
    ctx = dash.callback_context

    if not ctx.triggered:
        return html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            children=[
                html.Div(
                    dcc.Graph(figure=create_reorder_pie()),
                    style={"flex": "1", "padding": "10px"}
                ),
                html.Div(
                    html.Img(
                        src="/assets/download.jpeg", 
                        style={
                            "width": "400px",
                            "height": "auto",
                            "borderRadius": "8px",
                            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
                        }
                    ),
                    style={"flex": "1", "textAlign": "center"}
                ),
            ]
        )

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-orders":
        return html.Div(
            [
                html.H3("Orders by Day and Time", style={"textAlign": "center"}),
                dcc.Graph(figure=create_dow_graph()),
                dcc.Graph(figure=create_heatmap()),
            ]
        )
    elif button_id == "btn-products":
        return html.Div(
            [
                html.H3("Product Information", style={"textAlign": "center"}),
                dcc.Graph(figure=create_product_comparison()),
                dcc.Graph(figure=create_reorder_scatter()),
                dcc.Graph(figure=create_category_bar()),
                dcc.Graph(figure=create_interactive_category()),
            ]
        )
    elif button_id == "btn-users":
        return html.Div(
            [
                html.H3("User Information", style={"textAlign": "center"}),
                dcc.Graph(figure=create_user_pie_chart()),
            ]
        )
    return html.Div("Select an option from the Overview menu.")

# Callback  "Market Basket Analysis" 
@app.callback(
    Output("mba-graph", "figure"),
    Input("mba-product-dropdown", "value")
)
def update_mba_graph(selected_product):
    if not selected_product:
        return go.Figure()

    filtered_rules = rules[rules["antecedents_str"] == selected_product].head(10)

    if filtered_rules.empty:
        return go.Figure()

    fig = px.bar(
        filtered_rules,
        x="confidence",
        y="consequents_str",
        orientation="h",
        title=f"Top 10 Recommendations for '{selected_product}' (MBA)",
        labels={"confidence": "Confidence", "consequents_str": "Recommended Products"},
        template="plotly_dark",
        color="lift",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Recommended Products",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# Callback 
@app.callback(
    Output("recommendation-graph", "figure"),
    [Input("recommendation-type-dropdown", "value"),
     Input("user-dropdown", "value")]
)
def update_recommendations(recommendation_type, selected_user):
    if not recommendation_type or not selected_user:
        print("Recommendation type or User ID not selected.")
        return go.Figure()

    if recommendation_type == "User-User":
        recommendations = recommend_user_user(selected_user, user_product_matrix, user_similarity)
    elif recommendation_type == "Item-Item":
        recommendations = recommend_item_item(selected_user, user_product_matrix, item_similarity)
    elif recommendation_type == "Content-Based":
        recommendations = recommend_content_based(selected_user, data, item_features_matrix, product_to_index)
    else:
        print(f"Invalid recommendation type: {recommendation_type}")
        return go.Figure()

    if recommendations.empty:
        print(f"No recommendations available for User ID {selected_user} with type {recommendation_type}.")
        fig = go.Figure()
        fig.update_layout(
            title="No Recommendations Available",
            xaxis_title="",
            yaxis_title="",
        )
        return fig

    try:
        recommendations_df = recommendations.reset_index()
        if recommendations_df.shape[1] == 2:
            recommendations_df.columns = ['product_id', 'count']
        elif recommendations_df.shape[1] > 2:
            recommendations_df = recommendations_df.iloc[:, :2]  
            recommendations_df.columns = ['product_id', 'count']
        else:
            raise ValueError(f"Unexpected structure in recommendations_df: {recommendations_df.shape}")
    except Exception as e:
        print(f"Error processing recommendations: {e}")
        return go.Figure()

    recommendations_with_names = recommendations_df.merge(product_names, on='product_id', how='left')

    if recommendations_with_names.empty:
        print(f"No product names found for recommendations. User ID: {selected_user}")
        fig = go.Figure()
        fig.update_layout(
            title="No Product Names Available",
            xaxis_title="",
            yaxis_title="",
        )
        return fig

    fig = px.bar(
        recommendations_with_names,
        x="product_name",
        y="count",
        color="count",
        title=f"{recommendation_type} Recommendations for User {selected_user}",
        labels={"product_name": "Product", "count": "Recommendation Count"},
        template="plotly_dark",
        color_continuous_scale=(
            "Oranges" if recommendation_type == "User-User" else
            "Greens" if recommendation_type == "Item-Item" else
            "Blues"
        ),
    )
    fig.update_layout(
        xaxis=dict(tickangle=45),
        xaxis_title="Recommended Product",
        yaxis_title="Recommendation Count",
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)