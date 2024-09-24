from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, ALL, Patch
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import calendar
import math

def generate_spend(row):
    return max(0, (row.product_amount + row.transaction_fee - row.cashback))

def preprocess_df(df):
    _df = df.copy()
    _df["day"] = pd.DatetimeIndex(_df['transaction_date']).day
    _df["month"] = pd.DatetimeIndex(_df['transaction_date']).month
    _df['year'] = pd.DatetimeIndex(_df['transaction_date']).year
    _df["year_month"] = _df.apply(lambda x: "{}/{:02d}".format(x.year, x.month), axis=1)
    _df["y_m_d"] = _df.apply(lambda x: "{}/{:02d}".format(x.year_month, x.day), axis=1)
    _df["spend"] = _df.apply(lambda x: generate_spend(x), axis=1)
    return _df


def map_month(val, reverse=False):

    if reverse:
        if isinstance(val, str):
            return d_m[val]
        elif isinstance(val, list):
            return [d_m[e] for e in val]
    else:
        if isinstance(val, int):
            return d_i[val]
        elif isinstance(val, list):
            return [d_i[e] for e in val]
        
def spawn_dropdown(value):
    return html.Div([
        html.Div(d_tr[value]),
        dcc.Dropdown(df[value].unique(), id={"type": "filter-dropdown", "index": trend_cols.index(value)})
    ], style={"width": "15%", 'display': 'inline-block',
              'margin-right': 10})


df = pd.read_csv('https://raw.githubusercontent.com/afkarshad/dash_app/refs/heads/main/digital_wallet_transactions.csv')
df = preprocess_df(df)
df_success = df[df["transaction_status"]=="Successful"].copy()

d_m = {month: index for index, month in enumerate(calendar.month_abbr) if month}
d_i = {index: month for index, month in enumerate(calendar.month_abbr) if month}

trend_cols = ["product_category", "payment_method", "transaction_status", "device_type", "location"]
trend_labels = ["Product Category", "Payment Method", "Transaction Status", "Device Type", "Location"]
d_tr = dict(zip(trend_cols, trend_labels))

unique_users = df.groupby(["year", "month"])["user_id"].nunique().reset_index()
rev = df_success.groupby(["year", "month"])["spend"].sum().reset_index()

daily_users = df.groupby("y_m_d")["user_id"].nunique().reset_index()
avg_daily_users = format(math.floor(daily_users["user_id"].mean()),",")

daily_trx = df.groupby("y_m_d")["spend"].sum().reset_index()
avg_daily_trx = format(math.floor(daily_trx["spend"].mean()),",")

exclude_cols = ["day", "month", "year", "year_month", "y_m_d"]
metrics_options = ["Monthly Active Users", "Daily Active Users", "Monthly Revenue", "Daily Revenue"]

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1('Dashboard'),

    html.A("[dataset]", href="https://www.kaggle.com/datasets/harunrai/digital-wallet-transactions", target="_blank"),
    html.P(),

    html.H2("Overview"),
    html.Div([
        html.Div([
            html.Div("Total Users"),
            html.H2(format(df["user_id"].nunique(),",")),
            html.Div("unique users")
        ], style={"width": "23%", 'display': 'inline-block',
                  'margin-right': 20, "textAlign": "center",
                  "border":"2px black solid"}),
        html.Div([
            html.Div("Lifetime Revenue"),
            html.H2(format(math.floor(df_success["spend"].sum()),",")),
            html.Div("Rupee")
        ], style={"width": "23%", 'display': 'inline-block',
                  'margin-right': 20, "textAlign": "center",
                  "border":"2px black solid"}),
        html.Div([
            html.Div("Avg Daily Active Users"),
            html.H2(avg_daily_users),
            html.Div("unique users")
        ], style={"width": "23%", 'display': 'inline-block',
                  'margin-right': 20, "textAlign": "center",
                  "border":"2px black solid"}),
        html.Div([
            html.Div("Avg Transaction Volume"),
            html.H2(avg_daily_trx),
            html.Div("Rupee")
        ], style={"width": "23%", 'display': 'inline-block',
                  "textAlign": "center", "border":"2px black solid"}),
    ]),
    html.Hr(),
    html.Div([
        html.Div([
            html.Div("Year"),
            dcc.Dropdown([2023, 2024], 2024,
                         id='dropdown-year',
                         clearable=False)
        ], style={"width": "10%", 'display': 'inline-block',
                  'margin-right': 10}),
        html.Div([
            html.Div("Month"),
            dcc.Dropdown({}, id='dropdown-month',
                         clearable=False)
        ], style={"width": "10%", 'display': 'inline-block'}),
        html.Button("Show details", id="btn-vis"),
    ]),

    html.Br(),
    html.Div([
        html.Div([
            html.Div("Monthly Active Users"),
            html.H2(id="monthly-unique-users"),
            html.Div("unique users")
        ], style={"width": "49%", 'display': 'inline-block',
                  'margin-right': 10, "textAlign": "center",
                  "border":"2px black solid"}),
        html.Div([
            html.Div("Monthly Revenue"),
            html.H2(id="monthly-revenue"),
            html.Div("Rupee")
        ], style={"width": "49%", 'display': 'inline-block',
                  "textAlign": "center", "border":"2px black solid"}),
    ]),

    html.Hr(),
    html.Div([
        dash_table.DataTable(
            id='table-content',
            columns=[
                {"name": i, 'id': i} for i in df.columns
                if i not in exclude_cols
            ],
            # filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_size=10,
            style_table={'overflowX': 'scroll'}
        ),
    ], id="table-div", style={"display": "none"}),

    html.Div([
        html.Div([
            dcc.Dropdown(metrics_options, metrics_options[0],
                         id='dropdown-metrics')
        ], style={"width": "30%"}),
        dcc.Graph(id="graph-metrics")
    ]),
    html.Hr(),

    html.H2("Trends"),

    html.Div([
        html.Div([
            html.Div("Filter"),
            dcc.Dropdown(trend_cols, trend_cols[0],
                         id='dropdown-trend-filter')
        ], style={"width": "30%"}),

        html.Div(id="graph-div"),

        html.Div([
            html.Div("Additional Filters"),
            dcc.Dropdown(trend_cols,
                         id='dropdown-trend-filter-options',
                         multi=True)
        ], style={"width": "30%"}),
        html.Div(id="dropdown-trend-filter-selector")
    ])
])

vis_tracker=0
@callback(
    Output('table-div', 'style'),
    Output('btn-vis', 'children'),
    Input('btn-vis', 'n_clicks'),
    State('btn-vis', 'children'),
    prevent_initial_call=True
)
def update_visibility(click, value):
    if click is not None:
        global vis_tracker
        if click != vis_tracker:
            vis_tracker = click
            if value == "Show details":
                return {"display": "inline"}, "Hide details"
            elif value == "Hide details":
                return {"display": "none"}, "Show details"


@callback(
    Output('monthly-unique-users', 'children'),
    Output('monthly-revenue', 'children'),
    Input('dropdown-year', 'value'),
    Input('dropdown-month', 'value')
)
def update_data(year, month):
    m = map_month(month, True)
    u = unique_users[(unique_users["year"]==year)&(unique_users["month"]==m)].user_id.values[0]
    r = rev[(rev["year"]==year)&(rev["month"]==m)].spend.values[0]
    r = math.floor(r)
    return format(u, ','), format(r, ',')

@callback(
    Output('dropdown-month', 'options'),
    Output('dropdown-month', 'value'),
    Input('dropdown-year', 'value')
)
def update_month_options(value):
    list_month = df[df.year==value]["month"].unique().tolist()
    months = map_month(list_month)
    if value == 2024:
        val_month = months[-1]
    else:
        val_month = months[0]
    return months, val_month

@callback(
    Output('graph-metrics', 'figure'),
    Input('dropdown-metrics', 'value'),
)
def update_graph_metrics(value):
    if value == "Monthly Active Users":
        dff = df.groupby("year_month")["user_id"].nunique().reset_index()
        x_col, y_col = "year_month", "total_users"
    elif value == "Monthly Revenue":
        dff = df_success.groupby("year_month")["spend"].sum().reset_index()
        x_col, y_col = "year_month", "Rupee"
    elif value == "Daily Active Users":
        dff = df.groupby("y_m_d")["user_id"].nunique().reset_index()
        x_col, y_col = "year_month_day", "total_users"
    elif value == "Daily Revenue":
        dff = df_success.groupby("y_m_d")["spend"].sum().reset_index()
        x_col, y_col = "year_month_day", "Rupee"
    dff.columns = [x_col, y_col]
    return px.line(dff, x=x_col, y=y_col)

@callback(
    Output('table-content', 'data'),
    Input('dropdown-year', 'value'),
    Input('dropdown-month', 'value')
)
def update_table_data(val1, val2):
    m = map_month(val2, True)
    dff = df[(df.year==val1) & (df.month==m)].copy()
    dff = dff.drop(["day", "month", "year"], axis=1)
    return dff.to_dict('records')

@callback(
    Output('graph-div', 'children', allow_duplicate=True),
    Input('dropdown-trend-filter', 'value'),
    prevent_initial_call="initial_duplicate"
)
def update_graph(value):
    if value is not None:
        df_ = df.copy()
        df_group = df_.groupby([value, "year_month"]).size().reset_index(name="total_transactions")
        line_fig = px.line(df_group, x="year_month", y="total_transactions", color=value)
        return dcc.Graph(figure=line_fig)
    else:
        raise PreventUpdate

@callback(
    Output('dropdown-trend-filter-selector', 'children'),
    Input('dropdown-trend-filter-options', 'value'),
    prevent_initial_call=True
)
def update_selector(value):
    if value is not None:
        dropdown_widgets = [spawn_dropdown(v) for v in value]
        if len(dropdown_widgets) > 0:
            update_button = html.Button("Update graph", id="btn-update")
            dropdown_widgets.append(update_button)
        return dropdown_widgets
    
@callback(
    Output('btn-update', 'disabled'),
    Input({"type": "filter-dropdown", "index": ALL}, "value")
)
def update_btn_state(values):
    state = list(set([v for i,v in enumerate(values)]))
    if state == [None]:
        return True
    return False

graph_click_tracker = 0
@callback(
    Output('graph-div', 'children'),
    Input('btn-update', 'n_clicks'),
    State({"type": "filter-dropdown", "index": ALL}, "value"),
    State({"type": "filter-dropdown", "index": ALL}, "id"),
    State('dropdown-trend-filter', 'value'),
    prevent_initial_call=True
)
def update_graph_2(click, value, ids, main_filter_value):
    if click is not None:
        global graph_click_tracker
        if click != graph_click_tracker:
            graph_click_tracker = click
            dff = df.sort_values("year_month").copy()
            cols = [trend_cols[elm["index"]] for elm in ids]
            # pd.DataFrame({'letter': ['A','B','C']})
            ym = pd.DataFrame({"year_month": df["year_month"].unique().tolist()})
            mf = pd.DataFrame({main_filter_value: df[main_filter_value].unique()})
            cross_ym_mf = pd.merge(ym, mf, how="cross")
            for c, v in zip(cols, value):
                dff = dff[dff[c]==v]
            dff_group = dff.groupby(["year_month", main_filter_value]).size().reset_index(name="total_transactions")
            df_final = pd.merge(cross_ym_mf, dff_group, how="left", on=["year_month", main_filter_value]).fillna(0)
            line_fig = px.line(df_final, x="year_month", y="total_transactions", color=main_filter_value)
            return dcc.Graph(figure=line_fig)
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)
