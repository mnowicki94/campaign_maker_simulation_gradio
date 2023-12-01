import gradio as gr

import sqlite3
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def create_coefficients_from_model(revenue):
    # IMPORT DATA

    # Create a SQL connection to our SQLite database
    dbfile = './Marketing_DS_task_1.db'
    con = sqlite3.connect(dbfile)
    cur = con.cursor()

    # reading all table names
    table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

    spend_data = pd.read_sql_query('SELECT * FROM spend_data', con)
    revenue_data = pd.read_sql_query('SELECT * FROM revenue_data', con)

    con.close()

    # prepare dataset

    spend_data_to_join = spend_data.drop('country', axis=1)

    df = pd.merge(revenue_data, spend_data_to_join,
                  on=['platform', 'country_group', 'channel', 'media_type', 'install_month'], how='left')

    # Filter data to those that have revenue after 30 days as we focus on 30 days revenue, not older

    df = df[df['install_month'] != '2023-10-01']

    df = df.drop(['D60_Revenue', 'D90_Revenue', 'D120_Revenue', 'D150_Revenue', 'D180_Revenue'], axis=1)

    df = df.dropna(how='any', axis=0)

    df_for_model = df.copy()

    df_for_model['key'] = df_for_model['install_month'] + '_' + df_for_model['campaign']

    df_for_model = df_for_model.drop(['channel', 'campaign', 'install_month'], axis=1)

    df_for_model = pd.get_dummies(df_for_model, columns=['platform', 'media_type', 'country_group'])

    df_for_model = df_for_model.drop('media_type_Unknown', axis=1)

    df_for_model = df_for_model.groupby('key').agg({
        revenue: "sum",
        'spend': "sum",
        'platform_android': "max",
        'platform_ios': "max",
        'media_type_Demand-Side-Platform': "max",
        'media_type_Incentivized': "max",
        'media_type_Pre-installed': "max",
        'media_type_Search': "max",
        'media_type_Social': "max",
        'media_type_Video Networks': "max",
        'country_group_Group_1': "max",
        'country_group_Group_2': "max",
        'country_group_Group_3': "max",
    })

    # setting up X and y dataset

    x = df_for_model.drop(revenue, axis=1)

    y = df_for_model[[revenue]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # basic linear regression model
    # lin=LinearRegression()

    np.random.seed(54321)

    x_train = sm.add_constant(x_train)

    mod = sm.OLS(endog=y_train, exog=x_train, hasconst=True)

    res = mod.fit()

    r2 = res.rsquared

    print('r2: ' + str(r2))

    coefficients = res.params
    coefficients = coefficients.reset_index()

    coefficients['column'] = coefficients['index'].astype(str) + '_coeff'
    coefficients = coefficients.drop('index', axis=1)
    coefficients = coefficients.rename(columns={0: 'coeff'})
    coefficients = coefficients[['column', 'coeff']]

    print(coefficients)
    return coefficients



def campaign_maker(spend,platform,country_group,media_type):

    coeffs = create_coefficients_from_model(revenue='D7_Revenue')

    const_coeff = coeffs[coeffs.column == 'const_coeff'].coeff.item()
    spend_coeff = coeffs[coeffs.column == 'spend_coeff'].coeff.item()
    platform_android_coeff = coeffs[coeffs.column == 'platform_android_coeff'].coeff.item()
    platform_ios_coeff = coeffs[coeffs.column == 'platform_ios_coeff'].coeff.item()
    media_type_Demand_Side_Platform_coeff = coeffs[coeffs.column == 'media_type_Demand-Side-Platform_coeff'].coeff.item()
    media_type_Incentivized_coeff = coeffs[coeffs.column == 'media_type_Incentivized_coeff'].coeff.item()
    media_type_Pre_installed_coeff = coeffs[coeffs.column == 'media_type_Pre-installed_coeff'].coeff.item()
    media_type_Search_coeff = coeffs[coeffs.column == 'media_type_Search_coeff'].coeff.item()
    media_type_Social_coeff = coeffs[coeffs.column == 'media_type_Social_coeff'].coeff.item()
    country_group_Group_1_coeff = coeffs[coeffs.column == 'country_group_Group_1_coeff'].coeff.item()
    country_group_Group_2_coeff = coeffs[coeffs.column == 'country_group_Group_1_coeff'].coeff.item()
    country_group_Group_3_coeff = coeffs[coeffs.column == 'country_group_Group_1_coeff'].coeff.item()

    platform_ios = float(platform_ios_coeff) if 'ios' in platform else 0
    platform_android = float(platform_android_coeff) if 'android' in platform else 0

    country_group_1 = float(country_group_Group_1_coeff) if 'country_group1' in country_group else 0
    country_group_2 = float(country_group_Group_2_coeff) if 'country_group2' in country_group else 0
    country_group_3 = float(country_group_Group_3_coeff) if 'country_group3' in country_group else 0

    demand_side_platform = float(media_type_Demand_Side_Platform_coeff) if 'demand_side_platform' in media_type else 0
    search = float(media_type_Search_coeff) if 'search' in media_type else 0
    social = float(media_type_Social_coeff) if 'social' in media_type else 0
    incentivized = float(media_type_Incentivized_coeff) if 'incentivized' in media_type else 0
    preinstalled = float(media_type_Pre_installed_coeff) if 'preinstalled' in media_type else 0

    return float(const_coeff) + (int(spend)*float(spend_coeff)) + platform_ios + platform_android + country_group_1 + country_group_2 + country_group_3\
        + demand_side_platform +  search + social + incentivized + preinstalled


demo = gr.Interface(
    campaign_maker,
    [
        gr.Slider(1, 1000000, value=50000, label="Campaign spend [PLN]", info="Insert budget for campaign"),

        gr.CheckboxGroup(["ios", "android"], label="Platform", info="What platform/s?"),

        gr.CheckboxGroup(["country_group1", "country_group2","country_group3"], label="Country group", info="Which country group/s will be targeted in the campaign?"),

        gr.CheckboxGroup(["demand_side_platform", "search", "social", "incentivized","preinstalled"], label="Media type/s",
                         info="Which media types you want to use in campaign?"),

    ],
    "text",

    title='Campaign Maker',
    description='Please insert budget, country and media types you want to use in your campaign and click button to predict estimated revenue after 7 days based on regression model'
                ' trained on the provided data from years 2022-2023. ',
)

if __name__ == "__main__":
    demo.launch()