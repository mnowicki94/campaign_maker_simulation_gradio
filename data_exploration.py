import sqlite3
import pandas as pd
import numpy as np
import sys
import gradio as gr
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)



#TASK
# ● Understand the overall Marketing performance picture
# ● Estimate future returns from investment by preparing a prediction or simulation (you can pick one or more models of your choice)
# ● Draw conclusions & shape possible next steps to be taken

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


#SPEND DATA ANALYSIS

stdoutOrigin=sys.stdout
sys.stdout = open("spend_data_stats.txt", "w")


for var in spend_data.columns:

    if var == 'spend':
        print('mean spend: '+str(spend_data[var].mean()))
        print('median spend: ' +str(spend_data[var].median()))
        print('quantiles spend: ' + str(spend_data[var].quantile([0.25, 0.5, 0.75, 1])))

    if var == 'country':
        continue


    else:
        print(var)
        # print ('value counts: ')
        # print(spend_data[var].value_counts(normalize=True) * 100)

        print ('generated spends: ')
        print(spend_data.groupby(var).sum('spend').sort_values(['spend'], ascending=False))



#end of saving logs
sys.stdout.close()
sys.stdout=stdoutOrigin

#REVENUE DATA ANALYSIS

stdoutOrigin=sys.stdout
sys.stdout = open("revenue_data_stats.txt", "w")


for var in revenue_data.columns:

    print(var)
    # print ('value counts: ')
    # print(revenue_data[var].value_counts(normalize=True) * 100)

    print ('generated revenue: ')
    print(revenue_data.groupby(var).sum(['installs','D1_Revenue','D7_Revenue']).sort_values(['installs','D1_Revenue','D7_Revenue'], ascending=False))



#end of saving logs
sys.stdout.close()
sys.stdout=stdoutOrigin


#checks:

country_groups = spend_data.groupby('country')['country_group'].first()

#join

spend_data_to_join = spend_data.drop('country',axis=1)


df = pd.merge(revenue_data, spend_data_to_join , on=['platform','country_group','channel','media_type','install_month'], how='left')

# Filter data to those that have revenue after 30 days as we focus on 30 days revenue, not older

df = df[df['install_month'] != '2023-10-01']

df = df.drop(['D60_Revenue','D90_Revenue','D120_Revenue','D150_Revenue','D180_Revenue'],axis=1)

df = df.dropna(how='any',axis=0)

df_campaigns = df.groupby('campaign').agg({'spend': "sum",'installs': "sum",'D1_Revenue': "sum",'D7_Revenue': "sum",'D14_Revenue': "sum",'D30_Revenue': "sum"})


df_campaigns['spend_installs_ratio'] = df_campaigns['installs'] / df_campaigns['spend']
df_campaigns['spend_D1_Revenue_ratio'] = df_campaigns['D1_Revenue'] / df_campaigns['spend']
df_campaigns['spend_D7_Revenue_ratio'] = df_campaigns['D7_Revenue'] / df_campaigns['spend']
df_campaigns['spend_D14_Revenue_ratio'] = df_campaigns['D14_Revenue'] / df_campaigns['spend']
df_campaigns['spend_D30_Revenue_ratio'] = df_campaigns['D30_Revenue'] / df_campaigns['spend']

print(df_campaigns)

