# -*- coding: utf-8 -*-

from glob import glob
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

def convert_to_list(bs4row):
    list_bs4row = bs4row.findAll(["td","th"])
    return [bs4.get_text().strip() for bs4 in list_bs4row]

'''
(1) Get abbreviations for US states
https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations
'''
website_url = requests.get('https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations').text
soup = BeautifulSoup(website_url,'lxml')# get the table
my_table = soup.find("table", { "class":"wikitable sortable"})

# get the table
rows=my_table.findAll("tr")

# convert to list of list
my_data = [convert_to_list(r) for r in rows[12:70]]
df_state = pd.DataFrame(my_data,columns=['Full_name','status','ISO','ANSI1','ANSI2','USPS','USCG','GPO','AP','Other_abv'])
df_state.dropna(subset=['GPO'],inplace=True)
state_abv = dict(zip(df_state['Full_name'].tolist(),df_state['USPS'].tolist()))
abv_state = dict(zip(df_state['USPS'].tolist(),df_state['Full_name'].tolist()))

'''
(2) Get annunal net migration rate for US states
https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=CF
'''

fns = glob('Data/*_PEPTCOMP_with_ann.csv')
col_match_tail = " - Net Migration - Total"
col_match_head = 'Annual Estimates'

dict_year_netimg = {}
for fn in fns:
    df = pd.read_csv(fn,skiprows=1)
    df_state = df[df['Geography'].isin(state_abv.keys())]
    target_col = [col for col in df.columns if col.endswith(col_match_tail) and col.startswith(col_match_head)][0]
    d_temp = dict(zip(df_state['Geography'].tolist(),df_state[target_col].to_list()))
    dict_year_netimg[fn.split("_")[1]] = d_temp
    
df_mig = pd.DataFrame(index = state_abv.keys())
for year in dict_year_netimg:
    df_mig[year] = df_mig.index.map(dict_year_netimg[year])

df_mig.dropna(axis=0,inplace=True)

'''
(3) Get population in US states in 2018
source: https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population
'''
website_url = requests.get('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population').text
soup = BeautifulSoup(website_url,'lxml')# get the table
my_table = soup.find("table", { "class":"wikitable sortable"})

# get the table
rows=my_table.findAll("tr")

# convert to list of list
header = convert_to_list(rows[0])
my_data = [convert_to_list(r) for r in rows[1:]]
df_pop = pd.DataFrame(my_data,columns=header)
list_state = df_mig.index
df_pop = df_pop[df_pop['Name'].isin(list_state)]
d_state_pop2018 = dict(zip(df_pop['Name'].tolist(),[int(pop.replace(',','')) for pop in df_pop['Population estimate, July 1, 2018[4]'].tolist()]))

'''
(4) caluculate population by states for each year
'''
d_year_pop = {2018:d_state_pop2018}

for i in range(8):
    year = 2018 - i
    net_mig = dict(zip(df_mig.index,df_mig[str(year)]))
    d_state_pop = {}
    for state in list_state:
        d_state_pop[state] = d_year_pop[year][state]+net_mig[state]
    d_year_pop[year-1] = d_state_pop
    
'''
(5) caluculate migration rate (per 1000 people) by states
'''

df_rate = pd.DataFrame(index = list_state)
for year in range(2011,2019):
    d_state_rate = {}
    for state in list_state:
        d_state_rate[state] = df_mig.loc[state,str(year)]/d_year_pop[year][state]*1000
    df_rate[year] = df_rate.index.map(d_state_rate)    


def plot_state_mig_rates(df_rate_T,list_state,fig_name):
    
    plt.figure(figsize=(12,2*((len(list_state)+4)//4)))

    for idx,state in enumerate(list_state):
        plt.subplot(len(list_state)//4+1, 4, idx+1)
        ax=sns.lineplot(data=df_rate_T[state])
        ax.set_title(state)
        ax.set_ylim([-30,30])

    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)

df_rate_T =df_rate.transpose()
plot_state_mig_rates(df_rate_T,list_state,'state_migration_rate_ALL.png')

'''
(6) Perform time series clusering to classify state based on their net migration rate
'''

def time_series_clustering_and_plot(df_rate,list_state,fig_name):

    '''
    the code is adapted from:
    https://stackoverflow.com/questions/34940808/hierarchical-clustering-of-time-series-in-python-scipy-numpy-pandas
    '''
    # Do the clustering
    df_temp = df_rate[df_rate.index.isin(list_state)]
    Z = hac.linkage(df_temp, method='single', metric='correlation')

    # Plot dendogram
    plt.figure(figsize=(12,len(list_state)//3))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('States')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        orientation='right',
        leaf_font_size=12,  # font size for the x axis labels
        labels=df_rate.index
    )
    plt.tight_layout()
    plt.savefig(fig_name,dpi=300)
    return Z

Z = time_series_clustering_and_plot(df_rate,list_state,"states_mig_rate_Dendrogram_ALL.png")

def get_dict_sub_cluster(Z,k):

    list_clusters = list(fcluster(Z,k,criterion='maxclust'))
    clusters = list(set(list_clusters))

    d_cluster_states = {}
    for cluster in clusters:
        idxes = [i for i,c in enumerate(list_clusters) if c==cluster]
        d_cluster_states[cluster] = df_rate.index[idxes].tolist()
    return d_cluster_states

d_cluster_states = get_dict_sub_cluster(Z,10)

with open('list_states_clustered_by_migration_pattern.csv','w') as f:
    for cluster in d_cluster_states:
        f.write(",".join([str(cluster),','.join(d_cluster_states[cluster])])+"\n")

for cluster in d_cluster_states:
    plot_state_mig_rates(df_rate_T,d_cluster_states[cluster],"state_cluster_"+str(cluster)+".png")

# get the list of growthing and shrinking states
p_decrease = d_cluster_states['7']
p_increase = d_cluster_states['3']

'''
(7) Collect industry information for companies from Linkedin Profile data 
'''

df_linkedin = pd.read_csv("Data/temp_datalab_records_linkedin_company.csv")
df_linkedin_clean = df_linkedin.dropna(subset=['industry'])
df_linkedin_clean.loc[:,'industry']=df_linkedin_clean['industry'].apply(lambda s: s.replace('&amp;','&'))
d_company_industry = dict(df_linkedin_clean.groupby('company_name')['industry'].apply(list))

'''
(8) Calculate the industrial composition of job postings in 2018 for each US states
'''

df_job = pd.read_csv("Data/job_listings_USA.csv", encoding = "ISO-8859-1",header=None)
df_job.columns = ['Title','Brand','Category','City','State','Date']
df_job['Date'] = pd.to_datetime(df_job['Date'], errors='coerce')
df_job['Industry'] = df_job['Brand'].apply(lambda s: d_company_industry[s][0] if s in d_company_industry else np.nan)
df_job = df_job.dropna(subset=['Industry'])

df_job = pd.read_csv("Data/job_listings_USA.csv", encoding = "ISO-8859-1",header=None)

def get_state(row,abv_state):
    if row['State'] in abv_state:
        return abv_state[row['State']]
    elif row['City'] in abv_state:
        return abv_state[row['City']]
    else:
        return np.nan


df_job.loc[:,'State_Full'] = df_job.apply(lambda row: get_state(row,abv_state), axis=1)
df_job = df_job.dropna(subset=['State_Full'])

# collect job postings in year 2018
df_job_2018 = df_job[df_job['Date'].dt.year==2018]
df_job_2018['Industry'].value_counts()

list_industry = df_job_2018['Industry'].value_counts().index[:20]
list_state = list(set(df_job_2018['State_Full'].tolist()))
df_state_ind_2018 = pd.DataFrame(index=list_state)


for ind in list_industry:
    df_ind = df_job_2018[df_job_2018['Industry']==ind]
    d_state_ind_2018 = dict(df_ind.groupby('State_Full')['Title'].count())
    df_state_ind_2018[ind] = df_state_ind_2018.index.map(d_state_ind_2018)

df_state_ind_2018=df_state_ind_2018.fillna(0)

# Calculate industrial composition for each state
df_state_ind_2018_norm = df_state_ind_2018.div(df_state_ind_2018.sum(axis=1), axis=0)
df_state_ind_2018_norm.dropna(inplace=True)

def get_status(p,p_decrease,p_increase):
    if p in p_decrease:
        return 0
    elif p in p_increase:
        return 1
    else:
        return np.nan
    
df_state_ind_2018_norm['p_status'] = df_state_ind_2018_norm.index
df_state_ind_2018_norm['p_status'] = df_state_ind_2018_norm['p_status'].apply(lambda p: get_status(p,p_decrease,p_increase))
df_state_ind_2018_norm.dropna(subset=['p_status'],inplace=True)

# Calculae the correlation of industrial composition and population growth
df_corr = pd.DataFrame(df_state_ind_2018_norm.corr()['p_status'])
df_corr.dropna(inplace=True)
df_corr = df_corr.head(len(df_corr)-1).sort_values(['p_status'],ascending =False)
df_corr.to_csv('corr_state_job.csv')

# Plot the industries with positive and negative correlation
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
ax = sns.barplot(x='p_status',y=df_corr.tail(5).index, data = df_corr.tail(5),palette="Reds_d",
                 order=df_corr.tail(5).index[::-1])
ax.set_title('negatively correlated with population growth')
plt.subplot(1,2,2)
ax = sns.barplot(x='p_status',y=df_corr.head(5).index, data = df_corr.head(5),palette="Blues_d")
ax.set_title('Positively correlated with population growth')
plt.tight_layout()


list_industry = set(df_job_2018['Industry'].tolist())
df_state_ind = pd.DataFrame(index=list_state)

df_job_2018 = df_job_2018[df_job_2018['Date'].dt.year==2018]
d_state_ind = df_job_2018.groupby('State_Full')['Industry']

for ind in list_industry:
    df_ind = df_job_2018[df_job_2018['Industry']==ind]
    d_state_ind = dict(df_ind.groupby('State_Full')['Title'].count())
    df_state_ind[ind] = df_state_ind.index.map(d_state_ind)
df_state_ind=df_state_ind.fillna(0)

df_state_ind_norm = df_state_ind.div(df_state_ind.sum(axis=1), axis=0)
df_state_ind_norm.dropna(inplace=True)

from sklearn.cluster import KMeans

kclusters = 3
kmeans = KMeans(n_clusters=kclusters).fit(df_state_ind_norm[list_industry].values)

df_state_cluster_by_ind = pd.DataFrame({'code':[state_abv[s] for s in list_state],
                                        'state':list_state,
                                        'clusters':list(kmeans.labels_)})

d_state_clster = dict(zip(df_state_cluster_by_ind['state'].tolist(),df_state_cluster_by_ind['clusters'].tolist()))
df_job_2018.loc[:,'cluster'] = df_job_2018['State_Full'].apply(lambda s: d_state_clster[s] if s in d_state_clster else np.nan)

df_cluster_ind_2018 = pd.DataFrame(index=list(range(3)))

for ind in list_industry:
    df_ind = df_job_2018[df_job_2018['Industry']==ind]
    d_c_ind = dict(df_ind.groupby('cluster')['Title'].count())
    df_cluster_ind_2018[ind] = df_cluster_ind_2018.index.map(d_c_ind)

df_cluster_ind_2018=df_cluster_ind_2018.fillna(0)    
df_cluster_ind_2018_norm = df_cluster_ind_2018.div(df_cluster_ind_2018.sum(axis=1), axis=0)
df_cluster_ind_2018_norm.transpose().to_csv('industry_dis_by_cluster.csv')

neg_corr = df_corr.tail(5).index
pos_corr = df_corr.head(5).index

def plot_job_dist_by_corr(df_cluster_ind_2018_norm,list_corr,file_name):
    plt.figure(figsize=(12,len(list_corr)//5*3))
    for i,industry in enumerate(list_corr):
        plt.subplot(len(list_corr)//5, 5, i+1)
        ax=sns.barplot(data=[[df_cluster_ind_2018_norm.loc[0,industry]],
                             [df_cluster_ind_2018_norm.loc[1,industry]],
                             [df_cluster_ind_2018_norm.loc[2,industry]],
                            ])
        ax.set_title(industry)

    plt.tight_layout()
    plt.savefig(file_name,dpi=300)

plot_job_dist_by_corr(df_cluster_ind_2018_norm,neg_corr,'job_distr_of_negative_corr.png')
plot_job_dist_by_corr(df_cluster_ind_2018_norm,pos_corr,'job_distr_of_positive_corr.png')

dict_cluster_statelist = dict(df_state_cluster_by_ind.groupby('clusters')['state'].apply(list))
dict_cluster_statelist