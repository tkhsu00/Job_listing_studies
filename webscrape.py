# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:20:44 2019

@author: TKHsu
"""
import pandas as pd
from bs4 import BeautifulSoup
import requests

# Function: Convert BeautifulSoup tags to string list
def convert_to_list(bs4row):
    list_bs4row = bs4row.findAll(["td","th"])
    return [bs4.get_text().strip() for bs4 in list_bs4row]

website_url = requests.get('https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations').text

# get the table
soup = BeautifulSoup(website_url,'lxml')
#my_table = soup.find(lambda tag: tag.name=='TABLE' and tag.has_attr('className') and tag['className']=="wikitable sortable jquery-tablesorter") 

my_table = soup.find("table", { "class":"wikitable sortable"})
# get the table
rows=my_table.findAll("tr")

# convert to list of list
my_data = [convert_to_list(r) for r in rows[12:]]
df_state = pd.DataFrame(my_data,columns=['Full_name','status','ISO','ANSI1','ANSI2','USPS','USCG','GPO','AP','Other_abv'])
df_state.to_csv('WebScraping_state_abv.csv',index=False)