# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:36:43 2019

@author: TKHsu
"""

with open("job_listings_USA.csv","w") as f:
    with open("temp_datalab_records_job_listings.csv", encoding="utf-8") as infile:
        for count,line in enumerate(infile):
            data = line.strip().split(",")
            print(count,"\r")
            if len(data)>14:
                if data[10] == 'USA' and data[14]:
                    try:
                        f.write(",".join([data[4]]+data[6:10]+[data[14]])+"\n")
                    except UnicodeEncodeError:
                        continue
            
            
        
