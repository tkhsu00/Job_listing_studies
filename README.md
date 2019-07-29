# Job_listing_studies


### __DATA ACQUISITION__

(1) Job Postings and Linkedin Profiles data sets from Thinknum:\
https://blog.thedataincubator.com/tag/data-sources/

(2) Annunal net migration rate for US states\
https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=CF

(3) Population in US states in 2018\
https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population

(4) Abbreviations for US states\
https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations

### __INTRODCUTION__

Population growth rate is a strong indicator of economic growth for US states. Population decline has a negative impact on economic growth due to labor shortage and low tax revenue. Here I compare the industrial composition of job postings between growing states and shrinking states, aiming to identify the industries that are positively or negatively correlated with population growth.

### __Methodology__

(1) In order to determine the growing states and shrinking states, I performed time-series cluster analysis to classify US states according to their net migration rate from 2011 to 2018.\ 
(2) I then processed the job posting and Linkedin Profiles data from Thinknum. From Linkedin Profiles dataset I collected the industry information for 5018 companies, and use these information to group the companies from 1.2 million job posting into different fields. I calculated the industrial composition of job postings in 2018 for each US states, and correlated the industry composition to population growth.


### __Results and Discussion__
(1) The results showed US states including Nevada, Idaho, and Arizona were experiencing population growth from year 2011 to 2018, while Puerto Rico, Alaska, Wyoming were experiencing population decline.\
(2) The industries such as ¡§logistics and supply chain¡¨, ¡§construction¡¨, and ¡§automotive¡¨ are positively correlated with population growth, while ¡§government administration¡¨, ¡§broadcast media¡¨, ¡§restaurant¡¨ are negatively correlated with population growth.\
(3) The future direction of this project is to use industrial composition of job posting data to predict population growth for each states.
