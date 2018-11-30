from pyspark import SparkConf, SparkContext
import sys
import re, string
import operator
import json
import requests
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from zipfile import ZipFile
from pyspark.sql import SparkSession, functions, types
from io import *
import csv
import pandas as pd
from urllib.request import *
import getCodeSets as codesets
import loadHousingPrice as housepriceindex
import loadLabourFource as laborfoce
import loadIncomeData as incomedata
import loadDiverseExpenditures as expenditures
import loadTouristInfo as tourist_data
import loadCrimeData as crime_incidents
spark = SparkSession.builder.appName('Load Data Sets').getOrCreate()
#conf = SparkConf().setAppName('reddit etl')
#sc = SparkContext(conf=conf)

def main():
    housing_df = housepriceindex.loadPriceIndex().createOrReplaceTempView("housing")
    income_df = incomedata.loadIncomeData().createOrReplaceTempView("income")
    expenditures_df = expenditures.loadExpenditures().createOrReplaceTempView("expenditures")
    tourism_df = tourist_data.loadTouristInfo().createOrReplaceTempView("tourist_info")
    labourforce_df = laborfoce.loadLabourForceData().createOrReplaceTempView("labour_force")
    crimes_df = crime_incidents.loadCrimeData().createOrReplaceTempView("crimes")

    house_date_transformed = spark.sql("SELECT h.province, h.REF_DATE, h.uom_houseindex, h.scalar_houseindex,h.avg_house_only,h.avg_land_only, h.avg_totalhouseland,SUBSTR(h.REF_DATE , 0, INSTR(h.REF_DATE , '-')-1) as ref_date_temp \
      FROM housing h").createOrReplaceTempView("housing_temp")
    join_houseindex_income = spark.sql("SELECT h.province, h.REF_DATE, h.uom_houseindex, h.scalar_houseindex,h.avg_house_only,h.avg_land_only, h.avg_totalhouseland,l.uom_income, l.scalar_income,l.statistic_income, round(l.avg_income/12,2) as avg_income \
    FROM housing_temp h LEFT JOIN income l ON h.province = l.province AND h.ref_date_temp = l.REF_DATE").createOrReplaceTempView("house_income")


    join_prev_expenditures = spark.sql("SELECT h.province, h.REF_DATE, h.uom_houseindex, h.scalar_houseindex,h.avg_house_only,h.avg_land_only, h.avg_totalhouseland, h.uom_income, h.scalar_income,h.statistic_income,h.avg_income, \
    e.uom_expenditure , e.scalar_expenditure,e.statistic_expenditure,round(e.avg_food_expenditures/12,2) as avg_food_expenditures, round(e.avg_income_taxes/12,2) as avg_income_taxes, round(e.avg_mortageinsurance/12,2) as avg_mortageinsurance, round(e.avg_mortagePaid/12,2) as avg_mortagePaid ,\
    round(e.avg_accomodation/12,2) as avg_accomodation, round(e.avg_rent/12,2) as avg_rent, round(e.avg_shelter/12,2) as avg_shelter, round(e.avg_total_expenditure/12,2) as avg_total_expenditure, round(e.avg_taxes_landregfees/12,2) as avg_taxes_landregfees \
    FROM house_income h LEFT JOIN expenditures e ON h.province = e.province AND SUBSTR(h.REF_DATE , 0, INSTR(h.REF_DATE , '-')-1) = e.REF_DATE ").createOrReplaceTempView("prev_plus_expenditures")
    
    join_prev_touristdata=spark.sql("SELECT h.province, h.REF_DATE, h.uom_houseindex, h.scalar_houseindex,h.avg_house_only,h.avg_land_only, h.avg_totalhouseland, h.uom_income, h.scalar_income,h.statistic_income,h.avg_income, \
    h.uom_expenditure , h.scalar_expenditure,h.statistic_expenditure,h.avg_food_expenditures, h.avg_income_taxes, h.avg_mortageinsurance, h.avg_mortagePaid ,\
    h.avg_accomodation, h.avg_rent, h.avg_shelter, h.avg_total_expenditure, h.avg_taxes_landregfees, \
    t.uom_tourist, t.scalar_tourist,t.avg_international_tourism as avg_international_tourism, t.avg_domestic_tourism as avg_domestic_tourism \
     FROM prev_plus_expenditures h LEFT JOIN tourist_info t ON h.province = t.province AND h.REF_DATE=t.REF_DATE").createOrReplaceTempView("prev_plus_labourforce")

    join_prev_labourforce = spark.sql("SELECT h.province ,h.REF_DATE,h.uom_houseindex, h.scalar_houseindex, h.avg_house_only,h.avg_land_only, h.avg_totalhouseland,h.uom_income, h.scalar_income,h.statistic_income, h.avg_income,\
    h.uom_expenditure,h.scalar_expenditure,h.statistic_expenditure,h.avg_food_expenditures, h.avg_income_taxes, h.avg_mortageinsurance, h.avg_mortagePaid, \
    h.avg_accomodation, h.avg_rent, h.avg_shelter, h.avg_total_expenditure, h.avg_taxes_landregfees,\
    h.uom_tourist, h.scalar_tourist,h.avg_international_tourism, h.avg_domestic_tourism, \
    l.uom_labourforce,l.scalar_labourforce,l.statistic_labourforce,l.avg_employment, l.avg_fulltime, l.avg_labourforce, l.avg_parttime, l.avg_population, l.avg_unemployment,\
    l.uom_lfperc,l.scalar_lfperc,l.statistic_labourforceperc,l.avg_employment_rate,l.avg_participationrate,l.avg_unemploymentrate \
    FROM prev_plus_labourforce h LEFT JOIN labour_force l ON h.province=l.province AND h.REF_DATE=l.REF_DATE").createOrReplaceTempView("prev_plus_crime")

    join_prev_crimes = spark.sql("SELECT h.province ,h.REF_DATE,h.uom_houseindex, h.scalar_houseindex, h.avg_house_only,h.avg_land_only, h.avg_totalhouseland,h.uom_income, h.scalar_income,h.statistic_income, h.avg_income,\
    h.uom_expenditure,h.scalar_expenditure,h.statistic_expenditure,h.avg_food_expenditures, h.avg_income_taxes, h.avg_mortageinsurance, h.avg_mortagePaid, \
    h.avg_accomodation, h.avg_rent, h.avg_shelter, h.avg_total_expenditure, h.avg_taxes_landregfees,\
    h.uom_tourist, h.scalar_tourist,h.avg_international_tourism, h.avg_domestic_tourism, \
    h.uom_labourforce,h.scalar_labourforce,h.statistic_labourforce,h.avg_employment, h.avg_fulltime, h.avg_labourforce, h.avg_parttime, h.avg_population, h.avg_unemployment,\
    h.uom_lfperc,h.scalar_lfperc,h.statistic_labourforceperc,h.avg_employment_rate,h.avg_participationrate,h.avg_unemploymentrate, \
    c.uom_crime, c.scalar_crime, c.statistic_crime,round(c.avg_crime_incidents/12,2) as avg_crime_incidents \
    FROM prev_plus_crime h LEFT JOIN crimes c ON h.province = c.province AND SUBSTR(h.REF_DATE , 0, INSTR(h.REF_DATE , '-')-1) = c.REF_DATE")
    join_prev_crimes.write.csv("data",mode='overwrite',header = 'true') 

if __name__ == '__main__':
    #type_ = sys.argv[1]
    #output = sys.argv[2]
    main()