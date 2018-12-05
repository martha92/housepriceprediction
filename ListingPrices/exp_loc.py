import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+

def main(inputs,output):
    # main logic starts here
    listing_schema = types.StructType([
    types.StructField('province', types.StringType(), False),
    types.StructField('listprice', types.StringType(), False),
    types.StructField('date_added', types.StringType(), False),
    types.StructField('locality', types.StringType(), False),
    types.StructField('postal_code', types.StringType(), False),
    types.StructField('year_built', types.StringType(), True),
    types.StructField('taxes', types.StringType(), True),
    types.StructField('Basement', types.StringType(), True),
    types.StructField('Lot_size', types.StringType(), True),
    types.StructField('Bed', types.StringType(), True),
    types.StructField('Baths', types.StringType(), True),
    types.StructField('PriceperSqft', types.StringType(), True),
    types.StructField('monthyr', types.StringType(), True)
    ])
    listingdata = spark.read.csv(inputs, schema=listing_schema).createOrReplaceTempView('listingdata')
    listingdata1 = spark.sql('select * from listingdata where locality is not null').createOrReplaceTempView('listingdata1')
    listingdata2 = spark.sql('select province, avg(PriceperSqft) as avgprice, locality from listingdata1 group by province,locality').createOrReplaceTempView('listingdata2')
    listingdata3 = spark.sql('select max(avgprice) as maxavg, province from listingdata2 group by province').createOrReplaceTempView('listingdata3')
    listingdata4 = spark.sql('select listingdata3.province, listingdata3.maxavg, listingdata2.locality from listingdata3,listingdata2 where listingdata3.maxavg = listingdata2.avgprice').coalesce(1)
    listingdata4.write.format("csv").save(output)
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs,output)



   