from pyspark import SparkContext
from pyspark import SparkConf
from operator import add

sc = SparkContext("local[*]", "First App")

m = sc.parallelize([1, 2, 3, 4, 5])

d = m.zip(m).map(lambda record : record[0]*record[1])


print(d.collect())