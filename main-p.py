import random

from pyspark import SparkContext
from pyspark import SparkConf
from operator import add, sub, div, mul, pow
import math
from decimal import Decimal

sc = SparkContext("local[*]", "First App")
epsilon = sc.broadcast(1e-5)
loss_list = []

def y_hat_calculator(rdd_records):
    return rdd_records.map(lambda record: (record[0], record[1], record[2], sum(list(map(mul, record[0], record[2])))))

def y_hat_deriviation(rdd_records, learning_rate, lambda_reg):
    rdd =  rdd_records.map(lambda record : [(record[3]-record[1])*element/4601 for element in record[0]])\
                      .reduce(lambda record_1, record_2: map(add, record_1, record_2))
    weight_rdd = [(lambda_reg/4601) * element for element in rdd_records.take(1)[0][2]]
    return sc.parallelize([list(map(add, rdd, weight_rdd))]).map(lambda record: record)

def treshhold(record):
    return 1.0 if record >= 0.5 else 0.0

def batch_predict(rdd_XyW):
    return y_hat_calculator(rdd_XyW).map(lambda record: (record[0], record[1], record[2], treshhold(record[3])))

def batch_detailed_accuracy_check(rdd_XyW):
    y_yhat = batch_predict(rdd_XyW).map(lambda record: (record[1], record[3]))
    true_positive = y_yhat.filter(lambda record: record[0]==1.0 and record[1]==1.0)
    false_positive = y_yhat.filter(lambda record: record[0] == 0.0 and record[1] == 1.0)
    false_negative = y_yhat.filter(lambda record: record[0] == 1.0 and record[1] == 0.0)
    true_negative = y_yhat.filter(lambda record: record[0] == 0.0 and record[1] == 0.0)
    Precision = float(true_positive) / float(true_positive + false_positive)
    Recall = float(true_positive) / float(false_negative + true_positive)
    print("Precision : " + str(Precision))
    print("Recall : " + str(Recall))

def cost_function(rdd_records, lambda_reg):
    with_y_hat = y_hat_calculator(rdd_records)
    total_number= rdd_records.count()
    process_log = with_y_hat\
        .map(lambda record: record[1] * Decimal(abs(record[3])).ln().__float__()   +   (1-record[1])* Decimal(abs(1-record[3])).ln().__float__()  )\
        .reduce(lambda record_1, record_2: record_1+record_2)
    process_weight = sc.parallelize(rdd_records.take(1)[0][2]).map(lambda element: element**2)\
        .reduce(lambda record_1, record_2: record_1+record_2)
    return (-1/total_number)*process_log + (lambda_reg/(2*total_number))*process_weight


# Read File
text_file = sc.textFile("spam.data")
records = text_file.map(lambda line: line.split(" "))\
                  .map(lambda record_str : map(float, record_str))\
                  .map(lambda record: (record[0:57], record[57]))
# Normalization
records_X = records.map(lambda record : record[0])
records_y = records.map(lambda record : record[1])
records_X_element_wise_sum = records_X.reduce(lambda record_1, record_2 : list(map(add, record_1, record_2)))
records_X_mean = sc.parallelize([[x / records_X.count() for x in records_X_element_wise_sum]])
records_X_element_mean_diff = records_X\
    .cartesian(records_X_mean)\
    .map(lambda record : list(map(sub, record[0], record[1])))
records_X_element_square_mean_diff = records_X_element_mean_diff.map(lambda record : [ x**2 for x in record])
records_X_element_sum_square_mean_diff = records_X_element_square_mean_diff.reduce(lambda record_1, record_2 : map(add, record_1, record_2))
records_X_standard_deviation = sc.parallelize([[ x / math.sqrt(records_X.count() - 1) for x in records_X_element_sum_square_mean_diff]])
record_X_normalized = records_X_element_mean_diff\
    .cartesian(records_X_standard_deviation)\
    .map(lambda record : list(map(div, record[0], record[1])))
records_with_mean = records.cartesian(records_X_mean).map(lambda record : (record[0][0], record[0][1], record[1]))
records_with_mean_std = records_with_mean.cartesian(records_X_standard_deviation).map(lambda record : (record[0][0], record[0][1],record[0][2], record[1]))
records_normalized = records_with_mean_std.map(lambda record: (list(map(div, list(map(sub, record[0], record[2])), record[3])), record[1]))


records_bayas = records_normalized.map(lambda record: (record[0] + [1], record[1]))
# Fix the Warning Message Here here
feature_size = len(records_bayas.take(1)[0][0])
weight_list = [float(random.random()) for i in range(feature_size)]
weight = sc.parallelize([weight_list])
record_weight = records_bayas.cartesian(weight).map(lambda record : (record[0][0], record[0][1], record[1]))

for i in range(20):
    record_y_hat = y_hat_calculator(record_weight)
    deriviation = y_hat_deriviation(record_y_hat, 0.25, 0.15)
    record_weight = record_y_hat.cartesian(deriviation)\
        .map(lambda record: (record[0][0], record[0][1], list(map(sub, record[0][2], record[1]))))
    loss_list.append(cost_function(record_weight, 0.15))

batch_detailed_accuracy_check(record_weight)




