import csv

with open('facebook_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Columns: {", ".join(row)}')
            line_count += 1
        else:
            line_count += 1
    print(f'Processed {line_count} lines.')

input = [[] for i in range(9)]
shares = []

with open('facebook_train.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_file.readline()
    for row in csv_reader:
        for i in range(10):
            if i != 1 and i != 9:
                input[i].append(int(row[i]))
            elif i == 1:
                input[i].append(row[i])
            else:
                shares.append(int(row[i]))

photos = []
links = []
stati = []
videos = []

for i in input[1]:
    if i != "Photo":
        photos.append(0)
    else:
        photos.append(1)

for i in input[1]:
    if i != "Link":
        links.append(0)
    else:
        links.append(1)

for i in input[1]:
    if i != "Status":
        stati.append(0)
    else:
        stati.append(1)

for i in input[1]:
    if i != "Video":
        videos.append(0)
    else:
        videos.append(1)

input.remove(input[1])
for i in [videos, stati, links, photos]:
    input.insert(1, i)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array(input).transpose()
y = np.array(shares)

reg = LinearRegression().fit(X, y)
reg.score(X, y)


with open('facebook_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Columns: {", ".join(row)}')
            line_count += 1
        else:
            line_count += 1
    print(f'Processed {line_count} lines.')

input_test = [[] for i in range(9)]
shares_test = []

with open('facebook_test.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_file.readline()
    for row in csv_reader:
        for i in range(10):
            if i != 1 and i != 9:
                input_test[i].append(int(row[i]))
            elif i == 1:
                input_test[i].append(row[i])
            else:
                shares_test.append(int(row[i]))

photos_test = []
links_test = []
stati_test = []
videos_test = []

for i in input_test[1]:
    if i != "Photo":
        photos_test.append(0)
    else:
        photos_test.append(1)

for i in input_test[1]:
    if i != "Link":
        links_test.append(0)
    else:
        links_test.append(1)

for i in input_test[1]:
    if i != "Status":
        stati_test.append(0)
    else:
        stati_test.append(1)

for i in input_test[1]:
    if i != "Video":
        videos_test.append(0)
    else:
        videos_test.append(1)

input_test.remove(input_test[1])
for i in [videos_test, stati_test, links_test, photos_test]:
    input_test.insert(1, i)

X_test = np.array(input_test).transpose()
y_pred = reg.predict(X_test)

from sklearn import metrics
print(metrics.mean_absolute_error(shares_test, y_pred))
print(metrics.mean_squared_error(shares_test, y_pred))
print(metrics.median_absolute_error(shares_test, y_pred))
print(metrics.r2_score(shares_test, y_pred))


#####casualties
with open('casualty_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Columns: {", ".join(row)}')
            line_count += 1
        else:
            line_count += 1
    print(f'Processed {line_count} lines.')
