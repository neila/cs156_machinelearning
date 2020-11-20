#benchmark: 462.libquantum
import csv

times = []
bases = []

with open('benchmarks.txt', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        if row == 0:
            pass
        if row["benchName"] == "462.libquantum":
            times.append(int(row["testID"].split("-")[1][0:4])+int(row["testID"].split("-")[1][4:6])*(1/12))
            bases.append(float(row["base"]))


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

baselog = np.log(bases).reshape(-1, 1)
timesfixed = np.array(times).reshape(-1,1)
reg = LinearRegression().fit(timesfixed, baselog)

preds = []
for i in times:
    preds.append(reg.intercept_[0] + reg.coef_[0][0]*i)

plt.semilogy(times,bases)
plt.plot(times,preds, "r-")
axes = plt.gca()
axes.set_xlim([min(times),max(times)])
axes.set_ylim([min(bases),max(bases)])
plt.title("Semilog plot for CPU performance")
plt.show()



################### MNIST digits
from sklearn import neighbors, datasets
digits = datasets.load_digits()

#3 examples
plt.matshow(digits.images[0])
plt.matshow(digits.images[43])
plt.matshow(digits.images[174])

index = [] #store 1&8 data
count = 0 #index count
for i in digits.target:
    if i == 1 or i == 8:
        index.append(count)
    count += 1

X = digits.data[index]
y = digits.target[index]

ks = range(4, 20, 2)
accuracy = []

#train
for k in ks:
	model = neighbors.KNeighborsClassifier(n_neighbors=k)
	model.fit(X, y)
	score = model.score(X, y)
	accuracy.append(score)

#highest accuracy
i = int(np.argmax(accuracy))
print("k=%d had the highest accuracy of %.2f%%" % (ks[i], accuracy[i] * 100))
model = neighbors.KNeighborsClassifier(n_neighbors=ks[i])
model.fit(X, y)

predictions = model.predict(digits.data)
results = []
for i in range(len(predictions)):
    if predictions[i] == digits.target[i]:
        results.append(1)
    else:
        results.append(0)

print("model accuracy on entire dataset:",sum(results)/len(results)*100,"%")
#Since the model can only make predictions of 1&8, assuming the data has no bias, 0.19 accuracy means most of 1&8 are correctly classified. (and all else wrong)
