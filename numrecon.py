import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digit = datasets.load_digits()
clf = svm.SVC(gamma = 0.001, C=100)
x,y = digit.data[:-10], digit.target[:-10]
print(x.shape)
clf.fit(x,y)
plt.imshow(digit.images[-10], cmap= plt.cm.gray, interpolation= "nearest") # change digits.images[] to any other number for other prediction
plt.show()
print('Prediction:',clf.predict(digit.data[-10].reshape(1,-1))) # digits.data[] should be same as digits.image[]