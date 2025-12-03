from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import math

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

X_train = mnist_train.data.numpy().astype("float32") / 255.0
y_train = mnist_train.targets.numpy()

X_test = mnist_test.data.numpy().astype("float32") / 255.0
y_test = mnist_test.targets.numpy()

def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

x_train = zero_pad(X_train, 1)
x_test = zero_pad(X_test, 1)

def convolute(x, conv1, conv2, linear1, linear2):
    res1 = np.zeros((32, 28, 28))
    for k in range(32):
      for i in range(28):
          for j in range(28):
              res1[k][i][j] += np.dot(conv1[k], x[i:i+3][j:j+3])
    res2 = np.zeros((64, 26, 26))
    res1 = res1.transpose(1, 2, 0)
    for k in range(64):
        for i in range(26):
            for j in range(26):
                res2[k][i][j] += np.dot(conv2[k], res1[i:i+3][j:j+3])
    res3 = np.zeros((64, 24, 24))
    for k in range(64):
        for i in range(26):
            for j in range(26):
                res3[k][i][j] = max(res2[i][j], res2[i][j+1], res2[i+1][j], res2[i+1][j+1])
    res4 = res3.reshape(-1) #64*26*26
    res5 = np.dot(res4, linear1)
    res6 = np.dot(res5, linear2)
    return res6

def loss(x, y):
    tar = np.zeros((y))
    tar[y] = 1
    s = 0
    for d in x:
        s += math.pow(math.e, d)
    l = -x[y] + math.log(s)
    return l

conv1 = np.random.rand(32, 3, 3)*np.sqrt(2/9)
conv2 = np.random.rand(64, 3, 3, 32)*np.sqrt(1/144)
linear1 = np.random.rand(100)
linear2 = np.random.rand(10)

#training 'em
print("Training...")
epochs = 5
lr = 0.1

for epoch in epochs:
    print("epoch:", epoch)
    for i in range(60000):
        img = x_train[i]
        y_logits = convolute(img, conv1=conv1, conv2=conv2, linear1=linear1, linear2=linear2)
        l = loss(y_logits, y_train[i])
        new_conv1 = conv1.copy()
        new_conv2 = conv2.copy()
        new_linear1 = linear1.copy()
        new_linear2 = linear2.copy()
        print("Updating conv1...")
        for k in range(32):
            for i in range(3):
                for j in range(3):
                    h = 1e-6
                    hconv1 = conv1.copy()
                    hconv1[k][i][j] += h
                    lh = convolute(img, conv1=hconv1, conv2=conv2, linear1=linear1, linear2=linear2)
                    grad = math.abs(l - lh)/h
                    new_conv1[k][i][j] -= lr * grad
        print("Updating conv2...")
        for l in range(64):
            for i in range(3):
                for j in range(3):
                    for k in range(32):
                        h = 1e-6
                        hconv2 = conv2.copy()
                        hconv2[l][i][j][k] += h
                        lh = convolute(img, conv1=conv1, conv2=hconv2, linear1=linear1, linear2=linear2)
                        grad = math.abs(l - lh)/h
                        new_conv2[l][i][j][k] -= lr * grad
        print("Updating linear1...")
        for i in range(100):
            h = 1e-6
            hlinear1 = linear1.copy()
            hlinear1[i] += h
            lh = convolute(img, conv1=conv1, conv2=conv2, linear1=hlinear1, linear2=linear2)
            grad = math.abs(l - lh)/h
            new_linear1[i] -= lr*grad
        print("Updating linear2...")
        for i in range(10):
            h = 1e-6
            hlinear2 = linear2.copy()
            hlinear2[i] += h
            lh = convolute(img, conv1=conv1, conv2=conv2, linear1=linear1, linear2=hlinear2)
            grad = math.abs(l - lh)/h
            new_linear2[i] -= lr*grad
        conv1 = new_conv1.copy()
        conv2 = new_conv2.copy()
        linear1 = new_linear1.copy()
        linear2 = new_linear2.copy()

#testing 'em
print("Testing...")
correct = 0
for i in range(10000):
    img = x_test[i]
    target = y_test[i]
    y_logits = convolute(img, conv1=conv1, conv2=conv2, linear1=linear1, linear2=linear2)
    pred = np.argmax(y_logits)
    if (pred == target):
        correct += 1
print("Correct percentage: ", correct/100)