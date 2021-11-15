import numpy as np 
from sklearn.naive_bayes import GaussianNB
import timeit
import matplotlib.pyplot as plt

def Bayesian(X_train,X_test,Y_train,Y_test,dim=2):
    bayes = GaussianNB()

    start = timeit.default_timer()
    bayes.fit(X_train,Y_train)
    stop = timeit.default_timer()
    
    preds = bayes.predict(X_test)
    accuracy = (preds==Y_test).mean()

    preds_train = bayes.predict(X_train)
    accuracy_train = (preds_train==Y_train).mean()
    


    print('Bayesian Time: ', stop - start) 
    print("Bayesian train:",accuracy_train)
    print("Bayesian test accuracy: ",accuracy)
    if(dim==2):
        # predict the classification probabilities on a grid
        xlim = (-4 ,4)
        ylim = (-8, 11)
        x = np.linspace(-4,4,50)
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                            np.linspace(ylim[0], ylim[1], 81))
        Z = bayes.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        
        g = 1.63*xx**2 + 1.63*yy**2 - 4.73*xx*yy - 0.13
        #------------------------------------------------------------
        # Plot the results
        fig = plt.figure(figsize=(5, 3.75))
        ax = fig.add_subplot(111)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, zorder=2)

        ax.contour(xx, yy, Z, [0.5], colors='k')
        ax.contour(xx,yy,g,[0],colors='red',label="Derived")
        # ax.plot(x,g,label="Derived")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.legend()
        plt.show()

means = [[0,0],[0,0]]
covs =  [[[1, 0.9], [0.9, 1]], [[0.5, 0], [0, 0.5]]]


# print(covs)

prior1 = 0.5
prior2 = 0.5

#No. of datas
train_samples = 2000
test_samples = 1000

#Training data
X_train = []

sample1_train = int(train_samples * prior1)
sample2_train = int(train_samples * prior2)
Y_train = np.append(np.zeros(sample1_train),np.ones(sample2_train))

# samples = [sample1_train,sample2_train]

# for mean, cov, sample in zip(means,covs,samples):
#   x = np.random.multivariate_normal(mean, cov, sample)
#   X_train += list(x)

# X_train = np.asarray(X_train)

# np.save("10dxtrain",X_train)

X_train = np.load("ex2dxtrain.npy")


#Testing data
X_test = []

sample1_test = int(test_samples * prior1)
sample2_test = int(test_samples * prior2)
Y_test = np.append(np.zeros(sample1_test),np.ones(sample2_test))

# samples = [sample1_test,sample2_test]

# for mean, cov, sample in zip(means,covs,samples):
#   x = np.random.multivariate_normal(mean, cov, sample)
#   X_test += list(x)

# X_test = np.asarray(X_test)

# np.save("10dxtest",X_test)

X_test = np.load("ex2dxtest.npy")

Bayesian(X_train,X_test,Y_train,Y_test)