from sklearn import svm
from sklearn import metrics

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import seaborn as sn
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(data=cm)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# define model
clf = Perceptron(eta0=1.0, random_state=0)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    plt.figure(figsize=(10,6))
plt.title("Linear kernel with C=0.1", fontsize=18)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, cmap='cool')
plot_svc_decision_function(clf)