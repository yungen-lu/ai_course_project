from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from collections import Counter

rus = RandomUnderSampler(random_state=42)


def custom_scoring(y_true, y_pred, verbose=False):
    '''
    customize the scoring function for the severity classification task (4 classes)

    Output:
        macro Beta f1 score
        weighted Beta f1 score
    '''
    report = classification_report(y_true, y_pred, output_dict=True)
    macro_beta_f1 = 0
    weighted_beta_f1 = 0
    beta_weights = {
        '1': 0.5,
        '2': 1,
        '3': 1,
        '4': 2,
    }
    total_data_count = report['weighted avg']['support']
    for cl in range(1, 5):
        pr = report[str(cl)]['precision']
        rc = report[str(cl)]['recall']
        beta = beta_weights[str(cl)]
        beta_f1 = ((1+beta**2)*pr*rc)/(pr*(beta**2) + rc)
        if verbose: 
            print(f'beta f1 for level [{cl}]: {beta_f1}, pr: {pr}, rc: {rc}')

        support_proportion = report[str(cl)]['support'] / total_data_count
        weighted_beta_f1 += beta_f1 * support_proportion
        macro_beta_f1 += beta_f1*0.25

    print(f"macro beta f1: {macro_beta_f1}")
    print(f"weighted beta f1: {weighted_beta_f1}")
    return macro_beta_f1, weighted_beta_f1


def cross_valid(X, y, estimator, cv=5, verbose=False, balance_cls=False):
    '''
    K-Fold cross validation for training data

    Print:

        Average macro f1 score in k-fold
        Average weighted f1 score in k-fold
    '''
    total_macro_beta_f1 = 0
    total_weighted_beta_f1 = 0
    X.reset_index()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    print('Validation data')
    for i, (train_index, valid_index) in enumerate(kf.split(X)):
        x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if balance_cls:
            x_train, y_train = rus.fit_resample(x_train, y_train)
            if verbose:
                print('After under sampling:')
                print(f'Length of training data: {len(x_train)}, and its distribution among each severity {Counter(y_train)}')

        estimator.fit(x_train, y_train)
        y_valid_pred = estimator.predict(x_valid)
        macro_beta_f1, weighted_beta_f1 = custom_scoring(y_valid, y_valid_pred, verbose=False)
        print(f'Round {i} macro beta f1-score: {macro_beta_f1}')
        print(f'Round {i} weighted beta f1-score: {weighted_beta_f1}')
        total_macro_beta_f1 += macro_beta_f1
        total_weighted_beta_f1 += weighted_beta_f1

    avg_macro_betaf1 = total_macro_beta_f1 / cv
    avg_weighted_betaf1 = total_weighted_beta_f1 / cv
    print(f'average macro beta f1-score after kfold: {avg_macro_betaf1}')
    print(f'average weighted beta f1-score after kfold: {avg_weighted_betaf1}')


def test(estimator, x_test, y_test):
    '''
    Output standard classification report and customize scoring output
    '''
    print('Testing data:')
    y_test_pred = estimator.predict(x_test)
    print(classification_report(y_test, y_test_pred))
    custom_scoring(y_test, y_test_pred, verbose=False)


def auc_pr(estimator, x_test, y_test):
    y_scores = estimator.predict_proba(x_test)[:, 1]
    precision = dict()
    recall = dict()
    n_classes = 4
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_scores[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()
