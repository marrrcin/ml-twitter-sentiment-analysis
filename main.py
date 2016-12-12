import multiprocessing
from multiprocessing import Process
from time import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier as XGBoostClassifier
from sklearn.svm import LinearSVC

from cleanup import TwitterCleanuper
from preprocessing import TwitterData
from word2vec import Word2VecProvider


def preprocess(results, data_path, is_testing, data_name, min_occurrences=5, cache_output=None):
    twitter_data = TwitterData()
    twitter_data.initialize(data_path, is_testing)
    twitter_data.build_features()
    twitter_data.cleanup(TwitterCleanuper())
    twitter_data.tokenize()
    twitter_data.stem()
    twitter_data.build_wordlist(min_occurrences=min_occurrences)
    #twitter_data.build_data_model()
    # twitter_data.build_ngrams()
    # twitter_data.build_ngram_model()
    # twitter_data.build_data_model(with_ngram=2)
    # word2vec = Word2VecProvider()
    # word2vec.load("H:\\Programowanie\\glove.twitter.27B.200d.txt")
    # twitter_data.build_word2vec_model(word2vec)
    if cache_output is not None:
        twitter_data.data_model.to_csv(cache_output, index_label="idx", float_format="%.6f")
    results[data_name] = twitter_data.data_model


def preprare_data(min_occurrences):
    import os
    training_data = None
    testing_data = None
    print("Loading data...")
    test_data_file_name = "data\\processed_test_word2vec_bow_" + str(min_occurrences) + ".csv"
    train_data_file_name = "data\\processed_train_word2vec_bow_" + str(min_occurrences) + ".csv"
    use_cache = os.path.isfile(train_data_file_name) and os.path.isfile(
        test_data_file_name)
    if use_cache:
        training_data = TwitterData()
        training_data.initialize(None, from_cached=train_data_file_name)
        training_data = training_data.data_model

        testing_data = TwitterData()
        testing_data.initialize(None, from_cached=test_data_file_name)
        testing_data = testing_data.data_model
        print("Loaded from cached files...")
    else:
        print("Preprocessing data...")
        with multiprocessing.Manager() as manager:

            results = manager.dict()

            preprocess_training = Process(target=preprocess, args=(
                results, "data\\train.csv", False, "train", min_occurrences, train_data_file_name,))

            preprocess_testing = Process(target=preprocess, args=(
                results, "data\\test.csv", True, "test", min_occurrences, test_data_file_name,))

            preprocess_training.start()
            preprocess_testing.start()
            print("Multiple processes started...")

            preprocess_testing.join()
            print("Preprocessed testing data...")

            preprocess_training.join()
            print("Preprocessed training data...")

            training_data = results["train"]
            testing_data = results["test"]

            print("Data preprocessed & cached...")

    return training_data, testing_data


def log(text):
    print(text)
    with open("log.txt", "a") as log_file:
        log_file.write(str(text) + "\n")


def test_classifier(X_train, y_train, X_test, y_test, classifier):
    log("")
    log("===============================================")
    classifier_name = str(type(classifier).__name__)
    log("Testing " + classifier_name)
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    log("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    log("Predicting time {0}s".format(time() - now))

    from sklearn.metrics import classification_report
    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    log("=================== Results ===================")
    log(classification_report(y_test, predictions, labels=list_of_labels))
    log("            Negative     Neutral     Positive")
    log("F1       " + str(f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)))
    log("Precision" + str(precision))
    log("Recall   " + str(recall))
    log("Accuracy " + str(accuracy))
    log("===============================================")

    return precision, recall, accuracy


def cv(classifier, X_train, y_train):
    log("")
    log("===============================================")
    classifier_name = str(type(classifier).__name__)
    log("Testing " + classifier_name)
    now = time()

    log("Crossvalidating...")
    # recall = [cross_val_score(classifier, X_train, y_train, scoring="recall_micro", cv=10, n_jobs=-1)]
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    # precision = [cross_val_score(classifier, X_train, y_train, scoring="precision_micro", cv=10, n_jobs=-1)]
    recall = -1
    precision = -1
    log("Crosvalidation completed in {0}s".format(time() - now))
    log("=================== Results ===================")
    log("Accuracy: " + str(accuracy))
    log("Precision: " + str(precision))
    log("Recall: " + str(recall))
    log("===============================================")
    log("CV time: {0}".format(time() - now))
    return accuracy, precision, recall


import numpy as np


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            log("Model with rank: {0}".format(i))
            log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            log("Parameters: {0}".format(results['params'][candidate]))
            log("")


def best_fit(X_train, y_train):
    log("")

    seed = 666
    import time as ttt
    attributes = len(X_train.columns)
    examples = len(X_train)
    now = time()
    log(ttt.ctime())
    # Parameters for SVM
    # parameters = {
    #     "dual": [True, False],
    #     "tol": [1e-3, 1e-4, 1e-5],
    #     "C": [1.0, 1.5, 2.0, 5.0, 10, 100, 1000]
    # }
    # rand_search = RandomizedSearchCV(LinearSVC(max_iter=5000), param_distributions=parameters, cv=8,n_jobs=-1,n_iter=20)
    #
    #
    # rand_search.fit(X_train,y_train)
    # report(rand_search.cv_results_, 10)
    # log(ttt.ctime())
    # log(time() - now)
    # return

    # Parameters for Bagging
    # parameters = {
    #     "n_estimators": [2, 3, 5, 13, 51, 201, 303, 403, 505],
    #     "max_features": list(map(lambda x: int(x),
    #                              [sqrt(attributes), 2 * sqrt(attributes), 3 * sqrt(attributes), attributes / 2,
    #                               attributes / 3, attributes / 4]))
    # }
    #
    # rand_search = RandomizedSearchCV(BaggingClassifier(
    #     base_estimator=LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000, C=1.0, tol=0.0001, dual=True),
    #     random_state=seed, n_jobs=1), param_distributions=parameters, n_jobs=-1, n_iter=3, cv=8,
    #     scoring=make_scorer(f1_score, average="micro", labels=["positive", "negative", "neutral"]))
    #
    # now = time()
    # log(ttt.ctime())
    # rand_search.fit(X_train, y_train)
    #
    # report(rand_search.cv_results_, 10)
    log(ttt.ctime())
    log(time() - now)

    # Parameters for RF
    # log("RF:")
    # parameters = {
    #     "n_estimators":[103, 201, 305, 403, 666, 1001, 5007, 10001],
    #     "max_depth":[None, 5, 20, 40, 73, 100, 1000, 2000],
    #     "criterion":["gini", "entropy"]
    # }
    #
    # rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=seed,n_jobs=-1),param_distributions=parameters,
    #                                  n_iter=15,scoring="accuracy",
    #                                  n_jobs=1,cv=10)
    # now = time()
    # log(ttt.ctime())
    # rand_search.fit(X_train, y_train)
    #
    # report(rand_search.cv_results_, 10)
    # log(ttt.ctime())
    # log(time() - now)

    # Parameters for XGBoost
    log("XGB:")
    parameters = {
        "n_estimators":[103,201, 403],
        "max_depth":[3,10,15],
        "objective":["multi:softmax","binary:logistic"],
        "learning_rate":[0.05, 0.1, 0.15, 0.3]
    }

    rand_search = RandomizedSearchCV(XGBoostClassifier(seed=seed),param_distributions=parameters,
                                     n_iter=5,scoring="accuracy",
                                     n_jobs=-1,cv=8)


    now = time()
    log(ttt.ctime())
    rand_search.fit(X_train, y_train)

    report(rand_search.cv_results_, 10)
    log(ttt.ctime())
    log(time() - now)

    parameters = {
        "n_estimators": [403, 666, 1000],
        "max_depth": [40,50,90,100,200],
        "subsample":[1.0, 0.6, 0.9],
        "objective": ["multi:softmax", "binary:logistic"],
        "learning_rate": [0.1, 0.15, 0.5]
    }

    rand_search = RandomizedSearchCV(XGBoostClassifier(seed=seed,), param_distributions=parameters,
                                     n_iter=5, scoring="accuracy",
                                     n_jobs=-1, cv=8)

    now = time()
    log(ttt.ctime())
    rand_search.fit(X_train, y_train)

    report(rand_search.cv_results_, 10)
    log(ttt.ctime())
    log(time() - now)

    return


    # Parameters for VotingClassifier
    # parameters = {
    #     "weights": [
    #         [1, 1, 1],
    #         [2, 1, 1],
    #         [2, 2, 1],
    #         [4, 1, 5],
    #         [1, 1, 2],
    #         [5, 1, 2],
    #         [5, 2, 1],
    #         [5, 3, 2],
    #         [6, 2, 1],
    #         [6, 1, 5],
    #         [6, 1, 2],
    #         [7, 1, 6],
    #         [7, 2, 3],
    #     ]
    # }
    log("Voting RF XGB NB:")
    parameters = {
        "weights": [
            [1, 1, 1],
            [2, 1, 1],
            [1, 1, 2],
            [4, 1, 5],
            [3, 1, 3],
            [3, 1, 4]
        ]
    }

    rand_search = GridSearchCV(VotingClassifier([
        ("randomforest", RandomForestClassifier(n_estimators=403, random_state=seed, max_depth=73, n_jobs=-1)),
        ("naivebayes", BernoulliNB()),
        ("xgboost", XGBoostClassifier(n_estimators=103, seed=seed, max_depth=3, objective="multi:softmax"))
    ], voting="soft", n_jobs=1), scoring="accuracy", n_jobs=-1, cv=8, param_grid=parameters)
    rand_search.fit(X_train, y_train)
    #
    report(rand_search.cv_results_, 10)
    log(ttt.ctime())
    log(time() - now)


def numbers_to_boolean(df):
    for column in filter(lambda col: col.startswith("number_of_"), df.columns):
        df[column] = (df[column] >= 1).astype(int)


if __name__ == "__main__":

    def main():
        result_col_names = ["min_occ", "precision", "recall", "accuracy"]
        result_col_names = ["min_occ", "precision_negative", "precision_neutral", "precision_positive",
                            "recall_negative",
                            "recall_neutral", "recall_positive", "accuracy"]

        results_df = pd.DataFrame(columns=result_col_names)
        for m in range(3, 4):
            print("Preparing data with min_occurrences=" + str(m))
            training_data, testing_data = preprare_data(m)
            log("********************************************************")
            log("Validating for {0} min_occurrences:".format(m))
            # drop idx & id columns
            if training_data.columns[0] == "idx":
                training_data = training_data.iloc[:, 1:]

            if testing_data.columns[0] == "idx":
                testing_data = testing_data.iloc[:, 1:]

            if "original_id" in training_data.columns:
                training_data.drop("original_id", axis=1, inplace=True)

            if "original_id" in testing_data.columns:
                testing_data.drop("original_id", axis=1, inplace=True)

            # continue
            import random
            seed = 666
            random.seed(seed)
            X_train, X_test, y_train, y_test = train_test_split(training_data.iloc[:, 1:], training_data.iloc[:, 0],
                                                                train_size=0.7, stratify=training_data.iloc[:, 0],
                                                                random_state=seed)

            use_full_set = True
            if use_full_set:
                X_train = training_data.iloc[:, 1:]
                y_train = training_data.iloc[:, 0]

                X_test = testing_data.iloc[:, 1:]
                y_test = testing_data.iloc[:, 0]

            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # scaler.fit(X_train)
            #
            # scaler.transform(X_train)
            # scaler.transform(X_test)

            # numbers_to_boolean(X_train)
            # numbers_to_boolean(X_test)

            from math import sqrt

            classifiers = [
                # MLPClassifier(hidden_layer_sizes=(900, 666, 500, 100, 50, 13), random_state=seed, max_iter=5000)
                # LinearSVC(random_state=seed,class_weight="balanced",max_iter=5000,C=1.0,tol=1e-5,dual=True),
                # SVC(random_state=seed, class_weight="balanced", max_iter=10000, kernel="linear",probability=True)
                # BaggingClassifier(base_estimator=LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000,
                #                                            C=1.0, tol=1e-5, dual=True),
                #                   n_estimators=403, n_jobs=-1, random_state=seed,
                #                   max_features=410),
                # VotingClassifier([
                #     ("svm", BaggingClassifier(
                #         base_estimator=LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000,
                #                                  C=2.0, tol=0.001, dual=True),
                #         n_estimators=8, n_jobs=-1, random_state=seed,
                #     )),
                #     ("naivebayes", BernoulliNB()),
                #     ("randomforest", RandomForestClassifier(max_depth=73, n_estimators=403, n_jobs=-1))
                # ], voting="soft", weights=[1,1,1]),
                # XGBoostClassifier(n_estimators=103,seed=seed,max_depth=4, objective="multi:softmax"),
                # LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000, C=2.0, tol=0.001, dual=True)
                # LogisticRegression(max_iter=5000,n_jobs=-1,solver="sag",random_state=seed),
                # RandomForestClassifier(n_jobs=-1,random_state=seed,n_estimators=403)
                # VotingClassifier([
                #     ("randomforest",
                #      RandomForestClassifier(n_estimators=403, random_state=seed, max_depth=73, n_jobs=-1)),
                #     ("naivebayes", BernoulliNB()),
                #     ("xgboost", XGBoostClassifier(n_estimators=103, seed=seed, max_depth=3, objective="multi:softmax"))
                # ], voting="soft", weights=[4, 1, 5], n_jobs=-1),
                BernoulliNB()
                # SVC(C=2.0,kernel="linear",tol=0.001,random_state=seed),

                # BaggingClassifier(
                #     base_estimator=LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000,
                #                              C=2.0, tol=0.001, dual=True),
                #     n_estimators=8, n_jobs=-1, random_state=seed,
                # )
                # BernoulliNB(),
                # LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000, C=2.0, tol=0.001, dual=True)
            ]
            #
            best_fit(X_train, y_train)
            continue
            # classifier = joblib.load("VotingClassifier_train.bin")
            # predictions = classifier.predict(X_test)
            #
            # not_predicted_idx = []
            # for idx, predicted in enumerate(predictions):
            #     if predicted == y_test.iloc[idx]:
            #         continue
            #     not_predicted_idx.append(idx)
            #
            # bad_predictions = X_test.iloc[not_predicted_idx, :]
            # bad_predictions = bad_predictions.assign(label=y_test.iloc[not_predicted_idx])
            # bad_predictions.to_csv("data\\not_predicted.csv", index_label="idx")
            #
            # continue
            for classifier in classifiers:
                # precision, recall, accuracy = test_classifier(X_train, y_train, X_test, y_test, classifier)
                precision, recall, accuracy = cv(classifier,X_train,y_train)
                continue
                _tmp = [m]
                for idx in range(0, len(precision)):
                    _tmp.append(precision[idx])

                for idx in range(0, len(recall)):
                    _tmp.append(recall[idx])

                _tmp.append(accuracy)
                results_df = results_df.append(pd.DataFrame([_tmp], columns=result_col_names))
                results_df.to_csv("results_" + str(type(classifier).__name__) + "_train.csv", index_label="idx")
                # joblib.dump(classifier, str(type(classifier).__name__)+"_train.bin")

        print(results_df)

        print("Done!")


    main()
