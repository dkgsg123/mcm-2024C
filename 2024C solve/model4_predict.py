import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import graphviz

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

from model1_flow_capture import flow_capture
from model3_detect_swing import mydetect

from sklearn.tree import export_text

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# # 创建模型
# model = RandomForestClassifier()
#
# # 定义超参数的分布
# param_dist = {
#     'n_estimators': randint(50, 200),
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # 使用随机搜索进行自动调参
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
# random_search.fit(X_train, y_train)
#
# # 输出最佳参数
# best_params = random_search.best_params_



# dtree for classification
class myclassifier:
    def __init__(self, X, y, type):

        self.data_2d = X
        self.target = y

        self.type = type

        if type == 'dtree':
            self.model = DecisionTreeClassifier()
        if type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=None)
        if type == 'xgb':
            self.model = xgb.XGBClassifier()

        if type == 'log':
            self.model = LogisticRegression()

        if type == 'bayes':
            self.model = MultinomialNB()


    def plot_dtree(self, feature_names, label_name, filename): # 基于所有的数据集
        self.model = self.model.fit(self.data_2d, self.target)
        dot_data = tree.export_graphviz(self.model, out_file=None, feature_names=feature_names,class_names=label_name, filled=True, rounded=True)  # 重要参数可定制
        graph = graphviz.Source(dot_data)
        graph.render(view=True, format="pdf", filename=filename)

        # tree.plot_tree(self.model, feature_names=feature_names, class_names=label_name, filled=True) # 彩色的
        # plt.show()

    # def print_text(self, names): # 基于所有的数据集
    #     self.model = self.model.fit(self.data_2d, self.target)
    #     r = export_text(self.model, feature_names=names)
    #     print(r)

    def auto_params(self):
        if self.type == 'rf':
            model = RandomForestClassifier()
            # param_grid = {
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [None, 10, 20],
            #     'min_samples_split': [2, 5, 10],
            #     'min_samples_leaf': [1, 2, 4]
            # }
            #
            #
            # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
            # grid_search.fit(self.data_2d, self.target)
            #
            # best_params = grid_search.best_params_
            # print(best_params)

            param_dist = {
                'n_estimators': randint(50, 200),
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
            random_search.fit(self.data_2d, self.target)

            best_params = random_search.best_params_

            self.model = RandomForestClassifier(**best_params)

        if self.type == 'xgb':

            model = xgb.XGBClassifier()
            param_dist = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 1, 2],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 1, 2]
            }

            # Create a RandomizedSearchCV object
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=10,  # Number of parameter settings that are sampled
                cv=5,  # Number of cross-validation folds
                scoring='accuracy',  # Scoring metric
                random_state=42
            )

            random_search.fit(self.data_2d, self.target)
            best_params = random_search.best_params_
            self.model = xgb.XGBClassifier(**best_params)






    def get_model_evaluation(self, test_size=0.3, random_state=None): # 一次打乱，很有用但是还是没kfold有用
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_2d, self.target, test_size=test_size, random_state=random_state
        )

        # Train the model on the training set
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        # Evaluate the model on the test set
        score = precision_score(y_test, y_pred) # precision

        print(score)

        return score

    def get_kfold_evaluation(self, n_splits=5, random_state=None): # 更平均
        # Create a KFold cross-validation splitter
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Perform cross-validation
        scores = cross_val_score(self.model, self.data_2d, self.target, cv=kf, scoring=make_scorer(precision_score, average='binary'))

        # Return the mean accuracy across all folds
        mean_scores = scores.mean()

        print(mean_scores)

        return mean_scores


from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    '''
    chi2 test
    :param confusion_matrix:
    :return:
    '''
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()
    phi2 = chi2 / n # x2/n
    r, k = confusion_matrix.shape
    # phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    # rcorr = r - ((r - 1) ** 2) / (n - 1)
    # kcorr = k - ((k - 1) ** 2) / (n - 1)
    # return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    d_min = min(r, k) - 1
    return np.sqrt((phi2)/d_min)




# def calculate_caremers_v(df, column_a, column_b):
#     """
#     calculate carmer v for the 2 input columns in dataframe
#     :param df: Pandas dataframe object
#     :param column_a: 1st column to study
#     :param column_b: 2nd column to study
#     :return: Pandas dataframe object with the duplicated recorders removed.
#     """
#     if column_a not in df.columns:
#         print("the input columne %s doesn't exit in the dataframe." % column_a)
#         return None
#     elif column_b not in df.columns:
#         print("the input columne %s doesn't exit in the dataframe." % column_b)
#         return None
#     else:
#         cross_tb = pd.crosstab(index=df[column_a], columns=df[column_b])
#         np_tb = cross_tb.to_numpy()
#         min_row_column = min(np_tb.shape[0], np_tb.shape[1])
#         colume_sum = np_tb.sum(axis=0)
#         row_sum = np_tb.sum(axis=1)
#         total_sum = np_tb.sum()
#         np_mid = np.matmul(row_sum.reshape(len(row_sum), 1),
#                            colume_sum.reshape(1, len(colume_sum))) / total_sum
#         new_tb = np.divide(np.power((np_tb - np_mid), np.array([2])),
#                            np_mid)
#
#         return np.sqrt(new_tb.sum() / (total_sum * (min_row_column - 1)))


if __name__ == '__main__':
    # get df with momentum
    model = flow_capture('2023-wimbledon-1408')
    model.run_prob_flow()
    df = model.get_final_df()  # df no+everything
    # df = df[(df['set_no'] == 5)]


    # gei labels
    model = mydetect(df)
    model.run_label_by_gradient('onehot') # 5次
    df = model.run_label_by_gradient_demo('onehot') # 3次 已经生成预测模型
    print(df.shape)
    # print(df.to_string())


    # feature enginerring

    my_feature_list = ['server', 'serve_no', 'point_victor', 'game_victor', 'set_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
                    'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won',
                    'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
                    'p1_consecutive_pt', 'p2_consecutive_pt', 'p1_consecutive_gm', 'p2_consecutive_gm', 'p1_consecutive_set', 'p2_consecutive_set',
                       'encoder_m1_demo_inflection']
    feature_list = []
    # 'winner_shot_type' 懒得编码了

    for feature in my_feature_list:
        confusion_matrix = pd.crosstab(df[feature], df['encoder_m1_inflection'])
        # print(confusion_matrix)
        stat = cramers_v(confusion_matrix)
        # print(f'{feature} value = ', stat)  # ???????
        # print(f'{feature} ', stat)
        if (stat >= 0.1) and not np.isnan(stat):
            feature_list.append(feature)
            print(f'{feature} ', stat)


    df.dropna(subset=feature_list, inplace=True)

    print(feature_list)
    # y = df[['encoder_m1_inflection']].values.reshape(-1, 1) # 5 in a row
    #
    # model = myclassifier(df[feature_list].values, y, 'bayes')
    # model.auto_params()
    # model.get_model_evaluation()
    # model.get_kfold_evaluation()

    y_pred = df[['encoder_m1_demo_inflection']].values.reshape(-1, 1) # 3 in a row
    y = df[['encoder_m1_inflection']].values.reshape(-1, 1)  # 5 in a row
    print('5:', y.sum(), '3:', y_pred.sum())

    # model = MultinomialNB()
    # model.fit(y_pred, y)
    # print(y_pred)
    # y_pred = model.predict(y_pred)
    # print(y_pred.reshape(-1, 1))

    score = precision_score(y, y_pred)
    score_recall = recall_score(y, y_pred)
    score_f1 = f1_score(y, y_pred)
    print('last hope precise:', 100*score, '%')
    print('last hope recall:', 100 * score_recall, '%')
    print('last hope F1:', 100 * score_f1, '%')