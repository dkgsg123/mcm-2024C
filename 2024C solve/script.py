# for i in range(1, 3):
#     player = i
#     model = myKmeans(df[[f'p{player}_unf_err', f'p{player}_winner', f'p{player}_break_pt_won', f'p{player}_break_pt_missed']].values.reshape(-1, 4))
#     # model.plot_find_k()
#     labels = model.get_labels(5)
#     plt.scatter(df['point_no'].values, df[f'p{player}_points_won'].values, c=labels)  # scatter+labels

# for i in range(1, 3):
#     player = i
#     var_list = ['server', 'serve_no', 'point_victor', 'game_victor', 'set_victor', f'p{player}_ace',
#                 f'p{player}_winner', f'p{player}_double_fault', f'p{player}_unf_err', f'p{player}_net_pt', f'p{player}_net_pt_won',
#                 f'p{player}_break_pt', f'p{player}_break_pt_won', f'p{player}_break_pt_missed', f'p{player}_distance_run', 'rally_count']
#     model = myKmeans(df[var_list].values.reshape(-1, len(var_list)))
#     model.plot_find_k()
#     labels = model.get_labels(5)
#     plt.scatter(df['point_no'].values, df[f'p{player}_points_won'].values, c=labels)  # scatter+labels

# plt.figure()
#
# for i in range(1, 3):
#     player = i
#     var_list = [f'p{player}_points_won', f'p{player}_unf_err', f'p{player}_winner', f'p{player}_break_pt_won', f'p{player}_break_pt_missed']
#     model = myKmeans(df[var_list].values.reshape(-1, len((var_list))))
#     model.plot_find_k()
#     labels = model.get_labels(3)
#     # plt.scatter(df['point_no'].values, df[f'p{player}_points_won'].values, c=labels)  # scatter+labels
#     # for label in labels:
#     #     print(label)
#     # print('1111111111111111111111111111111111111111111111111')
#
# plt.legend()
# plt.show()

# class myKmeans:
#     def __init__(self, data):
#         # data
#         self.data_2d = data
#         self.data_2d = zscore(self.data_2d)
#
#     def get_labels(self, k):
#         self.k = k
#         self.model = KMeans(n_clusters=k, n_init=10)
#         self.model.fit(self.data_2d)
#         return self.model.labels_
#
#     def plot_find_k(self):
#         self.inertias = []
#         for i in range(1, 21): # k 从 1 遍历到 10
#             kmeans = KMeans(n_clusters=i, n_init=10)
#             kmeans.fit(self.data_2d)
#             self.inertias.append(kmeans.inertia_)
#
#         plt.plot(np.arange(1, 21), self.inertias, marker='o')
#         plt.title('Elbow method')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('Inertia')
#         plt.show()

# model.plot_dtree(feature_names=feature_list, label_name=['Not Inflection', 'Inflection'], filename=r'./results/1701 final try')