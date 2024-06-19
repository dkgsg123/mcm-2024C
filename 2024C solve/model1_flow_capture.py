import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from my_dataset import dataset
import warnings
warnings.filterwarnings("ignore")

class flow_capture:
    def __init__(self, match_name, dist_type=None, dist_rate=0):
        self.dataset = dataset()
        self.df = self.dataset.get_match(match_name)
        self.df.set_index('point_no', inplace=True)
        # no index
        self.p1_name = self.df.iloc[0, 0]
        self.p2_name = self.df.iloc[0, 1]

        # initial
        self.df['p1_momentum'] = 0.5
        self.df['p2_momentum'] = 0.5

        self.df['p1_consecutive_pt'] = 0
        self.df['p2_consecutive_pt'] = 0

        self.df['p1_consecutive_gm'] = 0
        self.df['p2_consecutive_gm'] = 0

        self.df['p1_consecutive_set'] = 0
        self.df['p2_consecutive_set'] = 0



        # hyperparams
        self.p1_initial_momentum = 0.5
        self.p2_initial_momentum = 0.5
        self.rate_d_f = 0.002
        self.rate_u_e = 0.002
        self.rate_b_pt = 0.002
        self.rate_cons_p = 0.002
        self.rate_cons_g = 0.008
        self.rate_cons_s = 0.05
        self.rate_end_dist = 0.002
        self.rate_w_m = 0.002
        self.rate_server = 0.002

        if dist_type == 'd_f':
            self.rate_d_f = 0.002*(1+dist_rate)
        if dist_type == 'u_e':
            self.rate_u_e = 0.002 * (1 + dist_rate)
        if dist_type == 'b_pt':
            self.rate_b_pt = 0.002 * (1 + dist_rate)
        if dist_type == 'cons_p':
            self.rate_cons_p = 0.002 * (1 + dist_rate)
        if dist_type == 'cons_g':
            self.rate_cons_g = 0.008 * (1 + dist_rate)
        if dist_type == 'cons_s':
            self.rate_cons_s = 0.05 * (1 + dist_rate)
        if dist_type == 'end_dist':
            self.rate_end_dist = 0.002 * (1 + dist_rate)
        if dist_type == 'win_margin':
            self.rate_w_m = 0.002 * (1 + dist_rate)
        if dist_type == 'server':
            self.rate_server = 0.002 * (1 + dist_rate)



    def print_info(self):
        print(self.df.head().to_string())
        print(self.df.shape)

    def get_break_pt_perc(self, player):
        df = self.df
        print(f'p{player} perc of winning break pt:', (df[f'p{player}_break_pt_won'].values.sum())/(df[f'p{player}_break_pt'].values.sum()))

    def get_consecutive(self, no, type, player):
        df = self.df
        times = 0
        if type == 'point':
            while no > 0 and df.loc[no, 'point_victor'] == player:
                times += 1
                no = no - 1
        elif type == 'game':
            if df.loc[no, 'game_victor'] == player:
                while no > 0 and (df.loc[no, 'game_victor'] == 0 or df.loc[no, 'game_victor'] == player):
                    if df.loc[no, 'game_victor'] == player:
                        times += 1
                    no = no - 1
            # if df.loc[no, 'game_victor'] == player:
            #     times = 1
        elif type == 'set':
            if df.loc[no, 'set_victor'] == player:
                while no > 0 and (df.loc[no, 'set_victor'] == 0 or df.loc[no, 'set_victor'] == player) :
                    if df.loc[no, 'set_victor'] == player:
                        times += 1
                    no = no - 1
            # if df.loc[no, 'set_victor'] == player:
            #     times = 1
        else:
            print('consecutive error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        return times

    def get_win_margin(self, no, player):
        df = self.df

        def trans(score):
            if score == '0':
                value = 0
            elif score == '15':
                value = 1
            elif score == '25':
                value = 2
            elif score == '40':
                value = 3
            elif score == 'AD':
                value = 4
            else:
                value = int(score)

            return value

        value = trans(df.loc[no, f'p{player}_score'])

        return 1/(11-value)

    def get_end_distance(self, no):
        df = self.df
        value = max(df.loc[no, 'p1_sets'], df.loc[no, 'p2_sets'])
        return value

    def get_index(self, no, type, player):
        df = self.df

        if type == 'double_fault':
            if df.loc[no, f'p{player}_double_fault'] == 1:
                index = -1
            else:
                index = 0
        elif type == 'unf_err':
            if df.loc[no, f'p{player}_unf_err'] == 1:
                index = -1
            else:
                index = 0

        elif type == 'break_pt': # score
            if df.loc[no, f'p{player}_break_pt'] == 1:
                index = 1
            else:
                index = 0
        elif type == 'server': # factor
            if df.loc[no, 'server'] == player:
                index = 1
            else:
                index = 0
        else:
            index = 0

        return index

    def get_factor(self, no, type, player):
        '''
        factor for one player
        :param no:
        :param type:
        :param player:
        :return:
        '''
        df = self.df
        factor = {} # dict
        if type == 'double_fault':
            factor['double_fault'] = self.rate_d_f*self.get_index(no, type, player)
        elif type == 'unf_err':
            factor['unf_err'] = self.rate_u_e * self.get_index(no, type, player)
        elif type == 'break_pt':
            factor['break_pt'] = self.rate_b_pt * self.get_index(no, type, player)
        elif type == 'server':
            factor['server'] = self.rate_server * self.get_index(no, type, player)
        elif type == 'consecutive_p':
            factor['consecutive_p'] = self.rate_cons_p * self.get_consecutive(no, 'point', player)
        elif type == 'consecutive_g':
            factor['consecutive_g'] = self.rate_cons_g * self.get_consecutive(no, 'game', player)
        elif type == 'consecutive_s':
            factor['consecutive_s'] = self.rate_cons_s * self.get_consecutive(no, 'set', player)
        elif type == 'win_margin':
            factor['win_margin'] = self.rate_w_m * self.get_win_margin(no, player)
        elif type == 'end_distance':
            factor['end_distance'] = self.rate_end_dist * self.get_end_distance(no)
        else:
            print('factor error!!!!!!!!!!!!!!!!!')
        return factor[type]

    def run_prob_flow(self):
        '''
        predict
        :return:
        '''
        df = self.df



        for i in range(1, len(df)+1):

            fac_result1 = self.get_factor(i, 'double_fault', 1)+self.get_factor(i, 'unf_err', 1)+self.get_factor(i, 'server', 1)+self.get_factor(i, 'consecutive_p', 1) \
                          -self.get_factor(i, 'consecutive_p', 2)+self.get_factor(i, 'consecutive_g', 1)-self.get_factor(i, 'consecutive_g', 2) \
                          +self.get_factor(i, 'consecutive_s', 1)-self.get_factor(i, 'consecutive_s', 2)

            # fac_result1 = self.get_factor(i, 'double_fault', 1) + self.get_factor(i, 'unf_err', 1) + self.get_factor(i,
            #                                                                                                          'server',
            #                                                                                                          1) + self.get_factor(
            #     i, 'consecutive_p', 1) \
            #                + self.get_factor(i, 'consecutive_g',
            #                                                                          1) \
            #               + self.get_factor(i, 'consecutive_s', 1)

            fac_volatility1 = (1+self.get_factor(i, 'break_pt', 1))*(1+self.get_factor(i, 'win_margin', 1))*(1+self.get_factor(i, 'end_distance', 1))

            fac_result2 = self.get_factor(i, 'double_fault', 2) + self.get_factor(i, 'unf_err', 2) + self.get_factor(i,
                                                                                                                     'server',
                                                                                                                     2) + self.get_factor(
                i, 'consecutive_p', 2) - self.get_factor(
                i, 'consecutive_p', 1) + self.get_factor(i, 'consecutive_g', 2) - self.get_factor(i, 'consecutive_g', 1) + self.get_factor(i, 'consecutive_s', 2) - self.get_factor(i, 'consecutive_s', 1)

            # fac_result2 = self.get_factor(i, 'double_fault', 2) + self.get_factor(i, 'unf_err', 2) + self.get_factor(i,
            #                                                                                                          'server',
            #                                                                                                          2) + self.get_factor(
            #     i, 'consecutive_p', 2) + self.get_factor(i, 'consecutive_g', 2)  + self.get_factor(
            #     i, 'consecutive_s', 2)

            fac_volatility2 = (1 + self.get_factor(i, 'break_pt', 2)) * (1 + self.get_factor(i, 'win_margin', 2)) * (
                        1 + self.get_factor(i, 'end_distance', 2))

            if i == 1:
                # prob for prior
                df.loc[1, 'p1_momentum'] = self.p1_initial_momentum + fac_result1*fac_volatility1
                df.loc[1, 'p1_consecutive_pt'] = self.get_consecutive(1, 'point', 1)
                df.loc[1, 'p1_consecutive_gm'] = self.get_consecutive(1, 'game', 1)
                df.loc[1, 'p1_consecutive_set'] = self.get_consecutive(1, 'set', 1)

                # print('p1', fac_result1)
                df.loc[1, 'p2_momentum'] = self.p2_initial_momentum + fac_result2*fac_volatility2
                df.loc[1, 'p2_consecutive_pt'] = self.get_consecutive(1, 'point', 2)
                df.loc[1, 'p2_consecutive_gm'] = self.get_consecutive(1, 'game', 2)
                df.loc[1, 'p2_consecutive_set'] = self.get_consecutive(1, 'set', 2)


                # print('p2', fac_result2)

            else:
                df.loc[i, 'p1_momentum'] = df.loc[i - 1, 'p1_momentum'] + fac_result1*fac_volatility1
                df.loc[i, 'p1_consecutive_pt'] = self.get_consecutive(i, 'point', 1)
                df.loc[i, 'p1_consecutive_gm'] = self.get_consecutive(i, 'game', 1)
                df.loc[i, 'p1_consecutive_set'] = self.get_consecutive(i, 'set', 1)
                # print('p1', fac_result1)
                df.loc[i, 'p2_momentum'] = df.loc[i - 1, 'p2_momentum'] + fac_result2*fac_volatility2
                df.loc[i, 'p2_consecutive_pt'] = self.get_consecutive(i, 'point', 2)
                df.loc[i, 'p2_consecutive_gm'] = self.get_consecutive(i, 'game', 2)
                df.loc[i, 'p2_consecutive_set'] = self.get_consecutive(i, 'set', 2)
                # print('p2', fac_result2)

    def plot_p1_momentum(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)

        sns.set()
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=self.p1_m, label='final_m')
        sns.lineplot(x=self.df['elapsed_time'], y=0.5 * np.ones(len(self.df['elapsed_time'])), label='tie_up',
                     linestyle='--', color='red')

        plt.xlabel('Elapsed Time')
        plt.ylabel('Momentum')
        plt.title('Momentum Over Time')
        plt.legend()
        plt.show()




    def get_p1_final_momentum(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)
        self.p2_m = pd.Series(data=self.df['p2_momentum'].values, index=self.df['elapsed_time'], name=self.p2_name)
        self.p1_final_m = self.df.apply(lambda row: row['p1_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
                                        axis=1)
        self.p1_final_m = pd.Series(data=self.p1_final_m.values, index=self.df['elapsed_time'], name=self.p1_name)
        # print(self.p1_final_m.to_string())
        return self.p1_final_m

    def get_p2_final_momentum(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)
        self.p2_m = pd.Series(data=self.df['p2_momentum'].values, index=self.df['elapsed_time'], name=self.p2_name)
        self.p2_final_m = self.df.apply(lambda row: row['p2_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
                                        axis=1)
        self.p2_final_m = pd.Series(data=self.p2_final_m.values, index=self.df['elapsed_time'], name=self.p2_name)
        return self.p2_final_m

    def plot_p1_final_momentum(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)
        self.p2_m = pd.Series(data=self.df['p2_momentum'].values, index=self.df['elapsed_time'], name=self.p2_name)
        self.p1_final_m = self.df.apply(lambda row: row['p1_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
                                        axis=1)
        sns.set()
        plt.figure(figsize=(7, 5))
        sns.lineplot(x=self.df['elapsed_time'], y=self.p1_final_m, label='final_m')
        sns.lineplot(x=self.df['elapsed_time'], y=0.5*np.ones(len(self.df['elapsed_time'])), label='tie_up', linestyle='--', color='red')

        plt.xlabel('Elapsed Time')
        plt.ylabel('Momentum')
        plt.title('Momentum Over Time')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()

    def plot_p2_final_momentum(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)
        self.p2_m = pd.Series(data=self.df['p2_momentum'].values, index=self.df['elapsed_time'], name=self.p2_name)
        self.p2_final_m = self.df.apply(lambda row: row['p2_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
                                        axis=1)
        sns.set()
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=self.df['elapsed_time'], y=self.p2_final_m, label='final_m')
        sns.lineplot(x=self.df['elapsed_time'], y=0.5*np.ones(len(self.df['elapsed_time'])), label='tie_up', linestyle='--', color='red')

        plt.xlabel('Elapsed Time')
        plt.ylabel('Momentum')
        plt.title('Momentum Over Time')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()


    def get_final_df(self):
        self.p1_m = pd.Series(data=self.df['p1_momentum'].values, index=self.df['elapsed_time'], name=self.p1_name)
        self.p2_m = pd.Series(data=self.df['p2_momentum'].values, index=self.df['elapsed_time'], name=self.p2_name)
        self.df['p1_final_momentum'] = self.df.apply(lambda row: row['p1_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
                                        axis=1)

        self.df['p2_final_momentum'] = self.df.apply(
            lambda row: row['p2_momentum'] / (row['p1_momentum'] + row['p2_momentum']),
            axis=1)
        return self.df

def sensitive(player, type):
    # 扰动list对称扰动
    Deltas = np.linspace(-0.1, 0.1, 10)

    values = []
    # 对每一取值生成model，传入参数，生成多个model，取最末尾值
    for delta in Deltas:
        model = flow_capture('2023-wimbledon-1701', dist_type=type, dist_rate=delta)
        model.run_prob_flow()
        model.get_final_df()
        # print(model.df[f'p{player}_momentum'].tail())
        series = model.df[f'p{player}_momentum']
        values.append(series.iloc[-1])

    # lineplot
    # sns,set()
    # temp = pd.DataFrame({'Deltas': Deltas, 'Momentum': values})
    # sns.lineplot(x='Deltas', y='Momentum', data=temp)
    # plt.show()

    # return

    return values



if __name__ == '__main__':
    # model = flow_capture('2023-wimbledon-1701')
    # model.run_prob_flow()
    # model.get_final_df()
    # model.print_info() # head & shape

    # p1_final_momentum
    # p2_final_momentum
    list = ['d_f', 'u_e', 'b_pt', 'cons_p', 'cons_g', 'cons_s', 'end_dist', 'win_margin', 'server'] # 0-8
    sensitive(player=1, type=list[0])

    Deltas = np.linspace(-0.1, 0.1, 10)
    player = 2

    sens_df = pd.DataFrame({
        'Deltas': Deltas,
        list[0]: sensitive(player=player, type=list[0]),
        list[1]: sensitive(player=player, type=list[1]),
        list[2]: sensitive(player=player, type=list[2]),
        list[3]: sensitive(player=player, type=list[3]),
        list[4]: sensitive(player=player, type=list[4]),
        list[5]: sensitive(player=player, type=list[5]),
        list[6]: sensitive(player=player, type=list[6]),
        list[7]: sensitive(player=player, type=list[7]),
        list[8]: sensitive(player=player, type=list[8]),
    })

    print(sens_df.to_string())

    sens_df.to_excel(r'./evaluation/sensitive/all p2.xlsx', index=False)




    # model.plot_p1_momentum()
    # model.plot_p1_final_momentum()
    ################################# 好的数据存excel #######################################
    # model.get_final_df().to_excel(r'./results/1701/initial params.xlsx')