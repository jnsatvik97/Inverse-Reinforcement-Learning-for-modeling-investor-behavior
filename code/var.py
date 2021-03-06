import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pylab as plt
import os
from numpy.linalg import norm
from scipy.stats import entropy

from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.stattools import adfuller

import argparse

class var():

    
    def __init__(self, d=15):
       
        self.d = d


    def read_data(self, train='train_normalized_round2', train_start=1, train_end=18, test='test_normalized_round2', test_start=19, test_end=24, old_format=False):
        
        print("Reading train files")
        list_df = []
        idx = 0
        for num_day in range(train_start, train_end+1):
            if old_format:
                filename = "trend_distribution_day%d_reordered.csv" % num_day
            else:
                filename = "trend_distribution_day%d.csv" % num_day                
            path_to_file = train + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(self.d), usecols=range(self.d), dtype=np.float64)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16
            
        df_train = pd.concat(list_df)
        df_train.index = pd.to_datetime(df_train.index, unit="D")
        self.df_train = df_train
        
        print("Reading test files")
        list_df = []
        for num_day in range(test_start, test_end+1):
            if old_format:
                filename = "trend_distribution_day%d_reordered.csv" % num_day
            else:
                filename = "trend_distribution_day%d.csv" % num_day                
            path_to_file = test + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(self.d), usecols=range(self.d), dtype=np.float64)
            df.index = np.arange(idx, idx+16) # use the same idx that was incremented when reading training data
            list_df.append(df)
            idx += 16
        if len(list_df):
            df_test = pd.concat(list_df)
        else:
            df_test = pd.DataFrame()

        df_test.index = pd.to_datetime(df_test.index, unit="D")
        self.df_test = df_test
        
        return self.df_train, self.df_test


    def check_stationarity(self, topic):
        ts = self.df_train[topic]

        #Determing rolling statistics
        rolmean = pd.rolling_mean(ts, window=12)
        rolstd = pd.rolling_std(ts, window=12)
    
        
        orig = plt.plot(ts, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
        
       
        
    def train(self, max_lag, df_train):

        self.model = VAR(df_train)
        
        self.results = self.model.fit(maxlags=max_lag, ic='aic')


    def cross_validation(self, lag_range=range(1,21), validation_size=5, repetitions=5):
        
        list_error = []
        num_training_points = int(len(self.df_train.index) / 16)
        list_choices = range(num_training_points)
        num_selected = num_training_points - validation_size
        # For each lag value
        for lag in lag_range:
            print("Lag %d" % lag)
            avg_error = 0
            # Repeat for repetition times
            for rep in range(0, repetitions):
                # Randomly split into training set and validation set
                selected = np.random.choice(list_choices, num_selected, replace=False)
                the_rest = [x for x in list_choices if x not in selected]
                list_temp = []
                for point in selected:
                    list_temp.append( self.df_train[point*16:(point+1)*16] )
                df_selected = pd.concat(list_temp)
                list_temp = []
                for point in the_rest:
                    list_temp.append( self.df_train[point*16:(point+1)*16] )
                df_validation = pd.concat(list_temp)

                # Relabel indices to have increasing time order
                df_selected.index = np.arange(len(df_selected.index))
                df_selected.index = pd.to_datetime(df_selected.index, unit="D")
                df_validation.index = np.arange(len(df_selected.index), len(df_selected.index) + 16*validation_size)
                df_validation.index = pd.to_datetime(df_validation.index, unit="D")
                # Train
                self.train(max_lag=lag, df_train=df_selected)

                # Test on the validation set and accumulate error                
                avg_error += self.validation(validation_size*16, df_selected, df_validation)

            # Average error over repetitions
            avg_error = avg_error / repetitions
            print("Lag %d. avg_error" % lag, avg_error)
            # Record avg error for this lag value
            list_error.append(avg_error)

        print("Min error is", np.min(list_error))
        print("Best lag value is", lag_range[np.argmin(list_error)])
        f = open('eval_var_round2/var_cross_validation.csv', 'a')
        s = ','.join(map(str, lag_range))
        s += '\n'
        f.write(s)
        s = ','.join(map(str, list_error))
        s += '\n'
        f.write(s)


    def plot(self, topic, lag):
        
        #plt.plot(self.df_train.index[lag:], self.df_train[topic][lag:], color='r', label='data')
        plt.plot(self.df_train.index, self.df_train[topic], color='r', label='data')        
        plt.plot(self.df_train.index[lag:], self.results.fittedvalues[topic], color='b', label='time series')
        plt.legend(loc='best')
        plt.title('Topic %d data and fitted time series' % topic)
        plt.show()
        

    def JSD(self, P, Q):
       
        # Replace all invalid values by 1e-100
        P[P<=0] = 1e-100
        Q[Q<=0] = 1e-100

        P_normed = P / norm(P, ord=1)
        Q_normed = Q / norm(Q, ord=1)
        M = 0.5 * (P + Q)

        return 0.5 * (entropy(P,M) + entropy(Q,M))


    def evaluate_train(self):
        lag = self.results.k_ar

        # Total number of distributions in fitted time series
        len_fitted = len(self.results.fittedvalues.index)
        # Total number of distributions across all days and all hours in training set
        len_empirical = len(self.df_train.index)

       
        
        idx_fitted = 15 - lag
        # start at final distribution of day1
        idx_empirical = 15

        
        num_trajectories = int(len_empirical/16)
        array_l1_final = np.zeros(num_trajectories)
        array_JSD_final = np.zeros(num_trajectories)
        
        # Go through all final distributions
        idx =  0
        while idx_fitted < len_fitted and idx_empirical < len_empirical:
            l1_final = norm( self.df_train.ix[idx_empirical] - self.results.fittedvalues.ix[idx_fitted], ord=1)
            array_l1_final[idx] = l1_final

            JSD_final = self.JSD( self.df_train.ix[idx_empirical], self.results.fittedvalues.ix[idx_fitted] )
            array_JSD_final[idx] = JSD_final

            idx_fitted += 16
            idx_empirical += 16
            idx += 1

        

        array_l1_mean = np.zeros(len_fitted)
        array_JSD_mean = np.zeros(len_fitted)
        idx_fitted = 0 # start at very beginning

        while idx_fitted < len_fitted:
            l1 = norm( self.df_train.ix[idx_fitted+lag] - self.results.fittedvalues.ix[idx_fitted], ord=1)
            array_l1_mean[idx_fitted] = l1

            JSD_final = self.JSD( self.df_train.ix[idx_fitted+lag], self.results.fittedvalues.ix[idx_fitted] )
            array_JSD_mean[idx_fitted] = JSD_final            
            idx_fitted += 1

        # Mean over all days of the difference between final distributions
        mean_l1_final = np.mean(array_l1_final)
        mean_JSD_final = np.mean(array_JSD_final)
        print(array_l1_final)
        print(array_JSD_final)
        print(mean_l1_final)
        print(mean_JSD_final)

        # Mean over all hours of the difference between distributions at all hours
        mean_l1 = np.mean(array_l1_mean)
        mean_JSD = np.mean(array_JSD_mean)
        print(mean_l1)
        print(mean_JSD)
        

    def validation(self, steps, df_selected, df_validation):
       
       
        num_previous = len(df_selected.index)
        lag_order = self.results.k_ar
        future = self.results.forecast(df_selected.values[-lag_order:], steps)
        df_future = pd.DataFrame(future)
        df_future.index = np.arange(num_previous, num_previous+steps)
        df_future.index = pd.to_datetime(df_future.index, unit="D")

        len_validation = len(df_validation.index)
        num_trajectories = int(len_validation / 16)
        array_JSD_mean = np.zeros(num_trajectories)
        idx_day = 0
        while idx_day < num_trajectories:
            idx_hour = 0
            jsd = 0
            while idx_hour < 16:
                idx = 16*idx_day + idx_hour
                jsd += self.JSD( df_validation.ix[idx], df_future.ix[idx] )
                idx_hour += 1

            array_JSD_mean[idx_day] = jsd/16
            idx_day += 1

        mean_JSD_mean = np.mean(array_JSD_mean)
        return mean_JSD_mean


    def forecast(self, num_prior=416, steps=176, topic=0, plot=1, show_plot=1):
        
        lag = self.results.k_ar
        num_previous = len(self.df_train.index)
        
        future = self.results.forecast(self.df_train.values[-num_prior:], steps)

        self.df_future = pd.DataFrame(future)
        self.df_future.index = np.arange(num_previous, num_previous+steps)
        self.df_future.index = pd.to_datetime(self.df_future.index, unit="D")

        array_x_train = np.arange(num_previous)
        array_x_test = np.arange(num_previous, num_previous+len(self.df_future.index))

        if plot == 1:
            # For plotting future along with raw data and fitted time series
            plt.plot(array_x_train, self.df_train[topic], color='r', linestyle='-', label='train data')        
            plt.plot(array_x_train[lag:], self.results.fittedvalues[topic], color='b', linestyle='--', label='time series (train)')
            plt.plot(array_x_test, self.df_test[topic], color='k', linestyle='-', label='test data')
            plt.plot(array_x_test, self.df_future[topic], color='g', linestyle='--', label='time series (test)')
            plt.ylabel('Topic %d popularity' % topic)
            plt.xlabel('Time steps (hrs)')
            plt.legend(loc='best')
            plt.title('Topic %d data and fitted time series' % topic)
            if show_plot == 1:
                plt.show()

        return self.df_future


    def evaluate_test(self, outfile='eval_var_round2/test.csv'):
        lag = self.results.k_ar

        # Total number of distributions in future 
        len_future = len(self.df_future.index)
        # Total number of distributions across all days and all hours in test set
        len_empirical = len(self.df_test.index)
        
        if len_future != len_empirical:
            print("Lengths of test set and generated future differ!")
            return

        
        idx_future = 15
        idx_empirical = 15

        # num_trajectories = number of days
        num_trajectories = int(len_empirical/16)
        array_l1_final = np.zeros(num_trajectories)
        array_JSD_final = np.zeros(num_trajectories)
        
        # Go through all final distributions
        idx =  0
        while idx_future < len_future and idx_empirical < len_empirical:
            l1_final = norm( self.df_test.ix[idx_empirical] - self.df_future.ix[idx_future], ord=1)
            array_l1_final[idx] = l1_final

            JSD_final = self.JSD( self.df_test.ix[idx_empirical], self.df_future.ix[idx_future] )
            array_JSD_final[idx] = JSD_final

            idx_future += 16
            idx_empirical += 16
            idx += 1

        ### Part 2: evaluate distributions at all hours ###

        array_l1_mean = np.zeros(num_trajectories)
        array_JSD_mean = np.zeros(num_trajectories)
        idx_day = 0
        while idx_day < num_trajectories:
            idx_hour = 0
            l1 = 0
            jsd = 0
            while idx_hour < 16:
                idx = 16*idx_day + idx_hour
                l1 += norm( self.df_test.ix[idx] - self.df_future.ix[idx], ord=1 )
                jsd += self.JSD( self.df_test.ix[idx], self.df_future.ix[idx] )
                idx_hour += 1

            array_l1_mean[idx_day] = l1/16
            array_JSD_mean[idx_day] = jsd/16
            idx_day += 1

        
        mean_l1_final = np.mean(array_l1_final)
        std_l1_final = np.std(array_l1_final)
        mean_JSD_final = np.mean(array_JSD_final)
        std_JSD_final = np.std(array_JSD_final)
        print("Mean L1 final", mean_l1_final)
        print("Mean JSD final", mean_JSD_final)

        
        mean_l1_mean = np.mean(array_l1_mean)
        std_l1_mean = np.std(array_l1_mean)
        mean_JSD_mean = np.mean(array_JSD_mean)
        std_JSD_mean = np.std(array_JSD_mean)
        print("Mean L1 mean", mean_l1_mean)
        print("Mean JSD mean", mean_JSD_mean)        

        with open(outfile, 'ab') as f:
            np.savetxt(f, np.array(['array_l1_final']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_final]), fmt='%.3e')
            np.savetxt(f, np.array([std_l1_final]), fmt='%.3e')            
            np.savetxt(f, array_l1_final.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_l1_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_mean]), fmt='%.3e')
            np.savetxt(f, np.array([std_l1_mean]), fmt='%.3e')            
            np.savetxt(f, array_l1_mean.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_final']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_final]), fmt='%.3e')
            np.savetxt(f, np.array([std_JSD_final]), fmt='%.3e')
            np.savetxt(f, array_JSD_final.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_mean]), fmt='%.3e')
            np.savetxt(f, np.array([std_JSD_mean]), fmt='%.3e')
            np.savetxt(f, array_JSD_mean.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_lag', type=int, default=1)
    args = parser.parse_args()
    
    exp = var(d=15)
    print("reading data")
    df_train, df_test = exp.read_data(train='train_normalized_round2', train_start=1, train_end=16, test='test_normalized_round2', test_start=17, test_end=20)
    print("training")
    exp.train(max_lag=args.max_lag, df_train=exp.df_train)
    # print("evaluate training performance")
    # exp.evaluate_train()
    print("forecasting")
    exp.forecast(num_prior=args.max_lag, steps=64, topic=0, plot=1, show_plot=1)
    # print("evaluate test performance")
    # exp.evaluate_test()
