import numpy as np
import os
from scipy import special
import itertools
import time
import warnings

warnings.filterwarnings('error')

class actor_critic:

    def __init__(self, dim_theta=3, d=47):

        self.dim_theta = dim_theta
        # initialize theta as random column vector, entries [-1,1)
        # self.theta = np.random.rand(dim_theta, 1) * 2 - 1
        self.theta = np.array([[1],[-1],[3]]) # good initialization #here

        # initialize weight vector (column) for value function approximation
        self.w = self.init_w(d)

        # number of topics
        self.d = d

        # d x d x dim_theta tensor, computed within sample_action and used for
        # calculating gradient for theta update
        self.tensor_phi = np.zeros([self.d, self.d, self.dim_theta])        

        # d x d matrix, computed within sample_action and used for sampling action P
        # and also for calculating gradient for theta update
        self.mat_alpha = np.zeros([self.d, self.d])

    def init_w(self, d):
        
        num_features = int((d+1)*d / 2 + d + 1)
        return np.random.rand(num_features, 1)


    def init_pi0(self, path_to_dir='/home/t3500/devdata/mfg/distribution/train_reordered'):
       
        list_pi0 = []
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            f = open(path_to_file, 'r')
            list_lines = f.readlines()
            f.close()
            # Ignore the null topic (need to make a decision later)
            list_pi0.append( list(map(int, list_lines[0].strip().split(',')))[1:1+self.d] )
            
        num_rows = len(list_pi0)
        num_cols = len(list_pi0[0])

        self.mat_pi0 = np.zeros([num_rows, num_cols])
        for i in range(len(list_pi0)):
            total = np.sum(list_pi0[i])
            self.mat_pi0[i] = list(map(lambda x: x/total, list_pi0[i]))
        

    def reorder(self, list_rows):
       
        row1 = list_rows[0]
        # create mapping from index to value
        list_pairs = []
        for i in range(len(row1)):
            list_pairs.append( (i, row1[i]) )

        # sort by decreasing popularity
        list_pairs.sort(reverse=True, key=lambda x: x[1])

        # extract ordering
        order = []
        for pair in list_pairs:
            order.append( pair[0] )
        
        # apply ordering to all rows in list_rows
        for i in range(len(list_rows)):
            list_rows[i] = [ list_rows[i][j] for j in order ]

        return list_rows

    
    def reorder_files(self, path_to_dir='/home/t3500/devdata/mfg/distribution/train', output_dir='/home/t3500/devdata/mfg/distribution/train_reordered'):
        
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            f = open(path_to_file, 'r')
            f.readline() # skip the header line of topics
            list_lines = f.readlines()
            f.close()
            # strip away newline, convert csv format to list of entries,
            # remove the last empty entry (due to extra comma)
            list_lines = list(map(lambda x: x.strip().split(',')[:-1], list_lines))
            # convert to int
            for i in range(len(list_lines)):
                list_lines[i] = list(map(int, list_lines[i]))
            # reorder
            list_rows = self.reorder(list_lines)
            # write to new file
            index_dot = filename.index('.')
            filename_new = filename[:index_dot] + '_reordered' + filename[index_dot:]
            f = open(output_dir + '/' + filename_new, 'w')
            for row in list_rows:
                s = ','.join(map(str, row))
                s += '\n'
                f.write(s)
            f.close()


    def sample_action(self, pi):
        
        # Construct all alphas
        self.mat_alpha = np.zeros([self.d, self.d])
        # Create tensor phi(i,j,pi) for storing all phi matrices for later use
        self.tensor_phi = np.zeros([self.d,self.d,self.dim_theta])
        for i in range(self.d):
            # Construct d x (num_features) matrix by concatenating columns
            # This is 2x faster than looping through rows j of mat_phi and filling them in
            # with 1, pi[i], pi[j], pi[i]*pi[j], pi[i]**2, pi[j]**2
            # This assumes that dim_theta = 6 (now 3)
            col1 = np.ones([self.d, 1])
            col2 = col1 * pi[i]
            col3 = pi.reshape(self.d, 1)
#            
            mat_phi = np.concatenate([col1,col2,col3], axis=1) #here

            # Previous version
#            mat_phi = np.zeros([self.d, self.dim_theta])
#            for j in range(self.d):
#                phi = [1, pi[i], pi[j], pi[i]*pi[j], pi[i]**2, pi[j]**2]
#                mat_phi[j] = phi

           
            self.tensor_phi[i] = mat_phi
            temp = mat_phi.dot(self.theta) # d x 1
            # element-wise product, to get all entries nonzero
            alpha = temp * temp # d x 1
            # Insert alpha transpose into mat_alpha as the i-th row
            self.mat_alpha[i] = np.transpose(alpha)
        
        # Sample matrix P from Dirichlet
        P = np.zeros([self.d, self.d])
        for i in range(self.d):
            
            y = np.random.gamma(shape=self.mat_alpha[i,:], scale=1)
            # replace zeros with dummy value
            y[y == 0] = 1e-20
            total = np.sum(y)
            # Store into i-th row of matrix P
            # P[i] = [y_j/total for y_j in y]
            try:
                P[i] = y / total
            except Warning:
                P[i] = y / total
                print(y, total)

        return P


    def calc_cost(self, P, pi, d):
        
        
        P_squared = P * P # element-wise product
        # (P_squared product with all_ones) element-wise product with pi as column vector
        v1 = P_squared.dot(np.ones([d,1])) * pi.reshape(d, 1)
        # P_squared product with pi as column vector
        v2 = P_squared.dot(pi.reshape(d, 1))
        reward = pi.dot( v1 - v2 )

        
        
        return reward


    def calc_value(self, pi):
        
        # generate pairs of (pi_i, pi_j) for all i, for all j >= i
        list_tuples = list(itertools.combinations_with_replacement(pi, 2))
        # calculate products
        list_features = []
        for idx in range(len(list_tuples)):
            pair = list_tuples[idx]
            list_features.append(pair[0] * pair[1])
        # append first-order feature
        list_features = list_features + list(pi)
        # append bias
        list_features.append(1)
        # calculate value by inner product
        value = np.array(list_features).dot(self.w)


        return value


    def calc_features(self, pi):
        
        # generate pairs of (pi_i, pi_j) for all i, for all j >= i
        list_tuples = list(itertools.combinations_with_replacement(pi, 2))
        # calculate products
        list_features = []
        for idx in range(len(list_tuples)):
            pair = list_tuples[idx]
            list_features.append(pair[0] * pair[1])
        # append first-order feature
        list_features = list_features + list(pi)
        # append bias
        list_features.append(1)

        return np.array(list_features)


    def calc_gradient_vectorized(self, P, pi):
        
        
        mat1 = special.digamma(self.mat_alpha)
        # Each row of mat2 has same value along the row
        # each element in row i is psi(\sum_j alpha^i_j)
        mat2 = special.digamma( np.ones([self.d, self.d]) * np.sum(self.mat_alpha, axis=1, keepdims=True) )
        # (i,j) element of mat3 is ln(P_{ij})
        try:
            mat3 = np.log(P)
        except Warning:
            print(P)
            mat3 = np.log(P)
        # mat4 is the matrix whose (i,j) entry is 2<phi(i,j,pi), theta>
        # recall that phi is a 3d tensor and phi(i,j,pi) is a vector
        mat4 = 2 * np.tensordot( self.tensor_phi, self.theta.flatten(), axes=1 )

        mat_B = ( -mat1 + mat2 + mat3 ) * mat4

        
        gradient = np.tensordot( mat_B, self.tensor_phi, axes=2 )

        return gradient.reshape(self.dim_theta, 1)

   
    
    def calc_gradient(self, P, pi):
        
        # initialize gradient as column vector
        gradient = np.zeros([self.dim_theta, 1])

        for i in range(self.d):

            
            multiplier = special.digamma( np.sum(self.mat_alpha[i]) )

            for j in range(self.d):
                # 2 * (phi(i,j,pi) dot theta) phi(i,j,pi)
                common_term = 2 * (self.tensor_phi[i,j].dot(self.theta)) * np.transpose(self.tensor_phi[i, j:j+1, :])
                
                
                first_term = - special.digamma(self.mat_alpha[i,j]) * common_term

                
                second_term = multiplier * common_term

               
                third_term = np.log( P[i,j] ) * common_term

                gradient = gradient + first_term + second_term + third_term

        return gradient


    def train_log(self, vector, filename):
        f = open(filename, 'a')
        vector.tofile(f, sep=',', format="%f")
        f.write("\n")
        f.close()
    

    def train(self, num_episodes=4000, gamma=1, lr_critic=0.2, lr_actor=0.6, consecutive=100, file_theta='theta.csv', file_pi='pi.csv', file_cost='cost.csv', write_file=0):
        
        # initialize collection of start states
        self.init_pi0(path_to_dir=r'C:\Users\Satvik\Documents\Projects\Python\RL\MFG\data_train_reordered')
        self.num_start_samples = self.mat_pi0.shape[0] # number of rows

        list_cost = []
        for episode in range(num_episodes):
            print("Episode", episode)
            # Sample starting pi^0 from mat_pi0
            idx_row = np.random.randint(self.num_start_samples)
            # idx_row = 0 # for testing purposes, select the first row of day 1 always #here
            # print("idx_row", idx_row)
            pi = self.mat_pi0[idx_row, :] # row vector

            discount = 1
            total_cost = 0
            num_steps = 0

            # Stop after finishing the iteration when num_steps=15, because
            # at that point pi_next = the predicted distribution at midnight
            while num_steps < 16:
                num_steps += 1

                print("pi\n", pi)
                print(num_steps)
                print(self.theta)

                
                P = self.sample_action(pi)
            
                
                pi_next = np.transpose(P).dot(pi)

                cost = self.calc_cost(P, pi, self.d)
                
                # Calculate TD error
                vec_features_next = self.calc_features(pi_next)
                vec_features = self.calc_features(pi)
                
                delta = cost + gamma*(vec_features_next.dot(self.w)) - (vec_features.dot(self.w))

                
                length = len(vec_features)
                self.w = self.w + lr_critic * delta * vec_features.reshape(length,1)

                # theta update
                gradient = self.calc_gradient_vectorized(P, pi)
                self.theta = self.theta - lr_actor * delta * gradient

                discount = discount * gamma
                pi = pi_next
                total_cost += cost

            list_cost.append(total_cost)

            if (episode % consecutive == 0):
                print("Theta\n", self.theta)
                print("pi\n", pi)
                cost_avg = sum(list_cost)/consecutive
                print("Average cost during previous %d episodes: " % consecutive, str(cost_avg))
                list_cost = []
                if write_file:
                    self.train_log(self.theta, file_theta)
                    self.train_log(pi, file_pi)
                    self.train_log(np.array([cost_avg]), file_cost)


if __name__ == "__main__":
    ac = actor_critic()
    t_start = time.time()
    ac.train(num_episodes=1, gamma=1, lr_critic=0.1, lr_actor=0.1, consecutive=1, write_file=0)
    t_end = time.time()
    print("Time elapsed", t_end - t_start)
