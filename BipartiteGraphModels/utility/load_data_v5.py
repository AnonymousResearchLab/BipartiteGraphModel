import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pandas as pd

np.random.seed(1103)
rd.seed(1103)

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
        
        self.u2i_adj = self.R.tocsr()

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def get_ppr_adj_mat(self):
        start_ts = time()
        try:
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            ppr_adj_mat = sp.load_npz(self.path + '/ppr_adj_mat.npz')
            print('Load ppr adj matrix, time:{}, shape:{}'.format(time()-start_ts, ppr_adj_mat.shape))
        except Exception:
            adj_mat, ppr_adj_mat = self.create_ppr_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/ppr_adj_mat.npz', ppr_adj_mat)
        return ppr_adj_mat

    def create_ppr_adj_mat(self, alpha=0.1):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.tocsr()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def calc_A_hat(adj_matrix: sp.spmatrix):
            nnodes = adj_matrix.shape[0]
            A = adj_matrix + sp.eye(nnodes)
            D_vec = np.sum(A, axis=1).A1
            D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
            D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
            return D_invsqrt_corr @ A @ D_invsqrt_corr

        def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float):
            nnodes = adj_matrix.shape[0]
            M = calc_A_hat(adj_matrix)
            A_inner = sp.eye(nnodes) - (1 - alpha) * M
            return alpha * sp.linalg.inv(A_inner.tocsc()).tocsr()

        ppr_adj_mat = calc_A_hat(adj_mat)

        print('created ppr adj, time:{:5.3f}s, shape:{}'.format(time()-t2, ppr_adj_mat.shape))
        return adj_mat, ppr_adj_mat.tocsr()

    def create_adj_mat(self, alpha=0.1):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state



class DataV2(object):
    def __init__(self, path, batch_size, neighbors_num=[64], weighted_sample_num=5):
        self.path = path
        self.batch_size = batch_size
        self.neighbors_num = neighbors_num
        self.weighted_sample_num = weighted_sample_num
        self.construct_matrix()
        self.user_path, self.item_path = self.build_path()

    def construct_matrix(self):
        train_path = self.path + '/train.txt'
        test_path = self.path + '/test.txt'
        user_list_file = pd.read_csv(self.path + '/user_list.txt', sep=' ')
        item_list_file = pd.read_csv(self.path + '/item_list.txt', sep=' ')

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        start_ts = time()
        users_set = set()
        items_set = set()
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        # max_train_user_item = 0
        # with open(train_path, 'r') as f_train:
        #     for l in f_train.readlines():
        #         if len(l) == 0:
        #             break
        #         items = [int(i) for i in l.strip().split(' ')]
        #         user, train_items = items[0], items[1:]
        #         max_train_user_item = max(
        #             max_train_user_item, len(train_items))

        #         users_set.add(user)
        #         for item in train_items:
        #             u2i_matrix[user, item] = 1.
        #             items_set.add(item)

        # self.max_train_user_item = max_train_user_item

        # self.u2i_matrix = u2i_matrix.tocsr()
        # self.i2u_matrix = u2i_matrix.transpose().tocsr()
        # u2u_matrix = self.u2i_matrix * self.i2u_matrix
        # i2i_matrix = self.i2u_matrix * self.u2i_matrix

        # self.users_arr = np.sort(np.asarray(list(users_set)))
        # self.items_arr = np.sort(np.asarray(list(items_set)))
        # self.u2i_adj = self.construct_adj(self.u2i_matrix, self.n_users)
        # self.i2u_adj = self.construct_adj(self.i2u_matrix, self.n_items)
        # self.u2u_adj = self.construct_adj(u2u_matrix, self.n_users)
        # self.i2i_adj = self.construct_adj(i2i_matrix, self.n_items)

        # # self.test_u2i_dict = dict()
        # test_u2i_matrix = sp.dok_matrix(
        #     (self.n_users, self.n_items), dtype=np.float32)
        # max_test_user_item = 0
        # with open(test_path, 'r') as f_test:
        #     for l in f_test.readlines():
        #         if len(l) == 0:
        #             break
        #         items = [int(i) for i in l.strip().split(' ')]
        #         user, test_items = items[0], items[1:]
        #         # self.test_u2i_dict[user] = test_items
        #         for item in test_items:
        #             test_u2i_matrix[user, item] = 1.
        #         max_test_user_item = max(max_test_user_item, len(test_items))
        # self.max_test_user_item = max_test_user_item
        # self.test_u2i_adj = self.construct_adj(
        #     test_u2i_matrix.tocsr(), self.n_users)

        self.train_items, self.test_set = {}, {}
        with open(train_path) as f_train:
            with open(test_path) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1
                        items_set.add(i)

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
        
        self.items_arr = np.sort(np.asarray(list(items_set)))
        self.u2i_matrix = self.R.tocsr()
        self.i2u_matrix = self.R.transpose().tocsr()
        u2u_matrix = self.u2i_matrix * self.i2u_matrix
        i2i_matrix = self.i2u_matrix * self.u2i_matrix
        self.u2i_adj = self.construct_adj(self.u2i_matrix, self.n_users)
        self.i2u_adj = self.construct_adj(self.i2u_matrix, self.n_items)
        self.u2u_adj = self.construct_e2e_adj(u2u_matrix, self.n_users)
        self.i2i_adj = self.construct_e2e_adj(i2i_matrix, self.n_items)

        # print('Start construct negative')
        # t1 = time()
        # self.negative_u2i_pool = [self.get_negative_pool(self.i2i_adj, user_items, self.items_arr) for user_items in self.u2i_adj]
        # t2 = time()
        # print('Construct negative_u2i_pool. Time: {:5.3f}'.format(t2 - t1))
        # self.negative_u2u_pool = [self.get_negative_pool(self.u2u_adj, user_users, self.users_arr) for user_users in self.u2u_adj]
        # t3 = time()
        # print('Construct negative_u2u_pool. Time: {:5.3f}'.format(t3 - t2))
        # self.negative_i2i_pool = [self.get_negative_pool(self.i2i_adj, item_items, self.items_arr) for item_items in self.i2i_adj]
        self.u_i_neg_pool, self.i_u_neg_pool, self.u_u_neg_pool, self.i_i_neg_pool = self.construct_negative_pool()
        self.u2u_adj = self.construct_top_k_adj(self.u2u_adj)
        self.i2i_adj = self.construct_top_k_adj(self.i2i_adj)
        end_ts = time()
        print('Construct adjust matrix. Time: {:5.3f}'.format(end_ts - start_ts))

    def create_adj_mat(self, alpha=0.1):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def check_duplicate(self):
        for i, (a, b) in enumerate(zip(self.u2i_adj, self.test_u2i_adj)):
            intersect_list = np.intersect1d(a, b)
            if len(intersect_list) > 0:
                print('Find duplicate, id:{}, items:{}.'.format(i, intersect_list))
        # print(self.users_arr)
        # print(self.items_arr)

    def construct_adj(self, csr_mat, n_element):
        adj_list = []
        for i in range(n_element):
            next_arr = np.asarray(
                csr_mat.indices[csr_mat.indptr[i]:csr_mat.indptr[i+1]])
            adj_list.append(next_arr)
            # if i in [12, 30, 39, 52]:
            #     print(i, next_arr)
        adj_list = np.asarray(adj_list)
        return adj_list

    def construct_e2e_adj(self, csr_mat, n_element):
        adj_list = []
        for i in range(n_element):
            indices_arr = np.asarray(csr_mat.indices[csr_mat.indptr[i]:csr_mat.indptr[i+1]])
            data_arr = np.asarray(csr_mat.data[csr_mat.indptr[i]:csr_mat.indptr[i+1]])
            adj_list.append([indices_arr, data_arr])
        adj_list = np.asarray(adj_list)
        return adj_list

    def construct_top_k_adj(self, e2e_adj):
        adj_list = []
        for indices_arr, data_arr in e2e_adj:
            indices_arr, data_arr = self.top_k(indices_arr, data_arr)
            data_arr = np.cumsum(data_arr / np.sum(data_arr))
            data_arr[-1] = 1.0
            adj_list.append([indices_arr, data_arr])
        adj_list = np.asarray(adj_list)
        return adj_list

    def top_k(self, objects, weights, k=100):
        sorted_index = np.argsort(weights)[::-1]
        sorted_objects = objects[sorted_index][:k]
        sorted_weights = weights[sorted_index][:k]
        return sorted_objects, sorted_weights

    # def construct_negative_pool(self):
    #     negative_u2i_pool = [self.get_negative_pool(self.i2i_adj, user_items, self.items_arr) for user_items in self.u2i_adj]
    #     negative_u2u_pool = [self.get_negative_pool(self.u2u_adj, user_users, self.users_arr) for user_users in self.u2u_adj]
    #     negative_i2i_pool = [self.get_negative_pool(self.i2i_adj, item_items, self.items_arr) for item_items in self.i2i_adj]

    def get_negative_pool(self, adj, entities, all_entities):
        pos_neighbors = np.union1d(
            entities, np.unique(np.concatenate(adj[entities])))
        neg_entities = np.setdiff1d(all_entities, pos_neighbors)
        return neg_entities

    def random_select(self, arr, num):
        idxs = np.random.randint(len(arr), size=num)
        return arr[idxs]

    def negative_select(self, pool, positive, num):
        # negative_pool = np.setdiff1d(pool, positive)
        # idxs = np.random.randint(len(negative_pool), size=num)
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_id = np.random.randint(low=0, high=len(pool),size=1)[0]
            if neg_id not in positive and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    # uniform sampling
    def sample(self):
        # center users
        batch_users = np.random.choice(self.users_arr, size=self.batch_size)

        # all positive users of centor users
        pos_users = self.u2u_adj[batch_users]
        # select batch positive users
        batch_pos_users = np.asarray(
            [self.random_select(users, 1)[0] for users in pos_users])
        # select batch negative users
        batch_neg_users = np.asarray([self.negative_select(
            self.users_arr, users, 1)[0] for users in pos_users])

        batch_users_items = self.u2i_adj[batch_users]
        batch_items = np.asarray([self.random_select(item, 1)[0]
                                  for item in batch_users_items])

        pos_items = self.i2i_adj[batch_items]
        batch_pos_items = np.asarray(
            [self.random_select(item, 1)[0] for item in pos_items])

        batch_neg_items = np.asarray([self.negative_select(
            self.items_arr, items, 1)[0] for items in pos_items])

        batch_users = self.sample_hops(self.u2u_adj, batch_users)
        batch_pos_users = self.sample_hops(self.u2u_adj, batch_pos_users)
        batch_neg_users = self.sample_hops(self.u2u_adj, batch_neg_users)
        batch_items = self.sample_hops(self.i2i_adj, batch_items)
        batch_pos_items = self.sample_hops(self.i2i_adj, batch_pos_items)
        batch_neg_items = self.sample_hops(self.i2i_adj, batch_neg_items)

        return batch_users, batch_pos_users, batch_neg_users, batch_items, batch_pos_items, batch_neg_items

    def sample_v2(self):
        # batch_users = np.random.choice(self.users_arr, size=self.batch_size)
        # pos_users = self.u2u_adj[batch_users]
        # batch_pos_users = np.asarray([self.random_select(users, 1)[0] for users in pos_users])
        # batch_neg_users = np.asarray([self.negative_select(self.users_arr, users, 1)[0] for users in pos_users])

        # user item_pos, item_neg
        batch_users = np.random.choice(self.users_arr, size=self.batch_size)
        batch_pos_items = np.asarray(
            [self.random_select(item, 1)[0] for item in self.u2i_adj[batch_users]])
        # batch_neg_items = np.asarray([self.random_select(neg_items, 1) for neg_items in self.negative_u2i_pool[batch_users]])
        batch_neg_items = np.asarray([self.negative_select(self.items_arr, item, 1)[
                                     0] for item in self.u2i_adj[batch_users]])

        batch_u, batch_u_p, batch_u_n = self.sample_center(
            batch_users, self.u2u_adj)
        batch_p_i, batch_p_i_p, batch_p_i_n = self.sample_center(
            batch_pos_items, self.i2i_adj)
        batch_n_i, batch_n_i_p, batch_n_i_n = self.sample_center(
            batch_neg_items, self.i2i_adj)

        samples = [
            batch_u, batch_u_p, batch_u_n,
            batch_p_i, batch_p_i_p, batch_p_i_n,
            batch_n_i, batch_n_i_p, batch_n_i_n
        ]

        return samples

    def sample_v3(self, user_path=[25,10,25], item_path=[10,25,10]):
        batch_users = np.random.choice(self.users_arr, size=self.batch_size)
        batch_pos_items = np.asarray(
            [self.random_select(item, 1)[0] for item in self.u2i_adj[batch_users]])
        # batch_neg_items = np.asarray([self.random_select(neg_items, 1) for neg_items in self.negative_u2i_pool[batch_users]])
        batch_neg_items = np.asarray(
            [self.negative_select(self.items_arr, item, 1)[0] for item in self.u2i_adj[batch_users]])

        batch_user_samples = self.sample_user_path(batch_users, user_path)
        batch_pos_items_samples = self.sample_item_path(batch_pos_items, item_path)
        batch_neg_items_samples = self.sample_item_path(batch_neg_items, item_path)

        return batch_user_samples, batch_pos_items_samples, batch_neg_items_samples

    def sample_attention(self):
        batch_users = np.random.choice(self.users_arr, size=self.batch_size)
        batch_pos_items = np.asarray(
            [self.random_select(item, 1)[0] for item in self.u2i_adj[batch_users]])
        batch_neg_items = np.asarray(
            [self.negative_select(self.items_arr, item, 1)[0] for item in self.u2i_adj[batch_users]])

        users_samples = self.sample_attention_user(batch_users)
        pos_items_samples = self.sample_attention_item(batch_pos_items)
        neg_items_samples = self.sample_attention_item(batch_neg_items)

        return (users_samples, pos_items_samples, neg_items_samples)

    def sample_attention_user(self, batch_users):
        # users_samples = (batch_users, self.sample_u2i(batch_users, 50), self.sample_u2u(batch_users, 50))
        batch_items_pos = self.sample_u2i(batch_users, 50)
        batch_users_pos = self.sample_i2u(batch_items_pos, 1)
        return (batch_users, batch_items_pos, batch_users_pos)
    
    def sample_attention_item(self, batch_items):
        # items_samples = (batch_items, self.sample_i2u(batch_items, 50), self.sample_i2i(batch_items, 50))
        batch_users_pos = self.sample_i2u(batch_items, 50)
        batch_items_pos = self.sample_u2i(batch_users_pos, 1)

        # exp1 (batch_items, batch_items_pos, batch_users_pos)
        # exp2 (batch_items, batch_users_pos, batch_items_pos)
        return (batch_items, batch_items_pos, batch_users_pos)

    def sample_user_path(self, users, path_nums=[25,10,25]):
        one_hop_items = self.sample_u2i(users, path_nums[0])
        two_hop_users = self.sample_i2u(one_hop_items, path_nums[1])
        three_hop_items = self.sample_u2i(two_hop_users, path_nums[2])
        samples = [users, one_hop_items, two_hop_users, three_hop_items]
        return samples

    def sample_item_path(self, items, path_nums=[10,25,10]):
        one_hop_users = self.sample_i2u(items, path_nums[0])
        two_hop_items = self.sample_u2i(one_hop_users, path_nums[1])
        three_hop_users = self.sample_i2u(two_hop_items, path_nums[2])
        samples = [items, one_hop_users, two_hop_items, three_hop_users]
        return samples

    def sample_center(self, entities, adj):
        pos_entities = adj[entities]
        batch_pos_entities = np.asarray(
            [self.random_select(users, 1)[0] for users in pos_entities])
        batch_neg_entities = np.asarray([self.negative_select(
            self.users_arr, users, 1)[0] for users in pos_entities])
        # batch_neg_entities = np.asarray([self.random_select(neg_entities) for neg_entities in neg_pool[entities]])

        batch_entities = self.sample_hops(adj, entities)
        batch_pos_entities = self.sample_hops(adj, batch_pos_entities)
        batch_neg_entities = self.sample_hops(adj, batch_neg_entities)

        return batch_entities, batch_pos_entities, batch_neg_entities

    def sample_test_single(self, size=3):
        users = np.random.choice(list(self.test_u2i_dict.keys()), size=size)
        return self.sample_hops(self.u2u_adj, users)

    def sample_hops(self, adj, entities):
        samples = [entities]
        current_hop = entities
        for num in self.neighbors_num:
            neighbors_mat = adj[current_hop]
            chosen = np.concatenate(
                [self.random_select(neighbors, num) for neighbors in neighbors_mat])
            samples.append(chosen)
            current_hop = chosen
        return samples

    def sample_u2i(self, users, num_items):
        items_mat = self.u2i_adj[users]
        selected = np.concatenate([self.random_select(items, num_items) for items in items_mat])
        return selected

    def sample_i2u(self, items, num_users):
        users_mat = self.i2u_adj[items]
        selected = np.concatenate([self.random_select(users, num_users) for users in users_mat])
        return selected

    def sample_u2u(self, users, num_users):
        users_mat = self.u2u_adj[users]
        selected = np.concatenate([self.random_select(users, num_users) for users in users_mat])
        return selected

    def sample_i2i(self, items, num_items):
        items_mat = self.i2i_adj[items]
        selected = np.concatenate([self.random_select(items, num_items) for items in items_mat])
        return selected

    def sample_source_target(self, center, sample_num, source='u', target='i'):
        if source == 'u' and target == 'i':
            entities_mat = self.u2i_adj[center]
        elif source == 'u' and target == 'u':
            entities_mat = self.u2u_adj[center]
        elif source == 'i' and target == 'i':
            entities_mat = self.i2i_adj[center]
        elif source == 'i' and target == 'u':
            entities_mat = self.i2u_adj[center]
        else:
            raise ValueError('source and target should be \'u\' or \'i\'')
        selected = np.concatenate([self.random_select(entity, sample_num) for entity in entities_mat])
        return selected

    def sample_path(self, center, path=['u','i']):
        samples = [np.asarray(center)]
        current_center = center
        assert len(path) == len(self.neighbors_num) + 1, 'len(path) should be equals to len(neighbors_num) + 1'
        for i, neigh_num in enumerate(self.neighbors_num):
            source, target = path[i], path[i+1]
            sample_num = neigh_num
            current_center = np.asarray(self.sample_source_target(current_center, sample_num, source, target))
            samples.append(current_center)
        return samples

    def build_path(self):
        user_path, item_path = [], []
        for i in range(len(self.neighbors_num) + 1):
            if i % 2 == 0:
                user_path.append('u')
                item_path.append('i')
            else:
                user_path.append('i')
                item_path.append('u')
        return user_path, item_path

    def construct_negative_pool(self):
        users_items_negative_pool = []
        items_users_negative_pool = []
        users_users_negative_pool = []
        items_items_negative_pool = []
        for pos_items in self.u2i_adj:
            users_items_negative_pool.append(np.setdiff1d(self.items_arr, pos_items))
        for pos_users in self.i2u_adj:
            items_users_negative_pool.append(np.setdiff1d(self.exist_users, pos_users))
        for users_users in self.u2u_adj:
            users_users_negative_pool.append(np.setdiff1d(self.exist_users, users_users[0]))
        for items_items in self.i2i_adj:
            items_items_negative_pool.append(np.setdiff1d(self.items_arr, items_items[0]))
        result = [np.asarray(users_items_negative_pool), np.asarray(items_users_negative_pool),
                  np.asarray(users_users_negative_pool), np.asarray(items_items_negative_pool)]
        return result

    def sample_v5(self):
        batch_users = np.asarray(rd.sample(self.exist_users, self.batch_size))
        batch_pos_items = np.asarray([self.random_select(item, 1)[0] for item in self.u2i_adj[batch_users]])
        batch_neg_items = np.asarray([self.negative_select(self.items_arr, item, 1)[0] for item in self.u2i_adj[batch_users]])

        batch_user_samples = self.sample_path(batch_users, path=self.user_path)
        batch_pos_iu_samples = self.sample_path(batch_pos_items, path=self.item_path)
        batch_neg_iu_samples = self.sample_path(batch_neg_items, path=self.item_path)

        # batch_users_pos = np.asarray([self.weighted_sampling(user[0], user[1]) for user in self.u2u_adj[batch_users]])
        batch_users_pos = np.concatenate([self.weighted_sampling(user[0], user[1], self.weighted_sample_num) for user in self.u2u_adj[batch_users]])
        batch_users_neg = np.concatenate([self.random_select(user, self.weighted_sample_num) for user in self.u_u_neg_pool[batch_users]])

        # batch_pos_items_pos = np.asarray([self.weighted_sampling(item[0], item[1]) for item in self.i2i_adj[batch_pos_items]])
        batch_pos_items_pos = np.concatenate([self.weighted_sampling(item[0], item[1], self.weighted_sample_num) for item in self.i2i_adj[batch_pos_items]])
        batch_pos_items_neg = np.concatenate([self.random_select(item, self.weighted_sample_num) for item in self.i_i_neg_pool[batch_pos_items]])

        # batch_neg_items_pos = np.asarray([self.weighted_sampling(item[0], item[1]) for item in self.i2i_adj[batch_neg_items]])
        batch_neg_items_pos = np.concatenate([self.weighted_sampling(item[0], item[1], self.weighted_sample_num) for item in self.i2i_adj[batch_neg_items]])
        batch_neg_items_neg = np.concatenate([self.random_select(item, self.weighted_sample_num) for item in self.i_i_neg_pool[batch_neg_items]])

        return batch_user_samples, batch_pos_iu_samples, batch_neg_iu_samples, batch_users_pos, batch_users_neg, batch_pos_items_pos, batch_pos_items_neg, batch_neg_items_pos, batch_neg_items_neg
        # return batch_users, batch_pos_items, batch_neg_items

    def weighted_sampling(self, objects, cs, num=1):
        return [objects[np.searchsorted(cs, np.random.rand())] for i in range(num)]

    def sample_v4(self):
        batch_users = np.asarray(rd.sample(self.exist_users, self.batch_size))
        batch_pos_items = np.asarray([self.random_select(item, 1)[0] for item in self.u2i_adj[batch_users]])
        batch_neg_items = np.asarray([self.negative_select(self.items_arr, item, 1)[0] for item in self.u2i_adj[batch_users]])

        batch_user_samples = self.sample_path(batch_users, path=self.user_path)
        batch_pos_iu_samples = self.sample_path(batch_pos_items, path=self.item_path)
        batch_neg_iu_samples = self.sample_path(batch_neg_items, path=self.item_path)

        batch_users_pos = np.asarray([self.weighted_sampling(user[0], user[1]) for user in self.u2u_adj[batch_users]])
        batch_users_neg = np.asarray([self.negative_select(self.exist_users, user[0], 1)[0] for user in self.u2u_adj[batch_users]])

        batch_pos_items_pos = np.asarray([self.weighted_sampling(item[0], item[1]) for item in self.i2i_adj[batch_pos_items]])
        batch_pos_items_neg = np.asarray([self.negative_select(self.items_arr, item[0], 1)[0] for item in self.i2i_adj[batch_pos_items]])

        batch_neg_items_pos = np.asarray([self.weighted_sampling(item[0], item[1]) for item in self.i2i_adj[batch_neg_items]])
        batch_neg_items_neg = np.asarray([self.negative_select(self.items_arr, item[0], 1)[0] for item in self.i2i_adj[batch_neg_items]])

        return batch_user_samples, batch_pos_iu_samples, batch_neg_iu_samples, batch_users_pos, batch_users_neg, batch_pos_items_pos, batch_pos_items_neg, batch_neg_items_pos, batch_neg_items_neg

if __name__ == '__main__':
    data_loader = DataV2("Data/gowalla", 1024, neighbors_num=[64])
    t1 = time()
    for i in range(1000):
        result = data_loader.sample_v5()
    t2 = time()
    print('Time:{}'.format((t2 - t1) / 100))
    # print(batch_user_samples)
    # print(batch_pos_iu_samples)
    # print(batch_neg_iu_samples)

