import numpy as np
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test_v5 import *

class GACSE(object):
    def __init__(self, data_config, pretrain_data=None):
        self.model_type = 'GACSE'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.weighted_sample_num = args.weighted_sample_num

        self.n_heads = 3

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.neighbors_num = [1] + eval(args.neighbors_num)

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.node_dropout_flag = args.node_dropout_flag

        self.A_fold_hat = self.calc_split_A_hat()

        self.verbose = args.verbose

        self.users = [tf.placeholder(tf.int32, shape=(None,)) for i in self.neighbors_num]
        self.pos_items_u = [tf.placeholder(tf.int32, shape=(None,)) for i in self.neighbors_num]
        self.neg_items_u = [tf.placeholder(tf.int32, shape=(None,)) for i in self.neighbors_num]

        self.users_pos = tf.placeholder(tf.int32, shape=(None,))
        self.users_neg = tf.placeholder(tf.int32, shape=(None,))

        self.pos_items_pos = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_neg = tf.placeholder(tf.int32, shape=(None,))

        self.neg_items_pos = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items_neg = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        self.weights = self._init_weights()

        self.ua_embeddings, self.ia_embeddings = self._trans_all_embed(self.weights['user_embedding'], self.weights['item_embedding'])

        self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings = self._create_am_embed(self.ua_embeddings, self.ia_embeddings)
        # self.u_g_embeddings = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users[0])
        # self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items_u[0])
        # self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items_u[0])

        self.u_cse = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users[0])
        self.u_p_cse = tf.nn.embedding_lookup(self.weights['cse_user_embedding'], self.users_pos)
        self.u_n_cse = tf.nn.embedding_lookup(self.weights['cse_user_embedding'], self.users_neg)

        self.pos_i_cse = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items_u[0])
        self.pos_i_pos_cse = tf.nn.embedding_lookup(self.weights['cse_item_embedding'], self.pos_items_pos)
        self.pos_i_neg_cse = tf.nn.embedding_lookup(self.weights['cse_item_embedding'], self.pos_items_neg)

        self.neg_i_cse = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items_u[0])
        self.neg_i_pos_cse = tf.nn.embedding_lookup(self.weights['cse_item_embedding'], self.neg_items_pos)
        self.neg_i_neg_cse = tf.nn.embedding_lookup(self.weights['cse_item_embedding'], self.neg_items_neg)

        self.u_g_embeddings += [self.u_cse]
        self.pos_i_g_embeddings += [self.pos_i_cse]
        self.neg_i_g_embeddings += [self.neg_i_cse]

        self.u_g_embeddings = tf.concat(self.u_g_embeddings, axis=1)
        self.pos_i_g_embeddings = tf.concat(self.pos_i_g_embeddings, axis=1)
        self.neg_i_g_embeddings = tf.concat(self.neg_i_g_embeddings, axis=1)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)

        self.u_loss = self.create_point_wise_loss(self.u_cse,
                                                  self.u_p_cse,
                                                  self.u_n_cse)
        self.ip_loss = self.create_point_wise_loss(self.pos_i_cse,
                                                   self.pos_i_pos_cse,
                                                   self.pos_i_neg_cse)
        self.in_loss = self.create_point_wise_loss(self.neg_i_cse,
                                                   self.neg_i_pos_cse,
                                                   self.neg_i_neg_cse)
        
        self.pointwise_loss = 1e-4 * (self.u_loss + self.ip_loss + self.in_loss)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss + self.pointwise_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def calc_split_A_hat(self):
        A_hat = self.norm_adj

        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(A_hat)
        else:
            A_fold_hat = self._split_A_hat(A_hat)
        return A_fold_hat

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['cse_user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='cse_user_embedding')
        all_weights['cse_item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='cse_item_embedding')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        all_weights['W_trans'] = tf.Variable(
            initializer([self.emb_dim, self.weight_size_list[0]]), name='W_trans')
        all_weights['b_trans'] = tf.Variable(
            initializer([1, self.weight_size_list[0]]), name='b_trans')

        all_weights['W_att'] = tf.Variable(
            initializer([self.emb_dim*2, self.weight_size_list[0]]), name='W_att')
        all_weights['V_att'] = tf.Variable(
            initializer([self.weight_size_list[0], 1]), name='V_att')

        all_weights['W_att_l'] = tf.Variable(
            initializer([self.weight_size_list[0], self.weight_size_list[0]]), name='W_att_l')
        all_weights['W_att_r'] = tf.Variable(
            initializer([self.weight_size_list[0], self.weight_size_list[0]]), name='W_att_r')

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _trans_all_embed(self, users_embedings, item_embeddings):
        local_embeddings = tf.concat([users_embedings, item_embeddings], axis=0)

        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(self.A_fold_hat[f], local_embeddings))

        message_embeddings = tf.concat(temp_embed, 0)

        message_embeddings = tf.nn.leaky_relu(
            tf.matmul(message_embeddings, self.weights['W_trans']) + self.weights['b_trans']
        )

        users_embedings, item_embeddings = tf.split(message_embeddings, [self.n_users, self.n_items], 0)

        return users_embedings, item_embeddings

    def _create_am_embed(self, user_embeddings, item_embeddings):
        # u_g_embeddings = [tf.nn.embedding_lookup(self.weights['user_embedding'], self.users[0])]
        # pos_i_g_embeddings = [tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items_u[0])]
        # neg_i_g_embeddings = [tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items_u[0])]
        u_g_embeddings = []
        pos_i_g_embeddings = []
        neg_i_g_embeddings = []
        for i in range(len(self.neighbors_num) - 1):
            u_x, u_neib = tf.nn.embedding_lookup(user_embeddings, self.users[i]), tf.nn.embedding_lookup(item_embeddings, self.users[i+1])
            pos_iu_x, pos_iu_neib = tf.nn.embedding_lookup(item_embeddings, self.pos_items_u[i]), tf.nn.embedding_lookup(user_embeddings, self.pos_items_u[i+1])
            neg_iu_x, neg_iu_neib = tf.nn.embedding_lookup(item_embeddings, self.neg_items_u[i]), tf.nn.embedding_lookup(user_embeddings, self.neg_items_u[i+1])
            # pos_ii_x, pos_ii_neib = tf.nn.embedding_lookup(item_embeddings, self.pos_items_i[i]), tf.nn.embedding_lookup(item_embeddings, self.pos_items_i[i+1])
            # neg_ii_x, neg_ii_neib = tf.nn.embedding_lookup(item_embeddings, self.neg_items_i[i]), tf.nn.embedding_lookup(item_embeddings, self.neg_items_i[i+1])

            u_g = self._agg(u_x, u_neib, self.neighbors_num[i+1])
            pos_i_g = self._agg(pos_iu_x, pos_iu_neib, self.neighbors_num[i+1])
            neg_i_g = self._agg(neg_iu_x, neg_iu_neib, self.neighbors_num[i+1])
            # pos_ii_g = self._agg(pos_ii_x, pos_ii_neib)
            # neg_ii_g = self._agg(neg_ii_x, neg_ii_neib)

            # pos_i_g = alpha * pos_iu_g + (1 - alpha) * pos_ii_g
            # neg_i_g = alpha * neg_iu_g + (1 - alpha) * neg_ii_g

            u_g_embeddings.append(u_g)
            pos_i_g_embeddings.append(pos_i_g)
            neg_i_g_embeddings.append(neg_i_g)

        # u_g_embeddings = tf.concat(u_g_embeddings, axis=1)
        # pos_i_g_embeddings = tf.concat(pos_i_g_embeddings, axis=1)
        # neg_i_g_embeddings = tf.concat(neg_i_g_embeddings, axis=1)

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def _agg(self, x, neighbors, num_neighbors):
        x_att = tf.reshape(x, (-1, 1, self.weight_size_list[0]))
        neighbors_att = tf.reshape(neighbors, (-1, num_neighbors, self.weight_size_list[0]))
        # neighbors_att = tf.transpose(neighbors_att, perm=[0,2,1])
        x_att = tf.tile(x_att, [1, num_neighbors, 1])
        all_emb = tf.concat([x_att, neighbors_att], axis=2)
        all_emb = tf.reshape(all_emb, (-1, self.weight_size_list[0]*2))

        scores = tf.nn.tanh(tf.matmul(all_emb, self.weights['W_att']))
        scores = tf.matmul(scores, self.weights['V_att'])
        scores = tf.reshape(scores, (-1, 1, num_neighbors))
        scores = tf.nn.softmax(scores, axis=2)

        # neighbors_att = tf.transpose(neighbors_att, perm=[0,2,1])
        agg_neib = tf.squeeze(tf.matmul(scores, neighbors_att))

        # out = tf.nn.leaky_relu(agg_neib + tf.multiply(x, agg_neib))
        out_l = tf.nn.leaky_relu(tf.matmul(x + agg_neib, self.weights['W_att_l']))
        out_r = tf.nn.leaky_relu(tf.matmul(tf.multiply(x, agg_neib), self.weights['W_att_r']))
        # out = tf.matmul(out, self.weights['W_att_trans']) + self.weights['b_att_trans']
        out = out_l + out_r
        return tf.nn.l2_normalize(out, axis=1)

    # def create_point_wise_loss(self, entity, pos_entity, neg_entity):
    #     epsilon = 1e-7

    #     pos_scores = tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(entity, pos_entity), axis=1)) + epsilon)
    #     neg_scores = tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(entity, neg_entity), axis=1)) + epsilon)

    #     return tf.negative(tf.reduce_mean(pos_scores - neg_scores)) / self.batch_size
    def create_point_wise_loss(self, entity, pos_entity, neg_entity):
        epsilon = 1e-7
        entity = tf.reshape(entity, (-1, 1, self.emb_dim))
        pos_entity = tf.reshape(pos_entity, (-1, self.weighted_sample_num, self.emb_dim))
        pos_entity = tf.transpose(pos_entity, perm=[0,2,1])
        pos_scores = tf.squeeze(tf.reduce_sum(tf.clip_by_value(tf.log(tf.nn.sigmoid(tf.matmul(entity, pos_entity)) + epsilon), -10, 0), axis=2))
        neg_entity = tf.reshape(neg_entity, (-1, self.weighted_sample_num, self.emb_dim))
        neg_entity = tf.transpose(neg_entity, perm=[0,2,1])
        neg_scores = tf.squeeze(tf.reduce_sum(tf.clip_by_value(tf.log(tf.nn.sigmoid(tf.matmul(entity, neg_entity)) + epsilon), -10, 0), axis=2))
        # neg_scores = tf.log(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(entity, neg_entity), axis=1)) + epsilon)

        return tf.negative(tf.reduce_mean(pos_scores - neg_scores))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        pos_neg_scores = tf.reduce_sum(tf.multiply(pos_items, neg_items), axis=1)
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        
        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores  - tf.maximum(0.0, pos_neg_scores))))

        # emb_loss = self.decay * regularizer
        emb_loss = emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # if args.adj_type == 'plain':
    #     config['norm_adj'] = plain_adj
    #     print('use the plain adjacency matrix')

    # elif args.adj_type == 'norm':
    #     config['norm_adj'] = norm_adj
    #     print('use the normalized adjacency matrix')

    # elif args.adj_type == 'gcmc':
    #     config['norm_adj'] = mean_adj
    #     print('use the gcmc adjacency matrix')

    # else:
    #     config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
    #     print('use the mean adjacency matrix')
    config['norm_adj'] = norm_adj
    # config['u2i_adj'] = data_generator.u2i_adj
    # config['i2u_adj'] = data_generator.i2u_adj
    # config['u2u_adj'] = data_generator.u2u_adj
    # config['i2i_adj'] = data_generator.i2i_adj
    # config['i2u_adj'] = data_generator.i2u_adj

    t0 = time()

    # if args.pretrain == -1:
    #     pretrain_data = load_pretrained_data()
    # else:
    #     pretrain_data = None

    # users, pos_items_u, neg_items_u, pos_items_i, neg_items_i = data_generator.sample_v4()
    # print('debug:', [len(u) for u in users])

    model = GACSE(data_config=config, pretrain_data=None)

    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, pointwise_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        
        # feed_dict_user = {u : d for u, d in zip(model.users, users)}
        # feed_dict_pos_item_u = {i : d for i, d in zip(model.pos_items_u, pos_items_u)}
        # feed_dict_neg_item_u = {i : d for i, d in zip(model.neg_items_u, neg_items_u)}
        # feed_dict_pos_item_i = {i : d for i, d in zip(model.pos_items_i, pos_items_i)}
        # feed_dict_neg_item_i = {i : d for i, d in zip(model.neg_items_i, neg_items_i)}
        for idx in range(n_batch):
            # start_ts = time()
            users, pos_items_u, neg_items_u, users_pos, users_neg, pos_items_pos, pos_items_neg, neg_items_pos, neg_items_neg = data_generator.sample_v5()
            # data_ts = time()
            # print('sample time', data_ts - start_ts)
            # print([len(u) for u in users])
            feed_dict = {u : d for u, d in zip(model.users, users)}
            feed_dict.update({i : d for i, d in zip(model.pos_items_u, pos_items_u)})
            feed_dict.update({i : d for i, d in zip(model.neg_items_u, neg_items_u)})
            # feed_dict.update({i : d for i, d in zip(model.pos_items_i, pos_items_i)})
            # feed_dict.update({i : d for i, d in zip(model.neg_items_i, neg_items_i)})
            feed_dict.update({model.node_dropout: eval(args.node_dropout), 
                            model.mess_dropout: eval(args.mess_dropout),
                            model.users_pos: users_pos,
                            model.users_neg: users_neg,
                            model.pos_items_pos: pos_items_pos,
                            model.pos_items_neg: pos_items_neg,
                            model.neg_items_pos: neg_items_pos,
                            model.neg_items_neg: neg_items_neg})
            # feed_dict_ts = time()
            # print('feed_dict time', feed_dict_ts - data_ts)

            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_pointwise_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.pointwise_loss],
                               feed_dict=feed_dict)
            # train_ts = time()
            # print('train time', train_ts - feed_dict_ts)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            pointwise_loss += batch_pointwise_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, pointwise_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, pointwise_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()

