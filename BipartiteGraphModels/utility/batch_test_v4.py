'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data_v4 import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
neighbors_num = eval(args.neighbors_num)
data_generator = DataV2(path=args.data_path + args.dataset, batch_size=args.batch_size, neighbors_num=neighbors_num)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    # print(u)
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

# batch_user_samples = self.sample_path(batch_users, path=data_generator.user_path, neighbors_num=neighbors_num)
# batch_pos_iu_samples = self.sample_path(batch_pos_items, path=data_generator.item_path, neighbors_num=neighbors_num)
# batch_neg_iu_samples = self.sample_path(batch_neg_items, path=data_generator.item_path, neighbors_num=neighbors_num)
# batch_pos_ii_samples = self.sample_path(batch_pos_items, path=['i', 'i'], neighbors_num=neighbors_num)
# batch_neg_ii_samples = self.sample_path(batch_neg_items, path=['i', 'i'], neighbors_num=neighbors_num)
def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = data_generator.sample_path(test_users[start: end], path=data_generator.user_path)
        user_batch_u = data_generator.sample_path(test_users[start: end], path=['u', 'u'])

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch_u = data_generator.sample_path(range(i_start, i_end), path=data_generator.item_path)
                item_batch_i = data_generator.sample_path(range(i_start, i_end), path=['i', 'i'])

                # users, pos_items_u, neg_items_u, pos_items_i, neg_items_i = data_generator.sample_v4()
                # # print([len(u) for u in users])
                # feed_dict = {u : d.astype(np.int) for u, d in zip(model.users, users)}
                # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_u, pos_items_u)})
                # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.neg_items_u, neg_items_u)})
                # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_i, pos_items_i)})
                # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.neg_items_i, neg_items_i)})
                # feed_dict.update({model.node_dropout: eval(args.node_dropout), 
                #                 model.mess_dropout: eval(args.mess_dropout)})

                feed_dict = {u : d.astype(np.int) for u, d in zip(model.users, user_batch)}
                feed_dict.update({i : d.astype(np.int) for i, d in zip(model.users_u, user_batch_u)})
                feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_u, item_batch_u)})
                feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_i, item_batch_i)})

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)
                else:
                    feed_dict.update({model.node_dropout: [0.]*len(eval(args.layer_size)),
                                      model.mess_dropout: [0.]*len(eval(args.layer_size))})
                    i_rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            item_batch_u = data_generator.sample_path(item_batch, path=data_generator.item_path)
            item_batch_i = data_generator.sample_path(item_batch, path=['i', 'i'])
            # item_batch_i = data_generator.sample_path(item_batch, path=['i', 'i'], neighbors_num=neighbors_num)

            feed_dict = {u : d.astype(np.int) for u, d in zip(model.users, user_batch)}
            feed_dict.update({i : d.astype(np.int) for i, d in zip(model.users_u, user_batch_u)})
            feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_u, item_batch_u)})
            feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_i, item_batch_i)})
            # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_i, item_batch_i)})

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)
            else:
                feed_dict.update({model.node_dropout: [0.] * len(eval(args.layer_size)),
                                  model.mess_dropout: [0.] * len(eval(args.layer_size))})
                rate_batch = sess.run(model.batch_ratings, feed_dict=feed_dict)

        user_batch_rating_uid = zip(rate_batch, user_batch[0])
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result

# def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
#     result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
#               'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

#     pool = multiprocessing.Pool(cores)

#     u_batch_size = BATCH_SIZE * 2
#     i_batch_size = BATCH_SIZE

#     test_users = users_to_test
#     n_test_users = len(test_users)
#     n_user_batchs = n_test_users // u_batch_size + 1

#     count = 0

#     for u_batch_id in range(n_user_batchs):
#         start = u_batch_id * u_batch_size
#         end = (u_batch_id + 1) * u_batch_size

#         user_batch = test_users[start: end]

#         if batch_test_flag:

#             n_item_batchs = ITEM_NUM // i_batch_size + 1
#             rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

#             i_count = 0
#             for i_batch_id in range(n_item_batchs):
#                 i_start = i_batch_id * i_batch_size
#                 i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

#                 item_batch = range(i_start, i_end)

#                 if drop_flag == False:
#                     i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
#                                                                 model.pos_items: item_batch})
#                 else:
#                     i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
#                                                                 model.pos_items: item_batch,
#                                                                 model.node_dropout: [0.]*len(eval(args.layer_size)),
#                                                                 model.mess_dropout: [0.]*len(eval(args.layer_size))})
#                 rate_batch[:, i_start: i_end] = i_rate_batch
#                 i_count += i_rate_batch.shape[1]

#             assert i_count == ITEM_NUM

#         else:
#             item_batch = range(ITEM_NUM)

#             if drop_flag == False:
#                 rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
#                                                               model.pos_items: item_batch})
#             else:
#                 rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
#                                                               model.pos_items: item_batch,
#                                                               model.node_dropout: [0.] * len(eval(args.layer_size)),
#                                                               model.mess_dropout: [0.] * len(eval(args.layer_size))})

#         user_batch_rating_uid = zip(rate_batch, user_batch)
#         batch_result = pool.map(test_one_user, user_batch_rating_uid)
#         count += len(batch_result)

#         for re in batch_result:
#             result['precision'] += re['precision']/n_test_users
#             result['recall'] += re['recall']/n_test_users
#             result['ndcg'] += re['ndcg']/n_test_users
#             result['hit_ratio'] += re['hit_ratio']/n_test_users
#             result['auc'] += re['auc']/n_test_users


#     assert count == n_test_users
#     pool.close()
#     return result