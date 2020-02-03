from utility.parser import parse_args
from utility.load_data_v5 import *
import numpy as np

args = parse_args()
Ks = eval(args.Ks)
neighbors_num = eval(args.neighbors_num)
data_generator = DataV2(path=args.data_path + args.dataset, batch_size=args.batch_size, neighbors_num=neighbors_num, weighted_sample_num=args.weighted_sample_num)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def tsne(sess, model, model_name='gacse', drop_flag=False, batch_test_flag=False):
    users = [885, 1929, 4758, 11131, 14269, 24758]
    items = [[8160,2660,10103,159,140,10104,10105,24,4320,10106,376],
            [2086,16538,624,5275,7777,3827,5337,6066,3552,5532,11631],
            [27588,20154,17626,27589,16075,15997,27590,27591,16105],
            [9109,3023,6338,8321,21722,776,8834,782,5990,5982,5989],
            [1053,8960,3157,6940,5479,12805,3159,10927,39403,1389],
            [6174,43404,38793,14021,8093,877,2187,660,41152,9665]]

    items_array = np.concatenate(items)
    if model_name == 'gacse':
        user_batch = data_generator.sample_path(users, path=data_generator.user_path)
        item_batch_u = data_generator.sample_path(items_array, path=data_generator.item_path)
    else:
        user_batch = users
        item_batch_u = items_array

    feed_dict = {u : d.astype(np.int) for u, d in zip(model.users, user_batch)}
    feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_u, item_batch_u)})
    # feed_dict.update({i : d.astype(np.int) for i, d in zip(model.pos_items_i, item_batch_i)})

    users_embeddings, items_embeddings = sess.run([model.u_g_embeddings, model.pos_i_g_embeddings], feed_dict=feed_dict)

    users_embeddings = np.asarray(users_embeddings)
    items_embeddings = np.asarray(items_embeddings)

    np.save('{}_users.npy'.format(model_name), users_embeddings)
    np.save('{}_items.npy'.format(model_name), items_embeddings)

