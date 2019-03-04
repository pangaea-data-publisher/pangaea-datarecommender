import pandas as pd
import logging
import gc
import json
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
from sklearn.preprocessing import normalize
import numpy as np
import time
import datetime
import multiprocessing as mp

class InferRelData:
    sparse_mat = None
    dataset_u = None
    download_dict = None

    def __init__(self,cfg):
        #global config
        #config = cfg
        self.min_users = int(cfg['DATASOURCE']['min_users'])
        self.CHUNK_SIZE = int(cfg['DATASOURCE']['chunk_size'])
        self.JSONDOWNLOAD_FILE = cfg['DATASOURCE']['download_file']
        self.TOPK = int(cfg['DATASOURCE']['top_k'])
        self.SIM_THRESHOLD = float(cfg['DATASOURCE']['sim_threshold'])
        self.DATALIST_FILE = cfg['DATASOURCE']['datalist_file']
        self.IPLIST_FILE = cfg['DATASOURCE']['iplist_file']
        self.output_top_k = int(cfg['DATASOURCE']['output_top_k'])
        self.output_max_top_k = int(cfg['DATASOURCE']['output_max_top_k'])
        self.number_of_processes = int(cfg['DATASOURCE']['number_of_processes'])-1 #make an additional process for query
        #self.SIM_SPARSE_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['sim_sparse_file'])

    def get_Total_Related_Downloads(self, dfmain):
        filtered = dfmain.groupby('_id').agg({'ip': 'nunique'})
        filtered_by = filtered[filtered.ip >= self.min_users].reset_index()
        f_datasets = filtered_by._id.unique().tolist()
        group_df = dfmain[dfmain._id.isin(f_datasets)]  # 6079317
        #logging.info("Total_Related_Downloads dataframe size :%s", str(group_df.shape))

        # size includes NaN values, count does not:
        download_count = group_df.groupby(['_id'])['_id'].agg(['count']).reset_index()
        download_dict = dict(zip(download_count['_id'], download_count['count']))

        # build datasets vs ip similarity matrix
        group = pd.DataFrame({'download_count': group_df.groupby(['_id', 'ip']).size()}).reset_index()
        person_u = list(group.ip.unique())
        dataset_u = list(group._id.unique())
        data = group['download_count'].tolist()

        row = group._id.astype(pd.api.types.CategoricalDtype(categories=dataset_u)).cat.codes
        col = group.ip.astype(pd.api.types.CategoricalDtype(categories=person_u)).cat.codes
        len_dataset = len(dataset_u)
        len_person = len(person_u)
        logging.info("Datasets vs Ips :%s %s", str(len_dataset), str(len_person))  # 310170 81649
        sparse_mat = sparse.csr_matrix((data, (row, col)), dtype=np.int8, shape=(len_dataset, len_person))

        #normalize sparse matrix
        sparse_mat = normalize(sparse_mat, copy=False)

        #logging.info('Sparse matrix size in bytes:%s', str(df_sparse.data.nbytes))  # 4004550

        # with open(self.DATALIST_FILE, 'w') as f1:
        #     for item in dataset_u:
        #         f1.write("%s\n" % item)
        #
        # with open(self.IPLIST_FILE, 'w') as f2:
        #     for item in person_u:
        #         f2.write("%s\n" % item)

        del dfmain, filtered, filtered_by, f_datasets, group_df, download_count, group, person_u, data, row, col, len_person
        gc.collect()

        #json_data = self.cosine_similarity_n_space(df_sparse, dataset_u, download_dict, len_dataset, self.CHUNK_SIZE)

        #chunk.size , e.g., every 1000 rows
        #[range(0, 10), range(10, 15)...] if chunk_size =10
        chunks = list(self.get_chunks(range(0, sparse_mat.shape[0]), self.CHUNK_SIZE))
        #send the function along to the initializer
        pool = mp.Pool(self.number_of_processes, initializer=self.init_proc, initargs=[sparse_mat, dataset_u,download_dict])
        # pool.map(f, chunks)
        manager = mp.Manager()
        return_dict = manager.dict()

        # Execute the cosine sim task in parallel
        #the order of the results may not correspond to the order in which the pool.apply_async calls were made
        for seq in chunks:
            pool.apply_async(self.cosine_similarity_n_space_v2, args=(return_dict, seq))
        # Tell the pool that there are no more tasks to come and join
        pool.close()
        logging.info('Waiting for pool to finish')
        pool.join()
        logging.info('Start writing download json')
        # Print the results
        #for i in return_dict.keys():
            #print(i, return_dict[i])

        with open(self.JSONDOWNLOAD_FILE, 'w') as fp:
            json.dump(return_dict._getvalue(), fp) #Use dict_proxy._getvalue() to fetch the actual dict instance underlying the proxy, and pass that to json.dump
        logging.info('Writing download json done!')
        del return_dict

    def init_proc(self,m, d, dw):
        global sparse_mat
        global dataset_u
        global download_dict
        sparse_mat = m  # data is now accessible in all children
        dataset_u = d
        download_dict = dw

    def get_chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def cosine_similarity_n_space_v2(self,return_dict,seq):
        num_top_dataset = self.TOPK
        topk = num_top_dataset + 1
        rows = sparse_mat[seq]
        #similarities = cosine_similarity(rows, sparse_mat)
        similarities = linear_kernel(rows, sparse_mat)
        reverse_idx = np.argsort(-similarities)
        reverse_idx = reverse_idx[:, :topk]
        related_datasets = np.array([dataset_u[k] for k in reverse_idx.flat]).reshape(reverse_idx.shape)

        target_idx = seq[0]
        for i in range(reverse_idx.shape[0]):
            target_id = dataset_u[target_idx]
            relrow = list(map(int, related_datasets[i]))
            simrow = list(similarities[i, reverse_idx[i, :]])
            # remove target dataset
            if target_id in relrow:
                del simrow[relrow.index(target_id)]
                relrow.remove(target_id)
            remove_indices = {a for a, b in enumerate(simrow) if b < self.SIM_THRESHOLD}
            if remove_indices:
                relrow = [x for y, x in enumerate(relrow) if y not in remove_indices]
                simrow = [e for f, e in enumerate(simrow) if f not in remove_indices]
            if relrow:
                if (len(relrow) >= self.output_top_k):
                    indices_max = max([i for i, x in enumerate(simrow) if x == simrow[self.output_top_k - 1]]) + 1
                    indices_max = min(self.output_max_top_k, indices_max)
                    relrow = relrow[0:indices_max]
                    simrow = simrow[0:indices_max]
                dt = {}
                dt['related_datasets'] = relrow
                dt['related_datasets_similarities'] = simrow
                dt['total_downloads'] = int(download_dict[target_id])
                return_dict[str(target_id)] = dt
            target_idx = target_idx + 1
        del similarities, reverse_idx, related_datasets, remove_indices, relrow, simrow
        gc.collect()


def cosine_similarity_n_space(self,mat, dataset_u, download_dict,shape, batch_size):
        num_top_dataset = self.TOPK
        starttime = time.time()
        json_data = {}
        target_idx = 0
        topk = num_top_dataset + 1
        for row_i in range(0, int(shape / batch_size) + 1):
            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, shape])
            if end <= start:
                break  # to-do: handle this
            rows = mat[start: end]
            similarities = cosine_similarity(rows, mat)  # rows is O(1) size
            reverse_idx = np.argsort(-similarities)
            #indices = reverse_idx.ravel()
            # reverse_val = similarities.ravel()[indices]
            # reverse_val = reverse_val.reshape(A.shape)
            # select top-k
            reverse_idx = reverse_idx[:,:topk]
            related_datasets = np.array([dataset_u[k] for k in reverse_idx.flat]).reshape(reverse_idx.shape)
            for i in range(reverse_idx.shape[0]):
                target_id = dataset_u[target_idx]
                relrow = list(map(int, related_datasets[i]))
                simrow = list(similarities[i, reverse_idx[i, :]])
                # remove target dataset
                if target_id in relrow:
                    del simrow[relrow.index(target_id)]
                    relrow.remove(target_id)
                remove_indices = {a for a, b in enumerate(simrow) if b < self.SIM_THRESHOLD}
                if remove_indices:
                    relrow = [x for y, x in enumerate(relrow) if y not in remove_indices]
                    simrow = [e for f, e in enumerate(simrow) if f not in remove_indices]
                #downloads = download_count.loc[download_count._id == target_id, 'count'].values[0]
                if relrow:
                    if (len(relrow) >= self.output_top_k):
                        indices_max = max([i for i, x in enumerate(simrow) if x == simrow[self.output_top_k - 1]]) + 1
                        indices_max = min(self.output_max_top_k, indices_max)
                        relrow = relrow[0:indices_max]
                        simrow = simrow[0:indices_max]
                    dt = {}
                    dt['related_datasets'] = relrow
                    #dt['similarities'] = ['%.5f' % elem for elem in simrow]
                    dt['related_datasets_similarities'] = simrow
                    dt['total_downloads'] = int(download_dict[target_id])
                    json_data[str(target_id)] = dt
                target_idx = target_idx + 1
            del similarities, reverse_idx, related_datasets, remove_indices,relrow,simrow
            gc.collect()
        #logging.info("Cosine Sim Compute : %s mins " % ((time.time() - starttime) / 60))
        secs = (time.time() - starttime)
        logging.info('Cosine Sim Compute : ' + str(datetime.timedelta(seconds=secs)))
        return json_data
