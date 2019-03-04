import json
import configparser as ConfigParser
import argparse
import re
import pandas as pd
import time
import datetime
import logging
import published_datasets,process_logs,infer_reldataset2
import copy
from itertools import chain
from multiprocessing import Process
import gc

def main():
    #set logging info
    #logging.basicConfig(format='%(asctime)s %(message)s',filename='pangaea_recsys.log', level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO, filename='pangrecsys.log', filemode="a+",
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to import.ini config file")
    args = ap.parse_args()
    global config
    global configFile,query_file
    config = ConfigParser.ConfigParser()
    configFile = args.config
    config.read(configFile)
    #parent_dir = os.path.dirname(os.path.abspath(__file__)) C:\Users\anusu\python-workspace\pangaea-recsys\recommender
    query_file = config['DATASOURCE']['query_file']
    DATAFRAME_FILE = config['DATASOURCE']['dataframe_file']
    final_result_file = config['DATASOURCE']['final_result_file']
    global start_time
    #1. import recent datasets
    start_time = time.time()
    logging.info('Importing published datasets...')
    pubInst = published_datasets.PublishedDataset(config)
    list_published_datasets = pubInst.getDatasets()

    #2. read and clean logs
    c1 = process_logs.ProcessLogs(config,configFile)
    main_df = c1.readLogs()
    if not main_df.empty:
        # get request uri, extract data id
        logging.info("Extracting id from request param...")
        #main_df['_id'] = main_df['request'].str.extract(r'PANGAEA.\s*(\d+)')
        #main_df = main_df.dropna(subset=['_id'], how='all')
        #main_df['_id'] = main_df['_id'].astype(int)
        main_df.loc[:, '_id'] = main_df['request'].str.extract(r'PANGAEA.\s*(\d+)',expand=False)
        main_df = main_df.dropna(subset=['_id'], how='all')
        main_df.loc[:, '_id'] = main_df['_id'].astype(int)

        #if not main_df.empty:  # dataframe might be empty afterer dropna and extract operation
        df_old = None
        if c1.last_harvest_date != 'none':
            # append old dataframe
            df_old = pd.read_csv(c1.DATAFRAME_FILE)
            logging.info("Appending DF shape : %s ", str(main_df.shape))
            main_df = df_old.append(main_df, sort=True, ignore_index=True).reset_index(drop=True)
            del df_old
            gc.collect()
        else:
            logging.info("New DF (no append) : %s ", str(main_df.shape))

        #TO-DO:
        logging.info("Writing DF to an external file...")
        main_df.to_csv(DATAFRAME_FILE, index=False)

        logging.info("Updating harvest date in config file...")
        c1.updateConfigFile()
    else:
        main_df=pd.read_csv(c1.DATAFRAME_FILE)
        logging.info("No changes found!")

    if not main_df.empty:
        logging.info("Excluding non-published datasets...")
        main_df = main_df[main_df['_id'].isin(list_published_datasets)]
        logging.info("DF with only published datasets : %s", str(main_df.shape))

        # ####### 3. get query terms
        # # exlude rows that contains old data
        # logging.info("Start - Related Datasets By Query...")
        # df_query = main_df.copy()
        # # only select referer related to pangaea, get query terms for each datasets
        # domains = ['doi.pangaea.de', 'www.pangaea.de', '/search?']
        # domains_joins = '|'.join(map(re.escape, domains))
        # df_query = df_query[(df_query.referer.str.contains(domains_joins))]
        # df_query = c1.getQueryTerms(df_query)
        # df_query = df_query.reset_index()
        # df_query = df_query.set_index('_id')
        # #logging.info('Query Dataframe Shape: ' + str(df_query.shape))
        # df_query.to_json(query_file, orient='index')
        # secs = (time.time() - start_time)
        # del df_query
        # logging.info('Total Query Sim Time : ' + str(datetime.timedelta(seconds=secs)))

        # ####### 4. get usage related datasets
        # logging.info("Start - Related Datasets By Downloads...")
        # download_indicators = ['format=textfile', 'format=html', 'format=zip']
        # download_joins = '|'.join(map(re.escape, download_indicators))
        # main_df = main_df[(main_df.request.str.contains(download_joins))]
        # main_df = main_df.drop_duplicates(['time', 'ip', '_id'])
        # main_df = main_df[['ip', '_id']]
        #
        # dwnInst = infer_reldataset.InferRelData(config)
        # dwnInst.get_Total_Related_Downloads(main_df)
        # del main_df

        p1 = Process(target=computeRelDatasetsByQuery,args=[main_df,c1,query_file])
        logging.info("Start - Related Datasets By Query...")
        p1.start()
        logging.info("Start - Related Datasets By Downloads...")
        computeRelDatasetsByDownload(main_df,config)
        del main_df
        gc.collect()
        #p2 = Process(target=computeRelDatasetsByDownload,args=[main_df,config])
        #p2.start()
        p1.join() #wait for this [thread/process] to complete
        #p2.join()
        logging.info("End - Related Datasets By Query and Download...")

        ######## 5. merge results
        JSONDOWNLOAD_FILE = config['DATASOURCE']['download_file']
        JSONQUERY_FILE = config['DATASOURCE']['query_file']
        downloads = json.load(open(JSONDOWNLOAD_FILE))
        queries = json.load(open(JSONQUERY_FILE))
        logging.info("Merge - Len Downloads, Queries : %s %s", str(len(downloads.keys())), str(len(queries.keys())))
        logging.info("Merge - Difference : %s", str(len(list(set(queries.keys()) - set(downloads.keys())))))
        super_dict = {}
        for k, v in chain(downloads.items(), queries.items()):
            k = int(k)
            super_dict.setdefault(k, {}).update(v)
        logging.info("Len Merge : %s", str(len(super_dict.keys())))
        #timestr = time.strftime("%Y%m%d")
        with open(final_result_file, 'w') as outfile:
            json.dump(super_dict, outfile)
    else:
        logging.info("Empty dataframe after exclude operation...")

    secs = (time.time() - start_time)
    logging.info('Total Run Time: ' + str(datetime.timedelta(seconds=secs)))
    logging.info('--------------------------------')

def computeRelDatasetsByQuery(df_query,c1,query_file):
    #start_time1 = time.time()
    ####### 3. get query terms
    # exlude rows that contains old data
    # only select referer related to pangaea, get query terms for each datasets
    domains = ['doi.pangaea.de', 'www.pangaea.de', '/search?']
    domains_joins = '|'.join(map(re.escape, domains))
    df_query = df_query[(df_query.referer.str.contains(domains_joins))]
    df_query = c1.getQueryTerms(df_query)
    df_query = df_query.reset_index()
    df_query = df_query.set_index('_id')
    # logging.info('Query Dataframe Shape: ' + str(df_query.shape))
    df_query.to_json(query_file, orient='index')
    #secs = (time.time() - start_time1)
    del df_query
    gc.collect()
    #print('Total Query Sim Time : ' + str(datetime.timedelta(seconds=secs)))

def computeRelDatasetsByDownload(main_df,config):
    download_indicators = ['format=textfile', 'format=html', 'format=zip']
    download_joins = '|'.join(map(re.escape, download_indicators))
    main_df = main_df[(main_df.request.str.contains(download_joins))]
    main_df = main_df.drop_duplicates(['time', 'ip', '_id'])
    main_df = main_df[['ip', '_id']]

    #dwnInst = infer_reldataset.InferRelData(config)
    dwnInst = infer_reldataset2.InferRelData(config)
    dwnInst.get_Total_Related_Downloads(main_df)


if __name__ == "__main__":
    main()



