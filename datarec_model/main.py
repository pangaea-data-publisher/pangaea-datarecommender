import json
import configparser as ConfigParser
import argparse
import re
import pandas as pd
import time
import logging
import published_datasets,process_logs,infer_reldataset
from itertools import chain
import datetime
import gc
import sys
import urllib
#turn off the SettingWithCopyWarning globally
pd.options.mode.chained_assignment = None

def main():
    #set logging info
    #logging.basicConfig(format='%(asctime)s %(message)s',filename='pangaea_recsys.log', level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO, filename='pangrecsys.log', filemode="a+",
                        format="%(asctime)s %(levelname)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

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
    global num_years
    #num_years = int(config['DATASOURCE']['log_max_years'])
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
        main_df.loc[:, '_id'] = main_df['request'].str.extract(r'PANGAEA.\s?(\d+)',expand=False)
        main_df = main_df.dropna(subset=['_id'], how='all')
        main_df.loc[:, '_id'] = main_df['_id'].astype(int)
        gc.collect()

        #if not main_df.empty:  # dataframe might be empty afterer dropna and extract operation
        df_old = None
        if c1.last_harvest_date != 'none':
            # append old dataframe
            logging.info("Reading existing DF file...")
            df_old = pd.read_pickle(c1.DATAFRAME_FILE)
            logging.info("Append new DF...")
            main_df = pandas.concat([df_old, main_df], sort=True, ignore_index=True, copy=False)
            #main_df['_id'] = main_df['_id'].astype(int)
            #main_df = main_df.dropna(subset=['_id'], how='all')
            logging.info("Appended DF shape : %s ", str(main_df.shape))
            del df_old
            gc.collect()
        else:
            logging.info("New DF (no append) : %s ", str(main_df.shape))

        # 24.05.2019 only select data rows newer than the last XX years
        dt_years_ago = c1.log_max_dt.date()
        main_df = main_df[main_df['time'] >= dt_years_ago]

        #TO-DO:
        logging.info("Writing DF to an external file...")
        main_df.to_pickle(DATAFRAME_FILE)

        logging.info("Updating harvest date in config file...")
        c1.updateConfigFile()
    else:
        logging.info("No changes (new files) found, so old data (external file) will be used..")
        main_df=pd.read_pickle(c1.DATAFRAME_FILE)
        #main_df['_id'] = main_df['_id'].astype(int)
        #main_df = main_df.dropna(subset=['_id'], how='all')

    if not main_df.empty:
        logging.info("Excluding non-published datasets...")
        main_df = main_df[main_df['_id'].isin(list_published_datasets)]
        logging.info("DF with only published datasets : %s", str(main_df.shape))

        logging.info("Start - Related Datasets By Query...")
        computeRelDatasetsByQuery(main_df,c1,query_file)
        logging.info('Query completed...')
        logging.info("Start - Related Datasets By Downloads...")
        computeRelDatasetsByDownload(main_df,config)
        del main_df
        gc.collect()
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
    df_query.loc[:, 'query_1']=df_query['referer'].map(get_query)
    df_query.loc[:, 'query_2'] = ""
    df_query = c1.getQueryTerms(df_query)
    df_query = df_query.reset_index()
    df_query = df_query.set_index('_id')
    df_query.to_json(query_file, orient='index')
    del df_query
    gc.collect()
    #print('Total Query Sim Time : ' + str(datetime.timedelta(seconds=secs)))

def computeRelDatasetsByDownload(main_df,config):
    download_indicators = ['format=textfile', 'format=html', 'format=zip']
    download_joins = '|'.join(map(re.escape, download_indicators))
    main_df = main_df[(main_df.request.str.contains(download_joins))]
    main_df = main_df.drop_duplicates(['time', 'ip', '_id'])

    main_df = main_df[['ip', '_id']]
    dwnInst = infer_reldataset.InferRelData(config)
    dwnInst.get_Total_Related_Downloads(main_df)


def get_query(url):
    qparams = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    query_string = ""
    if len(qparams) > 0:
        for key in qparams:
            if re.match(r'f[.]|q|t|p', key):
                query_string += qparams[key] + " "
    return query_string

if __name__ == "__main__":
    main()



