from itertools import tee
import re
import pandas as pd
import datetime
import multiprocessing
import os
import urllib
import logging

class ProcessLogs:
    def __init__(self,cfg, cfgFile):
        global config,configFile
        config = cfg
        configFile = cfgFile
        #self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath('__file__'))))
        self.parent_dir = os.path.dirname(os.path.abspath(__file__))
        #self.source_dir = os.path.join(self.parent_dir, config['DATASOURCE']['source_path'])
        self.source_dir = config['DATASOURCE']['source_path']
        self.source_file_prefix = config['DATASOURCE']['source_file_prefix']
        self.source_file_suffix = config['DATASOURCE']['source_file_suffix']
        #self.JSONDOWNLOAD_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['download_file'])
        #self.PUBLISHED_DATA_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['published_data_file'])
        #self.SIM_THRESHOLD = float(config['DATASOURCE']['sim_threshold'])
        #self.CHUNK_SIZE = int(config['DATASOURCE']['chunk_size'])
        #self.TOPK = int(config['DATASOURCE']['top_k'])
        self.last_harvest_date = (config['DATASOURCE']['last_harvest_date'])
        self.date_pattern = re.compile(r'\b(\d{8})0000.bz2\b')
        self.DATAFRAME_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['dataframe_file'])
        self.last_date = None
        self.number_of_processes=  int(config['DATASOURCE']['number_of_processes'])

    def readLogs(self):
        #dirs = os.path.join(self.parent_dir, self.source_dir)
        # get a list of file names
        files = os.listdir(self.source_dir)
        file_list = [os.path.join(self.source_dir, filename) for filename in files if filename.startswith(self.source_file_prefix) and filename.endswith(self.source_file_suffix)]

        #only select new files, excludes old files
        if self.last_harvest_date !='none':
            # filter out old files
            filtered_file_list = []
            last_harv_date = datetime.datetime.strptime(self.last_harvest_date, '%Y%m%d')
            for f in file_list:
                file_date = self.get_date(f)
                if file_date > last_harv_date:
                    filtered_file_list.append(f)
            file_list = filtered_file_list
        len_file = str(len(file_list))
        logging.info('Number of new files : %s', len_file)
        if len(file_list)>0:
            # set up your pool
            pool = multiprocessing.Pool(self.number_of_processes)  # or whatever your hardware can support
            # have your pool map the file names to dataframes
            df_list = pool.map(self.read_csv, file_list)

            # Concatenate all data into one DataFrame
            df_final = pd.concat(df_list, ignore_index=True)
            pool.close()
            pool.join()

            #update config file
            dates = (self.get_date(fn) for fn in file_list)
            last_date = (d for d in dates if d is not None)
            last_date = max(last_date)
            self.last_date = last_date.strftime('%Y%m%d')
            logging.info('Last_date :%s', str(last_date))
            logging.info("df_final shape : %s", str(df_final.shape))
            return df_final
        else:
            return pd.DataFrame() #creates a new dataframe that's empty

    def updateConfigFile(self):
        # write to config file
        if self.last_date:
            config.set('DATASOURCE', 'last_harvest_date', self.last_date)
            with open(configFile, 'w+') as configfile:
                config.write(configfile)
            logging.info("Last Harvest Date Updated! :%s ", self.last_date)


    # wrap your csv importer in a function that can be mapped
    def read_csv(self, filename):
        data = pd.read_csv(filename, compression='bz2', encoding='ISO-8859-1',
                               sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', engine='python', header=0,
                               usecols=[0, 3, 4, 5, 7, 8],
                               names=['ip', 'time', 'request', 'status', 'referer', 'user_agent'],
                               converters={"request": self.parse_str})
        df = self.cleanLogs(data)
        return df

    def cleanLogs(self, dfmain):
        # Filter out non GET and non 200 requests
        request = dfmain.request.str.split()
        #print(dfmain.head())
        dfmain["status"] = dfmain["status"].apply(pd.to_numeric, errors='ignore')
        dfmain = dfmain[(request.str[0] == 'GET') & (dfmain.status == 200)]
        #unwanted resources
        dfmain = dfmain[~dfmain['request'].str.match(r'^/media|^/static|^/admin|^/robots.txt$|^/favicon.ico$')]
        # filter crawlers by User-Agent
        dfmain = dfmain[~dfmain['user_agent'].str.match(r'.*?bot|.*?spider|.*?crawler|.*?slurp', flags=re.I).fillna(False)]

        # added 21-02-20199
        dfmain['request'] = dfmain['request'].apply(urllib.parse.unquote)
        dfmain['request'] = dfmain['request'].map(lambda x: x.strip())

        # get request uri, extract data id
        #dfmain['_id'] = dfmain['request'].str.extract(r'PANGAEA.\s*(\d+)')
        #dfmain = dfmain.dropna(subset=['_id'], how='all')
        #dfmain['_id'] = dfmain['_id'].astype(int)

        #dfmain["_id"] = pd.to_numeric(dfmain["_id"],downcast='integer')

        # convert time
        dfmain['time'] = dfmain['time'].str.strip('[]').str[:-6]
        dfmain['time'] = pd.to_datetime(dfmain['time'], format='%d/%b/%Y:%H:%M:%S')
        dfmain['time'] = dfmain['time'].dt.date
        return dfmain

    def parse_str(self, x):
        # remove double quotes
        if x:
            return x[1:-1]
        else:
            return x

    def get_date(self, filename):
        matched = self.date_pattern.search(filename)
        if not matched:
            return None
        dates = datetime.datetime.strptime(matched.group(1), '%Y%m%d')
        return dates

    def getQueryTerms(self, df_query):
        # identify first and second degree queries
        #df_final['query_1'] = df_final['referer'].map(self.get_query)
        #df_final['query_1']=df_final['referer'].map(self.get_query)
        # df_final['query_2'] = ""
        #print('---------------------')
        #df_query.loc[:, 'query_2'] = ""
        #df_query.loc[:, 'query_1']=df_query['referer'].map(self.get_query)
        df_final = df_query[['ip', '_id', 'query_1', 'query_2', 'time']]
        first = df_final.groupby(by=['ip', 'time'])
        first_filtered = first.filter(lambda x: len(x[x['query_1'] != ""]) > 0)
        second = first_filtered.groupby(by=['ip', 'time'])
        filtered = second.filter(lambda x: len(x[x['query_1'] == ""]) > 0)
        for (i1, row1), (i2, row2) in self.pairwise(filtered.iterrows()):
            if ((row1["query_1"] != "") and (row2["query_1"] == "")):
                #filtered.set_value(i2, 'query_2', row1["query_1"]) #index, col, value
                filtered.at[i2, 'query_2'] = row1["query_1"]

        filtered = filtered[~((filtered.query_1 == "") & (filtered.query_2 == ""))]
        dfgroup = filtered.groupby('_id')['query_1', 'query_2'].apply(lambda x: x.sum())

        # strip white spaces
        dfgroup['query_1'] = dfgroup['query_1'].str.strip()
        dfgroup['query_2'] = dfgroup['query_2'].str.strip()
        dfgroup.loc[dfgroup.query_2 == "", 'query_2'] = None
        dfgroup.loc[dfgroup.query_1 == "", 'query_1'] = None
        #print('Query computation complete!')
        return dfgroup

    def pairwise(self,iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)