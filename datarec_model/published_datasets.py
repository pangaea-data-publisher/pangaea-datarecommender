# MIT License
#
# Copyright (c) 2017 Anusuriya Devaraju
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from elasticsearch import Elasticsearch
import logging
import pickle
from os.path import dirname, abspath
logging.getLogger("Elasticsearch").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

class PublishedDataset:

    def  __init__(self,config):
        self.es = Elasticsearch('http://ws.pangaea.de/es',port=80)
        self.ES_INDEX='pangaea'
        self.DOC_TYPE= 'panmd'
        self.size = 1000
        #self.data_file_dir = '/pang_datarec/results/ids.p'
        self.data_file_dir = config['DATASOURCE']['publised_data']

    def getDatasets(self):
        #start_time = time.time()
        #logging.info('Get Dataset Start Time: ' + time.strftime("%H:%M:%S"))
        # scan function in standard elasticsearch python API
        rs = self.es.search(index=self.ES_INDEX, doc_type =self.DOC_TYPE,
               scroll='10s',
               size=self.size, _source = "false",
               body={
                   "query": {"match_all": {}}
               })
        data = []
        sid = rs['_scroll_id']
        scroll_size = rs['hits']['total']
        #print(scroll_size)
        #before you scroll, process your current batch of hits
        data = rs['hits']['hits']

        while (scroll_size > 0):
            try:
                scroll_id = rs['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll='60s')
                data += rs['hits']['hits']
                scroll_size = len(rs['hits']['hits'])
            except:
                break
        #usage_dir = dirname(dirname(abspath(__file__)))
        ids =[]
        for dobj in data:
            ids.append(dobj["_id"])
        logging.info('Number of datasets: %s',str(len(ids)))

        with open(self.data_file_dir,'wb') as fp:
            pickle.dump(ids, fp)

        #secs =  (time.time() - start_time)
        #logging.info('Get Published Dataset Total Execution Time: '+str(dt.timedelta(seconds=secs)))
        return ids

