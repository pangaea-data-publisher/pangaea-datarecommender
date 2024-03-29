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
        self.es = Elasticsearch(config['DATASOURCE']['elastic_url'] )
        self.ES_INDEX=config['DATASOURCE']['elastic_index']
        self.DOC_TYPE= config['DATASOURCE']['elastic_type']
        self.size = 1000

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
        sid = rs['_scroll_id']
        scroll_size = len(rs['hits']['hits'])
        #print(scroll_size)
        #before you scroll, process your current batch of hits
        ids = set()
        for dobj in rs['hits']['hits']:
            ids.add(int(dobj["_id"]))

        while (scroll_size > 0):
            try:
                scroll_id = rs['_scroll_id']
                rs = self.es.scroll(scroll_id=scroll_id, scroll='60s')
                for dobj in rs['hits']['hits']:
                    ids.add(int(dobj["_id"]))
                scroll_size = len(rs['hits']['hits'])
            except:
                break

        logging.info('Number of datasets: %s',str(len(ids)))

        #secs =  (time.time() - start_time)
        #logging.info('Get Published Dataset Total Execution Time: '+str(dt.timedelta(seconds=secs)))
        return ids

