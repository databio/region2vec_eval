import requests
from bs4 import BeautifulSoup 
import pandas as pd
import urllib
import re
from tqdm import tqdm
import hashlib
import os
import gzip
import shutil
from config import DATA_FOLDER, DATA_URL


class TBFS_Downloader:
    #retrieve data and tokenize a chromosome regions
    def __init__(self, url, data_folder):
        self.url = url
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        self.links = links[7:]
        self.nfiles = len(self.links)
        self.data_prefix = os.path.join(data_folder, "data")
        os.makedirs(self.data_prefix, exist_ok=True)
        self.metadata_file = os.path.join(data_folder, "metadata.csv")
        self.file_list = os.path.join(data_folder, "file_list.txt")
        self.metadata = self.get_metadata(url+links[5])
        #download data files
        print('Downloading files....')
        for index in tqdm(range(self.nfiles)):
            filename = os.path.join(self.data_prefix, self.links[index])
            if (not os.path.exists(filename)) and (not os.path.exists(filename[:-3])):
                print(url+self.links[index],'\n',filename)
                urllib.request.urlretrieve(url+self.links[index], filename)
          # else:
          #   with open(filename, 'rb') as f:
          #     if hashlib.md5(f.read()).hexdigest()!= self.meta[index]['md5sum']:
          #       urllib.request.urlretrieve(url, filename)
        self.decompress()
        self.write_metadata()
        self.write_filelist()

    def decompress(self):
        count = 0
        for index in tqdm(range(self.nfiles)):
            new_file_name = os.path.join(self.data_prefix, self.links[index][:-3])
            filename = os.path.join(self.data_prefix, self.links[index])
            if not os.path.exists(new_file_name):
                count += 1
                with gzip.open(filename, 'rb') as f_in:
                    with open(new_file_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filename)
            key = self.links[index][:-3]
            self.metadata[key] = self.metadata[self.links[index]]
            del self.metadata[self.links[index]]
            self.links[index] = key
                
        print(f"{count}/{self.nfiles} decompressed")

    def retrieve_data(self, index):
        data_path = os.path.join(self.data_prefix, self.links[index])
        if not os.path.exists(data_path):
            print('Not found')
            return -1
        df = pd.read_csv(data_path, header=0, delimiter='\t', usecols=[0, 1, 2])
        df.columns=["chromosome", "start", "end"]
        chromosome = dict()
        for index, row in df.iterrows():
            chr = row[0]
            start = int(row[1])
            end = int(row[2])
            if chr not in chromosome.keys():
                chromosome[chr] = [(start, end-start)]
            else:
                chromosome[chr].append((start, end-start))
        return chromosome


    def get_metadata(self, url):
        metadata = {}
        f = urllib.request.urlopen(url)
        for line in f:
            info_dict = dict()
            str_line = line.decode("utf-8")
            data = str_line.split(';')
            filename = data[0].split('\t')[0]
            item = data[0].split('\t')[1].split('=')
            info_dict[item[0]] = item[1]
            for entry in data[1:]:
                items = entry.split('=')
                info_dict[items[0].strip()] = items[1].strip()
            metadata[filename] = info_dict
        return metadata

    def write_metadata(self):
        file_list = list(self.metadata.keys())
        key_list = list(self.metadata[file_list[0]].keys())
        with open(self.metadata_file, "w") as f:
            f.write("filename," + ','.join(key_list)+'\n')
            for filename in file_list:
                info_dict = self.metadata[filename]
                info_data = []
                for key in key_list:
                    if key in info_dict:
                        info_data.append(info_dict[key])
                    else:
                        info_data.append('None')
                    info = ','.join(info_data)
                f.write(f"{filename},{info}\n")
    def write_filelist(self):
        with open(self.file_list, "w") as f:
            for name in self.links:
                f.write(name)
                f.write('\n')

data_set = TBFS_Downloader(DATA_URL, DATA_FOLDER)
