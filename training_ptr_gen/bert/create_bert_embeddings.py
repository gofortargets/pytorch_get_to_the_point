import glob
import struct

import random
from tensorflow.core.example import example_pb2

from data_util import config

def text_generator(e):
  while True:
    e = next(example_generator) # e is a tf.Example
    try:
      article_text = e.features.feature['article'].bytes_list.value[0].decode() # the article text was saved under the key 'article' in the data files
      abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode() # the abstract text was saved under the key 'abstract' in the data files

    except ValueError:
      tf.logging.error('Failed to get article or abstract from example')
      continue
    if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
      #tf.logging.warning('Found an example with empty article text. Skipping it.')
      continue
    else:
      yield (article_text, abstract_text)

def example_generator(data_path, single_pass):
  res = []
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        res.append(example_pb2.Example.FromString(example_str))

    if single_pass:
      print ("example_generator completed reading all datafiles. No more data.")
      break

def main():
  example_generator(config.data_dir, single_pass=True)


if __name__ == "__main__":
    main()


