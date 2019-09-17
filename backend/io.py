# pylint: disable=unsubscriptable-object

"""
Bada Bing, Bada Boom
The thoughts, ideas, and opinions both spoken and internalized
are the forces, that traverse all platforms, all standards
and all norms to form their unique class, unique style, and a
unique kind of ideology that future generations will look up to.

We are #!/Shabang. Copyright 2019/2020 Shabang Systems, LLC.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import uuid

import pickle
import tempfile

import random
from tqdm import tqdm

from .utils import CompareEngine
from . import encoding

import numpy as np

class CorpusManager(object):
    def __init__(self, raw, encoder=encoding.SentenceOneHotEncoder()):
        self.raw = raw
        self.encoder = encoder
        self.compEngine = CompareEngine()
        self.id = str(uuid.uuid4()) 
        self.outputs = []
        self.input_a = []
        self.input_b = []
        self.isEncoderTrained = False

    @property
    def sequenceLength(self):
        return self.encoder.vocabSize

    def __save(self):
        if self.save_dir == "__temp__":
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "CM_compdata_"+self.id[:5])+".cpmgr", "wb") as datafile:
            pickle.dump(self, datafile)

    @staticmethod
    def load(directory):
        with open(directory, "rb") as datafile: 
            return pickle.load(datafile)

    def generate(self, size=1000, bar=False):
        # if size > len(self.input_a):
        #     s = size-len(self.input_a)
        #     sz = len(self.input_a)
        #     in_a, in_b, out = self.generate(size=s, bar=bar)
        # else:
        #     sz = size
        #     in_a = []
        #     in_b = []
        #     out = []

        in_a = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        in_b = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        out = np.empty((size,))
        
        if bar:
            iterator = tqdm(range(size))
        else:
            iterator = range(size)

        # selected_ins = random.sample(list(enumerate(self.input_a)), sz)
        
        input_a_enum = list(enumerate(self.input_a))

        for s in iterator:
            selected = random.sample(input_a_enum, 1)[0]

            in_a[s] = selected[1]
            in_b[s] = self.input_b[selected[0]]
            out[s] = self.outputs[selected[0]]

        return in_a, in_b, out

    def compile(self, size=1000, bar=True, save=True, save_dir="__temp__"):
        raw_split = self.raw.splitlines()
        if not self.isEncoderTrained:
            print("Training encoder...")
            self.encoder.train(raw_split)
            self.isEncoderTrained = True
            print("Done. Beginning compiliation.")
        else:
            print("Beginning compiliation...")
        if not save:
            print("DANGER AHEAD: you are compiling a corpus without saving... All work will be lost unless variables are saved.")
        else:
            if save_dir == "$temp":
                print("DANGER AHEAD: you are saving the compiled data to a temporary directory... Consider setting a permanent one with param save_dir.")
        self.save_dir = save_dir
        if bar:
            iterator = tqdm(range(size))
        else:
            iterator = range(size)
        data_len = len(raw_split)
        for _ in iterator:
            indx_sel_a = random.randint(0, data_len-1)
            indx_sel_b = random.randint(0, data_len-1)
            linea = raw_split[indx_sel_a]
            lineb = raw_split[indx_sel_b]
            while linea.strip() == "":
                indx_sel_a = random.randint(0, data_len)
                linea = raw_split[indx_sel_a]
            while lineb.strip() == "":
                indx_sel_b = random.randint(0, data_len)
                lineb = raw_split[indx_sel_b]
            similarity = self.compEngine.eval(linea, lineb)
            la_enc = self.encoder.encode(linea)
            lb_enc = self.encoder.encode(lineb)
            self.input_a = self.input_a + [la_enc[0]]
            self.input_b = self.input_b + [lb_enc[0]]
            self.outputs.append(similarity)
            self.__save()
    

    

        
