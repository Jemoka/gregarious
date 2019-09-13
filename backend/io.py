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

from . import encoding, CompareEngine

class CorpusManager(object):
    def __init__(self, raw, encoder=encoding.SentenceVectorizer, pad=True):
        self.raw = raw
        self.encoder = encoder(pad=pad)
        self.compEngine = CompareEngine()
        self.id = str(uuid.uuid4()) 
        self.outputs = []
        self.input_a = []
        self.input_b = []
        self.isEncoderTrained = False

    @property
    def sequenceLength(self):
        return self.encoder.sequenceLength

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
        if size > len(self.input_a):
            s = size-len(self.input_a)
            sz = len(self.input_a)
            in_a, in_b, out = self.generate(size=s, bar=bar)
        else:
            sz = size
            in_a = []
            in_b = []
            out = []

        if bar:
            iterator = tqdm(range(sz))
        else:
            iterator = range(sz)
        
        selected_ins = random.sample(list(enumerate(self.input_a)), sz)
        
        if bar:
            iterator = tqdm(selected_ins)
        else:
            iterator = selected_ins
        
        for idx, val in iterator:
            in_a.append(val)
            in_b.append(self.input_b[idx])
            out.append(self.outputs[idx])

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
    

    

        
