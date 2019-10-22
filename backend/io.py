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

from multiprocessing.dummy import Pool, Lock

class CorpusManager(object):
    def __init__(self, raw, encoder=encoding.SentenceOneHotEncoder()):
        self.raw = raw
        self.encoder = encoder
        self.compEngine = CompareEngine()
        self.id = str(uuid.uuid4()) 
        self.outputs = []
        self.input_a = []
        self.input_b = []
        self.__oplock = Lock()
        self.__ialock = Lock()
        self.__iblock = Lock()
        self.__wnlock = Lock()
        self.isEncoderTrained = False

    @property
    def sequenceLength(self):
        return self.encoder.vocabSize
    
    @property
    def sentSize(self):
        return self.encoder.sentSize

    def __save(self):
        if self.save_dir == "__temp__":
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "CM_compdata_"+self.id[:5])+".cpmgr", "wb") as datafile:
            pickle.dump([self.raw, self.encoder, self.compEngine, self.id, self.outputs, self.input_a, self.input_b, self.isEncoderTrained], datafile)

    @classmethod
    def load(cls, directory):
        with open(directory, "rb") as datafile: 
            dataObject = pickle.load(datafile)
            loaded = cls(dataObject[0], dataObject[1])
            loaded.compEngine = dataObject[2]
            loaded.id = dataObject[3]
            loaded.outputs = dataObject[4]
            loaded.input_a = dataObject[5]
            loaded.input_b = dataObject[6]
            loaded.isEncoderTrained = dataObject[7]
            return loaded

    def dump(self, bar=False):
        size = len(self.input_a)
        in_a = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        in_b = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        out = np.empty((size,))

        input_a_enum = list(enumerate(self.input_a))

        if bar:
            iterator = tqdm(input_a_enum)
        else:
            iterator = input_a_enum

        for indx, val in iterator:
            in_a[indx] = val
            in_b[indx] = self.input_b[indx]
            out[indx] = self.outputs[indx]

        return in_a, in_b, out

    def sample(self, size=1000, bar=False):
        in_a = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        in_b = np.empty((size, self.encoder.sentSize, self.encoder.vocabSize))
        out = np.empty((size,))
        
        if bar:
            iterator = tqdm(range(size))
        else:
            iterator = range(size)
        
        input_a_enum = list(enumerate(self.input_a))

        for s in iterator:
            selected = random.sample(input_a_enum, 1)[0]

            in_a[s] = selected[1]
            in_b[s] = self.input_b[selected[0]]
            out[s] = self.outputs[selected[0]]

        return in_a, in_b, out

    def __multithread_comp_straight(self, data_bundle):
        i, la, lb, dup_factor, dl = data_bundle
        print("I am processing", i, ". Data", round((self.__mtpsct/dl)*100, 2), "complete.")
        rfactor = random.randint(0, 100)
        if rfactor < 100*dup_factor:
            isDuplicate = True
        else:
            isDuplicate = False
        linea = la
        lineb = lb
        if linea.strip() == "":
            self.__mtpsct += 1
            return
        if lineb.strip() == "":
            self.__mtpsct += 1
            return
        if isDuplicate:
            self.__wnlock.acquire()
            similarity = self.compEngine.eval(linea, linea)
            self.__wnlock.release()
            la_enc = self.encoder.encode(linea)
            self.__ialock.acquire()
            self.input_a = self.input_a + [la_enc[0]]
            self.__iblock.acquire()
            self.input_b = self.input_a + [la_enc[0]]
            self.__ialock.release()
            self.__iblock.release()
        else:
            self.__wnlock.acquire()
            similarity = self.compEngine.eval(linea, lineb)
            self.__wnlock.release()
            la_enc = self.encoder.encode(linea)
            lb_enc = self.encoder.encode(lineb)
            self.__ialock.acquire()
            self.input_a = self.input_a + [la_enc[0]]
            self.__iblock.acquire()
            self.input_b = self.input_b + [lb_enc[0]]
            self.__iblock.release()
            self.__ialock.release()
        self.__oplock.acquire()
        self.outputs.append(similarity)
        self.__oplock.release()
        self.__mtpsct += 1
        

    def __do_straightCompile(self, raw_split, dup_factor, workers=4, multithread=True):
        data_len = len(raw_split)
        print("This is compile engine. You are about to compile a corpus of size", data_len, "into training data multithreadedly.")
        if int(data_len/2) != data_len/2: #removes the last item if odd so that pairing is possible
            del raw_split[-1]
            data_len -= 1
        
        print("Building compile bundle for multiprocessing...")
        compile_bundle = []
        for i in range(int(data_len/2)):
            compile_bundle.append([i, raw_split[i], raw_split[i+1], dup_factor, data_len])
        self.__mtpsct = 0
        print("Done. Creating workers...")
        pool = Pool(workers) 
        print("Done. On your marks, get set, pool.map()!")
        _ = pool.map(self.__multithread_comp_straight, compile_bundle)
        print("Done. Saving...")
        self.__save()
        print("Done. That was fun.")


    def __do_sampleCompile(self, size, raw_split, dup_factor, bar):
        if bar:
            iterator = tqdm(range(size))
        else:
            iterator = range(size)
        data_len = len(raw_split)
        for _ in iterator:
            rfactor = random.randint(0, 100)
            if rfactor < 100*dup_factor:
                isDuplicate = True
            else:
                isDuplicate = False
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
            if isDuplicate:
                similarity = self.compEngine.eval(linea, linea)
                la_enc = self.encoder.encode(linea)
                self.input_a = self.input_a + [la_enc[0]]
                self.input_b = self.input_a + [la_enc[0]]
            else:
                similarity = self.compEngine.eval(linea, lineb)
                la_enc = self.encoder.encode(linea)
                lb_enc = self.encoder.encode(lineb)
                self.input_a = self.input_a + [la_enc[0]]
                self.input_b = self.input_b + [lb_enc[0]]
            self.outputs.append(similarity)
            self.__save()

    def compile(self, straight=True, size=None, workers=4, bar=True, save=True, save_dir="__temp__", dup_factor=0.5):
        if straight:
            assert not size, "Use straight command with size of none!"
        if size:
            assert not straight, "Cannot sample compile the dataset with a None size."
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
        if size:
            self.__do_sampleCompile(size, raw_split, dup_factor, bar)
        elif straight:
            self.__do_straightCompile(raw_split, dup_factor, workers)
        
    

    

        
