import os
import pickle

def read_stub(read_from_stub,stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):    
        with open(stub_path,'rb') as f:
            object=pickle.load(f)
            return object
    return None

def save_stub(stub_path,object):
    if not os.path.exists(stub_path):
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    
    if stub_path is not None:
        with open(stub_path,'wb') as f:
            pickle.dump(object,f)
