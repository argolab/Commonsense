import pandas as pd
import json
import numpy as np

from openai import OpenAI
import dotenv
dotenv.load_dotenv('.env')
import json
import numpy as np



class DatasetQ():
    def __init__(self, path='airdat.csv', verbose=False):

        dataframe = pd.read_csv(path)
        self.dat = dataframe

        self.verbose = verbose

        if self.verbose:
            print("Columns in dataset: ", self.dat.columns)


    def describe(self, item='price'):
        return self.dat[item].describe()
    

    def get(self, ):
        return self.dat


    def print_col(self, ):
        print(self.dat.columns)


    def unique(self, item='price'):
        print(self.dat[item].unique())


    def marg(self, query_json, array=True):

        query_json = query_json.copy()

        tar = query_json['Target'][0]['Name']
        cond = query_json['Condition']

        uniques = self.dat[tar].unique()

        tmp = self.dat.copy()

        for c in cond:
            variable_name = c['Name']
            variable_value = c['Value']
            # filter dataset to get only the rows that satisfy the condition
            tmp = tmp[tmp[variable_name].isin(variable_value)]
        
        if self.verbose:
            print(tmp['room_type'].unique())


        tot = 0
        
        ret = {}
        names = []
        prob = []

        for u in sorted(uniques): # suppose we'll always use sorted to align the values CHECK IF THIS ALIGNS WITH OUR SCHEME
            percentage = (tmp[tar] == u).mean()
            names.append(u)
            prob.append(percentage)
            print(u, percentage)
            tot += percentage
        if abs(tot - 1) > 0.01:
            print("[Error]marginal do not add to 1: ", tot)

        ret['Value'] = names
        ret['Probability'] = prob

        query_json['Ground'] = ret

        if array:
            return np.array(prob), np.array(names)
        else:
            return query_json


class question_translate():
    def __init__(self, prompt):
        self.client = OpenAI()
        self.prompt = prompt
        self.model = 'gpt-3.5-turbo-0125'
        
    
    def get_text(self, js_question):
        tmp = self.prompt.copy()
        if type(js_question) == dict:
            js_question = json.dumps(js_question)
        tmp.append({'role': 'user', 'content': js_question})
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={ "type": "text" },
            temperature=0,
            messages=tmp
        )
        return response.choices[0].message.content
    
    def put_new(self, js_path, out_path=None):

        js = json.load(open(js_path)).copy()
        text = self.get_text(js)
        js['Text'] = text

        if out_path:
            json.dump(js, open(out_path, 'w'), indent=2)
        else:
            #out_path = js_path[:-5] + 'q.json'
            json.dump(js, open('./example_files/output.json', 'w'), indent=2)

        return text