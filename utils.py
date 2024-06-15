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


    def marg(self, query_json, target_values=None):

        if type(query_json) == str:
            query_json = json.loads(query_json)
        query_json = query_json.copy()


        tar = query_json['Target'][0]['Name']
        
        cond = query_json['Condition']

        if not target_values:
            target_values = ["<$50", "$51-$100", "$101-$200", "$201-$500", "$500+"]


        tmp = self.dat.copy()
        # print number of rows in the dataset
        tmp = tmp.dropna(subset=[tar])
        city = None

        #print(city, " : ", len(tmp))
        for c in cond:
            variable_name = c['Name']
            tmp = tmp.dropna(subset=[variable_name])
            variable_value = c['Value']
            if variable_name == 'City':
                city = variable_value
                continue
            #print(tmp[variable_name].unique())
            # filter dataset to get only the rows that satisfy the condition
            if type(variable_value) != list:
                variable_value = [variable_value]
            #print(tmp[variable_name].describe())
            tmp = tmp[tmp[variable_name].isin(variable_value)]
            #print(variable_name, variable_value)
            # drop nan columns
            #tmp = tmp.dropna(subset=[variable_name])
        #print(city, " : ", len(tmp))
        tot = 0
        
        prob = []

        for u in target_values:
            percentage = (tmp[tar] == u).mean()
            #names.append(u)
            prob.append(percentage)
            tot += percentage
        if abs(tot - 1) > 0.01:
            print("[Error]marginal do not add to 1: ", tot)

        return prob


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
            json.dump(js, open(js_path, 'w'), indent=2)

        return text
