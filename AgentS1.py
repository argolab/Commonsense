from pathlib import Path
import random
import json
from openai import OpenAI
import dotenv
import rich
import json
import torch
import numpy as np
import argparse
dotenv.load_dotenv('.env')

class AgentS1():
    def __init__(self, prompts, question_schema, price_bin='', log_dir='./example_files', log_name='s2ex.json', city='United States', model_dial='gpt-4o', model_trans='gpt-3.5-turbo', chat_temp=0.3):


        if type(question_schema) == str and question_schema.endswith('.json'):
            question_schema = json.load(open(question_schema))
        if type(question_schema) == str:
            question_schema = json.loads(question_schema)

        self.question_schema = question_schema
        self.schema = {'Question': json.loads(json.dumps(question_schema))}

        self.temperature = chat_temp
        # make log directory, recursively
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.log_path = Path(log_dir, log_name)

        if type(prompts) == str and prompts.endswith('.json'):
            prompts = json.load(open(prompts))
        if type(prompts) == dict:
            prompts = json.dumps(prompts)

        if price_bin == '':
            self.pbins = "Price: $0-$50, $51-$100, $101-$200, $201-$500, $501 and above"
        else:
            self.pbins = price_bin

        if city != 'United States':
            prompts = prompts.replace('United States', city)
        prompts = prompts.replace('[PRICEBINS]', self.pbins)

        self.prompts = json.loads(prompts)

        self.client = OpenAI()

        self.model_dial = model_dial
        self.model_trans = model_trans

        self.city = city

        self.records = {}
        self.records['schema'] = json.loads(json.dumps(self.schema))


    def log(self):
        # log self.records
        json.dump(self.records, open(self.log_path, 'w'), indent=2)

    def prompt(self, name):
        return [i.copy() for i in self.prompts[name]]

    def get_schema(self,):
        return json.loads(json.dumps(self.schema))
    
    def set_schema(self, schema):
        if type(schema) == str and schema.endswith('.json'):
            schema = json.load(open(schema))
        if type(schema) == str:
            schema = json.loads(schema)
        self.schema = json.loads(json.dumps(schema))
        self.records['schema'] = json.loads(json.dumps(self.schema))

    def question_text(self, js_question, add_city=True):

        question_prompt = self.prompt('eval_question_text')

        if type(js_question) == dict:
            js_question = json.dumps(js_question)
        if add_city:
            js_question = json.loads(js_question)
            js_question['Condition'].append({'Name': 'City', 'Value': [self.city]})
            js_question = json.dumps(js_question)

        js_question = json.dumps(js_question)

        question_prompt.append({'role': 'user', 'content': js_question})

        response = self.client.chat.completions.create(
            model=self.model_trans,
            response_format={ "type": "text" },
            temperature=0,
            messages=question_prompt
        )
        
        ques = response.choices[0].message.content

        return ques

    def translate_var(self, prop_assist):
        
        var_trans = self.prompt('var_trans')

        prop_assist += "Do not forget variable " + self.pbins

        var_trans.append({'role': 'user', 'content': prop_assist})

        response = self.client.chat.completions.create(
            model=self.model_trans,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=var_trans
        )

        assistant_message = response.choices[0].message.content

        var_trans.append({'role': 'assistant', 'content': assistant_message})

        return var_trans, assistant_message
    
    def translate_query(self, ):
        
        query_trans = self.prompt('query_trans')
        schema = self.get_schema()
        question_text = schema['Question']['Text']
        schema['Question'].pop('Text')
        question_js = json.dumps(schema['Question'])
        schema.pop('Question')
        var_schema = json.dumps(schema)
        new_message = "[Question] " + question_text + "\n\n"
        new_message += '[Database] ' + var_schema
        query_trans.append({'role': 'user', 'content': new_message})


        response = self.client.chat.completions.create(
            model=self.model_trans,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=query_trans
        )

        assistant_message = response.choices[0].message.content

        query_trans.append({'role': 'assistant', 'content': assistant_message})

        return query_trans, assistant_message
    

    def get_variable(self, question):

        prop = self.prompt('var_prop')

        
        prop[-1]['content'] += "Additionally, we would focus on question like: " + question
        prop[-1]['content'] += " Therefore, make sure the variable values at least cover those in the question. "
        prop[-1]['content'] += "Think about the question first. What can you infer to help with modeling? Feel free to choose other variables that help model that question. However, only provide the variables but don't proceed to solve the question. "


        response = self.client.chat.completions.create(
            model=self.model_dial,
            response_format={ "type": "text" },
            temperature=self.temperature,
            messages=prop
        )

        # Get the assistant's response
        assistant_message = response.choices[0].message.content
        
        # Append the assistant's response to the message list
        prop.append({"role": "assistant", "content": assistant_message})

        return prop.copy(), assistant_message


    def compile(self, ):

        
        schema = self.get_schema()
        if 'Text' not in schema['Question']:
            #print("Question text not found. Translating...")
            schema['Question']['Text'] = self.question_text(schema['Question'])
            self.set_schema(schema)
            schema = self.get_schema()
            #print(schema['Question']['Text'])

        #print("Getting variable proposal...")
        var_mess, var_assist = self.get_variable(schema['Question']['Text'])
        self.records['variable proposal'] = var_mess


        #print("Translating the variables...")
        js_mess, js_assist = self.translate_var(var_assist)
        self.records['variable translation'] = js_mess

        self.log()

        var_schema = json.loads(js_assist)
        schema['Variables'] = var_schema['Variables']
        self.set_schema(schema)
        schema = self.get_schema()

        #print("Translating the query...")
        query_mess, query_assist = self.translate_query()
        self.records['query translation'] = query_mess

        schema['Queries'] = json.loads(query_assist)['Queries']
        self.set_schema(schema)
        self.records['schema'] = schema

        return self.get_schema()

