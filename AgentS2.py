from pathlib import Path
import random
import json
from openai import OpenAI
import dotenv
import torch
import numpy as np
import argparse
dotenv.load_dotenv('.env')



class AgentS2():
    def __init__(self, prompts, schema, log_dir='./example_files', log_name='s2ex.json', city='United States', model_dial='gpt-4o', model_trans='gpt-3.5-turbo', chat_temp=0.3, other=True, insert_question=True):
        
        if type(schema) == str and schema.endswith('.json'):
            schema = json.load(open(schema))
        if type(schema) == str:
            schema = json.loads(schema)
        self.schema = schema
        self.temperature = chat_temp

        # make log directory, recursively
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.log_path = Path(log_dir, log_name)

        self.other = other

        # make a copy of the prompts, recursively since it's a list of dicts

        if type(prompts) == str and prompts.endswith('.json'):
            prompts = json.load(open(prompts))
        if type(prompts) == dict:
            prompts = json.dumps(prompts)
        
        if city != 'United States':
            prompts = prompts.replace('United States', city)
        if insert_question:
            prompts = prompts.replace('[INSERTQUESTION]', self.schema['Question']['Text'])

        self.prompts = json.loads(prompts)
            
        self.constraints_rec = None

        self.client = OpenAI()
        self.var_dict = {}
        for entry in schema['Variables']:
            self.var_dict[entry['Name']] = entry['Value']
        self.model_dial = model_dial
        self.model_trans = model_trans
        self.city = city

        self.records = {}
        self.records['schema'] = json.loads(json.dumps(self.schema))

        self.mega_records = []

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


    def schema_text(self, shuffle=False):
        schema = json.loads(json.dumps(self.schema))
        if shuffle:
            random.shuffle(schema['Variables'])

        text = "\n"
        vars = schema['Variables']
        var_counts = 0
        for i in vars:
            if i['Name'] == 'City':
                continue
            text += str(var_counts+1) + f". **{i['Name']}**: \n"
            for val in i['Value']:
                text += f"  - {val}\n"
            var_counts += 1
        return text
    
    def chat(self, messages, new_user_message=None):

        messages = messages.copy()

        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})

        # Send the updated message list to the API
        response = self.client.chat.completions.create(
            model=self.model_dial,
            response_format={ "type": "text" },
            temperature=self.temperature,
            messages=messages
        )

        # Get the assistant's response
        assistant_message = response.choices[0].message.content
        
        # Append the assistant's response to the message list
        messages.append({"role": "assistant", "content": assistant_message})

        return messages.copy(), assistant_message


    def question_text(self, js_question, add_city=True):

        question_prompt = self.prompt('question_text')

        if type(js_question) == dict:
            js_question = json.dumps(js_question)

        js_question = json.loads(js_question)

        cond_append = ''
        if len(js_question['Condition']) > 0:
            cond_append = ' given ' + js_question['Condition'][0]['Name'] + ' is ' + ", ".join(js_question['Condition'][0]['Value'])

        if add_city:
            js_question['Condition'].append({'Name': 'City', 'Value': [self.city]})
            cond_append += ' in ' + self.city
        
        tar = js_question['Target'][0]['Name']
        tar_vals = self.var_dict[tar]
        js_question = json.dumps(js_question)

        question_prompt.append({'role': 'user', 'content': js_question})

        response = self.client.chat.completions.create(
            model=self.model_dial,
            response_format={ "type": "text" },
            temperature=0,
            messages=question_prompt
        )
        
        ques = response.choices[0].message.content

        ques += '\ni.e. give ' + str(len(tar_vals)) + ' probabilities: ' + ", ".join(tar_vals) + cond_append + '. '
        ques += ' Ensure they sum to 1. '
        #print(ques)

        return ques


    def cons_prop_json(self, prop_assist):
        
        cons_trans = self.prompt('cons_trans')
        schema = self.get_schema()
        tmp_schema = {}
        tmp_schema['Variables'] = schema['Variables']
        schema = json.dumps(tmp_schema)
        cons_trans.append({'role': 'user', 'content': '[Schema] ' + schema + '\n\n[Transcribe]' + prop_assist})
        #print("using schema: ", schema)

        response = self.client.chat.completions.create(
            model=self.model_trans,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=cons_trans
        )
        cons_trans.append({'role': 'assistant', 'content': response.choices[0].message.content})

        return cons_trans, response.choices[0].message.content


    def get_prob(self, message, target_var, ):

        prob_prompt = self.prompt('get_prob')
        vals = self.var_dict[target_var]
        message += f"\nYou should extract {len(vals)} probabilities, in the order of " + ', '.join(vals) + "."
        prob_prompt.append({'role': 'user', 'content': message})

        response = self.client.chat.completions.create(
            model=self.model_trans,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=prob_prompt,
        )

        assistant_message = response.choices[0].message.content
        prob_prompt.append({"role": "assistant", "content": assistant_message})
        dialogue = prob_prompt.copy()



        prob_js = json.loads(assistant_message)

        return dialogue, prob_js
    

    def zero_cot(self, ):

        zero_prompt = self.prompt('zero')

        schema = self.get_schema()
        question_text = schema['Question']['Text']

        question_text += " Provide probabilities for: " + ", ".join(self.var_dict[schema['Queries'][0]['Target'][0]['Name']])

        zero_prompt.append({"role": "user", "content": question_text})

        response = self.client.chat.completions.create(
            model=self.model_dial,
            response_format={ "type": "text" },
            temperature=self.temperature,
            messages=zero_prompt
        )

        assist = response.choices[0].message.content

        zero_prompt.append({"role": "assistant", "content": assist})

        # get probabilties
        prob_mess, prob_js = self.get_prob(assist, schema['Queries'][0]['Target'][0]['Name'])
        self.records['zero shot'] = {}
        self.records['zero shot']['zero dialogue'] = zero_prompt
        self.records['zero shot']['translate dialogue'] = prob_mess
        self.records['zero shot']['result'] = prob_js
        self.log()


        # put the zero shot prompting in the beginning of the mega records
        cot = [i.copy() for i in self.prompt('zero')]
        self.mega_records.append({'role': 'user', 'content': question_text})
        cot += self.mega_records



        response_cot = self.client.chat.completions.create(
            model=self.model_dial,
            response_format={ "type": "text" },
            temperature=self.temperature,
            messages=cot
        )

        cot_assist = response_cot.choices[0].message.content
        cot.append({"role": "assistant", "content": cot_assist})

        cot_prob_mess, cot_prob_js = self.get_prob(cot_assist, schema['Queries'][0]['Target'][0]['Name'])
        self.records['cot'] = {}
        self.records['cot']['cot dialogue'] = cot
        self.records['cot']['translate dialogue'] = cot_prob_mess
        self.records['cot']['result'] = cot_prob_js
        self.log()


        return prob_js, cot_prob_js

        


    def propose_constraints(self, shuffle=False):
        """Get constraints proposal from an LLM and update the schema."""


        schema = self.get_schema()

        #print("Proposing constraints...")
        cons_prompt = self.prompt('cons_prop')
        schema_text = self.schema_text(shuffle)
        cons_prompt[-1]['content'] += schema_text
        prop_mess, prop_assist = self.chat(cons_prompt)

        self.mega_records += [i.copy() for i in prop_mess[-2:]]


        #print("Translating the constraints...")
        trans_mess, trans_assist = self.cons_prop_json(prop_assist)
        #print(trans_assist)

        self.records['constraint proposal'] = prop_mess
        self.records['constraint translation'] = trans_mess
        self.log()
        
        proposed = json.loads(trans_assist)
        schema['Constraints'] = proposed['Constraints']
        self.set_schema(schema)

        return
    

    def populate_constraint_prob(self, question_js, add_city=True, conf=False, reject=True):
        """Populate the probability of the target variable in the question_js, a Constraints entry."""

        if type(question_js) == dict:
            question_js = json.dumps(question_js)
        question_js = json.loads(question_js)
        # get number of variable values
        val_count = len(question_js['Target'][0]['Value'])
        question_text = self.question_text(question_js, add_city=add_city)

        if reject:
            got = 0
            looped = 0
            while got != val_count:
                val_mess, val_assist = self.chat(self.prompt('zero'), question_text)
                prob_mess, prob_js = self.get_prob(val_assist, question_js['Target'][0]['Name'])
                if 'Probability' in prob_js:
                    got = len(prob_js['Probability'])
                else:
                    got = 0
                looped += 1
                if looped > 5:
                    return None
            self.mega_records += [i.copy() for i in val_mess[-2:]]
        else:
            val_mess, val_assist = self.chat(self.prompt('zero'), question_text)
            prob_mess, prob_js = self.get_prob(val_assist, question_js['Target'][0]['Name'])
            self.mega_records += [i.copy() for i in val_mess[-2:]]
        
        question_js['Probability'] = prob_js['Probability']
        #if conf and 'Confidence' in prob_js:
        #    question_js['Confidence'] = prob_js['Confidence']

        rec = {}
        rec['question json'] = question_js.copy()
        rec['question text'] = question_text
        rec['value'] = val_mess
        rec['value translation'] = prob_mess
        self.records['constraint values'].append(rec)

        return question_js


    def compile(self, add_city=True, conf=True):

        schema = self.get_schema()

        if 'Constraints' not in schema:
            self.propose_constraints()
            schema = self.get_schema()

        if self.constraints_rec is not None:
            print('You should not reuse AgentS2')
            return
        
        self.log()
        self.constraints_rec = []
        self.records['constraint values'] = []

        #print("Populating the marginals...")
        # first ask for marginals on the variables
        for var in schema['Variables']:
            #print("Populating marginal for ", var['Name'])
            question_js = {'Target': [{'Name': var['Name'], 'Value': var['Value']}], 'Condition': []}
            question_js = self.populate_constraint_prob(question_js, add_city, conf)
            if question_js is not None:
                self.constraints_rec.append(question_js)

        self.records['schema'] = schema
        self.log()
        
        #print("\n--------------------------------------------\n")
        #print("Populating the constraints...")
        seen = set()
        for i, entry in enumerate(schema['Constraints']):
            if len(entry['Condition']) == 0 or len(entry['Target']) == 0:
                print("Warning: Target and Condition should both be proposed.")
                continue
            for cond in entry['Condition']:
                if 'Value' in cond:
                    print("Warning: Condition should not have Value.")
            #print("Populating constraint ", i)
            tar = entry['Target'][0]['Name']
            tar_vals = self.var_dict[tar]
            new_tar = [{'Name': tar, 'Value': tar_vals}]
            cond = entry['Condition'][0]['Name']

            if (tar, cond) in seen:
                continue
            else:
                seen.add((tar, cond))

            for cond_val in self.var_dict[cond]:
                if self.other:
                    new_cond = [{'Name': cond, 'Value': [cond_val], 'Other': [i for i in self.var_dict[cond] if i != cond_val]}]
                else:
                    new_cond = [{'Name': cond, 'Value': [cond_val]}]
                question_js = {'Target': new_tar, 'Condition': new_cond}
                question_js = self.populate_constraint_prob(question_js, add_city, conf)
                if question_js is not None:
                    self.constraints_rec.append(question_js)


        self.log()

        schema['Constraints'] = self.constraints_rec
        self.set_schema(schema)
        self.records['schema'] = schema
        schema = self.get_schema()
        self.log()

        zero_prob, cot_prob = self.zero_cot()


        return self.records

