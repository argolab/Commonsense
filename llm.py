from openai import OpenAI
import dotenv
import rich
import json
import torch
import numpy as np

dotenv.load_dotenv('.env')

class LLM:
    def __init__(self, prompt='./example_files/prompt.json', model="gpt-4o", question=None, sequences=None, json_model="gpt-4o", prob_model='gpt-3.5-turbo', verbose=False):
        self.json = []

        self.model = model
        self.json_model = json_model
        self.prob_model = prob_model

        self.client = OpenAI()
        # load prompt
        with open(prompt, 'r') as f:
            self.prompt = json.load(f)

        self.verbose = verbose
        
        self.current_dialogue = None


        self.current_json = None
        self.var_json = None

        self.cons_json = []

        if 'json_dialogue' in self.prompt:
            self.json_dialogue = self.prompt['json_dialogue'].copy()
        else:
            raise ValueError('No json dialogue provided')

        self.response = None

        self.question = question
        if self.question:
            self.add_condition()

        if sequences is None:
            self.sequences = [['binq', 'brainq', 'broadq', 'moreq', 'varq'], [None, 'margq', 'pricecondq', 'interactq']]
        else:
            self.sequences = sequences

        self.query_q = None

        self.zero = []
        self.zerojs = []


        self.dialogues = {}
        self.dialogues['mrf'] = {}
        self.dialogues['cot'] = {}
        self.dialogues['zero'] = {}

        self.results = {}

        self.error = False
        


    def get_zero(self, ):
        #self.reset()
        question = self.question['Text']
        if 'bins' in self.prompt:
            question += '\n' + self.prompt['bins']
        else:
            question += self.prompt['freebins']
        question += '\n' + 'Provide the probability of each bin for the variable price. Give a definite value, not a range.'
        messages = self.chat(self.get_initial_message(version='zero'), question, update=False).copy()
        self.dialogues['zero'] = {}
        self.dialogues['zero']['main dialogue'] = messages.copy()
        messages, prob = self.get_prob(messages)
        self.dialogues['zero']['probability dialogue'] = messages.copy()
        self.results['zero'] = prob.tolist()
        return prob


    def set_verbose(self, verbose):
        self.verbose = verbose


    def get_cot(self, sequences=None):
        if 'cot' not in self.dialogues or 'main dialogue' not in self.dialogues['cot']:
            self.button(sequences=sequences, js=False)
        

        question = self.question['Text']
        question += '\n' + 'Provide the probability of each bin for the variable price. Give a definite value, not a range.'
        messages = self.chat(self.dialogues['cot']['main dialogue'], question, update=False).copy()
        self.dialogues['cot']['main dialogue'] = messages.copy()
        short_mes = messages[:1] + messages[-2:]
        latest, prob = self.get_prob(short_mes)
        self.dialogues['cot']['probability dialogue'] = latest.copy()
        self.results['cot'] = prob.tolist()
        return prob


    def get_prob(self, messages):

        messages = messages.copy()
        messages.append({"role": "user", "content": self.prompt['getprob']})
        response = self.client.chat.completions.create(
            model=self.prob_model,
            response_format={ "type": "json_object" },
            temperature=0,
            messages=messages,
        )
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        latest = messages.copy()[-2:]

        try:
            js = json.loads(assistant_message)
            array = np.array(js['Probability'])
        except:
            print('Error: Could not parse JSON')
            self.error = True
            array = None

        return latest, array



    def get_current_dialogue(self):
        if self.current_dialogue is None:
            self.current_dialogue = self.get_initial_message()
        return self.current_dialogue.copy()

    def set_current_dialogue(self, messages):
        self.current_dialogue = messages.copy()


    def get_initial_message(self, version='default'):

        if 'initial_message' not in self.prompt:
            raise ValueError('No initial message provided in prompt.')
        if version not in self.prompt:
            raise ValueError('No version of initial message provided in prompt.')
        tmp = (self.prompt['initial_message'] + self.prompt[version])

        return [{'role': 'system', 'content': tmp}].copy()


    def get_json_dialogue(self):
        return self.json_dialogue.copy()
    
    def set_json_dialogue(self, messages):
        self.json_dialogue = messages.copy()



    def reset(self):
        self.json = []
        self.current_json = None
        self.var_json = None
        self.current_dialogue = self.get_initial_message()
        self.cons_json = []

    def big_button(self, ):
        self.reset()
        js, query = self.button()
        cot = self.get_cot()
        zero = self.get_zero()
        self.log()
        return js, query, cot, zero
    

    def button(self, sequences=None, question=True, js=True, linear=False):
        if sequences is None:
            sequences = self.sequences
        for dialogue in sequences[0]:
            messages = self.continue_from_last(self.prompt[dialogue])

        self.dialogues['mrf']['main dialogue'] = self.get_current_dialogue()
        self.dialogues['cot']['main dialogue'] = self.get_current_dialogue()
        if js:
            self.dialogues['mrf']['json dialogue'] = []
        
        for dialogue in sequences[1]:
            if dialogue is not None:
                messages = self.continue_from_last(self.prompt[dialogue], update=linear)
                self.dialogues['cot']['main dialogue'] += messages[-2:].copy()
                self.dialogues['mrf'][dialogue] = messages.copy()
            if js:
                messages = self.get_json(new_message=messages[-1]['content'])
                self.dialogues['mrf']['json dialogue'] += messages[-2:].copy()
        if not js:
            return
        if js and question and self.question:
            messages = self.get_json(new_message=self.question['Text'], query=True)
            self.dialogues['mrf']['json dialogue'] += messages[-2:].copy()
            query_q = messages[-1]['content']
            self.dialogues['mrf']['query'] = messages.copy()
            return self.current_json, query_q
        return self.current_json, None


    def get_json(self, new_message=None, temperature=0, query=False):

        messages = self.get_json_dialogue()
        if new_message:
            if not query:
                messages.append({"role": "user", "content": '[real][record] ' + new_message})
            else:
                messages.append({"role": "user", "content": '[real][query] ' + new_message})


        response = self.client.chat.completions.create(
                model=self.json_model,
                response_format={ "type": "json_object" },
                temperature=temperature,
                messages=messages,
            )

        self.response = response
        # Get the assistant's response
        assistant_message = response.choices[0].message.content
        
        # Append the assistant's response to the message list
        messages.append({"role": "assistant", "content": assistant_message})

        self.prent(messages[-1:])

        if not self.current_json:
            self.set_json_dialogue(messages.copy())

        current_j = json.loads(assistant_message)

        if not query:
            if not self.current_json:
                self.current_json = current_j.copy()
                if 'Constraints' not in self.current_json:
                    self.current_json['Constraints'] = []
            if not self.var_json:
                self.var_json = current_j['Variables'].copy()
            if 'Constraints' in current_j and len(current_j['Constraints']) > 0:
                self.cons_json += current_j['Constraints'].copy()
                self.current_json['Constraints'] += current_j['Constraints'].copy()

        return messages.copy()


    def continue_from_last(self, new_user_message=None, temperature=0.5, update=True):
        return self.chat(self.get_current_dialogue(), new_user_message, temperature, update=update)
    

    def chat(self, messages, new_user_message=None, temperature=0.5, update=True):
        """
        Continue the chat with the GPT model.
        
        :param messages: List of previous messages in the conversation
        :param new_user_message: New message from the user
        :param temperature: Temperature setting for the model (default is 0.5)
        :return: Assistant's response
        """
        # Append the new user message to the message list

        if messages is None:
            messages = self.get_initial_message()
        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})

        messages = messages.copy()
        
        # Send the updated message list to the API
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={ "type": "text" },
            temperature=temperature,
            messages=messages
        )
        self.response = response
        # Get the assistant's response
        assistant_message = response.choices[0].message.content
        
        # Append the assistant's response to the message list
        messages.append({"role": "assistant", "content": assistant_message})


        self.prent(messages[-2:])

        if update:
            self.set_current_dialogue(messages.copy())


        return messages.copy()


    def backspace(self, messages=None, cnt=1):
        """
        Remove the last user and assistant message from the list of messages.

        :param messages: List of previous messages in the conversation
        :return: Updated list of messages
        """
        # Remove the last user message
        if messages:
            messages = messages[:-int(cnt*2)]
        else:
            self.set_current_dialogue(self.get_current_dialogue()[:-int(cnt*2)])
            messages = self.get_current_dialogue()

        return messages


    def prent(self, messages):
        if self.verbose:
            for mes in messages:
                print(mes['role'], ":", mes['content'])
                print()
                print()
                print("--------------------------------------------------------------------------------------")
                print()
                print()



    def add_condition(self, place='varq'):

        exist = False
        for check in ['Target', 'Condition']:
            if check in self.question and len(self.question[check]) > 0:
                exist = True
        if not exist:
            return
        text = self.question['Text']
        if place not in self.prompt:
            raise ValueError('No suggested condition insertion place in prompt')
        #self.prompt[place] += ' However, you must also include the following variable and possibly some values (you can add more): \n'
        
        if place == 'initial_message':
            self.prompt[place][0]['content'] += ' Specifically, we would like to be able to answer the question: ' + text
        else:
            #self.prompt[place] += ' Specifically, we would like to be able to answer the question: ' + text
            self.prompt[place] += '\nAdditionally without loss of generality, we would like to be able to answer questions such as: ' + text
            self.prompt[place] += '\nMake sure that the updated variables can express such questions with existing variables. '
            """ self.prompt[place] += ' For each of the mentioned conditions, you could either\n' + '\n1. Include variables like: \n'


            for check in ['Target', 'Condition']:
                if check in self.question:
                    for var in self.question[check]:
                        self.prompt[place] += var['Name']
                        if 'Value' in var:
                            self.prompt[place] += ' that can express ' + ', '.join(var['Value']) + '\n'
                        else:
                            self.prompt[place] += '\n'
            
            self.prompt[place] += 'However, you can choose the variable values as you see fit, e.g. when a_1 and a_2 are mentioned, it may be more reasonable to represent them with a_12, while giving additional values a_3, a_4. This allows us to consider such questions without loss of generality. \n'
            self.prompt[place] += '\n\n2. Restrict the domain of our discussion to those specific scenarios to simplify the problem, especially if it can be leveraged to produce more meaningful variables specific to the scenario. In this case, it will not count towards the maximum amount of variables used. Remember to try to leverage this specific scenario with other possible variables.\n'  """
            
        if self.verbose:
            print("Modified variable question: ", self.prompt[place])


    def get_variable_json(self):
        ret = []
        for check in ['Target', 'Condition']:
            if check in self.question:
                for tar in self.question[check]:
                    tmp = {}
                    if 'Name' not in tar:
                        print('Error: Variable does not have a name')
                        return None
                    else:
                        tmp['Name'] = tar['Name']
                    if 'Value' in tar:
                        tmp['Value'] = tar['Value']
                    ret.append(tmp)
        ret = {'Variables': ret}
        return ret
    
    def log(self, ):
        with open('log.json', 'w') as f:
            json.dump(self.dialogues, f, indent=4)
        with open('results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        with open('prompt.json', 'w') as f:
            json.dump(self.current_json, f, indent=4)

