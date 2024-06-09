from openai import OpenAI
import dotenv
import rich
import json

dotenv.load_dotenv('/Users/shepardxia/Desktop/directory/commonsense/.env')

class LLM:
    def __init__(self, prompt='./prompt.json', model="gpt-4o", question=None, sequences=None):
        self.chache = []
        self.json = []
        self.model = model
        self.client = OpenAI()
        # load prompt
        with open(prompt, 'r') as f:
            self.prompt = json.load(f)
        if 'initial_message' in self.prompt:
            self.set_initial_message(self.prompt['initial_message'])
        else:
            raise ValueError('No initial message provided')
        
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


    

    def get_current_dialogue(self):
        return self.current_dialogue.copy()
    
    def set_current_dialogue(self, messages):
        self.current_dialogue = messages.copy()
    
    def get_initial_message(self):
        return self.initial_message.copy()
    
    def set_initial_message(self, message):
        self.initial_message = [{'role': 'system', 'content': message}]

    def get_json_dialogue(self):
        return self.json_dialogue.copy()
    
    def set_json_dialogue(self, messages):
        self.json_dialogue = messages.copy()
    

    def reset(self):
        self.chache = []
        self.json = []
        self.current_json = None
        self.var_json = None
        self.current_dialogue = self.get_initial_message()
        self.cons_json = []

    def cache(self, messages=None):
        if messages:
            self.chache.append(messages)
        else:
            self.chache.append(self.get_current_dialogue())

    def get_cache(self):
        return self.chache
    
    def load_dialogue(self, messages=None, index=None):
        if messages:
            self.set_current_dialogue(messages)
        else:
            if index:
                self.set_current_dialogue(self.chache[index])
            else:
                self.set_current_dialogue(self.chache[-1])
    
    def cache_json(self, message):
        self.json.append(message)
    
    def get_cache_json(self):
        return self.json
    

    def button(self, sequences=None, question=True):
        if sequences is None:
            sequences = self.sequences
        self.reset()
        for dialogue in sequences[0]:
            self.continue_from_last(self.prompt[dialogue])
        for dialogue in sequences[1]:
            if dialogue is not None:
                self.continue_from_last(self.prompt[dialogue])
            self.get_json(new_message=self.get_current_dialogue()[-1]['content'])
            self.cache()
            if dialogue is not None:
                self.backspace(cnt = 1)
        if question and self.question:
            self.query_q = self.get_json(new_message=self.question['Text'], query=True)
            return self.current_json, self.query_q
        return self.current_json

    
    def iter_chat(self, messages=None, new_messages=[], temperature=0.5):
        if messages is None:
            messages = self.get_initial_message()
        for prompt in new_messages:
            messages = self.continue_chat(messages, prompt, temperature=temperature)
        return messages


    def get_json(self, new_message=None, temperature=0, query=False):

        messages = self.get_json_dialogue()
        if new_message:
            if not query:
                messages.append({"role": "user", "content": '[real][record] ' + new_message})
            else:
                messages.append({"role": "user", "content": '[real][query] ' + new_message})


        response = self.client.chat.completions.create(
                model=self.model,
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

        return assistant_message

    def continue_from_last(self, new_user_message=None, temperature=0.5, json=False):
        return self.continue_chat(self.get_current_dialogue(), new_user_message, temperature, json)

    def continue_chat(self, messages, new_user_message=None, temperature=0.5, json=False):
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
        
        # Send the updated message list to the API
        if json:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={ "type": "json_object" },
                temperature=temperature,
                messages=messages,
            )
        else:
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

        self.set_current_dialogue(messages.copy())


        #return assistant_message, messages
        return messages

        

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

        #self.prent(messages)
        
        # Return the updated list of messages
        return messages

    def prent(self, messages):
        for mes in messages:
            print(mes['role'], ":", mes['content'])
            print()
            print()
            print("--------------------------------------------------------------------------------------")
            print()
            print()

    def modify_prompt(self):
        self.add_condition()


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

