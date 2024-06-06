from openai import OpenAI
import dotenv
import rich
import json

dotenv.load_dotenv('/Users/shepardxia/Desktop/directory/commonsense/.env')

class LLM:
    def __init__(self, initial_message, json_format=None, model="gpt-4o"):
        self.chache = []
        self.json = []
        self.model = model
        self.client = OpenAI()
        if initial_message:
            self.initial_message = initial_message.copy()
        else:
            raise ValueError('No initial message provided')
        self.current_dialogue = None

        self.current_json = None
        self.var_json = None

        self.cons_json = []

        self.json_format = json_format

    def get_current_dialogue(self):
        return self.current_dialogue.copy()
    
    def set_current_dialogue(self, messages):
        self.current_dialogue = messages.copy()
    
    def get_initial_message(self):
        return self.initial_message.copy()
    

    def set_initial_message(self, message):
        self.initial_message = message.copy()
    

    def reset(self):
        self.chache = []
        self.json = []
        self.current_json = []
        self.var_json = None
        self.current_dialogue = self.get_initial_message()



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
    

    def combine_json(self):
        print('not implemented')
        return self.json
    

    def button(self, seq_dialogues, para_dialogues):
        self.reset()
        for i, dialogue in enumerate(seq_dialogues):
            self.continue_from_last(dialogue)
        for dialogue in para_dialogues:
            if dialogue is not None:
                self.continue_from_last(dialogue)
            self.get_json()
            self.cache()
            self.backspace(cnt= 2 if dialogue else 1)
        return self.current_json
            

    
    def para_chat(self, messages, new_messages=[], temperature=0.5):
        ret = []
        for prompt in new_messages:
            mes = self.continue_chat(messages, prompt, temperature=temperature)
            ret.append(mes)
        return ret
    
    def iter_chat(self, messages=None, new_messages=[], temperature=0.5):
        if messages is None:
            messages = self.get_initial_message()
        for prompt in new_messages:
            messages = self.continue_chat(messages, prompt, temperature=temperature)
        return messages


    def get_json(self, new_user_message=None, messages=None, temperature=0):
        if messages is None:
            messages = self.get_current_dialogue()
        if new_user_message is None:
            new_user_message = self.json_format
            if self.current_json:
                new_user_message += '\nWe already have the following JSON for variable definitons:\n'
                new_user_message += self.current_json

        ret = self.continue_chat(messages, new_user_message, temperature, json=True)
        assistant_message = ret[-1]['content']

        if not self.current_json:
            self.current_json = assistant_message
        current_j = json.loads(assistant_message)
        if not self.var_json:
            self.var_json = current_j['variables']
        if 'constraints' in current_j and len(current_j['constraints']) > 0:
            self.cons_json += current_j['constraints']

        return ret

    def continue_from_last(self, new_user_message=None, temperature=0.5, json=False):
        return self.continue_chat(self.get_current_dialogue(), new_user_message, temperature, json)


    def continue_chat(self, messages, new_user_message=None, temperature=0.5, json=False):
        """
        Continue the chat with the GPT model.
        
        :param messages: List of previous messages in the conversation
        :param new_user_message: New message from the user
        :param model: Model to use for generating responses (default is "gpt-4")
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

        self.prent(messages)
        
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

