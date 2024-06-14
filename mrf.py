import torch
from torch import nn
import pandas as pd
import json
import tqdm


class Brute(nn.Module):
    def __init__(self, verbose=False):
        super(Brute, self).__init__()
        
        self.total_pos = []
        self.optim = None

        self.w0 = 0

        self.verbose = verbose


        self.universe = None
        self.universe_fire = None
        self.constraints = []
        self.targets = []
        self.conditions = []
        self.constraints_values = []
        self.var_dict = dict()
        self.val_dict = dict()
        self.num_features = 0

        self.batch_target = "Target"
        self.batch_probability = "Probability"


    def forward(self, x):
        """
        Not implemented
        """
        return x
    
    def set_w0(self, w0):
        """
        Set the weight for the entropy term, positive for maximization
        """
        self.w0 = w0

    
    def from_json(self, entire_json=None, vars=None, constraints=None):
        """
        Load the model from json
        either use (entire_json) or (vars and constraints)

        vars should be a dictionary with the variable names as keys and the possible values as a list
        constraints should be a list of lists, each tuple should have 3 elements

        vars <-> llm.
        """
        if entire_json:
            if type(entire_json) == str:
                data = json.loads(entire_json)
            else:
                data = entire_json
            #self.update_white(data)
            self.json_vars(data['Variables'])
            self.json_constraints(data['Constraints'])
        else:
            if vars:
                self.json_vars(vars)
            if constraints:
                self.json_constraints(constraints)
        return self.finish_build()



    def json_constraints(self, constraints):
        """
        Adding constraints from a list of constraints"""
        if type(constraints) != list:
            print("ERROR: Constraints must be a list")
            return
        for entry in constraints:
            if entry is None:
                continue
            condition = self.translate(entry['Condition']) if 'Condition' in entry else None
            if '(Deprecated)Target' not in entry:
                if self.batch_target not in entry or self.batch_probability not in entry:
                    print("ERROR: Invalid constraint type 1", entry)
                    continue
                batch_target, full = self.batch_translate(entry[self.batch_target])
                if full:
                    # sum of all probabilities should be 1
                    tot = 0
                    for prob in entry[self.batch_probability]:
                        if type(prob) not in [int, float]:
                            print(prob)
                            continue
                        tot += prob
                    if abs(tot - 1) > 0.05:
                        print("[Error]marginal do not add to 1: ", tot, " from ", entry[self.batch_probability], batch_target)
                        continue
                        entry[self.batch_probability] = [prob / tot for prob in entry[self.batch_probability]]
                        print("Normalized to ", entry[self.batch_probability])
                if batch_target is None:
                    continue
                if len(batch_target) != len(entry[self.batch_probability]):
                    print("ERROR: length of target does not match probability", entry)
                    continue
                for target, prob in zip(batch_target, entry[self.batch_probability]):
                    if target is None:
                        print("ERROR: invalid constraint type 2", entry)
                        continue
                    if type(prob) not in [int, float]:
                        print("ERROR: invalid probability", entry)
                        continue
                    self.add_constraint(target, condition, prob)
                    #if entry[self.batch_target][0]['Name'] == 'Price Ranges':
                    #    print(target, condition, prob)
            else:
                target = self.translate(entry['(Deprecated)Target'])
                if target is None:
                    print("ERROR: invalid constraint type 3", entry)
                    continue
                self.add_constraint(target, condition, entry['Probability'])


    def json_vars(self, json):
        """
        Add variables from a dictionary containing the variable names and their possible values"""


        for j, entry in enumerate(json):
            name = str(entry['Name'])
            self.add_var(len(entry['Value']))
            self.val_dict[name] = dict()
            for i, val in enumerate(entry['Value']):
                val = str(val)
                self.val_dict[name][val] = i
                self.val_dict[name][i] = val
            self.val_dict[j] = self.val_dict[name]
            self.var_dict[name] = j
            self.var_dict[j] = name
        self.N = len(self.total_pos)
        if self.verbose:
            self.vars_print()

    def vars_print(self):
        """
        Print the variables and their possible values"""

        for key, value in self.var_dict.items():
            if type(key) == str:
                print("\nVariable: ", key)
                to_print = ''
                for kkey, vvalue in self.val_dict[value].items():
                    if type(kkey) == str:
                        to_print += kkey + ', '
                print('Value: ', to_print[:-2], "\n")


    def infer(self, target, condition):
        """
        Calculate the marginal probability of the target given the condition (optional)"""
        self.eval()
        with torch.no_grad():
            cond = self.get_mask(condition)
            tar = self.get_mask(target, condition=cond)
            firing = torch.exp((self.features_firing * self.weights).sum(dim=-1))

            tar_firing = (tar * firing).sum()
            cond_firing = (cond * firing).sum()

            targets = tar_firing
            conditions = cond_firing

            return targets / conditions


    def calculate_entropy(self,):
        """
        Calculate the model entropy"""
        entropy_each = torch.exp((self.features_firing * self.weights).sum(dim=-1))

        bot = entropy_each.sum()
        px = (entropy_each / bot)
        return 1 * (px * torch.log(px)).sum()
        
        
    def add_var(self, possible):
        """
        update the total possible values for variable"""
        self.total_pos.append(possible)


    def add_constraint(self, y, x, constraints):
        """
        Generic add single constraint, inequality or equality"""
        self.constraints.append([y, x, constraints])


    def build_universe(self, ):

        entire = torch.zeros(list(self.total_pos))
        universe = torch.nonzero(entire == 0).float()
        return universe


    def get_mask(self, inputs, condition=None):

        checker = torch.arange(0, self.universe.shape[0])
        ret = torch.ones(self.universe.shape[0]).bool() if condition is None else condition


        if torch.is_tensor(inputs):
            inputs = inputs.int().tolist()
        if inputs is None:
            return ret.flatten()

        for i, input in enumerate(inputs[::-1]):
            if torch.is_tensor(input):
                input = input.int().tolist()
            if type(input) != list and type(input) != tuple and input != -1:
                ret = ret & (checker % self.total_pos[-(i+1)] == input)
            elif type(input) == list or type(input) == tuple:
                div = torch.zeros(self.universe.shape[0]).bool()
                for inp in input:
                    div |= (checker % self.total_pos[-(i+1)] == inp)
                ret = ret & div

            checker = checker // self.total_pos[-(i+1)]


        ret = ret.flatten()

        return ret
    
    def translate(self, inputs):
        ret = [-1 for i in range(self.N)]
        for input in inputs:
            possible_values = []
            var_ind = -1
            if type(input['Name']) == str:
                var_ind = self.var_name2ind(input['Name'])
            elif type(input['Name']) == int:
                var_ind = input['Name']
            if var_ind == -1:
                print("ERROR: Invalid input")
                return None
            for val in input['Value']:
                val_ind = -1
                if type(val) == str:
                    val_ind = self.val_name2ind(var_ind, val)
                elif type(val) == int:
                    val_ind = val
                if val_ind == -1:
                    print("ERROR: Invalid input")
                    return None
                possible_values.append(val_ind)
            ret[var_ind] = tuple(possible_values)
        return ret

    def batch_translate(self, inputs):
        ret = [-1 for i in range(self.N)]
        rets = []
        full = False
        for input in inputs:
            var_ind = self.var_name2ind(input['Name'])
            if var_ind == -1:
                print("Target Variable not found in batch translate", input['Name'])
                return None
            if len(input['Value']) == self.total_pos[var_ind]:
                full = True
            for val in input['Value']:
                possible_values = []
                val_ind = self.val_name2ind(var_ind, val)
                if val_ind == -1:
                    #print(input['Name'], val)
                    print("Target Value not found in batch translate")
                    return None
                possible_values.append(val_ind)
                tmp = ret.copy()
                tmp[var_ind] = tuple(possible_values)
                rets.append(tmp)
        return rets, full



    def finish_build(self, ):

        self.N = len(self.total_pos)

        self.total_pos = torch.tensor(self.total_pos, dtype=torch.int32)
        self.universe = self.build_universe()

        feature_firing = []

        features = set()
        constraints = {}

        # tar and cond should have the same format, a list of total possible values
        # equality only
        for i, (tar, cond, prob) in enumerate(self.constraints):
            
            condition = self.get_mask(cond)
            target = self.get_mask(tar, condition=condition)

            tar_tup = tuple(target.tolist())
            cond_tup = tuple(condition.tolist())


            if (tar_tup, cond_tup) in constraints.keys():
                print("ERROR: Duplicate constraints with ", constraints[(tar_tup, cond_tup)], tar, cond, prob)
                continue
            else:
                constraints[(tar_tup, cond_tup)] = i
            self.targets.append(target)
            self.conditions.append(condition)

            if tar_tup not in features:
                features.add(tar_tup)
                feature_firing.append(target)
            if cond_tup not in features:
                features.add(cond_tup)
                feature_firing.append(condition)

            if type(prob) == float:
                prob = [prob]
            elif type(prob) == int:
                prob = [prob]
            #if len(prob) == 2:
            #    prob = [(prob[0] + prob[1])/2]
            self.constraints_values.append(prob)

        #print(features)
        self.targets = torch.stack(self.targets, dim=0)
        self.conditions = torch.stack(self.conditions, dim=0)
        self.top_bottom = torch.stack([self.targets, self.conditions], dim=0)
        self.features_firing = torch.stack(feature_firing, dim=0).t()
        self.constraints_values = torch.tensor(self.constraints_values).squeeze()
        
        self.num_features = self.features_firing.shape[1]
        self.weights = nn.Parameter(torch.randn(self.num_features))
        self.optim = torch.optim.SGD(self.parameters(), lr=0.1)

        if self.verbose:
            print('-----------------------------------------------')
            print("Finished building")
            #self.vars_print()
            print("Features: ", self.num_features)
            print("Constraints: ", len(self.constraints))
            print('-----------------------------------------------')
        return features


    def update(self, epochs=100, slack=False):

        """
        Training"""
        self.losses = []
        self.train()
        if self.verbose:
            loop = tqdm.tqdm(range(epochs), desc="Training Progress")
        else:
            loop = range(epochs)
        loss_ent = 0
        #for n in tqdm.tqdm(range(epochs)):
        for n in loop:
            self.optim.zero_grad()
            

            firing = torch.exp((self.features_firing * self.weights).sum(dim=-1))

            top_bot_firing = (self.top_bottom * firing)
            top_bot = top_bot_firing.sum(dim=-1)

            loss = top_bot[0]/top_bot[1] - self.constraints_values
            # cut out ones with loss less than 0.02
            if not slack:
                loss = loss.pow(2).sum()
            else:
                loss = loss[torch.abs(loss) > 0.01].pow(2).sum()

            if self.w0 != 0:
                loss_ent = self.w0 * self.calculate_entropy()
                loss += loss_ent
                loss_ent = loss_ent.detach().item()

            loss.backward()
            self.optim.step()
            self.losses.append(loss.detach())
            if self.verbose:
                if n % (epochs // 10) == 0 or n == epochs - 1:
                    tmp = top_bot[0]/top_bot[1] - self.constraints_values
                    if loss_ent != 0:
                        print("Entropy: ", loss_ent)
                    print("Loss: ", loss, tmp.topk(5))
                    #print("Violated: ", violated, " out of ", tot)


    # ---------------------------Below are implementations for API reference------------------------------------------------------------------

    def nvars(self):
        """
        Return the number of variables
        """
        return self.N
    

    def nfeatures(self):
        """
        Return the number of features
        """
        return self.num_features
    
    def var_name2ind(self, name):
        """
        Return the index of a variable given the name
        """
        if name not in self.var_dict.keys():
            print("ERROR: Variable not found", name)
            return -1
        return self.var_dict[name]
    

    def var_ind2name(self, i):
        """
        Return the name of a variable given the index
        NOTE: the value index is 0-based
        """
        if i >= self.N:
            print("ERROR: Index out of range")
            return -1
        return self.var_dict[i]
    
    def val_name2ind(self, var, val):
        """
        Return the index of a value given the variable name and the value name
        NOTE: the value index is 0-based
        """
        if type(var) == str:
            var_ind = self.var_name2ind(var)
        else:
            var_ind = var
        if var_ind == -1:
            return -1
        if type(val) != str:
            print("ERROR: Value name must be a string")
            return -1
        if val not in self.val_dict[var]:
            print("ERROR: Value not found", val)
            return -1
        return self.val_dict[var][val]


    def val_ind2name(self, var, val):
        """
        Return the name of a value given the variable name and the value index
        """
        if type(var) == str:
            var_ind = self.var_name2ind(var)
        else:
            var_ind = var
        if type(val) != int:
            print("ERROR: Value index must be an integer")
            return -1
        if val >= self.total_pos[var_ind]:
            print("ERROR: Index out of range")
            return -1
        return self.val_dict[var][val]
    


    def marg(self, target=-1, condition=None):
        """
        Calculate the marginal probability of the target given the condition (optional)
        """
        # translate the condition for input to translate()
        proc_cond = []
        if type(condition) != list:
            for key, values in condition.items():
                proc_cond.append({'Name': key, 'Value': values})
        else:
            proc_cond = condition

        proc_tar = -1
        if type(target) == str:
            proc_tar = self.var_dict[target]
        elif target == -1:
            proc_tar = self.N - 1

        cond = self.translate(proc_cond)
        #print("Condition: ", cond, proc_cond)

        ret = []
        tot = 0
        for i in range(self.total_pos[proc_tar]):
            tar_tmp = [{"Name": proc_tar, "Value": [i]}]
            tar = self.translate(tar_tmp)
            ret.append(self.infer(tar, cond).item())
            tot += ret[-1]
        if abs(tot - 1) > 0.001:
            print("[Error]marginal do not add to 1: ", tot)
        return ret


    def query(self, query):
        """
        Query the model with a dictionary of variables and their values
        """
        if type(query) == str:
            query = json.loads(query)
        ret = []
        if 'Queries' not in query:
            print("ERROR: Invalid query")
            return 'Invalid query'
        for entry in query['Queries']:
            query_target_var = -1
            for check in ['(Deprecated)Target', self.batch_target]:
                if check in entry:
                    for target in entry[check]:
                        query_target_var = target['Name']
                        break
                    if 'Condition' not in entry:
                        condition = None
                    else:
                        condition = entry['Condition']
                    ret.append(self.marg(target=query_target_var, condition=condition))

        if len(ret) == 1:
            ret = ret[0]

        return ret

            
        
        