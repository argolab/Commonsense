import torch
from torch import nn
import pandas as pd
import json
import tqdm


class Brute(nn.Module):
    def __init__(self, ):
        super(Brute, self).__init__()
        
        self.total_pos = []
        self.optim = None

        self.w0 = 0


        self.universe = None
        self.universe_fire = None
        self.constraints = []
        self.targets = []
        self.conditions = []
        self.constraints_values = []
        self.var_dict = dict()
        self.val_dict = dict()
        self.num_features = 0
        #self.abstract_universe = []


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
            self.json_vars(data['variables'])
            self.json_constraints(data['probabilities'])
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
            condition = self.translate(entry['Condition'])
            if 'Target' not in entry:
                if 'Batch Target' not in entry or 'Batch Probability' not in entry:
                    print("ERROR: Invalid constraint type 1", entry)
                    continue
                batch_target = self.batch_translate(entry['Batch Target'])
                
                for target, prob in zip(batch_target, entry['Batch Probability']):
                    if condition is None or target is None:
                        print("ERROR: invalid constraint type 2", entry)
                        continue
                    self.add_constraint(target, condition, prob)
                    #if entry['Batch Target'][0]['Name'] == 'Price Ranges':
                    #    print(target, condition, prob)
            else:
                target = self.translate(entry['Target'])
                if condition is None or target is None:
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
        #self.vars_print()

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
        if self.tot_fire is None:
            denom = self.build_denom(torch.zeros(self.N))
            matching = denom.unsqueeze(1) * self.mask
            self.tot_fire = (matching == self.features_rem).all(dim=-1).float()
        entropy_each = self.tot_fire * self.weights
        entropy_each = torch.exp(entropy_each.sum(dim=-1))
        bot = entropy_each.sum()
        px = (entropy_each / bot)
        return 1 * (px * torch.log(px)).sum()
        
    
    def calculate_potential(self, x):
        """
        For inference, calculate the sum of potential function"""
        denom = self.build_denom(x)
        matching = denom.unsqueeze(1) * self.mask
        fire = (matching == self.features_rem).all(dim=-1).float()
        entropy_each = fire * self.weights
        entropy_each = torch.exp(entropy_each.sum(dim=-1))
        return entropy_each.sum()
        
        
    def add_var(self, possible):
        """
        update the total possible values for variable"""
        self.total_pos.append(possible)

    
    def add_entire_constraints(self, var, base, constraints):
        """
        deprecated"""
        for i, constraint in enumerate(constraints):
            tmp = base.clone().detach()
            tmp[var] = i + 1
            self.add_constraint(tmp, base, constraint)

    def add_entire_ineq_constraints(self, var, base, constraints):
        """
        deprecated"""
        for i, constraint in enumerate(constraints):
            tmp = base.clone().detach()
            tmp[var] = i + 1
            self.add_constraint(tmp, base, constraint)
            #print("adding constraint: ", tmp, base, constraint[0], constraint[1])

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

        #print("ret: ", ret, " for ", inputs, " in ", self.universe, " gives ", ret.unsqueeze(-1) * self.universe)
        if inputs is None:
            print(ret)
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
        for input in inputs:
            var_ind = self.var_name2ind(input['Name'])
            if var_ind == -1:
                return None
            for val in input['Value']:
                possible_values = []
                val_ind = self.val_name2ind(var_ind, val)
                if val_ind == -1:
                    return None
                possible_values.append(val_ind)
                tmp = ret.copy()
                tmp[var_ind] = tuple(possible_values)
                rets.append(tmp)
        return rets



    def finish_build(self, ):

        self.N = len(self.total_pos)

        self.total_pos = torch.tensor(self.total_pos, dtype=torch.int32)
        self.universe = self.build_universe()

        feature_firing = []

        features = set()

        # tar and cond should have the same format, a list of total possible values
        # equality only
        for (tar, cond, prob) in self.constraints:
            condition = self.get_mask(cond)
            target = self.get_mask(tar, condition=condition)

            self.targets.append(target)
            self.conditions.append(condition)
            # check if condition is all true
            if tuple(target.tolist()) in features and tuple(condition.tolist()) in features:
                print("ERROR: Duplicate features?")
            if tuple(target.tolist()) not in features:
                features.add(tuple(target.tolist()))
                feature_firing.append(target)
            if tuple(condition.tolist()) not in features:
                features.add(tuple(condition.tolist()))
                feature_firing.append(condition)

            if len(prob) == 2:
                prob = [(prob[0] + prob[1])/2]
            self.constraints_values.append(prob)
        #print(features)
        self.targets = torch.stack(self.targets, dim=0)
        self.conditions = torch.stack(self.conditions, dim=0)
        self.features_firing = torch.stack(feature_firing, dim=0).t()
        self.constraints_values = torch.tensor(self.constraints_values).squeeze()
        
        self.num_features = self.features_firing.shape[1]
        self.weights = nn.Parameter(torch.randn(self.num_features))
        self.optim = torch.optim.SGD(self.parameters(), lr=0.1)


        print('-----------------------------------------------')
        print("Finished building")
        #self.vars_print()
        print("Features: ", self.num_features)
        print("Constraints: ", len(self.constraints))
        print('-----------------------------------------------')
        return features


    def update(self, epochs=100):

        """
        Training"""

        self.losses = []
        self.train()
        for n in tqdm.tqdm(range(epochs)):
            self.optim.zero_grad()

            firing = torch.exp((self.features_firing * self.weights).sum(dim=-1))

            tar_firing = (self.targets * firing)
            cond_firing = (self.conditions * firing)


            targets = tar_firing.sum(dim=-1)
            conditions = cond_firing.sum(dim=-1)

            loss = (targets / conditions - self.constraints_values).pow(2).sum()
            #print("weights: ", self.weights)
            if self.w0 != 0:
                loss += self.w0 * self.calculate_entropy()
            #loss += 0.1 * self.weights.pow(2).mean()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.detach())
            if n % (epochs // 5) == 0 or n == epochs - 1:
                print("Loss: ", loss, (targets / conditions - self.constraints_values).pow(2).mean())
                #print("Violated: ", violated, " out of ", tot)


    def infer_print(self, target=-1, conditions=None):
        cond = torch.zeros(self.N).float().fill_(-1)
        ret = []
        #if conditions:
        #    for condition in conditions:
        #        index = self.variable_lookup[condition]
        #        cond[index[0]] = index[1]
        print("Conditions: ", conditions)
        df = pd.DataFrame()
        tar = cond.clone().detach()
        tot = 0
        if target == -1:
            target = self.N - 1
        for i in range(self.total_pos[target]):
            tar[target] = i
            print(tar)
            inf = self.infer(tar, cond).item()
            df[i] = [inf]
            tot += inf
        if abs(tot-1) > 0.001:
            print("ERROR: Inference sum not 1")
        df = df.T
        print(df)
        return ret
    

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
            print("ERROR: Value not found")
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
        for key, values in condition.items():
            proc_cond.append({'Name': key, 'Value': values})

        proc_tar = -1
        if type(target) == str:
            proc_tar = self.var_dict[target]
        elif target == -1:
            proc_tar = self.N - 1

        cond = self.translate(proc_cond)

        ret = torch.zeros(self.total_pos[proc_tar])
        for i in range(self.total_pos[proc_tar]):
            tar_tmp = [{"Name": proc_tar, "Value": [i]}]
            tar = self.translate(tar_tmp)
            ret[i] = self.infer(tar, cond).item()
        return ret
        
        