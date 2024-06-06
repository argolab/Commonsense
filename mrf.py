import torch
from torch import nn
import pandas as pd
import json
import tqdm


class Brute(nn.Module):
    def __init__(self, ):
        super(Brute, self).__init__()
        self.features = []
        self.feature_constraints = []
        self.total_pos = []
        self.features_cond = []
        self.optim = None
        self.constraints = []
        self.features_rem = []
        self.mask = []
        self.fire = []
        self.marginals = []

        self.w0 = 0

        self.tot_fire = None

        self.variable_lookup = dict()

        self.jvars = None

        self.dict = []


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
        self.finish_build()


    def json_constraints(self, constraints):
        """
        Adding constraints from a list of constraints"""
        if type(constraints) != list:
            print("ERROR: Constraints must be a list")
            return
    
        for conds, tars, prob in constraints:
            condition = torch.zeros(self.N).float()
            # check if all conditions and targets are in the variable lookup
            exist = True
            if conds:
                for cond in conds[1:]:
                    if cond not in self.variable_lookup.keys():
                        print("ERROR: Condition not found")
                        print(conds, tars, prob)
                        exist = False
                        continue
                    index = self.variable_lookup[cond]
                    condition[index[0]] = index[1]
            target = condition.clone().detach()
            for tar in tars[1:]:
                if tar not in self.variable_lookup.keys():
                    print("ERROR: Target not found: ", tar)
                    #print(conds, tars, prob)
                    exist = False
                    continue
                target_index = self.variable_lookup[tar]
                if type(target_index) == int:
                    print("ERROR: Target not found")
                    print(conds, tars, prob)
                target[target_index[0]] = target_index[1]
            if exist:
                self.add_constraint(target, condition, prob)


    def json_vars(self, json):
        """
        Add variables from a dictionary containing the variable names and their possible values"""
        copy_to_store = json.copy()
        for i, (key, value) in enumerate(json.items()):
            if isinstance(value, float):
                # remove this
                #copy_to_store.pop(key)
                continue
            self.add_var(len(value))
            self.variable_lookup[key] = i
            self.variable_lookup[i] = key
            tmp = []
            for j, val in enumerate(value):
                self.variable_lookup[val] = (i, float(j+1))
                self.variable_lookup[(i, j+1)] = val
                tmp.append(val)
            self.dict.append(tmp)
        
        self.jvars = copy_to_store
        self.N = len(self.total_pos)


    def vars_print(self):
        """
        Print the variables and their possible values"""
        if self.jvars is None:
            print("No variables")
            return
        for key, value in self.jvars.items():
            print(key, value)
        print(self.total_pos)


    def infer(self, target, condition):
        """
        Calculate the marginal probability of the target given the condition (optional)"""
        self.eval()
        with torch.no_grad():
            marginalized_potential = self.calculate_potential(target)
            normalization = self.calculate_potential(condition)
            return marginalized_potential/normalization

    
    def update(self, epochs=100):
        """
        Training"""
        self.losses = []
        self.train()
        for n in tqdm.tqdm(range(epochs)):
            loss = 0
            violated = 0
            tot = 0
            for i, (y, x) in enumerate(self.constraints):
                if (type(self.marginals[i]) == tuple or type(self.marginals[i]) == list) and (len(self.marginals[i]) > 1):
                    tot += 1
                    l = (max(0, (-torch.exp((y * self.weights).sum(dim=-1)).sum() / torch.exp((x * self.weights).sum(dim=-1)).sum()) + (self.marginals[i][0]))) ** 2
                    l += (max(0, (torch.exp((y * self.weights).sum(dim=-1)).sum() / torch.exp((x * self.weights).sum(dim=-1)).sum()) - (self.marginals[i][1]))) ** 2
                    if l > 0:
                        violated += 1
                else:
                    l = ((torch.exp((y * self.weights).sum(dim=-1)).sum() / torch.exp((x * self.weights).sum(dim=-1)).sum()) - (self.marginals[i][0])) ** 2
                loss += l
            if self.w0 != 0:
                loss += self.w0 * self.calculate_entropy()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.losses.append(loss.detach())
            if n % 100 == 0:
                print("Loss: ", loss)
                print("Violated: ", violated, " out of ", tot)


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
        self.features.append(y)
        self.features_cond.append(x)
        self.feature_constraints.append(constraints)



    def insert(self, tens):
        """
        Helper function"""
        for index, tensor in enumerate(self.features_rem):
            if tensor.eq(tens).all():
                return index
        self.features_rem.append(tens)
        self.mask.append(tens > 0)
        return len(self.features_rem) - 1



    def finish_build(self, ):
        """
        Finish building the model, call after adding variables and constraints"""
        self.total_pos = torch.tensor(self.total_pos, dtype=torch.int32)

        constraints_tmp = []
        for i in range(len(self.features)):
            y = self.insert(self.features[i])
            x = self.insert(self.features_cond[i])
            constraints_tmp.append((y, x, self.feature_constraints[i]))
        self.features_rem = torch.stack(self.features_rem)
        self.mask = torch.stack(self.mask).float().unsqueeze(0)
        self.weights = nn.Parameter(torch.zeros(len(self.features_rem)))


        for i, (feat) in enumerate(self.features_rem):
            denom = self.build_denom(feat)
            matching = denom.unsqueeze(1) * self.mask
            firing = (matching == self.features_rem).all(dim=-1).float()
            self.fire.append(firing)
    

        for i, (y, x, constraint) in enumerate(constraints_tmp):
            self.constraints.append([self.fire[y], self.fire[x]])
            self.marginals.append(constraint)
        
        #self.marginals = torch.tensor(self.marginals)

        self.num_features, _ = self.features_rem.shape
        self.optim = torch.optim.Adam(self.parameters(), lr=0.1)
        print('-----------------------------------------------')
        print("Finished building")
        self.vars_print()
        print("Features: ", len(self.features_rem))
        print("Constraints: ", len(self.constraints))
        print('-----------------------------------------------')



    def build_denom(self, input):
        """
        Enumerate all the possible combinations
        """
        mask = torch.Tensor(input) > 0
        ind = torch.nonzero(~mask)
        left = 1
        if len(ind):
            right = torch.prod(self.total_pos[ind]).item()
        else:
            right = 1
        tot = []
        for i, j in enumerate(list(input)):
            if j > 0:
                re = torch.Tensor([j]).unsqueeze(0).repeat(left*right, 1)
                tot.append(re)
            else:
                right //= self.total_pos[i].item()
                re = torch.arange(1, self.total_pos[i]+1).unsqueeze(0).repeat(left, 1).t().flatten().unsqueeze(0).t().repeat(right, 1)
                left *= self.total_pos[i]
                tot.append(re)
        return torch.cat(tot, 1)


    def infer_print(self, target=-1, conditions=None):
        cond = torch.zeros(self.N).float()
        ret = []
        if conditions:
            for condition in conditions:
                index = self.variable_lookup[condition]
                cond[index[0]] = index[1]
        print("Conditions: ", conditions)
        df = pd.DataFrame()
        tar = cond.clone().detach()
        if target == -1:
            target = self.N - 1
        for i in range(self.total_pos[target]):
            tar[target] = i + 1
            #print(self.variable_lookup[(target, float(i+1))], ": \t", self.infer(tar, cond))
            df[self.variable_lookup[(target, float(i+1))]] = [self.infer(tar, cond).item()]
            ret.append(self.infer(tar, cond).item())
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
        return len(self.features_rem)
    
    def var_name2ind(self, name):
        """
        Return the index of a variable given the name
        """
        return self.variable_lookup[name]
    

    def var_ind2name(self, i):
        """
        Return the name of a variable given the index
        NOTE: the value index is 1-based
        """
        return self.variable_lookup[i]
    
    def val_name2ind(self, var, val):
        """
        Return the index of a value given the variable name and the value name
        NOTE: the value index is 1-based
        """
        return int(self.variable_lookup[val][1])
    

    def val_ind2name(self, var, val):
        """
        Return the name of a value given the variable name and the value index
        """
        if type(var) == str:
            var = self.var_name2ind(var)
        return self.variable_lookup[(var, float(val))]
    

    def marg(self, target=-1, condition=None):
        """
        Calculate the marginal probability of the target given the condition (optional)
        """
        if type(target) == str:
            target = self.variable_lookup[target]
        elif target == -1:
            target = self.N - 1
        cond = torch.zeros(self.N).float()
        if condition:
            for key, values in condition.items():
                if type(key) == str:
                    key = self.variable_lookup[key]
                if len(values) != 1:
                    print("ERROR: Only one value allowed for condition")
                    return cond
                for val in values:
                    if type(val) == str:
                        val = self.variable_lookup[val][1]
                    cond[key] = val
        tar = cond.clone().detach()
        ret = torch.zeros(self.total_pos[target])
        for i in range(self.total_pos[target]):
            tar[target] = i + 1
            ret[i] = self.infer(tar, cond).item()
        return ret
        
        
    
