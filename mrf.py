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
        return x
    
    def set_w0(self, w0):
        self.w0 = w0

    
    def from_json(self, entire_json=None, vars=None, constraints=None):
        """ if type(json) == str:
            data = pd.read_json(json)
        else:
            data = json """
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

    
    def from_json_old(self, json):
        if type(json) == str:
            data = pd.read_json(json)
        else:
            data = json
        self.json_vars(data['aspects'])
        self.json_constraints(data['conditionals'])
        self.json_marginals(data['marginals'])
        self.finish_build()

    def json_vars(self, json):
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
        if self.jvars is None:
            print("No variables")
            return
        for key, value in self.jvars.items():
            print(key, value)
        print(self.total_pos)



    def json_constraints_old(self, json):
        """ for cond_var_name, entry in json.items():
            if isinstance(entry, float):
                continue
            for condition, subentry in entry.items():
                index = self.variable_lookup[condition]
                cond = torch.zeros(self.N).float()
                cond[index[0]] = index[1]
                for tar_var_name, item in subentry.items():
                    for target, constraints in item.items():
                        tar = cond.clone().detach()
                        target_index = self.variable_lookup[target]
                        tar[target_index[0]] = target_index[1]
                        if len(constraints) == 2:
                            #print(tar, cond)
                            self.add_ineq_constraint(tar, cond, constraints[0], constraints[1]) """
        for cond_var_name, entry in json.items():
            if isinstance(entry, float):
                continue
            for condition, item in entry.items():
                index = self.variable_lookup[condition]
                cond = torch.zeros(self.N).float()
                cond[index[0]] = index[1]
                for target, constraints in item.items():
                    tar = cond.clone().detach()
                    target_index = self.variable_lookup[target]
                    tar[target_index[0]] = target_index[1]
                    print(target, condition, constraints)
                    if type(constraints) == list and len(constraints) == 2:
                        #print(tar, cond, constraints[0], constraints[1])
                        self.add_ineq_constraint(tar, cond, constraints[0], constraints[1])
                    elif type(constraints) == list and len(constraints) == 1:
                        #print(tar, cond, constraints)
                        self.add_constraint(tar, cond, constraints[0])
                    else:
                        #print(tar, cond, constraints)
                        self.add_constraint(tar, cond, constraints)

    def json_marginals(self, json):
        for target, item in json.items():
            #target_index = self.variable_lookup[target]
            cond = torch.zeros(self.N).float()
            #tar[target_index[0]] = target_index[1]
            if isinstance(item, float):
                continue
            for condition, constraints in item.items():
                index = self.variable_lookup[condition]
                tar = torch.zeros(self.N).float()
                tar[index[0]] = index[1]
                print("marginals:", condition, constraints)
                if type(constraints) == list and len(constraints) == 2:
                    self.add_ineq_constraint(tar, cond, constraints[0], constraints[1])
                    #print(tar, cond, constraints)
                else:
                    self.add_constraint(tar, cond, constraints)
                    #print(tar, cond, constraints)


    def infer(self, target, condition):
        self.eval()
        with torch.no_grad():
            marginalized_potential = self.calculate_potential(target)
            normalization = self.calculate_potential(condition)
            return marginalized_potential/normalization

    
    def update(self, epochs=100):
        self.losses = []
        self.train()
        for n in tqdm.tqdm(range(epochs)):
        #for n in range(epochs):
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
        denom = self.build_denom(x)
        matching = denom.unsqueeze(1) * self.mask
        fire = (matching == self.features_rem).all(dim=-1).float()
        entropy_each = fire * self.weights
        entropy_each = torch.exp(entropy_each.sum(dim=-1))
        return entropy_each.sum()
        
        
    def add_var(self, possible):
        self.total_pos.append(possible)

    
    def add_entire_constraints(self, var, base, constraints):
        for i, constraint in enumerate(constraints):
            tmp = base.clone().detach()
            tmp[var] = i + 1
            self.add_constraint(tmp, base, constraint)

    def add_entire_ineq_constraints(self, var, base, constraints):
        for i, constraint in enumerate(constraints):
            tmp = base.clone().detach()
            tmp[var] = i + 1
            self.add_constraint(tmp, base, constraint)
            #print("adding constraint: ", tmp, base, constraint[0], constraint[1])

    def add_constraint(self, y, x, constraints):
        self.features.append(y)
        self.features_cond.append(x)
        #if (type(constraints) == list or type(constraints) == tuple) and len(constraints) > 1:
        #    self.feature_constraints.append([(constraints[0]+constraints[1])/2])
        #else:
        self.feature_constraints.append(constraints)


    def add_ineq_constraint(self, y, x, left, right):
        self.features.append(y)
        self.features_cond.append(x)
        self.feature_constraints.append((left, right))

    def insert(self, tens):
        for index, tensor in enumerate(self.features_rem):
            if tensor.eq(tens).all():
                return index
        self.features_rem.append(tens)
        self.mask.append(tens > 0)
        return len(self.features_rem) - 1
    

    def finish_build(self, ):
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

    def dictionary(self):
        print(self.dict)

    def variable_names(self):
        p = [self.variable_lookup[i] for i in range(self.N)]
        print(p)


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




