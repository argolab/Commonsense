import pandas as pd


class data():
    def __init__(self, path='airdat.csv', drop_zero=True, clean=['bedrooms', 'log_price'], cols=['log_price', 'bathrooms', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'review_scores_rating', 'city'], dataframe=None):
        if dataframe is None:
            dataframe = pd.read_csv(path)
        self.original_data = dataframe
        if 'price' not in self.original_data.columns:
            self.original_data['price'] = self.original_data['log_price'].apply(lambda x: 2.71828 ** x)
        print("All columns available: ", self.original_data.columns)
        print("Columns used: ", cols)

        df = self.original_data[cols].copy()

        df.dropna(inplace=True, subset=clean, )
        if 'price' not in df.columns:
            df['price'] = df['log_price'].apply(lambda x: 2.71828 ** x)

        if drop_zero:
            df = df[df['price'] > 1]

        self.dat = df

        price_bins = [-1, 50, 100, 200, 400, self.original_data['price'].max()+1]

        self.bins = dict()
        self.bins['price'] = price_bins

    def describe(self, item='price'):
        print(self.dat[item].describe())
    
    def get(self, ):
        return self.dat
    
    def print_og_col(self, ):
        print(self.original_data.columns)

    def print_col(self, ):
        print(self.dat.columns)

    def set_columns(self, add=None, cols=['log_price', 'bathrooms', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'amenities', 'review_scores_rating']):
        if add is None:
            self.dat = self.original_data[cols]
        else:
            self.dat = self.original_data[cols + add]

    def unique(self, item='price'):
        print(self.dat[item].unique())


    def set_bins(self, bins, name='price'):
        if name == 'price':
            price_bins = bins
            self.bins[name] = [-1] + price_bins + [self.original_data['price'].max()+1]
        else:
            self.bins[name] = bins


    def marg(self, tar='price', cond=None):

        if cond is None:
            pp = self.dat
        else:
            pp = cond(self.dat)

        tot = 0
        
        p = []
        if tar == 'price':
            for i in range(len(self.bins[tar]) - 1):
                percentage = ((pp[tar] < self.bins[tar][i+1]) & (pp[tar] >= self.bins[tar][i])).mean() * 100
                print(f"{int(self.bins[tar][i])} - {int(self.bins[tar][i+1])}: \t{percentage:.2f}%")
                #ret += f"{int(self.price_bins[i])} - {int(self.price_bins[i+1])}: \t{percentage:.2f}%\n"
                tot += percentage
                p.append(percentage / 100)
            if abs(tot - 100) > 0.01:
                print("[Error]marginal do not add to 1: ", tot)

        else:
            uniques = self.dat[tar].unique()
            for u in sorted(uniques): # suppose we'll always use sorted to align the values
                percentage = (pp[tar] == u).mean() * 100
                print(f"{u}: \t{percentage:.2f}%")
                #ret += f"{u}: \t{percentage:.2f}%\n"
                tot += percentage
                p.append(percentage / 100)
            if abs(tot - 100) > 0.01:
                print("[Error]marginal do not add to 1: ", tot)
        return p
        #return ret