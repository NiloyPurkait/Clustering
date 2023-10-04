import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_proj(data,c,target,name=None):
    # Plot the data points 'data' with colors based on their labels 'target'
    plt.scatter(data[:,0], data[:,1], label='data', c=target, cmap='viridis')

    # Plot the main PCA line using the provided PCA components 'c'
    plt.plot(np.linspace(-1,1), np.linspace(-1,1)*(c[1]/c[0]), color='black', linestyle='--', linewidth=1.5, label=name)
    
    # Loop over each data point in 'data' to plot the projection lines
    for i in range(len(data[:,0])-1):
        # Define the current data point 'w'
        w = data[i,:]

        # Calculate the projection of current data point on the PCA line
        cv = (np.dot(data[i,:], c)) / np.dot(c, np.transpose(c)) * c

        # Draw a line from the data point to its projection on the PCA line
        plt.plot([data[i,0], cv[0]], [data[i,1], cv[1]], 'r--', linewidth=1.5)

    # Draw the projection line for the last data point (to include in the legend)
    plt.plot([data[-1,0], cv[0]], [data[-1,1], cv[1]], 'r--', linewidth=1.5, label='projections')

    # Add a legend to the plot
    plt.legend()
    # Display the plot
    plt.show()


def preprocess_wish_dataset(root):
    summer_products_path = f"{root}/summer-products-with-rating-and-performance_2020-08.csv"
    unique_categories_path = f"{root}/unique-categories.csv"
    unique_categories_sort_path = f"{root}/unique-categories.sorted-by-count.csv"

    summer_products = pd.read_csv(summer_products_path)
    unique_categories = pd.read_csv(unique_categories_path)
    unique_categories_sort = pd.read_csv(unique_categories_sort_path)

    df = summer_products

    C = (df.dtypes == 'object')
    CategoricalVariables = list(C[C].index)
    Integer = (df.dtypes == 'int64') 
    Float   = (df.dtypes == 'float64') 
    NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)

    df[NumericVariables]=df[NumericVariables].fillna(0)
    df=df.drop('has_urgency_banner', axis=1) # 70 % NA's

    df[CategoricalVariables]=df[CategoricalVariables].fillna('Unknown')
    df=df.drop('urgency_text', axis=1) # 70 % NA's
    df=df.drop('merchant_profile_picture', axis=1) # 86 % NA's

    C = (df.dtypes == 'object')
    CategoricalVariables = list(C[C].index)
    Integer = (df.dtypes == 'int64') 
    Float   = (df.dtypes == 'float64') 
    NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)

    Size_map  = {'NaN':1, 'XXXS':2,'Size-XXXS':2,'SIZE XXXS':2,'XXS':3,'Size-XXS':3,'SIZE XXS':3,
                'XS':4,'Size-XS':4,'SIZE XS':4,'s':5,'S':5,'Size-S':5,'SIZE S':5,
                'M':6,'Size-M':6,'SIZE M':6,'32/L':7,'L.':7,'L':7,'SizeL':7,'SIZE L':7,
                'XL':8,'Size-XL':8,'SIZE XL':8,'XXL':9,'SizeXXL':9,'SIZE XXL':9,'2XL':9,
                'XXXL':10,'Size-XXXL':10,'SIZE XXXL':10,'3XL':10,'4XL':10,'5XL':10}

    df['product_variation_size_id'] = df['product_variation_size_id'].map(Size_map)
    df['product_variation_size_id']=df['product_variation_size_id'].fillna(1)
    OrdinalVariables = ['product_variation_size_id']

    Color_map  = {'NaN':'Unknown','Black':'black','black':'black','White':'white','white':'white','navyblue':'blue',
                'lightblue':'blue','blue':'blue','skyblue':'blue','darkblue':'blue','navy':'blue','winered':'red',
                'red':'red','rosered':'red','rose':'red','orange-red':'red','lightpink':'pink','pink':'pink',
                'armygreen':'green','green':'green','khaki':'green','lightgreen':'green','fluorescentgreen':'green',
                'gray':'grey','grey':'grey','brown':'brown','coffee':'brown','yellow':'yellow','purple':'purple',
                'orange':'orange','beige':'beige'}

    df['product_color'] = df['product_color'].map(Color_map)
    df['product_color']=df['product_color'].fillna('Unknown')

    NominalVariables = [x for x in CategoricalVariables if x not in OrdinalVariables]
    Lvl = df[NominalVariables].nunique()

    ToDrop=['title','title_orig','currency_buyer', 'theme', 'crawl_month', 'tags', 'merchant_title','merchant_name',
                'merchant_info_subtitle','merchant_id','product_url','product_picture','product_id']
    df = df.drop(ToDrop, axis = 1)
    FinalNominalVariables = [x for x in NominalVariables if x not in ToDrop]

    df_dummy = pd.get_dummies(df[FinalNominalVariables], columns=FinalNominalVariables)

    df_clean = df.drop(FinalNominalVariables, axis = 1)
    df_clean = pd.concat([df_clean, df_dummy], axis=1)

    NumericVariablesNoTarget = [x for x in NumericVariables if x not in ['units_sold']]

    print("The number of categorical variables: " + str(len(FinalNominalVariables)+len(OrdinalVariables)) +"; where 1 ordinal variable and 35 dummy variables")
    print("The number of numeric variables: " + str(len(NumericVariables)))
    return df_clean
