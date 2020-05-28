import pandas as pd
import tkinter as tk
#import tkinter.ttk as tk
import tkinter.filedialog


def close():
    root.destroy()
    root.quit()

def load_screen1():
    dataset = tkinter.filedialog.askopenfilename()
    global train_full
    train_full = pd.read_csv(dataset)
    #print(train_full.shape)
    btn1.destroy()
    global msg1, msg2, button1, v, fealist
    msg1 = tk.Label(root, text="Shape : \n"+str(train_full.shape))
    msg2 = tk.Label(root, text="    Summary : \n"+str(train_full.describe()))
    button1 = tk.Button(root, text="Proceed to Preprocess > ", command=prep_screen2)
    msg1.pack()
    msg2.pack()
    button1.pack(pady = 20)
    top = tk.Toplevel()
    top.minsize(300, 200)
    top.title("Set Output")
    v=tk.StringVar()
    tk.Label(top,text="Choose Output Feature:",justify = tk.LEFT,padx = 20).pack()
    for feat in train_full.columns:
        tk.Radiobutton(top,text=feat,padx = 20, variable=v,value=feat).pack(anchor=tk.W)
    bu1 = tk.Button(top, text="Select", command=top.destroy)
    bu1.pack()

#####--------Input feature Selection
    # top = tk.Toplevel()
    # top.minsize(300, 200)
    # top.title("Set Inputs")
    # tk.Label(top,text="Choose Input Features \n Warning: Don't select the output feature:",justify = tk.LEFT,padx = 20).pack()
    # #feat_dict = {new_list: [] for new_list in train_full.columns}
    # cols=train_full.columns
    # fealist=[0 for i in range(len(cols))]

    # print(len(cols))
    # for i in range(len(train_full.columns)):
    #     print(i)
    #     tk.Checkbutton(top,text=cols[i],padx = 20, variable=fealist[i]).pack(anchor=tk.W)
    # bu1 = tk.Button(top, text="Select", command=top.destroy)
    # bu1.pack()

def prep_screen2():
    msg1.destroy()
    msg2.destroy()
    button1.destroy()
    global X,y
    # print(v.get()+"------V.get------")
    # print(fealist)
    X=train_full.drop(v.get(), axis=1)
    y=train_full[v.get()]


    global w,b1,b2,cols_with_missing
    cols_with_missing = [col for col in train_full.columns if train_full[col].isnull().any()]
    if len(cols_with_missing):
        w = tk.Label(root,text=" Found missing values, How to Proceed? \n")
        b1 = tk.Button(root, text="Drop Missing Data", command=lambda: imp_screen3(1))
        b2 = tk.Button(root, text="Simple Imputation", command=lambda: imp_screen3(2))
        w.pack()
        b1.pack(pady = 20)
        b2.pack(pady = 20)
    else:
        w = tk.Label(root, text="No Missing Values Found!\n")
        b1 = tk.Button(root, text = "Proceed > ", command=lambda :imp_screen3(0))
        w.pack(padx = 10, pady = 20)
        b1.pack(pady = 20)

def imp_screen3(inp):
    from sklearn.model_selection import train_test_split
    w.destroy()
    b1.destroy()
    global X_train, X_valid, y_train, y_valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=3)
    if inp == 1:
        b2.destroy()
        X_train = X_train.drop(cols_with_missing, axis=1)
        X_valid = X_valid.drop(cols_with_missing, axis=1)
    if inp == 2:
        b2.destroy()
        from sklearn.impute import SimpleImputer
        my_imputer = SimpleImputer()
        X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
        X_valid = pd.DataFrame(my_imputer.transform(X_valid))
        X_train.columns = X_train.columns
        X_valid.columns = X_valid.columns

    global c,c1,c2,c3,object_cols
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)
    if len(object_cols) == 0:
        c = tk.Label(root,text=" No Categorical Values Found!\n")
        c1 = tk.Button(root, text="Continue > ", command=lambda: modsel_screen4(0))
        c.pack()
        c1.pack(pady = 20)
    else:
        c = tk.Label(root,text=str(len(object_cols))+"  Categorical Values Found!\n What to do?\n Hint: Use One-Hot Encoder with less than 10 Categorical Features and less than 5 cardianlity \n")
        #low_cardinality_cols = [col for col in object_cols if X_test[col].nunique() < 10]
        #high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
        c1 = tk.Button(root, text="Drop Missing Data", command=lambda: modsel_screen4(1))
        c2 = tk.Button(root, text="Label Encoding", command=lambda: modsel_screen4(2))
        c3 = tk.Button(root, text = "One-Hot Encoding", command=lambda :modsel_screen4(3))

        c.pack()
        c1.pack(pady = 20)
        c2.pack(pady = 20)
        c3.pack(pady = 20)



def modsel_screen4(enc):
    c.destroy()
    c1.destroy()
    global X_train, X_valid
    if enc == 1:
        c2.destroy()
        c3.destroy()
        X_train = X_train.drop(object_cols, axis=1)
        X_valid = X_valid.drop(object_cols, axis=1)
    if enc == 2:
        c2.destroy()
        c3.destroy()
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        for col in object_cols:
            X_train[col] = label_encoder.fit_transform(X_train[col])
            X_valid[col] = label_encoder.transform(X_valid[col])
    if enc == 3:
        c2.destroy()
        c3.destroy()
        from sklearn.preprocessing import OneHotEncoder
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index
        num_X_train = X_train.drop(object_cols, axis=1)
        num_X_valid = X_valid.drop(object_cols, axis=1)
        X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    global d,d1,m

    models = ["Linear Regression", "XGBoost","Decision Tree Regression", "Random Forest Regression"]

    d = tk.Label(root, text="Select Model\n")
    d1 = tk.Button(root, text="Train > ", command=res_screen5)
    d.pack()
    d1.pack(pady = 20)

    t = tk.Toplevel()
    t.minsize(300, 200)
    t.title("Set Input and Ouput")
    m=tk.StringVar()
    tk.Label(t,text="Choose Model for training:",justify = tk.LEFT,padx = 20).pack()
    for model in models:
        tk.Radiobutton(t,text=model,padx = 20, variable=m,value=model).pack(anchor=tk.W)
    bu1 = tk.Button(t, text="Select", command=t.destroy)
    bu1.pack()

def res_screen5():
    d.destroy()
    d1.destroy()

    from sklearn.metrics import mean_absolute_error
    #print(m.get())
    if m.get() == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(random_state = 0)
        model.fit(X_train, y_train)
    if m.get() == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=800, learning_rate=0.05,random_state = 0)
        model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_valid, y_valid)],verbose=False)
    if m.get() == "Decision Tree Regression":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state = 0)
        model.fit(X_train, y_train)
    if m.get() ==  "Random Forest Regression":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators = 500, random_state = 0)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_pred, y_valid)
    #print(m.get()+" gives Mean Absolute Error: " + str(mae))

    f = tk.Label(root, text="Results : \n")
    f1 = tk.Label(root, text=m.get()+" gives Mean Absolute Error: \n\n"+ str(mae))
    f2 = tk.Label(root,text="To Compare, Mean of "+v.get()+" is "+str(train_full[v.get()].mean()))
    fa = tk.Label(root, text="That is an error of "+ str(mae/train_full[v.get()].mean()*100)+"%")
    f3 = tk.Label(root,text = train_full.describe())
    f.pack(pady = 10)
    f1.pack(pady = 10)
    f2.pack(pady = 10, padx = 20)
    fa.pack(pady = 10)
    f3.pack(pady = 10)
#########################################################################################

root = tk.Tk()
root.minsize(600, 400)
root.title("Easy_ML")
btn1 = tk.Button(root, text="Select dataset csv file", command=load_screen1)
btn2 = tk.Button(root, text="Quit", command=close)

# btn1.place(in_= b2, relx = 0.5, rely = 0.5, anchor = CENTER)
# btn1.grid(row = 0, column = 0, padx = 10, pady = 10)
btn1.pack(padx = 20, pady = 80)
btn2.place(in_= root, width=60, height=40, relx = 0.9, rely = 0.9, anchor = "se")
root.mainloop()