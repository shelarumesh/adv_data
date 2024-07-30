import pickle

# open pickle model 
with open('model training/EN_Model.pickle','rb') as file:
    model = pickle.load(file)


#predict on model
result = model.predict([[12,25,2]])
print(result)