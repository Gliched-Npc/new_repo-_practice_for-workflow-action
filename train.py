from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import joblib 

def train():
    data=pd.read_csv('data/Iris.csv')
    X=data.drop(columns=['Id','Species'])
    Y=data['Species']
    
    model=RandomForestClassifier()
    model.fit(X,Y)
    
    joblib.dump(model,"model/model.pkl")

if __name__=="__main__":
    train()
    