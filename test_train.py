import joblib 
from train import train

def test_train():
    train()
    model=joblib.load("model/model.pkl")
    assert model is not None

if __name__=="__main__":
    test_train()
    print("Test Passed")