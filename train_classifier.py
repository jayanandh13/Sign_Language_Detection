import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
                            classification_report, 
                            confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    try:
        with open('./data.pickle', 'rb') as f:
            data_dict = pickle.load(f)
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
        return data, labels
    except FileNotFoundError:
        raise FileNotFoundError("data.pickle not found - run create_dataset.py first")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")

def train_model(data, labels):
    # Split data with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=0.2, 
        shuffle=True, 
        stratify=labels,
        random_state=42
    )
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(x_train, y_train)
    
    return model, x_test, y_test

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy*100:.2f}%')
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def save_model(model):
    try:
        with open('model.p', 'wb') as f:
            pickle.dump({'model': model}, f)
        print("Model saved successfully as model.p")
    except Exception as e:
        raise RuntimeError(f"Error saving model: {str(e)}")

def main():
    try:
        print("Loading dataset...")
        data, labels = load_data()
        
        print("Training model...")
        model, x_test, y_test = train_model(data, labels)
        
        print("Evaluating model...")
        evaluate_model(model, x_test, y_test)
        
        print("Saving model...")
        save_model(model)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()