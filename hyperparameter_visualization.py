import optuna
import sqlite3
from optuna.importance import get_param_importances
import optuna.visualization as vis
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tensorflow as tf

def visualize_study(trial_id):
    study_name = get_study_name(trial_id) # 148,149,150
    study = optuna.load_study(study_name=study_name, storage="sqlite:///example.db")

    param_importances = get_param_importances(study)

    # Print the importance of each parameter
    for param, importance in param_importances.items():
        print(f"Hyperparameter: {param}, Importance: {importance}")


def get_study_name(trial_id):

    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute('SELECT study_id FROM trials WHERE trial_id = ?',  (trial_id,))
    study_id = cursor.fetchone()[0]
    cursor.execute('SELECT study_name FROM studies WHERE study_id = ?', (study_id,))
    study_name = cursor.fetchone()[0]
    conn.close()

    print(f"Study Name: {study_name}")
    return study_name

if __name__ == "__main__":
    trial_id = 2 #148,149,150
    visualize_study(trial_id)

