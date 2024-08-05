import argparse
import optuna
import subprocess
import os
from datetime import datetime, date
import pandas as pd

# TODO before running : check available GPU , set model_type, set task(in main.py default=)
model = 'clam_sb'
gpu = 0

def get_date_time():
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    return curr_date, curr_time

def parse_metrics(log_dir):
    df = pd.read_csv(log_dir)

    # Find the rows with the highest val_auc and val_acc
    best_val_auc_row = df.loc[df['val_auc'].idxmax()]
    best_val_auc = best_val_auc_row['val_auc']

    return best_val_auc


def run_training_script(args):
    """Run the training script with the specified arguments."""
    command = [
        "python", os.path.join(os.path.dirname(__file__), "main.py"),
        "--lr", str(args.lr),
        "--reg", str(args.reg),
        "--opt", args.opt,
        "--drop_out", str(args.drop_out),
        "--bag_loss", args.bag_loss,
        "--model_type", args.model_type,
        "--exp_code", args.exp_code,
        "--model_size", str(args.model_size),
        "--log_data",
        "--weighted_sample",
        "--early_stopping",

    ]

    # Additional arguments if model_type is 'mil'
    if args.model_type == 'mil':
        command.extend([
            "--bag_weight", str(args.bag_weight),
            "--B", str(args.B)
        ])
        if args.inst_loss:
            command.extend([
                "--inst_loss", args.inst_loss,
            ])
        if str(args.no_inst_cluster):
            command.extend([
                "--no_inst_cluster",
            ])

    # Run the script
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    print("Running: {}".format(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode != 0:
        print(f"Script failed with error: {result.stderr}")
    else:
        print(f"Script executed successfully: {result.stdout}")



def objective(trial):
    # Define the search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    reg = trial.suggest_float('reg', 1e-6, 1e-3, log=True)
    opt = trial.suggest_categorical('opt', ['adam', 'sgd'])
    drop_out = trial.suggest_float('drop_out', 0.0, 0.5)
    bag_loss = trial.suggest_categorical('bag_loss', ['svm', 'ce'])
    # model_type = trial.suggest_categorical('model_type', ['clam_sb', 'clam_mb', 'mil'])
    model_type = f'{model}'
    model_size = trial.suggest_categorical('model_size', ['small', 'big'])

    curr_date, curr_time = get_date_time()
    exp_code = (model_type + curr_date + "_" + curr_time)

    # Additional hyperparameters for 'mil' model_type
    no_inst_cluster = None
    inst_loss = None
    bag_weight = None
    B = None
    if model_type == 'mil':
        no_inst_cluster = trial.suggest_categorical('no_inst_cluster', [True, False])
        inst_loss = trial.suggest_categorical('inst_loss', ['svm', 'ce', None])
        bag_weight = trial.suggest_float('bag_weight', 0.5, 1.0)
        B = trial.suggest_int('B', 4, 12)

    # Setup args namespace
    args = argparse.Namespace(lr=lr, reg=reg, opt=opt, drop_out=drop_out, bag_loss=bag_loss,
                              model_type=model_type, model_size=model_size, no_inst_cluster=no_inst_cluster,
                              inst_loss=inst_loss, bag_weight=bag_weight, B=B, exp_code=exp_code)


    # Define the log directory for TensorBoard logs
    log_dir = f"./results/{exp_code}_s1/summary.csv"

    # Run the training script
    run_training_script(args)

    # Extract the metric from TensorBoard logs
    metric = parse_metrics(log_dir)  # Example metric

    # Handle cases where the metric couldn't be extracted
    if metric is None:
        return float('inf')  # Return a high value if metric extraction fails

    return metric


if __name__ == "__main__":
    curr_date, _ = get_date_time()
    study = optuna.create_study(direction="maximize", storage="sqlite:///example.db",study_name=(model + "_" + curr_date), load_if_exists=True)
    study.optimize(objective, n_trials=100)

    # Print the best found parameters
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Best hyperparameters: ", trial.params)