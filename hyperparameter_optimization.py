import argparse
import optuna
import subprocess
import os
from datetime import datetime, date
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

""" before running : check available GPU , set model_type, set task(in main.py default=), 
    adjust weights according to class per sample distribution  
    study can be run with multiple instances if study_name is set to existing name      """

model = 'clam_sb'
gpu = 0

def get_date_time():
    curr_date = "".join(date.today().strftime("%d/%m").split("/")) + date.today().strftime("%Y")[2:]
    curr_time = "".join(str(datetime.now()).split(" ")[1].split(".")[0].split(":"))
    return curr_date, curr_time


def parse_metrics(log_dir, weights=(0.8, 0.2)):
    best_weighted_mean = 0
    best_trial = None

    event_files = []
    for dirpath, _, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename.startswith('events.out'):
                event_files.append(os.path.join(dirpath, filename))

    for idx, event_file in enumerate(event_files):
        event_accumulator = EventAccumulator(event_file)
        event_accumulator.Reload()

        # Get the scalar values for each class
        class_0_acc = event_accumulator.Scalars('final/val_class_0_acc')
        class_1_acc = event_accumulator.Scalars('final/val_class_1_acc')

        if class_0_acc and class_1_acc:
            # Assuming you want to use the latest values
            latest_class_0_acc = max(scalar.value for scalar in class_0_acc)
            latest_class_1_acc = max(scalar.value for scalar in class_1_acc)

            # Compute the weighted mean
            weighted_mean = (weights[0] * latest_class_0_acc +
                             weights[1] * latest_class_1_acc)

            # Update the best weighted mean
            if weighted_mean > best_weighted_mean:
                best_weighted_mean = weighted_mean
                best_trial = idx

    return best_weighted_mean


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
    log_dir = f"./results/{exp_code}_s1"

    run_training_script(args)
    metric = parse_metrics(log_dir)

    if metric is None:
        return float('inf')  # Return a high value if metric extraction fails

    return metric


if __name__ == "__main__":
    curr_date, _ = get_date_time()
    # study = optuna.create_study(direction="maximize", storage="sqlite:///example.db",study_name=(model + "_max_weighted_acc_" + curr_date), load_if_exists=True)
    study = optuna.create_study(direction="maximize", storage="sqlite:///example.db",study_name='clam_sb_max_weighted_acc_060824', load_if_exists=True)
    study.optimize(objective, n_trials=100)

    # Print the best found parameters
    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Best hyperparameters: ", trial.params)