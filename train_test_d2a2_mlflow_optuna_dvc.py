import mlflow
import optuna
import torch

from dvc_training_utils import prepare_dvc_data
from mlflow_utils import log_dvc_metadata, setup_seed
from option import args
from train_test_d2a2_mlflow_optuna import objective


def main():
    setup_seed(20)
    prepare_dvc_data(args)
    # n_trials = getattr(args, "n_trials", 10)
    mlflow.set_experiment(args.mlflow_experiment)

    parent_run_name = args.mlflow_run_name or "dvc_optuna_hpo"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("pipeline", "dvc_optuna_hpo_parent")
        mlflow.set_tag("parent_run_id", parent_run.info.run_id)
        log_dvc_metadata(args)

        study = optuna.create_study(
            direction="minimize",
            study_name="d2a2_dvc_hpo",
        )

        study.optimize(
            objective,
            n_trials=getattr(args, "n_trials", 10),
        )

        best_trial = study.best_trial

        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.log_metric("best_objective_rmse", float(study.best_value))

        for key, value in study.best_params.items():
            mlflow.log_param("best_%s" % key, value)

        torch.cuda.empty_cache()

        print("MLflow parent run_id: %s" % parent_run.info.run_id)
        print("best trial:", best_trial.number)
        print("best params:", study.best_params)
        print("best rmse: %.10f" % float(study.best_value))


if __name__ == "__main__":
    main()
