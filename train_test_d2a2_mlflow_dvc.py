import mlflow

from dvc_training_utils import prepare_dvc_data
from mlflow_utils import MLFlowTrainer, Tester, load_state_dict_flexible, setup_seed
from option import args
from train_test_d2a2_mlflow import (
    build_criterion,
    build_model,
    build_optimizer_scheduler,
    build_test_loader,
    build_train_loader,
)


def main():
    setup_seed(20)
    prepare_dvc_data(args)
    mlflow.set_experiment(args.mlflow_experiment)

    model = build_model()
    train_loader = build_train_loader()
    test_loader = build_test_loader()
    optimizer, scheduler = build_optimizer_scheduler(model)
    criterion = build_criterion()

    run_kwargs = {"run_name": args.mlflow_run_name or "dvc_train_then_test"}
    mlflow.log_param("dvc_data_path", args.dvc_data_path if hasattr(args, "dvc_data_path") else "")
    mlflow.log_param("seed", 20)
    mlflow.log_artifact("option.py")
    with mlflow.start_run(**run_kwargs) as run:
        mlflow.set_tag("pipeline", "dvc_train_then_test_best_model")
        mlflow.set_tag("run_id", run.info.run_id)

        trainer = MLFlowTrainer(
            args, model, optimizer, scheduler, criterion, train_loader, test_loader
        )
        train_summary = trainer.train()

        best_parameter_path = train_summary["best_parameter_path"]
        if best_parameter_path is None:
            best_parameter_path = train_summary["last_parameter_path"]
        if best_parameter_path is None:
            raise RuntimeError("No checkpoint was produced by training.")

        load_state_dict_flexible(model, best_parameter_path)
        mlflow.log_param("tested_checkpoint_path", best_parameter_path)

        tester = Tester(args, model, test_loader)
        test_rmse = tester.validate(
            mlflow_prefix="best_model_test",
            mlflow_step=train_summary["best_epoch"],
            log_mlflow=True,
            mlflow_artifact_path="best_model_test",
        )
        mlflow.log_metric("best_model_test_rmse_for_gate", float(test_rmse.mean()))

        print("MLflow run_id: %s" % run.info.run_id)
        print("best checkpoint: %s" % best_parameter_path)
        print("best_model_test_mean_rmse: %.10f" % float(test_rmse.mean()))


if __name__ == "__main__":
    main()
