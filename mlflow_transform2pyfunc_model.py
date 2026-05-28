# convert_old_pytorch_to_pyfunc.py
import os
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import torch
import pandas as pd


class D2A2ServingWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = mlflow.pytorch.load_model(
            context.artifacts["pytorch_model"]
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame):
        outputs = []

        for _, row in model_input.iterrows():
            rgb = torch.load(row["rgb_path"], map_location=self.device)
            depth = torch.load(row["depth_path"], map_location=self.device)
            mde = torch.load(row["mde_path"], map_location=self.device)

            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)
            if mde.dim() == 3:
                mde = mde.unsqueeze(0)

            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            mde = mde.to(self.device)

            with torch.no_grad():
                sr = self.model(rgb=rgb, depth=depth, MDE=mde)

            outputs.append(sr.detach().cpu().numpy().tolist())

        return outputs


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")

    old_run_id = "4a682a668e7643e7978540d626d7433e"
    old_model_uri = f"runs:/{old_run_id}/best_model"

    mlflow.set_experiment("d2a2_pyfunc")

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="best_model_pyfunc",
            python_model=D2A2ServingWrapper(),
            artifacts={
                "pytorch_model": old_model_uri,
            },
            code_path=["."],
        )

        print("NEW_RUN_ID =", run.info.run_id)
        print("MODEL_URI =", f"runs:/{run.info.run_id}/best_model_pyfunc")