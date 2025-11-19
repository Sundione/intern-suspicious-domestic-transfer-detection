from experiment.training_pipeline import TrainingPipeline


def suspicious_domestic_transfer_model_training(
    train_data_path: str,
    target: str = "is_fraud",
):
    training_pipeline = TrainingPipeline(
        target=target,
    )
    training_pipeline.training_pipeline(train_data_path)


suspicious_domestic_transfer_model_training("experiment/train_pipeline_test.csv")
