from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.seer.models.seer_model import SEERModel

# model hyper params
MODEL_HYPER_PARAMS = {
    "e_layers": 1,
    "d_model": 512,
    "d_score": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "n_heads": 8,
    "factor": 3,
    "patch_len": 16,
    "stride": 8,
    "output_attention": 0,
    "dropout": 0.1,
    "activation": "gelu",
    "head_dropout": 0.1,
    "batch_size": 32,
    "lradj": "type3",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "patience": 5,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "parallel_strategy": "DP",
    "loss": "MAE",
    "scaling": 0.5,
    "score_threshold": 0.2,
}


class SEER(DeepForecastingModelBase):
    """
    seer adapter class.

    This class adapts the SEERModel to the DeepForecastingModelBase framework.
    It handles model initialization and the processing logic.
    """

    def __init__(self, **kwargs):
        """
        Initializes the seer adapter.

        It passes the model-specific hyperparameters and any user-provided overrides
        to the base class constructor.
        """
        super(SEER, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "seer"

    def _init_model(self):
        """
        Initializes the underlying SEERModel instance using the unified configuration.
        """
        return SEERModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        """
        Executes the model's forward pass and formats the output for the training loop.

        Args:
            input (torch.Tensor): The input data for the model.
            target (torch.Tensor): The target data (not used by this model's forward pass).
            input_mark (torch.Tensor): The time features for the input.
            target_mark (torch.Tensor): The time features for the target.

        Returns:
            dict: A dictionary containing the model's raw output and any additional loss.
                  - 'output': The forecasting result from the model.
                  - 'additional_loss': The importance loss from the model, to be added to the main loss.
        """
        # SEERModel's forward pass requires 'input' and 'input_mark'.
        output, loss_importance = self.model(input, input_mark)

        # The training loop in the base class expects a dictionary.
        # It will use 'output' to calculate the primary loss (e.g., MSE, MAE)
        # and add 'additional_loss' to it.
        # We call .mean() on the additional loss to ensure it's a scalar,
        # consistent with the original implementation's logic.
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss