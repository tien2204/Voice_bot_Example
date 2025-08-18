












import torch


class CosyVoiceModel:

    def __init__(
        self,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flow = flow
        self.hift = hift

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()
