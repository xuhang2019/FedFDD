from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from client.fedfdd import FedFDDClient


class FedFDDServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedFDD",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedFDDClient(deepcopy(self.model), self.args, self.logger, self.device)


if __name__ == "__main__":
    server = FedFDDServer()
    server.run()
