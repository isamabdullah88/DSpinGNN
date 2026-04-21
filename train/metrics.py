import torch
import logging

class MetricsTracker:
    def __init__(self, device):
        self.device = device
        self.reset()
        self.logger = logging.getLogger(__name__)

    def reset(self):
        # Keep everything on the GPU as zero-tensors
        self.loss = torch.tensor(0.0, device=self.device)
        self.losse = torch.tensor(0.0, device=self.device)
        self.lossf = torch.tensor(0.0, device=self.device)
        self.lossx = torch.tensor(0.0, device=self.device)
        
        self.maee = torch.tensor(0.0, device=self.device)
        self.maef = torch.tensor(0.0, device=self.device)
        self.maex = torch.tensor(0.0, device=self.device)
        self.maex1 = torch.tensor(0.0, device=self.device)
        self.maex2 = torch.tensor(0.0, device=self.device)
        self.maexmini = torch.tensor(0.0, device=self.device)
        self.maexrest = torch.tensor(0.0, device=self.device)

        # Counters
        self.graphs, self.atoms, self.edges, self.edges1, self.edges2 = 0, 0, 0, 0, 0
        self.edgesmini, self.edgesrest = 0, 0

    def update_loss(self, loss, losse, lossf, lossx, num_graphs):
        self.graphs += num_graphs
        self.loss += loss.detach() * num_graphs
        self.losse += losse.detach() * num_graphs
        self.lossf += lossf.detach() * num_graphs
        self.lossx += lossx.detach() * num_graphs

    def update_mae(self, energy, forces, exchange, batch):
        self.atoms += batch.pos.shape[0]
        self.edges += batch.cr_edge_index.shape[1]

        # Global MAEs
        self.maee += torch.abs(energy.detach().view(-1) - batch.y_energy.view(-1)).sum()
        self.maef += torch.abs(forces.detach() - batch.y_forces).sum()
        self.maex += torch.abs(exchange.detach().view(-1) - batch.y_exchange.view(-1)).sum()

        # Ranged MAEs
        j1mask = batch.cr_edge_dist < 4.5
        j2mask = batch.cr_edge_dist >= 4.5
        
        self.edges1 += j1mask.sum().item()
        self.edges2 += j2mask.sum().item()
        
        self.maex1 += torch.abs(exchange[j1mask].detach().view(-1) - batch.y_exchange[j1mask].view(-1)).sum()
        self.maex2 += torch.abs(exchange[j2mask].detach().view(-1) - batch.y_exchange[j2mask].view(-1)).sum()

        # self.logger.info(f"Y-exchange shape: {batch.y_exchange.shape}")
        # jmask = ((batch.y_exchange.view(-1, 6) > -3.0) & (batch.y_exchange.view(-1, 6) < 3.0)).all(dim=1)
        # emask = jmask.unsqueeze(1).expand(-1, 6).reshape(-1)
        # self.maexmini += torch.abs(exchange[emask].detach().view(-1) - batch.y_exchange[emask].view(-1)).sum()
        # self.edgesmini += emask.sum().item()

        # rmask = ~emask
        # self.maexrest = torch.abs(exchange[rmask].detach().view(-1) - batch.y_exchange[rmask].view(-1)).sum()
        # self.edgesrest += rmask.sum().item()


    def get_averages(self):
        # Calculates averages and calls .item() EXACTLY ONCE at the end!
        return {
            "loss": (self.loss / self.graphs).item() if self.graphs > 0 else 0,
            "losse": (self.losse / self.graphs).item() if self.graphs > 0 else 0,
            "lossf": (self.lossf / self.graphs).item() if self.graphs > 0 else 0,
            "lossx": (self.lossx / self.graphs).item() if self.graphs > 0 else 0,
            "maee": (self.maee / self.atoms).item() if self.atoms > 0 else 0,
            "maef": (self.maef / (self.atoms * 3)).item() if self.atoms > 0 else 0,
            "maex": (self.maex / self.edges).item() if self.edges > 0 else 0,
            "maex1": (self.maex1 / self.edges1).item() if self.edges1 > 0 else 0,
            "maex2": (self.maex2 / self.edges2).item() if self.edges2 > 0 else 0,
            # "maexmini": (self.maexmini / (self.edgesmini)).item() if self.edgesmini > 0 else 0,
            # "maexrest": (self.maexrest / (self.edgesrest)).item() if self.edgesrest > 0 else 0,
        }