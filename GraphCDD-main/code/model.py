import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GraphMultisetTransformer

torch.backends.cudnn.enabled = False

class GCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(GCN, self).__init__()
        self.args = args
        self.gcn_cir1_f = GCNConv(self.args.fcir, self.args.fcir)
        self.gcn_cir2_f = GCNConv(self.args.fcir, self.args.fcir)

        self.gcn_drug1_f = GCNConv(self.args.fdrug, self.args.fdrug)
        self.gcn_drug2_f = GCNConv(self.args.fdrug, self.args.fdrug)
    
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        
        

        self.cnn_cir = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fcir, 1),
                               stride=1,
                               bias=True)
        
        self.cnn_drug = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdrug, 1),
                               stride=1,
                               bias=True)
        self.cnn_dis = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fdis, 1),
                               stride=1,
                               bias=True)

        self.gat_cir1_f = GATConv(self.args.fcir, self.args.fcir,heads=4,concat=False,edge_dim=1)
        self.gat_drug1_f = GATConv(self.args.fdrug, self.args.fdrug,heads=4,concat=False,edge_dim=1)
        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis,heads=4,concat=False,edge_dim=1)
      


    def forward(self, data):
        torch.manual_seed(1)
        x_cir = torch.randn(self.args.circRNA_number, self.args.fcir)
        x_drug = torch.randn(self.args.drug_number, self.args.fdrug)
        x_dis = torch.randn(self.args.disease_number, self.args.fdis)

 
        x_cir_f1 = torch.relu(self.gcn_cir1_f(x_cir.cuda(), data['circ']['edges'].cuda(), data['circ']['data_matrix'][data['circ']['edges'][0], data['circ']['edges'][1]].cuda()))
        x_cir_att= torch.relu(self.gat_cir1_f(x_cir_f1,data['circ']['edges'].cuda(),data['circ']['data_matrix'][data['circ']['edges'][0], data['circ']['edges'][1]].cuda()))
        x_cir_f2 = torch.relu(self.gcn_cir2_f(x_cir_att, data['circ']['edges'].cuda(), data['circ']['data_matrix'][data['circ']['edges'][0], data['circ']['edges'][1]].cuda()))
        
        x_drug_f1 = torch.relu(self.gcn_dis1_f(x_drug.cuda(), data['drug']['edges'].cuda(), data['drug']['data_matrix'][data['drug']['edges'][0], data['drug']['edges'][1]].cuda()))
        x_drug_att =torch.relu(self.gat_dis1_f(x_drug_f1, data['drug']['edges'].cuda(),data['drug']['data_matrix'][data['drug']['edges'][0], data['drug']['edges'][1]].cuda()))        
        x_drug_f2 = torch.relu(self.gcn_dis2_f(x_drug_att, data['drug']['edges'].cuda(), data['drug']['data_matrix'][data['drug']['edges'][0], data['drug']['edges'][1]].cuda()))
        
        
        x_dis_f1 = torch.relu(self.gcn_dis1_f(x_dis.cuda(), data['dis']['edges'].cuda(), data['dis']['data_matrix'][data['dis']['edges'][0], data['dis']['edges'][1]].cuda()))
        x_dis_att =torch.relu(self.gat_dis1_f(x_dis_f1, data['dis']['edges'].cuda(),data['dis']['data_matrix'][data['dis']['edges'][0], data['dis']['edges'][1]].cuda()))        
        x_dis_f2 = torch.relu(self.gcn_dis2_f(x_dis_att, data['dis']['edges'].cuda(), data['dis']['data_matrix'][data['dis']['edges'][0], data['dis']['edges'][1]].cuda()))
        
 
        X_cir = torch.cat((x_cir_f1, x_cir_f2), 1).t()
        X_cir = X_cir.view(1, self.args.gcn_layers, self.args.fcir, -1)

        X_drug = torch.cat((x_drug_f1, x_drug_f2), 1).t()
        X_drug = X_drug.view(1, self.args.gcn_layers, self.args.fdis, -1)

        X_dis = torch.cat((x_dis_f1, x_dis_f2), 1).t()
        X_dis = X_dis.view(1, self.args.gcn_layers, self.args.fdis, -1)
        


        cir_fea = self.cnn_cir(X_cir)
        cir_fea = cir_fea.view(self.args.out_channels, self.args.circRNA_number).t()
        
        drug_fea = self.cnn_dis(X_drug)
        drug_fea = drug_fea.view(self.args.out_channels, self.args.drug_number).t()
        
        dis_fea = self.cnn_dis(X_dis)
        dis_fea = dis_fea.view(self.args.out_channels, self.args.disease_number).t()
       
        return cir_fea.mm(dis_fea.t()),cir_fea.mm(drug_fea.t()),drug_fea.mm(dis_fea.t()),cir_fea,drug_fea,dis_fea









