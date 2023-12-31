from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.losses.losses import L2_loss
import torch


class AE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,drop_prob,batch_size,pca):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,drop_prob=drop_prob,batch_size=batch_size,pca=pca)
            
        def forward(self,x):
            z=self.encoder_base(x)
            return z
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, batch_size,data_shape,drop_prob,pca):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,batch_size=batch_size,drop_prob=drop_prob,pca=pca)

        def forward(self,x):
            return self.decoder_base(x)
    
    def __init__(self,data_shape,latent_dim,batch_size,pca,drop_prob,hidden_dim: int= 500,**kwargs):
        super().__init__()
        self.drop_prob=drop_prob
        self.pca=pca
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,drop_prob=self.drop_prob,batch_size=self.batch_size,pca=self.pca)
        self.decoder = self.Decoder(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,drop_prob=self.drop_prob,batch_size=batch_size,pca=self.pca)
        self.automatic_optimization=False
        self.train_losses=[]
        self.eval_losses=[]


    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = L2_loss(x_hat,x)
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.1)
        opt.step()
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.log("val_ae_loss", loss)
        self.eval_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        z=self.encoder(x)
        x_hat=self.decoder(z)
        x_hat=x_hat.reshape(x.shape)
        loss = torch.linalg.norm(x-x_hat)/torch.linalg.norm(x)
        self.log("test_ae_loss", loss)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.decoder_base.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        z=z.to(device)
        temp_interior=self.decoder(z)
        return temp_interior,z
     
