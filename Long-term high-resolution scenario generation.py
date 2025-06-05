
import time
import argparse
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()

parser.add_argument("--rate_value1", type=int, default=1000, help="Maximum turbine hub wind speed m/s; maximum surface radiation J/m2")
parser.add_argument("--rate_value2", type=int, default=0, help="Maximum turbine hub wind speed m/s; maximum surface radiation J/m2")
parser.add_argument("--VAE_epoch", type=int, default=1000, help="number of epochs of model training")
parser.add_argument("--DF_epoch", type=int, default=1000, help="number of epochs of model training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=12, help="Latent space dimension")
parser.add_argument("--sample_step", type=int, default=24, help="define the sampling step")
parser.add_argument("--input_step", type=int, default=24, help="define the input temporal step")
parser.add_argument("--target_step", type=int, default=24, help="define the target temporal step")
parser.add_argument("--scenario_number", type=int, default=100, help="define the scenario number")
parser.add_argument("--train_path", default='ERA.xlsx', help="train data path and data file name")
parser.add_argument("--test_path", default='SSP126.xlsx', help="test data path and data file name")
parser.add_argument("--seed", default=0, type=int)

args = parser.parse_args()
print(args)

# Set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

#gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("train_type:",device.type)

pwd = os.getcwd()
grader_father=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")

"""-----------------------------------------Prepare Test Train Data--------------------------------------------------"""

def Train_Data(rate_value1,rate_value2,step,input_time_step,target_time_step,data_file_path):

    raw_data = pd.read_excel(data_file_path,'sheet1')
    data = raw_data[['ssrd(W)']].values
    data = (data-rate_value2) / (rate_value1-rate_value2)

    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0            # delete data if data = nan

    Len_data = data.shape[0]
    num = (( Len_data - input_time_step - target_time_step) // step) + 1  # the finally data cant use to train ,lack the target data ,so lack 1 num
    print("sample-number:",num)

    # get input data (Previous day's data)
    node = np.arange(0, input_time_step, 1)
    x = node
    for i in range(1, num):
        x = np.append(x, node + i * step, axis=0)
    Input_data = data[x]
    Input_data = torch.from_numpy(Input_data).float()
    Input_data = Input_data.view(num, input_time_step, 1)
    Input_data = Input_data.clamp(min=1e-5,max=1-(1e-5)) # for numerical stability

    # get target data (Target day data)
    target_node = np.arange(input_time_step, input_time_step + target_time_step, 1)
    y = target_node
    for i in range(1, num):
        y = np.append(y, target_node + i * step, axis=0)
    Target_data = data[y]
    Target_data = torch.from_numpy(Target_data).float()
    Target_data = Target_data.view(num, target_time_step, 1)
    Target_data = Target_data.clamp(min=1e-5,max=1-(1e-5)) # for numerical stability

    # get Climate input data (Target day average data)
    Input_Climate_data = data[y]
    Input_Climate_data = Input_Climate_data.reshape((num, 24), order='C')
    Input_Climate_data = np.mean(Input_Climate_data,axis=1)
    Input_Climate_data = torch.from_numpy(Input_Climate_data).float()
    Input_Climate_data = Input_Climate_data.view(num, 1, 1)

    # reshape size
    Input_data = Input_data.permute(0, 2, 1) # (N,1,24)
    Target_data = Target_data.permute(0, 2, 1) # (N,1,24)
    Input_Climate_data = Input_Climate_data.permute(0, 2, 1)  # (N,1,1)

    return  Input_data,Input_Climate_data,Target_data

def Test_Data(rate_value1,rate_value2,step,climate_time_step,data_file_path):

    raw_data = pd.read_excel(data_file_path,'sheet1')
    data = raw_data[['rsds']].values
    data = (data-rate_value2) / (rate_value1-rate_value2)

    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0            # delete data if data = nan

    Len_data = data.shape[0]
    num = (( Len_data - climate_time_step) // step) + 1
    print("sample-number:",num)

    # get climate condition data
    node = np.arange(0, climate_time_step, 1)
    x = node
    for i in range(1, num):
        x = np.append(x, node + i * step, axis=0)
    Input_Climate_data = data[x]
    # Input_Climate_data = Input_Climate_data / rate_value
    Input_Climate_data = Input_Climate_data.reshape((num, climate_time_step), order='C')
    Input_Climate_data = np.mean(Input_Climate_data,axis=1,keepdims=True)
    Input_Climate_data = torch.from_numpy(Input_Climate_data).float().unsqueeze(dim=1)
    Input_Climate_data = Input_Climate_data.clamp(min=1e-5,max=1-(1e-5)) # for numerical stability

    return  Input_Climate_data

def dataload_data(target_data, BATCH_SIZE):
    train_loader = Data.DataLoader(
        dataset=target_data,  # torch Tensor Dataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
    )
    return train_loader

def latent_dataload_data(input_data1,input_data2, target_data1, target_data2, BATCH_SIZE):
    torch_dataset = Data.TensorDataset(input_data1,input_data2, target_data1, target_data2)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch Tensor Dataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
    )
    return train_loader

"""--------------------------------------------------Define VAE------------------------------------------------"""

class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append((nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))))
        self.encoder.append((nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))))
        self.encoder.append((nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))))

        self.miu = nn.Sequential(nn.Linear(64*args.latent_dim, args.latent_dim))
        self.sigma = nn.Sequential(nn.Linear(64*args.latent_dim, args.latent_dim))

    def forward(self, x):

        x = self.encoder[0](x)
        per3 = x
        x = self.encoder[1](x)
        per2 = x
        x = self.encoder[2](x)
        per1 = x

        x = x.view(x.size(0), -1)
        miu = self.miu(x)
        sigma = self.sigma(x)

        perceptual = [per1,per2,per3]

        return miu, sigma, perceptual

class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__()

        self.input = nn.Sequential(nn.Linear(args.latent_dim, args.latent_dim*64))

        self.decoder = nn.ModuleList()
        self.decoder.append((nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))))
        self.decoder.append((nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))))
        self.decoder.append((nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Tanh())))

    def forward(self, noise):

        x = torch.reshape(self.input(noise), (noise.size(0), 64, args.latent_dim))
        per1 = x
        x = self.decoder[0](x)
        per2 = x
        x = self.decoder[1](x)
        per3 = x
        output = self.decoder[2](x)

        perceptual = [per1,per2,per3]

        return  (1+output)/2, perceptual

def VAE_train():

    VE = VAE_Encoder().to(device)
    print(VE)
    print("Total number of paramerters in VE networks is {} ".format(sum(x.numel() for x in VE.parameters())))

    VD = VAE_Decoder().to(device)
    print(VD)
    print("Total number of paramerters in VD networks is {} ".format(sum(x.numel() for x in VD.parameters())))

    opt_VE = torch.optim.Adam(VE.parameters(), lr=args.lr)
    opt_DE = torch.optim.Adam(VD.parameters(), lr=args.lr)

    MAE = torch.nn.L1Loss(reduction='mean')

    for epoch in range(args.VAE_epoch):
        total_loss = 0

        for step, scenario in enumerate(train_loader):
            current_batch_size = scenario.shape[0]

            eps = torch.randn(scenario.size(0),args.latent_dim).to(device)

            scenario = scenario.to(device)

            miu, sigma, real_per = VE(scenario)
            z = miu + eps * torch.exp(sigma)
            Fake, real_per = VD(z)

            perceptual_loss = MAE(real_per[0],real_per[0]) + MAE(real_per[1],real_per[1]) + MAE(real_per[2],real_per[2])

            lamda = 0.02
            MAE_loss = MAE(Fake, scenario)
            KL_loss = -0.5 * torch.mean(1 + sigma - miu.pow(2) - sigma.exp())

            loss = MAE_loss + perceptual_loss + lamda * KL_loss

            total_loss += loss.item() * current_batch_size

            opt_VE.zero_grad()
            opt_DE.zero_grad()
            loss.backward()
            opt_VE.step()
            opt_DE.step()

        total_loss = total_loss*args.batch_size/len(train_loader.dataset)

        print('epoch:', epoch, 'loss:', total_loss)

    data_file_name = 'VE{ep}.pkl'.format(ep=args.VAE_epoch)
    torch.save(VE.state_dict(), data_file_name)  # save model parameters

    data_file_name = 'VD{ep}.pkl'.format(ep=args.VAE_epoch)
    torch.save(VD.state_dict(), data_file_name)  # save model parameters

def VAE_test(Train_target):

    VD = VAE_Decoder()
    VD.load_state_dict(torch.load('VD{ep}.pkl'.format(ep=args.VAE_epoch), map_location='cpu'))

    VE = VAE_Encoder()
    VE.load_state_dict(torch.load('VE{ep}.pkl'.format(ep=args.VAE_epoch), map_location='cpu'))

    time = np.arange(1, args.sample_step + 1, 1)
    time = np.expand_dims(time, axis=0)  # define time point

    # norm data encode
    Scenario_encode_miu,Scenario_encode_sigma,_ = VE(Train_target)

    test_number = 3320
    scenario_number = 30
    test_lamda = 1
    eps = torch.randn(scenario_number,args.latent_dim)
    Scenario_encode = Scenario_encode_miu[test_number, :].repeat(scenario_number,1) \
                      + test_lamda * eps* torch.exp(Scenario_encode_sigma[test_number, :]).repeat(scenario_number,1)
    Re_scenario,_ = VD(Scenario_encode)
    Re_scenario = Re_scenario.detach().numpy()

    plt.title("VAE reconstruct")
    for i in range(scenario_number):
        plt.plot(time[0, :], Re_scenario[i,0,:], linewidth=0.5, color='black')
    plt.plot(time[0, :], Train_target[test_number,0, :].numpy(), linewidth=2.0, color='blue')
    plt.show()

    # random generate
    test_noise = torch.randn(args.scenario_number, args.latent_dim)
    scenario,_ = VD(test_noise)
    scenario = scenario.detach().numpy()

    plt.title("VAE random generate")
    for step in range(args.scenario_number):
        plt.plot(time[0, :], scenario[step,0, :], linewidth=0.5)
    plt.show()

    K_S(np.squeeze(scenario, axis=1), np.squeeze(Train_target.numpy(), axis=1))

    return Scenario_encode_miu.detach(),Scenario_encode_sigma.detach()

def K_S(scenario,data):
    # scenario size: (scenario number:N, target_step:J)
    # history data size: (scenario number:M, target_step:J)
    # data size belong [0,1]

    # CDF plot - hour
    hour_scenario = scenario.flatten()
    hour_data = data.flatten()
    plt.hist(hour_scenario,bins=100,range=(0,1),cumulative=True,histtype='step',density=True)
    plt.hist(hour_data,bins=100,range=(0,1),cumulative=True,histtype='step',density=True)
    plt.legend(['hour_scenario', 'hour_data'],loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    hist,edge = np.histogram(hour_scenario,bins=100,range=(0,1), density=False)
    hour_scenario_cdf = np.cumsum(hist/np.sum(hist))
    hist, edge = np.histogram(hour_data, bins=100, range=(0, 1), density=False)
    hour_data_cdf = np.cumsum(hist / np.sum(hist))
    K_S_hour = np.max(np.abs(hour_scenario_cdf-hour_data_cdf))
    print("K_S score of hour data:", K_S_hour)

    # CDF plot - day
    day_scenario = np.mean(scenario,axis=1)
    day_data = np.mean(data,axis=1)
    plt.hist(day_scenario, bins=100, range=(0, 1), cumulative=True, histtype='step', density=True)
    plt.hist(day_data, bins=100, range=(0, 1), cumulative=True, histtype='step', density=True)
    plt.legend(['day_scenario','day_data'],loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

    hist,edge = np.histogram(day_scenario,bins=100,range=(0,1), density=False)
    day_scenario_cdf = np.cumsum(hist/np.sum(hist))
    hist, edge = np.histogram(day_data, bins=100, range=(0, 1), density=False)
    day_data_cdf = np.cumsum(hist / np.sum(hist))
    K_S_day = np.max(np.abs(day_scenario_cdf-day_data_cdf))
    print("K_S score of day mean data:", K_S_day)

    # CDF plot - volatility
    vol_scenario = (scenario[:,1:args.sample_step] - scenario[:,0:args.sample_step-1]).flatten()
    vol_data = (data[:, 1:args.sample_step] - data[:, 0:args.sample_step - 1]).flatten()
    plt.hist(vol_scenario,bins=100,range=(-1,1),cumulative=True,histtype='step',density=True)
    plt.hist(vol_data,bins=100,range=(-1,1),cumulative=True,histtype='step',density=True)
    plt.legend(['vol_scenario', 'vol_data'],loc = 'lower right')
    plt.xlim([-1, 1])
    plt.show()

    hist, edge = np.histogram(vol_scenario, bins=100, range=(-1, 1), density=False)
    vol_scenario_cdf = np.cumsum(hist / np.sum(hist))
    hist, edge = np.histogram(vol_data, bins=100, range=(-1, 1), density=False)
    vol_data_cdf = np.cumsum(hist / np.sum(hist))
    K_S_vol = np.max(np.abs(vol_scenario_cdf - vol_data_cdf))
    print("K_S score of volatility:", K_S_vol)

"""--------------------------------------------------Define Diffusion------------------------------------------------"""
#  diffusion beta~aplha
diffusion_step = 50 # T of diffusion process
betas = torch.linspace(1e-4, 0.05,diffusion_step)
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def q_x(x_0, t):
    # x0-→x1...-→xt...-→xT
    # norm randn
    noise = torch.randn_like(x_0)
    # choose alphas_bar_sqrt, one_minus_alphas_bar_sqrt of t
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]

    x_t = alphas_t * x_0 + alphas_1_m_t * noise
    return x_t

def p_sample_loop(model, c_power, c_climate, noise, n_steps, betas, one_minus_alphas_bar_sqrt):
    # xT-→xT-1...-→xt...-→x0
    xt = noise

    for i in reversed(range(n_steps)):
        # print("diffusion step:", i)
        xt,_ = p_sample(model, xt, c_power, c_climate, i, betas, one_minus_alphas_bar_sqrt)
        xt = xt.detach()  # clear memory

    cur_x = xt
    return cur_x

def p_sample(model, x, c_power, c_climate, t, betas, one_minus_alphas_bar_sqrt):
    # input Xt and output Xt-1
    # denoise process
    t = torch.tensor([t])
    t1 = t.repeat(x.size(0))

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, c_power, c_climate, t1)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x) if t>2 else 0
    sigma_t = betas[t].sqrt()

    # mean + noise
    sample = mean + sigma_t * z

    return sample,eps_theta

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()

        self.channel = 32

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.channel, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=self.channel, out_channels=args.latent_dim, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2))
        self.power = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=self.channel, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=self.channel, out_channels=self.channel, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.xt = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.channel, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(0.2))

        self.embed = nn.Embedding(num_embeddings=100,embedding_dim=32)

        self.hidden = nn.Sequential(
            nn.Conv1d(in_channels=self.channel*4, out_channels=self.channel, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(0.2))

        self.out = nn.Sequential(
            nn.Conv1d(in_channels=self.channel, out_channels=self.channel, kernel_size=1, stride=1,padding=0),
            nn.Tanh(),
            nn.Conv1d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1,padding=0)
        )

    def forward(self, xt, c_power, c_climate, t):
        # input xt; ouput Epsilon(xt,c,t)
        # dim x: [batch,1,target_step]
        # dim c_power: [batch,1,target_step]
        # dim c_climate: [batch,1,input_step]

        c_power = self.encoder(c_power)
        c_power = c_power.view(c_power.size(0), 1, args.latent_dim) # (B,1,12)
        c_power = self.power(c_power)

        c_climate = self.decoder(c_climate) # (B,32,12)

        embed_t = self.embed(t).unsqueeze(dim=2) # (B,32,1)
        embed_t = embed_t.repeat(1, 1, args.latent_dim) # (B,32,12)

        xt = self.xt(xt) # (B,32,12)

        x = torch.cat([c_power, c_climate, xt, embed_t], dim=1)
        x = self.hidden(x) + xt
        x = x/math.sqrt(2)

        epsilon = self.out(x) # (B,1,12)

        return epsilon

def diffusion_loss_fn(model, x_0, c_power, c_climate, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps=diffusion_step):
    # sample random t for loss
    batch_size = x_0.shape[0]

    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    if batch_size % 2 == 0:
        t = torch.cat([t, n_steps - 1 - t], dim=0)
    else:
        odd = torch.randint(0, n_steps, size=(1,))
        t = torch.cat([t, n_steps - 1 - t, odd], dim=0)

    a = alphas_bar_sqrt[t].to(device)
    a = a.unsqueeze(1).unsqueeze(2)

    aml = one_minus_alphas_bar_sqrt[t].to(device)
    aml = aml.unsqueeze(1).unsqueeze(2)

    # sample noise
    e = torch.randn_like(x_0).to(device)

    # model input xt
    x = x_0 * a + e * aml

    # output = DDPM model output noise
    output = model(x, c_power, c_climate, t.to(device))
    return e,output

def DF_train():

    DF = Diffusion().to(device)
    print("Total number of paramerters in DF networks is {} ".format(sum(x.numel() for x in DF.parameters())))

    optimizer = torch.optim.Adam(DF.parameters(), lr=args.lr, weight_decay=0.0)

    Loss = nn.MSELoss()

    DF_loss = np.zeros([args.DF_epoch, 1])

    for epoch in range(args.DF_epoch):

        for step, (Train_input_power,Train_input_climate,latent_miu,latent_sigma) in enumerate(latent_train_loader):

            Train_input_power = Train_input_power.to(device)
            Train_input_climate = Train_input_climate.to(device)

            eps = torch.randn(latent_miu.size(0), 12)
            latent_scenario = latent_miu + eps * torch.exp(latent_sigma)
            latent_scenario = latent_scenario.to(device).unsqueeze(1)

            e,output = diffusion_loss_fn(DF, latent_scenario, Train_input_power, Train_input_climate, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            loss = Loss(e,output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        DF_loss[epoch, :] = loss.cpu().detach().numpy()

        if epoch % 1 == 0:
            print('epoch:', epoch, 'DF loss:', DF_loss[epoch, :])

    data_file_name = 'DF{ep}.pkl'.format(ep=args.DF_epoch)
    torch.save(DF.state_dict(), data_file_name)  # save model parameters

"""-----------------------------------------------test function------------------------------------------------------"""

def Test(Test_input_climate):
    DF = Diffusion()
    DF.load_state_dict(torch.load('DF{ep}.pkl'.format(ep=args.DF_epoch), map_location='cpu'))

    VD = VAE_Decoder()
    VD.load_state_dict(torch.load('VD{ep}.pkl'.format(ep=args.VAE_epoch), map_location='cpu'))

    num = Test_input_climate.size(0)
    noise = torch.randn(args.scenario_number, 1, args.latent_dim)
    Test_input = torch.zeros(args.scenario_number, 1, args.target_step)
    c = Test_input_climate[0:1,:,:].repeat(args.scenario_number,1,1)
    latent_scenario = p_sample_loop(DF, Test_input, c, noise, diffusion_step, betas,
                                    one_minus_alphas_bar_sqrt)
    scenario, _ = VD(latent_scenario)
    S=scenario.detach().numpy()

    for i in range(1,num):
        print("test sample:", i)

        noise = torch.randn(args.scenario_number, 1, args.latent_dim)
        c = Test_input_climate[i:i+1, :, :].repeat(args.scenario_number, 1, 1)

        latent_scenario = p_sample_loop(DF, scenario, c, noise, diffusion_step, betas,
                                 one_minus_alphas_bar_sqrt)

        scenario, _ = VD(latent_scenario)

        S = np.concatenate((S,scenario.detach().numpy()),axis=2)

    S = S[:,0,:].T
    data_df = pd.DataFrame(S)
    data_df.to_csv('Rsds_Scenario_SSP126.csv', float_format='%.5f', header=None, index=False, index_label=False)

    return scenario

"""-----------------------------------------------main function------------------------------------------------------"""

if __name__ == '__main__':
    Train_input,Train_input_climate,Train_target = Train_Data(args.rate_value1, args.rate_value2, args.sample_step, args.input_step, args.target_step, args.train_path)

    Test_input_climate = Test_Data(args.rate_value1, args.rate_value2,  1, 1, args.test_path)

    train_loader = dataload_data(target_data=Train_target, BATCH_SIZE=args.batch_size)

    VAE_train()
    latent_miu, latent_sigma = VAE_test(Train_target)

    latent_train_loader = latent_dataload_data(input_data1=Train_input, input_data2=Train_input_climate,
                                               target_data1=latent_miu, target_data2=latent_sigma, BATCH_SIZE=args.batch_size)

    DF_train()

    scenario = Test(Test_input_climate)