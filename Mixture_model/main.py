from model import Model
from model import Model_Mod

import torch
import numpy as np
import matplotlib.pyplot as plt

#     Arguments:
#         in_features (int): the number of dimensions in the input
#         out_features (int): the number of dimensions in the output
#         num_gaussians (int): the number of Gaussians per output dimensions

if __name__ == "__main__":
    # initialize the model
    in_features, out_features, num_gaussians=1 ,1 ,5

    mdn_net = 'MDN'

    if mdn_net == 'MDN':
        model =  Model(in_features, out_features, num_gaussians)
    elif mdn_net == 'MDN_mod':
        model = Model_Mod (in_features, out_features, num_gaussians, softmax_temperature=0.9)


    # dataset
    NSAMPLE = 1000
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

    # plt.figure(figsize=(8, 8))
    # plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
    # plt.show()

    # training loop
    from IPython.display import clear_output
    loss_cat = []
    for epoch in range(500):
        chunks = 20
        x_batch = torch.chunk(torch.tensor(x_data), chunks, dim=0)
        y_batch = torch.chunk(torch.tensor(y_data), chunks, dim=0)
        for input, target in zip(x_batch, y_batch):
            model.train_()
            pi, sigma, mu = model.compute(torch.tensor(input))
            loss = model.mdn_loss(pi, sigma, mu, target)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            loss_cat.append(loss)

            model.eval_()
            with torch.no_grad():
                pi, sigma, mu = model.compute(torch.tensor(x_data))
            samples = model.sample(pi, sigma, mu)
            if epoch%5==0:
                print(f'epoch: {epoch} loss {loss}')
    clear_output(wait=True)
    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro',alpha=0.3)
    plt.plot(x_data,samples,'bo',alpha=0.3)
    plt.savefig('sample.png')
    plt.show()