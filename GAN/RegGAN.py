import torch
from GAN.testData import TestDataset
from GAN.Generator import Generator
from optim import RiemannianAdam, RiemannianSGD
from GAN.Critic import Critic
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import manifolds
import time
import io

LEARNING_RATE = 1e-3
# DIM = 256
DIM = 3
BATCH_SIZE = 64
# Z_DIM = 16
Z_DIM = 8
NUM_EPOCHS = 40
N = 10000
# CRITIC_HID = 512
CRITIC_HID = 64
CRITIC_DEP = 3
# GEN_HID = 512
GEN_HID = 64
GEN_DEP = 3
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

def gen_plot(x):
    x = manifold.logmap0(x).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:,1], x[:,2], s = 0.5)
    return plt.gcf()

device = "cuda" if torch.cuda.is_available() else "cpu"
manifold = manifolds.Lorentz()
dataset = TestDataset(manifold, 1, DIM, N, std = 1)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

gen = Generator(Z_DIM, GEN_HID, DIM, GEN_DEP).to(device)
critic = Critic(DIM, CRITIC_HID, CRITIC_DEP).to(device)

# initializate optimizer
opt_gen = RiemannianAdam(gen.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.5, 0.999))
opt_critic = RiemannianAdam(critic.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()
Act = torch.nn.Sigmoid()

# for tensorboard plotting
fixed_noise = manifold.random_normal((1000, Z_DIM)).to(device)
ts = str(int(time.time()))
writer_real = SummaryWriter(f"GAN/logs/Reg{ts}/real")
writer_fake = SummaryWriter(f"GAN/logs/Reg{ts}/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        for _ in range(CRITIC_ITERATIONS):
            # noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            noise = manifold.random_normal((cur_batch_size, Z_DIM)).to(device)
            fake = gen(noise)
            # print(real.shape)
            # print(fake.shape)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            critic_real = Act(critic_real)
            critic_fake = Act(critic_fake)
            loss_critic_real = criterion(critic_real, torch.ones_like(critic_real))
            loss_critic_fake = criterion(critic_fake, torch.zeros_like(critic_fake))
            loss_critic = (loss_critic_real + loss_critic_fake) / 2
            # gp = gradient_penalty(critic, real, fake, device=device)
            # loss_critic = (
            #     -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            # )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        gen_fake = critic(fake).reshape(-1)
        gen_fake = Act(gen_fake)
        # loss_gen = -torch.mean(gen_fake)
        loss_gen = criterion(gen_fake, torch.ones_like(gen_fake))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                # img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                # img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_figure("Real", gen_plot(dataset[:1000]), global_step=step)
                writer_fake.add_figure("Fake", gen_plot(fake), global_step=step)

            step += 1