import torch
from GAN.testData import TestDataset
from GAN.Generator import Generator
# import torch.optim as optim
from optim import RiemannianAdam, RiemannianSGD
from GAN.Critic import Critic
from torch.utils.data import Dataset, DataLoader
from geoopt import ManifoldParameter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import manifolds
import time
import io

LEARNING_RATE = 1e-4
# DIM = 256
DIM = 3
BATCH_SIZE = 64
# Z_DIM = 16
Z_DIM = 8
NUM_EPOCHS = 20
N = 10000
# CRITIC_HID = 512
CRITIC_HID = 64
CRITIC_DEP = 3
# GEN_HID = 512
GEN_HID = 64
GEN_DEP = 3
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
WD = 0.0

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, N = real.shape
    # alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_points = []
    for i in range(BATCH_SIZE):
        interpolated_points.append(manifold.geodesic(torch.rand(1).to(device), real[i], fake[i]))
    interpolated_points = torch.stack(interpolated_points)

    # Calculate critic scores
    mixed_scores = critic(interpolated_points)

    # Take the gradient of the scores with respect to the points
    gradient = torch.autograd.grad(
        inputs=interpolated_points,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    # gradient_norm = gradient.norm(2, dim=1)
    gradient_norm = manifold.norm(gradient)
    # print(gradient.norm(2, dim=1)[0], gradient_norm[0])
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

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
no_decay = ['bias', 'scale']
gen_optimizer_grouped_parameters = [{
    'params': [
        p for n, p in gen.named_parameters()
        if p.requires_grad and not any(
            nd in n
            for nd in no_decay) and not isinstance(p, ManifoldParameter)
    ],
    'weight_decay':
    WD
}, {
    'params': [
        p for n, p in gen.named_parameters() if p.requires_grad and any(
            nd in n
            for nd in no_decay) or isinstance(p, ManifoldParameter)
    ],
    'weight_decay':
    0.0
}]
cri_optimizer_grouped_parameters = [{
    'params': [
        p for n, p in critic.named_parameters()
        if p.requires_grad and not any(
            nd in n
            for nd in no_decay) and not isinstance(p, ManifoldParameter)
    ],
    'weight_decay':
    WD
}, {
    'params': [
        p for n, p in critic.named_parameters() if p.requires_grad and any(
            nd in n
            for nd in no_decay) or isinstance(p, ManifoldParameter)
    ],
    'weight_decay':
    0.0
}]
opt_gen = RiemannianAdam(gen_optimizer_grouped_parameters, lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))
opt_critic = RiemannianAdam(cri_optimizer_grouped_parameters, lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))

# opt_gen = RiemannianSGD(gen.parameters(), lr=LEARNING_RATE, stabilize=10)
# opt_critic = RiemannianSGD(critic.parameters(), lr=LEARNING_RATE, stabilize=10)

# for tensorboard plotting
fixed_noise = manifold.random_normal((1000, Z_DIM)).to(device)
ts = str(int(time.time()))
writer_real = SummaryWriter(f"GAN/logs/{ts}/real")
writer_fake = SummaryWriter(f"GAN/logs/{ts}/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            noise = manifold.random_normal((cur_batch_size, Z_DIM)).to(device)
            fake = gen(noise)
            # print(real.shape)
            # print(fake.shape)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # print(critic_real)
            # print(critic_fake)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            # print(loss_critic.item())
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
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