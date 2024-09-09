import torch
from GAN.EmbedDataset import TreeDataset, GraphDataset, GTDataset
from GAN.Generator import Generator
# import torch.optim as optim
from optim import RiemannianAdam, RiemannianSGD
from GAN.Critic import Critic
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import manifolds
import time
import io
import os
from config import parser

args = parser.parse_args()

LEARNING_RATE = args.lr
DIM = args.dim
BATCH_SIZE = args.batch_size
Z_DIM = args.z_dim
NUM_EPOCHS = args.epochs
CRITIC_HID = args.critic_dim
CRITIC_DEP = args.critic_depth
GEN_HID = args.generator_dim
GEN_DEP = args.generator_depth
CRITIC_ITERATIONS = args.critic_iterations
LAMBDA_GP = args.lambda_gp

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
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def gen_plot(x):
    x = manifold.logmap0(x).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:,1], x[:,2], s = 0.5)
    return plt.gcf()

device = "cuda" if torch.cuda.is_available() else "cpu"
manifold = manifolds.Lorentz()
dataset = GTDataset(gpath = os.path.join(args.embed_path, 'graph.pt'), 
                    tpath = os.path.join(args.embed_path, 'tree.pt'))
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

tree_gen = Generator(Z_DIM, GEN_HID, DIM, GEN_DEP).to(device)
graph_gen = Generator(Z_DIM, GEN_HID, DIM, GEN_DEP).to(device)
tree_critic = Critic(DIM, CRITIC_HID, CRITIC_DEP).to(device)
graph_critic = Critic(DIM, CRITIC_HID, CRITIC_DEP).to(device)

# initializate optimizer
opt_tree_gen = RiemannianAdam(tree_gen.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))
opt_graph_gen = RiemannianAdam(graph_gen.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))
opt_tree_critic = RiemannianAdam(tree_critic.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))
opt_graph_critic = RiemannianAdam(graph_critic.parameters(), lr=LEARNING_RATE, stabilize=10, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = manifold.random_normal((1000, Z_DIM)).to(device)
ts = str(int(time.time()))
writer_real = SummaryWriter(f"2_HWGAN/logs/{ts}/real")
writer_fake = SummaryWriter(f"2_HWGAN/logs/{ts}/fake")
step = 0

tree_gen.train()
graph_gen.train()
tree_critic.train()
graph_critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            noise = manifold.random_normal((cur_batch_size, Z_DIM)).to(device)
            # print(real.shape)
            # print(fake.shape)

            # Tree
            tree_fake = tree_gen(noise)
            critic_tree_real = tree_critic(real[0]).reshape(-1)
            critic_tree_fake = tree_critic(tree_fake).reshape(-1)
            gp = gradient_penalty(tree_critic, real[0], tree_fake, device=device)
            tree_loss_critic = (
                -(torch.mean(critic_tree_real) - torch.mean(critic_tree_fake)) + LAMBDA_GP * gp
            )
            tree_critic.zero_grad()
            tree_loss_critic.backward(retain_graph=True)
            opt_tree_critic.step()

            # Graph
            graph_fake = graph_gen(noise)
            critic_graph_real = graph_critic(real[1]).reshape(-1)
            critic_graph_fake = graph_critic(graph_fake).reshape(-1)
            gp = gradient_penalty(graph_critic, real[1], graph_fake, device=device)
            graph_loss_critic = (
                -(torch.mean(critic_graph_real) - torch.mean(critic_graph_fake)) + LAMBDA_GP * gp
            )
            graph_critic.zero_grad()
            graph_loss_critic.backward(retain_graph=True)
            opt_graph_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        tree_gen_fake = tree_critic(tree_fake).reshape(-1)
        loss_tree_gen = -torch.mean(tree_gen_fake)
        tree_gen.zero_grad()
        loss_tree_gen.backward()
        opt_tree_gen.step()

        graph_gen_fake = graph_critic(graph_fake).reshape(-1)
        loss_graph_gen = -torch.mean(graph_gen_fake)
        graph_gen.zero_grad()
        loss_graph_gen.backward()
        opt_graph_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {tree_loss_critic + graph_loss_critic:.4f}, \
                  loss G: {loss_tree_gen + loss_graph_gen:.4f}"
            )

            with torch.no_grad():
                tree_fake = tree_gen(fixed_noise)
                graph_fake = graph_gen(fixed_noise)

                writer_real.add_figure("RealTree", gen_plot(dataset[:1000][0]), global_step=step)
                writer_real.add_figure("RealGraph", gen_plot(dataset[:1000][1]), global_step=step)
                writer_fake.add_figure("FakeTree", gen_plot(tree_fake), global_step=step)
                writer_fake.add_figure("FakeGraph", gen_plot(graph_fake), global_step=step)

            step += 1

        # break
    torch.save(tree_gen.state_dict(), f"2_HWGAN/logs/{ts}/tree-model.epoch-" + str(epoch))
    torch.save(graph_gen.state_dict(), f"2_HWGAN/logs/{ts}/graph-model.epoch-" + str(epoch))