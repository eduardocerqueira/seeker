#date: 2024-09-25T17:01:44Z
#url: https://api.github.com/gists/0146b134f40fe51a9c0d3a91bc00518c
#owner: https://api.github.com/users/MaximeVandegar

if __name__ == "__main__":
    device = 'cuda'

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0., 0.9))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0., 0.9))

    data = DataLoader(Dataset('../data'), batch_size=64, shuffle=True, num_workers=0)
    loss = train(generator, discriminator, optimizer_g, optimizer_d, data, 25_000)

    loss_critic = moving_average(loss["critic"], window_size=1000)
    plt.plot(-np.array(loss["critic"]))
    plt.plot(-loss_critic)
    plt.xlabel("Discriminator iterations", fontsize=13)
    plt.ylabel("Negative critic loss", fontsize=13)
    plt.savefig("Imgs/wgan_loss.png")
    plt.close()

    generator.eval()
    NB_IMAGES = 8 ** 2
    img = generator(sample_noise(NB_IMAGES, device))
    plt.figure(figsize=(12, 12))
    for i in range(NB_IMAGES):
        plt.subplot(8, 8, 1 + i)
        plt.axis('off')
        plt.imshow(img[i].data.cpu().transpose(0, 1).transpose(1, 2).numpy() / 2 + .5)
    plt.savefig("Imgs/generated_samples.png")
    plt.close()