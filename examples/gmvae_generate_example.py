import numpy as np

from polyergalio.models.layers import GMVAELayer
from polyergalio.models.layers.basic_layers import RNG
from polyergalio.models.optimizers import Adam

SEED = 0
FEATURES = 6
FACTORS = 2
LATENT = 2
COMPONENTS = 4
NUM_TRAIN = 4000
NUM_HELD_OUT = 500
NUM_GENERATED = 4000
STEPS = 4000
BATCH_SIZE = 128
LEARNING_RATE = 3e-3
OBSERVATION_VARIANCE = 0.3
KL_WEIGHT = 1.0
READOUT_EVERY = 1000


def make_data(num_samples, mixing, rng):
    """Samples that sit near a FACTORS-dimensional plane inside FEATURES dimensions."""
    return rng.normal(size=(num_samples, FACTORS)) @ mixing + 0.05 * rng.normal(size=(num_samples, FEATURES))


def train(net, data):
    """
    Fit the layer on reconstruction plus the mixture's KL.

    The reconstruction error is scaled by OBSERVATION_VARIANCE, which sets how hard it pulls against the KL.
    """
    optimizer = Adam(LEARNING_RATE)
    order = np.random.default_rng(SEED)
    net.train()
    for step in range(STEPS):
        batch = data[order.integers(0, len(data), BATCH_SIZE)]
        net.zero_gradients()
        error = net(batch) - batch
        net.backward(error / (len(batch) * OBSERVATION_VARIANCE))
        optimizer.step(net)
        if step % READOUT_EVERY == 0 or step == STEPS - 1:
            reconstruction = 0.5 * (error ** 2).sum(axis=1).mean() / OBSERVATION_VARIANCE
            print(f"step {step:5d}  reconstruction {reconstruction:7.2f}  kl {net.divergence:5.2f}")


def compare(label, samples, data):
    """Print how closely samples match the data's spread and covariance."""
    spread = samples.std(axis=0) / data.std(axis=0)
    covariance_error = np.abs(np.cov(samples.T) - np.cov(data.T)).max()
    print(f"{label:10s} spread ratio {np.round(spread, 2)}  covariance error {covariance_error:5.2f}")


def main():
    RNG.bit_generator.state = np.random.default_rng(SEED).bit_generator.state
    rng = np.random.default_rng(SEED + 1)
    mixing = rng.normal(size=(FACTORS, FEATURES))
    data = make_data(NUM_TRAIN, mixing, rng)
    held_out = make_data(NUM_HELD_OUT, mixing, rng)

    net = GMVAELayer(FEATURES, LATENT, COMPONENTS, kl_weight=KL_WEIGHT, temperature=0.5)

    net.eval()
    compare("untrained", net.generate(NUM_GENERATED), data)

    train(net, data)

    net.eval()
    reconstruction = net(held_out)
    explained = 1 - ((reconstruction - held_out) ** 2).sum() / ((held_out - held_out.mean(axis=0)) ** 2).sum()
    print(f"held-out variance explained by reconstruction: {explained:.1%}")

    compare("generated", net.generate(NUM_GENERATED), data)
    compare("held-out", held_out, data)


if __name__ == "__main__":
    main()
