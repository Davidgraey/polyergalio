"""
Vision + Memory + Controller (VMC)
Toy world model: Perception (NVPVAE) -> Memory (multi-step Spectre generator, mixture density head) -> dreaming.

Three vehicles drive a square arena. Vehicle 0 is controlled by an action (turn, accelerate). The others drift between
turning left, going straight and turning right at random, so the next frame is a mixture and not a point.

Perception trains first and is then frozen. Memory trains on Perception's latents and the ego action. Dreaming primes
Memory with real frames, then feeds its own sampled latents back with the ego actions replayed, with no World involved.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers import NVPVAELayer
from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer
from polyergalio.models.layers.model_heads.prediction_heads import MixtureDensityHead
from polyergalio.models.layers.spectre_generator import SpectreCausalGenerator
from polyergalio.models.model_loss import MixtureNLLLoss, MultiStepLossWrapper
from polyergalio.models.network import Network
from polyergalio.models.optimizers import Adam

SEED = 0
OUTPUT_PATH = Path(__file__).with_name("world_model_toy.png")
LATENT_PATH = Path(__file__).with_name("world_model_latent.png")

# ------------- run
READOUT_EVERY = 200

# ------------- simulation: ToyWorld
NUM_VEHICLES = 3
DATA_DIMENSION = 5 # (x, y, cos, sin, speed)
ACTION_DIMENSION = 2
EPISODE_LENGTH = 64
NUM_TRAIN_EPISODES = 1000
NUM_HELD_OUT_EPISODES = 500
TIME_STEP = 0.15
MAX_TURN = 0.66
NPC_TURN = 0.33
NPC_SWITCH_PROBABILITY = 0.25
WALL_RADIUS = 0.8

# ------------- vision: NVPVAELayer and its training
INPUT_DIMENSION = NUM_VEHICLES * DATA_DIMENSION
LATENT_DIMENSION = 12
VISION_HIDDEN_DIMENSION = 128
FLOW_HIDDEN_DIMENSION = 128
NUM_COUPLINGS = 3
ACTIVATION_TYPE = "swish"
FLOW_ACTIVATION_TYPE = "tanh"
VISION_TRAIN_STEPS = 8000
VISION_LEARNING_RATE = 5e-4
VISION_BATCH_SIZE = 100

# ------------- memory: SpectreCausalGenerator, MixtureDensityHead, MultiStepLossWrapper and their training
SEQUENCE_LENGTH = 50
HIDDEN_DIMENSION = 256
NUM_HEADS = 8
MEMORY_TOKENS = 5
NUM_FUTURE_STEPS = 2
MIXTURE_COMPONENTS = 3
LOSS_DECAY = 0.66
MEMORY_TRAIN_STEPS = 2000
MEMORY_LEARNING_RATE = 5e-4
MEMORY_BATCH_SIZE = 10

# ------------- memory: dreaming through generate
PROMPT_LENGTH = 12
HORIZON = 50
TEMPERATURE = 0.2
NUM_DREAMS = 5


def wrap_angle(angle: NDArray) -> NDArray:
    """Wrap angles into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ToyWorld:
    """
    Vehicles on a square arena, vehicle 0 controlled and the rest drifting.

    Parameters
    ----------
    rng : source of randomness for starts, the ego's wandering and the drifting vehicles' switches
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def steer_to_center(self, position: NDArray, heading: NDArray) -> tuple[NDArray, NDArray]:
        """Heading error towards the arena centre and a mask of vehicles outside the wall radius."""
        to_center = np.arctan2(-position[..., 1], -position[..., 0])
        outside = np.hypot(position[..., 0], position[..., 1]) > WALL_RADIUS
        return wrap_angle(to_center - heading), outside

    def rollout(self, num_episodes: int, length: int) -> dict:
        """
        Parameters
        ----------
        num_episodes : number of independent episodes
        length : ticks per episode

        Returns
        -------
        dict with
            frames : (episodes, length, vehicles * 5) per vehicle [x, y, cos heading, sin heading, speed], every
                value inside [0, 1]
            actions : (episodes, length, 2) the ego's [turn, accelerate] chosen at each tick
        """
        count, vehicles = num_episodes, NUM_VEHICLES
        position = self.rng.uniform(-0.5, 0.5, size=(count, vehicles, 2))
        heading = self.rng.uniform(-np.pi, np.pi, size=(count, vehicles))
        speed = self.rng.uniform(0.4, 0.8, size=(count, vehicles))
        mode = self.rng.integers(-1, 2, size=(count, vehicles))
        wander = np.zeros((count, ACTION_DIMENSION))
        frames, actions = [], []

        for _ in range(length):
            frames.append(np.concatenate(
                [(position + 1) / 2, (np.cos(heading)[..., None] + 1) / 2, (np.sin(heading)[..., None] + 1) / 2,
                 speed[..., None]], axis=-1
            ).reshape(count, -1))

            wander = 0.85 * wander + 0.5 * self.rng.normal(size=wander.shape)
            steer, outside = self.steer_to_center(position, heading)
            turn = np.where(outside[:, 0], np.clip(2.0 * steer[:, 0], -1, 1), np.clip(wander[:, 0], -1, 1))
            accelerate = np.clip(wander[:, 1] + 2.0 * (0.6 - speed[:, 0]), -1, 1)
            actions.append(np.stack([turn, accelerate], axis=-1))

            switch = self.rng.random(size=(count, vehicles)) < NPC_SWITCH_PROBABILITY
            mode = np.where(switch, self.rng.integers(-1, 2, size=mode.shape), mode)
            heading_change = np.where(outside, np.clip(steer, -MAX_TURN, MAX_TURN), mode * NPC_TURN)
            heading_change[:, 0] = turn * MAX_TURN
            heading = heading + heading_change
            speed[:, 0] = np.clip(speed[:, 0] + 0.1 * accelerate, 0.2, 1.0)
            position = position + TIME_STEP * speed[..., None] * np.stack([np.cos(heading), np.sin(heading)], axis=-1)

        return {"frames": np.stack(frames, axis=1), "actions": np.stack(actions, axis=1)}


def vehicle_positions(frames: NDArray) -> NDArray:
    """(..., vehicles, 2) arena x, y of every vehicle from (..., vehicles * 5) frames."""
    return 2 * frames.reshape(*frames.shape[:-1], NUM_VEHICLES, DATA_DIMENSION)[..., :2] - 1


def train_perception(perception: NVPVAELayer, frames: NDArray) -> None:
    """Fit the NVPVAE on single frames with its optimal-variance reconstruction loss plus the flow prior's KL."""
    optimizer = Adam(VISION_LEARNING_RATE)
    order = np.random.default_rng(SEED)
    flat = frames.reshape(-1, frames.shape[-1])
    perception.train()

    for step in range(VISION_TRAIN_STEPS):
        batch = flat[order.integers(0, len(flat), VISION_BATCH_SIZE)]
        perception.zero_gradients()
        reconstruction, gradient = perception.reconstruction_loss(perception(batch), batch)
        perception.backward(gradient)
        optimizer.step(perception)

        if step % READOUT_EVERY == 0 or step == VISION_TRAIN_STEPS - 1:
            print(f"perception step {step:5d}  reconstruction {reconstruction:7.2f}  kl {perception.kl_divergence:6.2f}  "
                  f"variance {perception.observation_variance:.5f}")


def encode(perception: NVPVAELayer, frames: NDArray) -> NDArray:
    """(episodes, length, LATENT_DIMENSION) deterministic latents of every frame."""
    perception.eval()
    flat = frames.reshape(-1, frames.shape[-1])
    latents, _ = perception.encode(flat, perception.read_labels(None, len(flat), flat.dtype))
    return latents.reshape(*frames.shape[:2], -1)


def decode(perception: NVPVAELayer, latents: NDArray) -> NDArray:
    """Frames from latents of any leading shape."""
    flat = latents.reshape(-1, latents.shape[-1])
    frames = perception.decode(flat, perception.read_labels(None, len(flat), flat.dtype))
    return frames.reshape(*latents.shape[:-1], -1)


class PackedMixtureNLL(MixtureNLLLoss):
    """MixtureNLLLoss that reads the head's packed output, so MultiStepLossWrapper can call it like any other loss."""
    def __init__(self, head: MixtureDensityHead):
        super().__init__(head.components, head.latent_dim)
        self.head = head

    def forward(self, prediction: NDArray, targets: NDArray, mask: NDArray | None = None) -> float:
        return super().forward(*self.head.unpack(prediction), targets, mask)


def build_memory() -> tuple[Network, SpectreCausalGenerator]:
    """Embedding of [z_t, a_t] into a multi-step Spectre generator, which builds its own mixture density head."""
    generator = SpectreCausalGenerator(SEQUENCE_LENGTH, HIDDEN_DIMENSION, LATENT_DIMENSION,
                                       mixture_components=MIXTURE_COMPONENTS, num_heads=NUM_HEADS,
                                       memory_tokens=MEMORY_TOKENS, num_future_steps=NUM_FUTURE_STEPS)
    embed = FullyConnectedLayer(LATENT_DIMENSION + ACTION_DIMENSION, HIDDEN_DIMENSION, "linear")
    net = Network([embed, generator], name="memory")
    print("constructed memory network: ", net)
    return net, generator


def train_memory(network: Network, generator: SpectreCausalGenerator, latents: NDArray, actions: NDArray) -> list[float]:
    """Fit the multi-step mixture density loss on random windows of the stored latents; returns the loss history."""
    loss = MultiStepLossWrapper(PackedMixtureNLL, NUM_FUTURE_STEPS, decay=LOSS_DECAY, loss_kwargs={"head": generator.head})
    optimizer = Adam(MEMORY_LEARNING_RATE)
    order = np.random.default_rng(SEED + 1)
    inputs = np.concatenate([latents, actions], axis=-1)
    window = np.arange(SEQUENCE_LENGTH)
    history = []

    network.train()
    for step in range(1, MEMORY_TRAIN_STEPS + 1):
        episode = order.integers(0, len(latents), MEMORY_BATCH_SIZE)[:, None]
        start = order.integers(0, EPISODE_LENGTH - SEQUENCE_LENGTH + 1, MEMORY_BATCH_SIZE)[:, None]
        network.zero_gradients()
        prediction = network.forward(inputs[episode, start + window])
        history.append(loss.forward(prediction, latents[episode, start + window]))
        network.backward(loss.backward())
        optimizer.step(network)
        if step % READOUT_EVERY == 0:
            steps = " ".join(f"{value:7.2f}" for value in loss.step_losses)
            print(f"memory step {step:5d}  weighted nll {np.mean(history[-READOUT_EVERY:]):7.2f}  per step {steps}")
    return history


def dream(network: Network, generator: SpectreCausalGenerator,
          prompt: NDArray, future_actions: NDArray, temperature: float, rng: np.random.Generator) -> dict:
    """
    Parameters
    ----------
    prompt : (batch, PROMPT_LENGTH, LATENT_DIMENSION + ACTION_DIMENSION) real [z_t, a_t]
    future_actions : (batch, HORIZON - 1, ACTION_DIMENSION) ego actions replayed during the dream
    temperature : tau of the mixture head's sampler

    Returns
    -------
    the generator's result dict; samples is (batch, HORIZON, LATENT_DIMENSION)
    """
    embed = network.layers[0]

    return generator.generate(
        embed.forward(prompt),
        num_future_steps=HORIZON,
        sample=lambda output, tau, draw_rng: generator.head.sample(output, tau, draw_rng),
        feedback=lambda draw, step: embed.forward(np.concatenate([draw, future_actions[:, step]], axis=-1)[:, None])[:, 0],
        temperature=temperature,
        rng=rng,
    )


def main():
    RNG.bit_generator.state = np.random.default_rng(SEED).bit_generator.state
    rng = np.random.default_rng(SEED)
    world = ToyWorld(rng)
    train = world.rollout(NUM_TRAIN_EPISODES, EPISODE_LENGTH)
    held_out = world.rollout(NUM_HELD_OUT_EPISODES, EPISODE_LENGTH)

    perception = NVPVAELayer(INPUT_DIMENSION, LATENT_DIMENSION, hidden_dimension=VISION_HIDDEN_DIMENSION,
                             flow_hidden_dimension=FLOW_HIDDEN_DIMENSION, num_couplings=NUM_COUPLINGS,
                             activation_type=ACTIVATION_TYPE, flow_activation_type=FLOW_ACTIVATION_TYPE)
    print("built perception: ", perception)
    train_perception(perception, train["frames"])
    perception.eval()
    perception.plot_latent(x=train["frames"].reshape(-1, INPUT_DIMENSION)[:2000]).savefig(LATENT_PATH, dpi=140)

    train_latents = encode(perception, train["frames"])
    held_latents = encode(perception, held_out["frames"])
    center, scale = train_latents.mean(axis=(0, 1)), train_latents.std(axis=(0, 1)) + 1e-6
    train_latents, held_latents = (train_latents - center) / scale, (held_latents - center) / scale

    held_reconstruction = decode(perception, held_latents * scale + center)
    print(f"held-out position error through perception alone: "
          f"{np.abs(vehicle_positions(held_reconstruction) - vehicle_positions(held_out['frames'])).mean():.4f}")

    network, generator = build_memory()
    print(generator, f"\n{network.num_parameters} memory parameters\n")
    history = train_memory(network, generator, train_latents, train["actions"])

    network.eval()
    end = PROMPT_LENGTH + HORIZON
    truth = vehicle_positions(held_out["frames"][:, PROMPT_LENGTH:end])
    last = vehicle_positions(held_out["frames"][:, PROMPT_LENGTH - 2:PROMPT_LENGTH])
    ticks = np.arange(1, HORIZON + 1)[None, :, None, None]
    forecasts = {
        "hold last frame": np.repeat(last[:, 1:], HORIZON, axis=1),
        "constant velocity": last[:, 1:] + (last[:, 1] - last[:, 0])[:, None] * ticks,
    }
    errors = {label: np.linalg.norm(forecast - truth, axis=-1).mean(axis=0) for label, forecast in forecasts.items()}

    scenarios = (("dream, tau 0", 0.0, 0), (f"dream, tau {TEMPERATURE:g}", TEMPERATURE, 0), ("dream, tau 0, wrong actions", 0.0, 1))
    for label, temperature, action_shift in scenarios:
        actions = np.roll(held_out["actions"], action_shift, axis=0)
        prompt = np.concatenate([held_latents[:, :PROMPT_LENGTH], actions[:, :PROMPT_LENGTH]], axis=-1)
        result = dream(network, generator, prompt, actions[:, PROMPT_LENGTH:end - 1], temperature, rng)
        dreamed = decode(perception, result["samples"] * scale + center)
        errors[label] = np.linalg.norm(vehicle_positions(dreamed) - truth, axis=-1).mean(axis=0)
        print(f"{label}: largest cached-vs-training difference {result['cache_difference']:.1e}")

    reported = (1, 5, 10, HORIZON)
    print("\nmean position error over the first n dreamed steps")
    print(f"  {'':40s}" + "".join(f"{f'n={upto}':>8s}" for upto in reported))
    for label, error in errors.items():
        for name, part in (("ego", error[:, 0]), ("drifters", error[:, 1:].mean(axis=1))):
            print(f"  {label + ', ' + name:40s}" + "".join(f"{part[:upto].mean():8.3f}" for upto in reported))

    episode = 0
    own_prompt = np.concatenate([held_latents[:, :PROMPT_LENGTH], held_out["actions"][:, :PROMPT_LENGTH]], axis=-1)
    repeated = lambda values: np.repeat(values[episode:episode + 1], NUM_DREAMS, axis=0)
    result = dream(network, generator, repeated(own_prompt), repeated(held_out["actions"][:, PROMPT_LENGTH:end - 1]),
                   TEMPERATURE, rng)
    dreams = vehicle_positions(decode(perception, result["samples"] * scale + center))
    path = vehicle_positions(held_out["frames"][episode])

    figure, (curve, horizon, arena) = plt.subplots(1, 3, figsize=(16, 4.5))
    curve.plot(history, linewidth=0.6)
    curve.set(xlabel="step", ylabel="weighted mixture nll", title="memory training")

    for index, (label, error) in enumerate(errors.items()):
        horizon.plot(np.arange(1, HORIZON + 1), error[:, 0], color=f"C{index}", label=f"{label}, ego")
        horizon.plot(np.arange(1, HORIZON + 1), error[:, 1:].mean(axis=1), color=f"C{index}", linestyle="--")
    horizon.set(xlabel="dreamed steps", ylabel="position error", title="held-out error by horizon (dashed: drifters)")
    horizon.legend(fontsize=6)

    for vehicle in range(NUM_VEHICLES):
        arena.plot(path[:end, vehicle, 0], path[:end, vehicle, 1], color=f"C{vehicle}", alpha=0.35, linewidth=3)
        arena.plot(path[:PROMPT_LENGTH, vehicle, 0], path[:PROMPT_LENGTH, vehicle, 1], color=f"C{vehicle}", linewidth=3)
        for sample in dreams[:, :, vehicle]:
            arena.plot(sample[:, 0], sample[:, 1], color=f"C{vehicle}", linestyle="--", linewidth=0.8)
    arena.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal",
              title="truth (faint), prompt (solid), dreams (dashed)")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=140)
    plt.show()


if __name__ == "__main__":
    main()
