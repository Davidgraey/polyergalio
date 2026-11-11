from contextlib import contextmanager
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import Layer, RMSNormLayer, FullyConnectedLayer, DropoutLayer
from polyergalio.models.layers.model_heads.prediction_heads import MixtureDensityHead
from polyergalio.models.layers.spectre_layers import SpectreDecoderAttention


class SpectreCausalBlock(Layer):
    """
    Causal decoder as a block of layers:
        - Pre-norm as an RMSNorm (for stacking)
        - SpectreDecoderAttention
        - 2x FC layers (scale up{linear}, project down{nonlinear}) -> RMSNorm -> dropout -> residual sum.


    Parameters
    ----------
    sequence_length : window of the attention, and the length forward expects
    hidden_dimension : width of the inputs and of the block
    ffn_dimension : feed-forward width, 2 * hidden_dimension if None
    num_heads, memory_tokens, chunk_size : see SpectreDecoderAttention
    dropout_prob : 0 for no dropout
    """
    preserves_shape = True
    parameter_names = ("prenorm", "attention", "scale_up_fc", "out_fc", "postnorm")

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        upscale_dimension: Optional[int] = None,
        num_heads: int = 1,
        memory_tokens: int = 0,
        chunk_size: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dimension = hidden_dimension
        self.upscale_dimension = upscale_dimension or 2 * hidden_dimension
        self.num_heads = num_heads
        self.memory_tokens = memory_tokens
        self.chunk_size = chunk_size
        self.dropout_prob = dropout_prob

        self.prenorm = RMSNormLayer(hidden_dimension)
        self.attention = SpectreDecoderAttention(
            sequence_length=sequence_length,
            hidden_dimension=hidden_dimension,
            num_heads=num_heads,
            memory_tokens=memory_tokens,
            chunk_size=chunk_size,
        )
        self.scale_up_fc = FullyConnectedLayer(hidden_dimension, self.upscale_dimension, "linear")
        self.out_fc = FullyConnectedLayer(self.upscale_dimension, hidden_dimension, "swish")
        self.postnorm = RMSNormLayer(hidden_dimension)
        self.dropout = DropoutLayer(dropout_prob) if dropout_prob > 0 else None

        self.declare_shapes(inputs=((hidden_dimension,),), outputs=((hidden_dimension,),))
        self.zero_gradients()

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence_length, hidden_dim)
        mask : (batch, sequence_length), 1 for a real token and 0 for padding

        Returns
        -------
        (batch, sequence_length, hidden_dim), each position reading only itself and earlier positions
        """
        mixed = self.attention.forward(self.prenorm.forward(input_data), mask=mask)
        hidden = self.postnorm.forward(self.out_fc.forward(self.scale_up_fc.forward(mixed)))
        if self.dropout is not None:
            hidden = self.dropout.forward(hidden)
        return hidden + input_data

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        gradient = incoming_gradient
        if self.dropout is not None:
            gradient = self.dropout.backward(gradient)
        gradient = self.scale_up_fc.backward(self.out_fc.backward(self.postnorm.backward(gradient)))
        return self.prenorm.backward(self.attention.backward(gradient)) + incoming_gradient

    def reset(self, batch_size: int) -> None:
        """Empty the attention cache for `batch_size` sequences."""
        self.attention.reset_cache(batch_size)

    def step(self, step_input: NDArray, valid=True) -> NDArray:
        """
        One token through the block, extending the cache.

        Parameters
        ----------
        step_input : (batch, hidden_dim)
        valid : bool or (batch,) bools; False for sequences that already finished

        Returns
        -------
        (batch, hidden_dim)
        """
        if self.training:
            raise RuntimeError("Call eval() or use inference() before stepping")
        mixed = self.attention.decode_step(self.prenorm.forward(step_input), valid=valid)
        hidden = self.postnorm.forward(self.out_fc.forward(self.scale_up_fc.forward(mixed)))
        return hidden + step_input

    def __str__(self):
        return (f"SpectreCausalBlock, window {self.sequence_length}, hidden {self.hidden_dimension}, "
                f"ffn {self.upscale_dimension}, {self.num_heads} heads, {self.memory_tokens} memory, chunk {self.chunk_size}")

    def __repr__(self):
        return self.__str__()


def shift_left(values: NDArray, steps: int) -> NDArray:
    """
    # TODO: move to sequence utilities
    shift data earlier (left by {steps} position axis 1 (sequence), zero-padding the newly created empty
        states at the end.
    """
    shifted = np.zeros_like(values)
    shifted[:, :values.shape[1] - steps] = values[:, steps:]
    return shifted


def shift_right(values: NDArray, steps: int) -> NDArray:
    """
    # TODO: move to sequence utilities
    shift data later (right by {steps} position axis 1 (sequence), zero-padding the newly created empty states at
        the beginning.
    """
    shifted = np.zeros_like(values)
    shifted[:, steps:] = values[:, :values.shape[1] - steps]
    return shifted


class MultiStepHead(Layer):
    """
    adds additional / extra sequence depth:
    merges the previous sequence depth's hidden state together with the "Peek" input (external ground truth) from
        the future step

    Parameters
    ----------
    takes the config of a SpectreCausalBlock
    sequence_length, hidden_dimension, upscale_dimension, num_heads, memory_tokens, chunk_size, dropout_prob :
        see SpectreCausalBlock
    """
    parameter_names = ("norm_hidden", "norm_peek", "projection", "block")

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        upscale_dimension: Optional[int] = None,
        num_heads: int = 1,
        memory_tokens: int = 0,
        chunk_size: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dimension = hidden_dimension
        self.upscale_dimension = upscale_dimension or 4 * hidden_dimension
        self.num_heads = num_heads
        self.memory_tokens = memory_tokens
        self.chunk_size = chunk_size
        self.dropout_prob = dropout_prob

        self.norm_hidden = RMSNormLayer(hidden_dimension)
        self.norm_peek = RMSNormLayer(hidden_dimension)
        self.projection = FullyConnectedLayer(2 * hidden_dimension, hidden_dimension, "linear")
        self.block = SpectreCausalBlock(
            sequence_length=sequence_length,
            hidden_dimension=hidden_dimension,
            upscale_dimension=self.upscale_dimension,
            num_heads=num_heads,
            memory_tokens=memory_tokens,
            chunk_size=chunk_size,
            dropout_prob=dropout_prob,
        )
        self.declare_shapes(inputs=((hidden_dimension,), (hidden_dimension,)), outputs=((hidden_dimension,),))
        self.zero_gradients()

    def forward(self, hidden_state: NDArray, peek_state: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        hidden_state: (batch, sequence_length, hidden_dim) the previous sequence depth's hidden state
        peek_state: (batch, sequence_length, hidden_dim) the LATER step's input, already shifted into place
        mask: (batch, sequence_length)

        Returns
        -------
        (batch, sequence_length, hidden_dim)
        """
        # TODO: concat-then-norm? sum-then-norm?
        merged = self.projection.forward(
            np.concatenate([
                self.norm_hidden.forward(hidden_state),
                self.norm_peek.forward(peek_state) ],
                axis=-1)
        )
        return self.block.forward(merged, mask=mask)

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """Returns the gradients with respect to hidden and 'peek' data."""
        gradient = self.projection.backward(self.block.backward(incoming_gradient))
        # apportion the gradients wrt the hidden state from previous blocks, and the peek input
        return (
            self.norm_hidden.backward(gradient[..., :self.hidden_dimension]),
            self.norm_peek.backward(gradient[..., self.hidden_dimension:])
        )


class SpectreCausalGenerator(Layer):
    """
    A SpectreCausalBlock with an output head: trained on whole windows, run one step at a time to generate.

    With num_future_steps > 1 it also trains to predict further ahead, akin to multi-token prediction in DeepSeek

    Causal Block --> head
                 |----> MultiStepHead (recursive to each prior step's output, up to N)

    MultiStepHead builds on the previous state's hidden state and the true input of the step before, and reads out
    through the same head.

    This has the effect of training step_T to consider both the prior and future states

    Parameters
    ----------
    sequence_length : window of the attention, and the length forward expects
    hidden_dimension : width of the inputs and of the block
    latent_dimension : Z, the size of the latent the head predicts a density over
    mixture_components : J, gaussians in the head's mixture
    ffn_dimension, num_heads, memory_tokens, chunk_size, dropout_prob : see SpectreCausalBlock
    num_future_steps : M, the furthest step predicted; 1 is a plain next-step generator
    head_kwargs : further keyword arguments of the MixtureDensityHead, e.g. {"sigma_min": 0.02}

    Attributes
    ----------
    head : the MixtureDensityHead; its unpack and sample read the packed output of forward and generate
    """
    parameter_names = ("block", "head")
    cache_names = ("has_peek",)

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        latent_dimension: int,
        mixture_components: int = 5,
        upscale_dimension: Optional[int] = None,
        num_heads: int = 1,
        memory_tokens: int = 0,
        chunk_size: int = 1,
        dropout_prob: float = 0.0,
        num_future_steps: int = 1,
        head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if num_future_steps < 1:
            raise ValueError("num_future_steps must be at least 1")
        self.sequence_length = sequence_length
        self.hidden_dimension = hidden_dimension
        self.latent_dimension = latent_dimension
        self.mixture_components = mixture_components
        self.head_kwargs = dict(head_kwargs or {})
        self.head = MixtureDensityHead(hidden_dimension, latent_dimension, mixture_components, **self.head_kwargs)
        self.num_future_steps = num_future_steps
        self.upscale_dimension = upscale_dimension or 4 * hidden_dimension
        self.num_heads = num_heads
        self.memory_tokens = memory_tokens
        self.chunk_size = chunk_size
        self.dropout_prob = dropout_prob

        self.block = SpectreCausalBlock(
            sequence_length=sequence_length,
            hidden_dimension=hidden_dimension,
            upscale_dimension=self.upscale_dimension,
            num_heads=num_heads,
            memory_tokens=memory_tokens,
            chunk_size=chunk_size,
            dropout_prob=dropout_prob,
        )

        self.parameter_names = ("block", "head") + tuple(f"sequence_depth_{m}" for m in range(2, num_future_steps + 1))
        for m in range(2, num_future_steps + 1):
            setattr(self, f"sequence_depth_{m}", MultiStepHead(
                sequence_length=sequence_length,
                hidden_dimension=hidden_dimension,
                upscale_dimension=self.upscale_dimension,
                num_heads=num_heads,
                memory_tokens=memory_tokens,
                chunk_size=chunk_size,
                dropout_prob=dropout_prob,
            ))

        output_shape = self.head.shapes["output"]
        if num_future_steps > 1:
            output_shape = ((num_future_steps, *output_shape[0]),)
        self.declare_shapes(inputs=((hidden_dimension,),), outputs=output_shape)
        self.zero_gradients()

    # ------------- training
    def get_sequence_depth(self, num_future_steps: int) -> MultiStepHead:
        """The extra depth that predicts `num_future_steps` steps ahead, for 2 to num_future_steps."""
        return getattr(self, f"sequence_depth_{num_future_steps}")

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None, peek_data: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence_length, hidden_dim)
        mask : (batch, sequence_length), 1 for a real token and 0 for padding
        peek_data : (batch, sequence_length, hidden_dim) the 'side' / peek-ahead input the extra depths read for the
            later steps. The correct next step labels / target data
            defaults to input_data when None; UNUSED when we have only one future step

        Returns
        -------
        the head's output at every position, each reading only itself and earlier positions;
        with more than one future step (batch, sequence_length, steps, ...), where entry m - 1 at position t
            describes step t + m
        """
        if self.num_future_steps == 1:
            return self.head.forward(self.block.forward(input_data, mask=mask))

        self.has_peek = peek_data is not None
        peek = input_data if peek_data is None else peek_data
        mask = np.ones(input_data.shape[:2]) if mask is None else mask

        hidden = self.block.forward(input_data, mask=mask)
        hiddens = [hidden]
        for m in range(2, self.num_future_steps + 1):
            reach = m - 1
            hidden = self.get_sequence_depth(m).forward(hidden,
                                                        shift_left(peek, reach),
                                                        mask=mask * shift_left(mask, reach))
            hiddens.append(hidden)
        return self.head.forward(np.stack(hiddens, axis=2))

    def backward(self, incoming_gradient: NDArray) -> NDArray | tuple[NDArray, NDArray]:
        """
        Returns
        -------
        the gradient wrt input_data, or paired with the gradient wrt peek_data when forward was given one
        """
        head_gradient = self.head.backward(incoming_gradient)
        if self.num_future_steps == 1:
            return self.block.backward(head_gradient)

        carried = np.zeros_like(head_gradient[:, :, 0])
        peek_gradient = np.zeros_like(carried)
        for m in range(self.num_future_steps, 1, -1):
            carried, from_peek = self.get_sequence_depth(m).backward(head_gradient[:, :, m - 1] + carried)
            peek_gradient += shift_right(from_peek, m - 1)

        input_gradient = self.block.backward(head_gradient[:, :, 0] + carried)
        if self.has_peek:
            return input_gradient, peek_gradient
        return input_gradient + peek_gradient

    # ------------- generation
    # TODO: apply this `with contextmanager:` to all the layers for spot-inference
    @contextmanager
    def inference(self):
        """Run the generator in inference mode, then restore the previous mode."""
        was_training = self.training
        self.eval()
        try:
            yield self
        finally:
            self.train(was_training)

    def reset(self, batch_size: int) -> None:
        """Empty the attention cache for `batch_size` sequences."""
        self.block.reset(batch_size)

    def step(self, step_input: NDArray, valid=True) -> NDArray:
        """
        One step through the layers and head, extending the cache.

        Parameters
        ----------
        step_input : (batch, hidden_dim)
        valid : bool or (batch,) bools; False for sequences that already finished

        Returns
        -------
        (batch, ...) the head's output for this position
        """
        return self.head.forward(self.block.step(step_input, valid=valid))

    def prime(self, inputs: NDArray) -> NDArray:
        """
        Reset the cache and step through real inputs.

        Parameters
        ----------
        inputs : (batch, sequence, hidden_dim)

        Returns
        -------
        (batch, sequence, ...) the network's forward pass at each input
        """
        self.reset(inputs.shape[0])
        return np.stack([self.step(inputs[:, t]) for t in range(inputs.shape[1])], axis=1)

    def forward_difference(self, inputs: NDArray, outputs: NDArray, live: Optional[NDArray] = None) -> float:
        """
        Largest difference between stepped outputs and the training forward over the same inputs.

        Only the first sequence_length positions are compared, since the forward cannot see past the window.

        Parameters
        ----------
        inputs : (batch, sequence, hidden_dim) what was fed, one per step
        outputs : (batch, sequence, ...) what the steps returned
        live : (batch, sequence) bool, False where a sequence had finished; those positions are not compared

        Returns
        -------
        maximum absolute difference over the compared positions, 0.0 if there are none
        """
        batch, length, _ = inputs.shape
        window = min(length, self.sequence_length)
        padded = np.zeros((batch, self.sequence_length, self.hidden_dimension), dtype=inputs.dtype)
        padded[:, :window] = inputs[:, :window]
        mask = np.zeros((batch, self.sequence_length))
        mask[:, :window] = 1.0 if live is None else live[:, :window]

        with self.inference():
            full = self.head.forward(self.block.forward(padded, mask=mask))

        compared = mask[:, :window].astype(bool)
        if not compared.any():
            return 0.0
        return float(np.abs(outputs[:, :window] - full[:, :window])[compared].max())

    def decode_error(self, inputs: NDArray) -> float:
        """Largest difference between stepping through `inputs` (batch, sequence, hidden_dim) and the training forward."""
        with self.inference():
            outputs = self.prime(inputs)
        return self.forward_difference(inputs, outputs)

    def generate(
        self,
        current_state: NDArray,
        num_future_steps: int,
        sample: Callable,
        feedback: Callable,
        stop: Optional[Callable] = None,
        temperature: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Step through a current_state, then repeatedly draw from the head's output and feed the draw back as the next input.

        Parameters
        ----------
        current_state : (batch, sequence, hidden_dim) real inputs, sequence at least 1 and at most sequence_length
        num_future_steps : number of draws
        sample : sample(output, temperature, rng) -> (batch, ...) draw from the head's output
        feedback : feedback(draw, step) -> (batch, hidden_dim) the next input
        stop : stop(draw, step) -> (batch,) bool, True for sequences that end with this draw
        temperature : passed to `sample`; 0 for deterministic
        rng : passed to `sample`

        Returns
        -------
        dict with
            samples : (batch, n, ...) the draws, n <= steps, fewer only if every sequence stopped
            lengths : (batch,) draws made by each sequence, `steps` if it never stopped
            outputs : (batch, n, ...) the head's output behind each draw
            inputs : (batch, sequence + n - 1, hidden_dim) everything fed to the block
            cache_difference : largest difference between the cached outputs and the training forward over the
                inputs, within the first sequence_length positions
        """
        batch = current_state.shape[0]
        finished = np.zeros(batch, dtype=bool)
        lengths = np.full(batch, num_future_steps)

        with self.inference():
            outputs = list(np.moveaxis(self.prime(current_state), 1, 0))
            feed = list(np.moveaxis(current_state, 1, 0))
            alive = [np.ones(batch, dtype=bool)] * current_state.shape[1]
            samples = []

            for step in range(num_future_steps):
                draw = sample(outputs[-1], temperature, rng)
                samples.append(draw)

                if step == num_future_steps - 1:
                    break
                feed.append(feedback(draw, step))
                alive.append(~finished)
                outputs.append(self.step(feed[-1], valid=~finished))

        stacked_inputs = np.stack(feed, axis=1)
        stacked_outputs = np.stack(outputs, axis=1)

        # TODO: revisit this -- maybe we don't need all of these datapoints
        return {
            "samples": np.stack(samples, axis=1),
            "lengths": lengths,
            "outputs": stacked_outputs[:, current_state.shape[1] - 1:],
            "inputs": stacked_inputs,
            "cache_difference": self.forward_difference(stacked_inputs, stacked_outputs, np.stack(alive, axis=1)),
        }

    def __str__(self):
        steps = f", {self.num_future_steps} future steps" if self.num_future_steps > 1 else ""
        return (f"SpectreCausalGenerator, window {self.sequence_length}, hidden {self.hidden_dimension}, "
                f"ffn {self.upscale_dimension}, {self.num_heads} heads, {self.memory_tokens} memory, chunk {self.chunk_size}{steps}")

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # toy dataset for validation ---- # TODO: move this to examples
    # Data: each sequence is a sum of two sines with its own frequencies, phases and amplitudes, plus a little noise. Fresh batches are generated every step.
    # Model: a linear embedding from 1 value to 16, then generator (window 32, 2 model_heads, a 3-component mixture head over 1 value), wired as a Network and trained with MixtureNLLLoss and Adam.
    # Rollout: after training, generate gets 24 real steps, then feeds back its own predictions for 40 steps.
    # Checks: the cached steps matched the training forward pass to within 1e-14.
    import time

    import matplotlib.pyplot as plt

    from polyergalio.models.model_loss import MixtureNLLLoss
    from polyergalio.models.network import Network
    from polyergalio.models.optimizers import Adam

    SEQUENCE_LENGTH = 32
    HIDDEN = 16
    BATCH = 32
    STEPS = 3000
    WARMUP = 8
    ORDER = 4
    PROMPT = 24
    HORIZON = 40
    NOISE = 0.02
    COMPONENTS = 3

    rng = np.random.default_rng(0)

    def sine_waves(count: int, length: int) -> NDArray:
        """(count, length) sums of two sines, each sequence with its own frequencies, phases and amplitudes."""
        t = np.arange(length)
        frequency = rng.uniform(0.03, 0.15, size=(count, 2, 1))
        phase = rng.uniform(0, 2 * np.pi, size=(count, 2, 1))
        amplitude = rng.uniform(0.3, 0.7, size=(count, 2, 1))
        waves = (amplitude * np.sin(2 * np.pi * frequency * t + phase)).sum(axis=1)
        return waves + NOISE * rng.normal(size=waves.shape)

    def lagged(waves: NDArray) -> NDArray:
        """(count, length - ORDER, ORDER) the ORDER values before each position from ORDER on."""
        return np.stack([waves[:, i:waves.shape[1] - ORDER + i] for i in range(ORDER)], axis=-1)

    embed = FullyConnectedLayer(1, HIDDEN, "linear")
    generator = SpectreCausalGenerator(SEQUENCE_LENGTH, HIDDEN, latent_dimension=1, mixture_components=COMPONENTS,
                                       upscale_dimension=4 * HIDDEN, num_heads=2)
    network = Network([embed, generator], name="sine_generator")
    loss = MixtureNLLLoss(COMPONENTS, 1)
    optimizer = Adam(learning_rate=3e-3)

    print(generator)
    print(f"{network.num_parameters} parameters, predicting the next value of noisy two-sine waves\n")

    history = []
    start = time.time()
    for step in range(1, STEPS + 1):
        waves = sine_waves(BATCH, SEQUENCE_LENGTH + 1)
        prediction = network.forward(waves[:, :-1, None])
        history.append(loss.forward(*generator.head.unpack(prediction), waves[:, 1:, None]))
        network.backward(loss.backward())
        optimizer.step(network)
        if step % 500 == 0:
            print(f"step {step:5d}  train nll {np.mean(history[-500:]):.5f}  ({time.time() - start:.0f}s)")

    network.eval()
    test = sine_waves(512, SEQUENCE_LENGTH + 1)
    targets = test[:, 1:]
    predicted = generator.head.sample(network.forward(test[:, :-1, None]))[..., 0]
    position_mse = np.mean((predicted - targets) ** 2, axis=0)

    train = sine_waves(4096, SEQUENCE_LENGTH + 1)
    coefficients, *_ = np.linalg.lstsq(lagged(train).reshape(-1, ORDER), train[:, ORDER:].reshape(-1), rcond=None)
    global_ar_mse = np.mean((lagged(test) @ coefficients - test[:, ORDER:]) ** 2, axis=0)
    persistence_mse = np.mean((test[:, :-1] - targets) ** 2, axis=0)

    scored = slice(WARMUP, None)
    print(f"\nnext-step mse on held-out waves, positions with at least {WARMUP} steps of context")
    print(f"  generator             {position_mse[scored].mean():.5f}")
    print(f"  repeat last value     {persistence_mse[scored].mean():.5f}")
    print(f"  global linear AR({ORDER})  {global_ar_mse[WARMUP + 1 - ORDER:].mean():.5f}  (one set of coefficients for all waves)")
    print(f"  target noise          {NOISE ** 2:.5f}")

    rollout = sine_waves(64, PROMPT + HORIZON)
    result = generator.generate(
        embed.forward(rollout[:, :PROMPT, None]),
        num_future_steps=HORIZON,
        sample=lambda output, temperature, generator_rng: generator.head.sample(output, temperature, generator_rng),
        feedback=lambda draw, step: embed.forward(draw[:, None, :])[:, 0],
    )
    dreamed = result["samples"][..., 0]
    truth = rollout[:, PROMPT:]

    features, values = lagged(rollout[:, :PROMPT]), rollout[:, ORDER:PROMPT]
    gram = np.einsum("bnk,bnl->bkl", features, features) + 1e-6 * np.eye(ORDER)
    fitted = np.linalg.solve(gram, np.einsum("bnk,bn->bk", features, values)[..., None])[..., 0]
    extended = rollout[:, :PROMPT].copy()
    for _ in range(HORIZON):
        extended = np.concatenate([extended, np.einsum("bk,bk->b", extended[:, -ORDER:], fitted)[:, None]], axis=1)
    in_context_ar = extended[:, PROMPT:]

    generator_horizon = np.mean((dreamed - truth) ** 2, axis=0)
    ar_horizon = np.mean((in_context_ar - truth) ** 2, axis=0)

    print(f"\nopen-loop rollout: {PROMPT} real steps, then {HORIZON} steps fed its own predictions")
    print(f"  {'steps':10s} {'generator':>10s} {f'AR({ORDER}) fit per wave on the prompt':>36s}")
    for upto in (1, 5, 10, 20, HORIZON):
        print(f"  1-{upto:<8d} {generator_horizon[:upto].mean():10.5f} {ar_horizon[:upto].mean():36.5f}")
    print(f"  signal power {np.mean(truth ** 2):.5f}; an mse near it means no skill")
    print(f"  largest difference between cached steps and the training forward: {result['cache_difference']:.1e}")
    network.train()

    figure, (curve, context, examples) = plt.subplots(1, 3, figsize=(16, 4))
    curve.plot(history, linewidth=0.6)
    curve.set(xlabel="step", ylabel="train nll", title="training")

    context.plot(np.arange(1, SEQUENCE_LENGTH + 1), position_mse, marker=".", label="generator")
    context.plot(np.arange(ORDER, SEQUENCE_LENGTH + 1), global_ar_mse, color="gray", linestyle=":", label=f"global AR({ORDER})")
    context.axhline(NOISE ** 2, color="gray", linewidth=0.6, label="target noise")
    context.set(yscale="log", xlabel="steps of context", ylabel="next-step mse", title="held-out error by context length")
    context.legend()

    time_axis = np.arange(PROMPT + HORIZON)
    for row in range(3):
        offset = 2.0 * row
        color = f"C{row}"
        examples.plot(time_axis, rollout[row] + offset, color=color, alpha=0.3, linewidth=2.5)
        examples.plot(time_axis[PROMPT:], dreamed[row] + offset, color=color, linestyle="--")
        examples.plot(time_axis[PROMPT:], in_context_ar[row] + offset, color=color, linestyle=":", linewidth=0.8)
    examples.axvline(PROMPT - 0.5, color="black", linewidth=0.8)
    examples.set(xlabel="step", title="truth (faint), generator (dashed), AR fit on prompt (dotted)")
    examples.set_yticks([])
    figure.tight_layout()
    plt.show()