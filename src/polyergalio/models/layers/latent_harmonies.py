"""
Each head attends to a band of frequencies --> frequency set is downsampled to (latent + gaussian prior)
# take real-only upper half
Each head becomes a latent projection

Not all heads are activated - Mixture of experts

"""


# gaussian mixture model with learnable params
# self.max_sequence_length
# self.num_frequencies = (self.max_sequence_length // 2) +1
# self.latent_dimension: int
# frequency_space = np.fft.rfft(ordered, n=self.max_sequence, axis=-1).real
# split band per head --
# 


