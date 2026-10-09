0.1.7
=====================
- Tweaks to Nodes and connections to support a "multiple output" layer.
- Continued refinement to modules and code organization.
- Added a "recast" behavior -- allows us to downcast array precision (eg float64 -> float32). May need refinement...
- TESTS are still broken - I'm not fixing these until I get to a more stable build
- ** AI usage ** - generated docstrings for the new sections of network.py and the new Node type.

0.1.6
=====================
Breaking Change - reworked the basal types used. Previously, we have a Model, Layer and other mix-ins that was just
getting out of hand. We reworked these to have a clearer hierarchy, and to make the subtypes interchangable with minimal
effort.
|--->Serializable
|-------> Composite
|-----------> Network
|-----------> Feature Pipeline
|-----------> CompositeNode (NOT SUBCLASSES)
|
|-------> BasalEstimator
|-----------> Layer
|-----------> Processor (feature processing)
|-----------> FittedModel


0.1.5
=====================
Added a distributed training process - polyergalio.models.training/
- check ./distributed_quickstart.md for more context - but the gradient-averaging local-network training is functional and validated
- Updated a couple of layer behaviors- mostly in MoE module and the need for consistent interfaces to weight updates. (voting weights were not a layer, so were not being captured / registered. They're still not a layer, but are manually injected.)
- ** AI usage ** - Had LLM generate some missing docstrings.
- ** AI usage ** - Building tests for individual modules. Tests are still largely underutilized and definitely need culling.
- ** AI usage ** - App/Streamlit App and examples were built with coding tools' help.

0.1.4
=====================
Serialization methods updated


0.1.3
=====================
Aligned BasalModel with the Layer ABC
    - types.Composite / CompositeNode: shared container base for NeuralNetwork and UniversalPipeline (connect, node lookup, edges, summary, serialize / deserialize with nested component payloads); NeuralNetwork payload is now {type, config, weights}, so files saved by earlier versions do not load
    - Composite.reconnect(node, upstream_edge=..., downstream_edge=...) places a node between an upstream and a downstream node (the reader takes it in place of the upstream it read); disconnect(node, *sources) rewire a container after construction with the same checks as connect(); connect(strict=False) places an unwired node; NeuralNetwork additionally refuses cycles, re-checks downstream shapes, recomputes execution order and restores itself on failure; forward skips unwired nodes and validate() reports them; disconnect(node) with no sources cuts a node off in both directions, and delete(node) removes a fully disconnected node and its layer (it refuses while any edge remains, or if the node is the network's output)
    - encoders: Processor inherits Serializable (registry, config, fitted state, serialize / deserialize); variable_idx / col_idx and additional_targets removed, multi-column processors are connected to their columns in order; SentencePieceTokenizer serializes its model bytes (Tokenizer family)
    - UniversalPipeline rebuilt as a flat Composite: connect(processor, *columns, name=None) replaces add_encoder / finalize_setup and can be called at any time, fit(unfitted_only=True) for later nodes; save_pipeline / load_pipeline removed in favor of serialize(path) / Composite.deserialize
    - types.Serializable mixin (registry, get_config, get_weights / set_weights, serialize / deserialize, rebuild hook) shared by Layer, BasalModel and BasalTransform; serialize(path) pickles the payload and deserialize takes a payload or a path (NeuralNetwork too)
    - Serializable.serialize() payload now carries a version field (the installed polyergalio.__version__ at save time), alongside type / config / weights
    - operator_layers.LatentSum / LatentProduct / LatentDifference no longer report their input gradients through get_gradients(): those layers have no parameters, and exposing input gradients made Adam cache moment-estimate arrays sized to one batch, crashing on the next differently-sized batch
    - tokenizer.TokenSequenceBuilder.build_decision_batch(samples, max_length=None): single entry point from raw decision sample dicts (decisiontype, instructions, options / num_levels, state, answer) to a batch ready for the encoder and DecisionHead -- token_ids, mask (attention mask), marker_pos, token_mask, decisiontypes, and answers when every sample has one; build_decision_row does one sample
    - DecisionHead now mean-pools each option's hidden state across its own span (marker_pos to marker_end) instead of reading a single [MARK] position, so a multi-token option isn't squeezed through one vector; build_decision_sequence / build_binary_decision / build_choice_decision / build_score_decision / build_decision_row / build_decision_batch and the synthetic generators.data_generators.build_decision_sequence all now also return marker_end, and DecisionHead.forward requires it -- no fallback to the old single-position gather
    - Layer.purge() no longer nulls parameter gradients (FullyConnectedLayer, NormalizeLayer, RMSNormLayer, PrototypeLayer, LatentSum / LatentProduct / LatentDifference): zero_gradients() is now the sole owner of gradient state, purge() only clears forward-pass caches. A purged layer whose backward() is then skipped for a step (e.g. WaveletRefinementModule's stochastic early-termination gate) was leaving get_gradients() returning None, which crashed the optimizer's update_weights with a UFuncTypeError on `array -= None`

0.1.2
=====================
Modified our layers -- check examples.
    - Modified clustering mechanisms to work as Network Layers -- need to continue work here
    - added text and MLM examples
    - added Hyena implementation 

0.1.1
=====================
Reformatted several key layers (MoE) and Decision System-One. check examples.
    - models/layers Spectre layers now truer-to-paper (individual heads, not a shared represetnation)
    - models/layers MoE router now has routing, bias and factor included
    - models/layers DecisionHead extended with MOE trunks
    - models/layers added in the GatherLayer and other token / attention mask / target mask behaviors for Language models
    - encoders/ Text Encoder and tokenizers extended with additional training tasks (BART, Bert, electra)
- TODO: add Hyena / H3 FFT as an option. 

0.1.0
=====================
Renamed the package from ml_tools to polyergalio and prepared for PyPI
publication:
    - src/ml_tools became src/polyergalio, all imports updated
    - pyproject.toml: dynamic version resolution, explicit src-layout
      package discovery
    - removed import-time logging

0.0.1
=====================
Version algorithm aggregations - serialization and unserialize functions added
    - Clusterings, SOM, PLSOM, free-SOM and CentNN modified
    - Classification models updated with 
    - NNet layers, network and DAG
    - Generator "shapes patterns" added for clustering and images

0.0.0 
=====================
Version zero - collecting algorithms and tools from my other repos and 
projects.
