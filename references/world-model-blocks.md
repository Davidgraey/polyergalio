# World Model blocks

Perception compresses the whole world into a latent. Memory predicts the next latent and can dream. The Controller acts. Teal parts see everything; the Controller sees only what its own entity senses.

```mermaid
flowchart TB
    subgraph WORLD["World · simulation"]
        direction LR
        sense["sense()"]
        observe["observe()"]
        act["act()"]
        sense -. "step()" .-> observe -.-> act
    end

    subgraph PERCEPTION["PERCEPTION · GMVAE: encode, add noise, decode, reconstruct · privileged: full state"]
        direction LR
        WE["WorldEncoder<br/>tokens + planes"]
        ENC["Encoder<br/>q(z | x_t)"]
        EPS(["noise ε"])
        NOISE(("+"))
        Z(["z_t"])
        ZPREV(["z_prev"])
        DEC["Decoder<br/>p(x | z_t, z_prev)"]
        REC["Reconstruction<br/>tokens + planes"]
        PRIOR["Mixture prior<br/>K Gaussians, γ"]
        LOSS["Loss<br/>recon vs x_t + β·KL"]

        WE -->|"x_t"| ENC
        ENC -->|"μ, logσ²"| NOISE
        EPS --> NOISE
        NOISE --> Z
        Z --> DEC
        ZPREV -->|"stop-grad"| DEC
        DEC --> REC --> LOSS
        NOISE -->|"KL"| PRIOR --> LOSS
    end

    subgraph MEMORY["MEMORY · attention with memory slots, mixture density output · privileged: full state"]
        direction LR
        INPUT["Input<br/>[z_t, a_t]"]
        CORE["Attention core<br/>encoder blocks, causal window<br/>S memory slots carry state<br/>(an RNN fits the same slot)"]
        MDN["MDN head<br/>π, μ, σ · J components"]
        SAMPLE["Sample<br/>temperature τ"]

        INPUT --> CORE
        CORE -->|"slots carried to the next step"| CORE
        CORE -->|"h_t"| MDN
        MDN -->|"p(z_t+1)"| SAMPLE
    end

    subgraph CONTROLLER["CONTROLLER · a policy, or direct control · filtered: its own senses"]
        direction LR
        POLICY["Action policy<br/>instincts or a learned model<br/>reads the filtered Observation"]
        DIRECT["Direct control<br/>InputController commands"]
        ACTION["Action a_t<br/>turn, accelerate"]
        POLICY --> ACTION
        DIRECT --> ACTION
    end

    CRITIC["Critic or planner, later<br/>reads [z_t, h_t]<br/>never the acting policy"]

    sense -->|"Frame_t · all state"| WE
    Z -->|"z_t to Memory"| INPUT
    observe -->|"Observation · what it senses"| POLICY
    ACTION -->|"commands"| act
    ACTION -->|"a_t into Memory's input"| INPUT
    CORE -. "[z_t, h_t]" .-> CRITIC
    SAMPLE -. "dreamed z_t+1 to the decoder" .-> DEC
    SAMPLE -. "dreaming: sampled z_t+1 replaces z_t, no World involved" .-> INPUT

    classDef priv fill:#e2f0ec,stroke:#0b7466,stroke-width:1.5px,color:#1b2422
    classDef entity fill:#ffffff,stroke:#34423f,stroke-width:1.5px,color:#1b2422
    classDef later fill:none,stroke:#5b6a66,stroke-width:1.4px,stroke-dasharray:5 4
    classDef world fill:#e9edea,stroke:#34423f,color:#1b2422
    classDef gpriv fill:#f3f9f7,stroke:#0b7466,stroke-width:1.5px,color:#1b2422
    classDef gfree fill:#f0ece2,stroke:#34423f,stroke-width:1.5px,color:#1b2422
    classDef gworld fill:#e9edea,stroke:#34423f,color:#1b2422
    class PERCEPTION,MEMORY gpriv
    class CONTROLLER gfree
    class WORLD gworld
    class WE,ENC,EPS,NOISE,Z,ZPREV,DEC,REC,PRIOR,LOSS,INPUT,CORE,MDN,SAMPLE priv
    class POLICY,DIRECT,ACTION entity
    class CRITIC later
    class sense,observe,act world
```

**Legend.** Teal nodes are privileged: the full world state, or derived from it. White nodes are per entity: what it senses and what it does. Dashed edges are dreaming or an optional later consumer.

Perception turns each full-state frame into a latent `z_t`. Memory turns the latent sequence plus the controlled vehicles' actions into a density over `z_t+1`. The Controller closes the loop on the World and feeds its action into Memory.

## Notes

**Training order.** Perception first, then frozen. Memory trains on Perception's stored latents. The Controller is independent of both.

**Previous latent.** `z_prev` is the previous tick's mean with no gradient. A learned start vector replaces it at the start of a segment and half the time elsewhere.

**Dreaming.** Memory samples `z_t+1` at temperature τ and feeds it back as the next input. Perception's decoder renders each dreamed latent as a world state.

**Privilege boundary.** The action policy reads only its filtered observation. `[z_t, h_t]` is for a critic or planner, so partial observability survives.
