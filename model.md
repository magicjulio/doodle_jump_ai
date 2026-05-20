# Model

## Input

The model receives a flattened stack of game-state vectors, not pixels.

Each single-frame state has 12 values:

- Player center X, normalized by window width.
- Player screen-relative center Y, normalized by window height.
- Player horizontal velocity, normalized by `PLAYER_MAX_SPEED`.
- Player vertical velocity, normalized by `PLAYER_MAX_SPEED`.
- Four nearby platforms. For each platform:
  - Horizontal distance from the player, normalized by window width.
  - Vertical distance from the player, normalized by window height.

Platform distances are camera-relative on the Y axis, so the same local situation looks similar no matter how high the player has climbed. Horizontal platform distance is wrap-aware because the player wraps around the screen edges.

The agent uses frame stacking with `frame_stack_size = 16`, so the network input is:

```text
12 values per frame * 16 frames = 192 input values
```

The stack is updated once per agent decision, after the repeated action finishes. With the default `ACTION_REPEAT = 4`, the model receives every 4th simulated frame, not every consecutive frame.

At `FPS = 60`, the 16-frame stack covers:

```text
16 stacked observations * 4 simulated frames = 64 simulated frames
64 / 60 FPS = about 1.07 seconds of history
```

In headless mode the same simulated-frame spacing is used, but the loop runs as fast as the machine allows.

## Rewards

Rewards are calculated every simulated game frame. When action repeat is active, rewards from the repeated frames are summed into one replay-memory transition.

The visible score is tracked as:

```text
score = -camera_y // 50
```

The training reward is the score gained since the previous simulated frame:

```text
reward = current_score - previous_score
```

When action repeat is active, those per-frame score gains are summed into one replay-memory transition. Over a full episode, the accumulated reward matches the score a normal player sees. Death ends the episode, but it does not add an extra reward penalty.

## Actions

The action space has 3 discrete actions:

- `0`: no horizontal input
- `1`: move left
- `2`: move right

The model does not output an action every frame. It chooses one action every `ACTION_REPEAT` simulated frames. The default is:

```text
ACTION_REPEAT = 4
```

At `FPS = 60`, that means a new action is calculated about 15 times per second in visual mode. The chosen action is held for the repeated frames, then the accumulated reward and final stacked state are stored as one learning transition.
