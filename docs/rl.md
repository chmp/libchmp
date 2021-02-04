## `chmp.rl`


### `chmp.rl.ReplayBuffer`
`chmp.rl.ReplayBuffer(schema, capacity=0)`

A flexible replay buffer


### `chmp.rl.FunctionalEnv`
`chmp.rl.FunctionalEnv()`

A environment that is implement in terms of pure functions

In contrast to standard OpenAI gym envs the state is explicitly passed from
function to function.


### `chmp.rl.add_reward_to_go`
`chmp.rl.add_reward_to_go(episode, reward_column='reward', reward_to_go_column='reward_to_go')`

Return a new dict with the reward_to_go added as a new column.

