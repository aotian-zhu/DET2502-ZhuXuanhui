# Snake DQN — 15-Slide Presentation Script

英文PPT与英文讲稿；每页按20秒安排，共5分钟。讲稿共696词，建议约139词/分钟，含自然停顿。实际时长需排练确认。PPT已为每页设置20秒换页时间；播放时需启用使用计时。以下讲稿与PPT speaker notes一致。

## Slide 1 — Snake DQN

**On slide:** TensorFlow to PyTorch; CNN v17.1; 400,000 training iterations; actual policy frame.

**Narration:** This is my Snake DQN project for DTE-2502. I replaced the TensorFlow agent with PyTorch while keeping the existing game and model configuration. I will show how the implementation works, what happened over four hundred thousand training iterations, and why I kept an earlier checkpoint.

## Slide 2 — Project roadmap

**On slide:** The task → the implementation → the evidence.

**Narration:** I will start with the Snake task and the information the agent receives. Then I will walk through the network and the PyTorch training loop. The last part compares the training history, checkpoint evaluations, and a recorded game that exposes a remaining weakness.

## Slide 3 — Port the agent, keep the game

**On slide:** Reused environment, boards, replay and utilities; replaced agent and adapted workflow. Original project credited to DragonWarrior15/snake-rl.

**Narration:** The task was a framework migration, not a new Snake game. I reused the environment, obstacle boards, replay buffer, and game utilities from DragonWarrior15's repository. The main change is the DQN agent, with training, checkpoint loading, evaluation, and visualization adapted around it.

## Slide 4 — What the agent sees and earns

**On slide:** 10 × 10 × 2 frames; cell-value legend; food +1, collision −1, normal step 0; four directions with reverse blocked.

**Narration:** The input contains two ten-by-ten board frames. Cells encode empty space, body, head, food, and walls or obstacles. Food gives plus one, a collision minus one, and ordinary movement zero. Four directions are available, but the environment mask only blocks an immediate reverse turn.

## Slide 5 — One DQN training iteration

**On slide:** Observe → choose → store → learn.

**Narration:** Each iteration collects experience from the parallel games and stores the transitions. The online CNN estimates four action values. Epsilon controls exploration among allowed directions. A random replay batch then updates the online network, while a separate target network supplies the next-state values.

## Slide 6 — CNN v17.1: two frames to four values

**On slide:** 2 × 10 × 10 → 16 × 10 × 10 → 32 × 8 × 8 → 64 × 4 × 4 → flatten 1,024 → dense 64 → 4 Q-values.

**Narration:** The network follows version seventeen point one. Three convolution layers turn the two input frames into sixty-four four-by-four feature maps. Flattening gives one thousand twenty-four features, followed by sixty-four hidden units and four linear outputs. These outputs are action values, not action probabilities.

## Slide 7 — TensorFlow to PyTorch

**On slide:** Actual input-preparation code; NHWC → NCHW; explicit inference, optimization and checkpoint changes.

**Narration:** The conversion required more than replacing layer names. I divide board values by four and move channels to the PyTorch position. Inference uses no-grad, while training explicitly runs backward and the optimizer step. One checkpoint stores online, target, and optimizer states for device-aware loading.

## Slide 8 — Replay memory + direction masking

**On slide:** 80,000 transitions; batch 64; stored transition fields; direction masking is not collision avoidance.

**Narration:** The replay buffer holds eighty thousand transitions, including the next state's direction mask. Training samples sixty-four at a time. Masking is applied both when choosing actions and when calculating targets. Importantly, allowed does not mean safe: the mask prevents reversing, but does not remove wall or body collisions.

## Slide 9 — Learning from the next state

**On slide:** Bellman equation; terminal target; executed-action update; target copy every 500; Huber/RMSprop/gradient clipping.

**Narration:** The Bellman target combines immediate reward with discounted future value from the target network. A terminal transition removes that future term. Only the executed action's target is replaced. I use mean Huber loss, clip gradient norm at ten, and copy online weights to the target every five hundred iterations.

## Slide 10 — The run configuration

**On slide:** 128 environments; batch 64; gamma 0.99; RMSprop 0.0005; epsilon 1.0 → 0.01; target synchronization every 500; replay refilled on resume.

**Narration:** Training uses one hundred twenty-eight parallel environments and one optimization update per iteration. Gamma is point nine nine, and RMSprop uses a learning rate of point zero zero zero five. Epsilon drops from one to point zero one. Resuming restores model and optimizer states, but refills the replay buffer.

## Slide 11 — 400,000 iterations: the full history

**On slide:** Full CSV-derived reward and loss curves; 20-point reward rolling mean; resume boundary; selected-iteration marker; 21:08 + 22:07 = 43:15 training-loop time.

**Narration:** The full history covers four hundred thousand iterations. The first half took twenty-one minutes eight seconds, and continuation took twenty-two minutes seven seconds. Logged rewards fluctuate, even after learning improves. These are small evaluations of about eight games, not the larger checkpoint comparisons on the next slide.

## Slide 12 — Which checkpoint did I keep?

**On slide:** 168,000: 100 games, reward 17.89, length 20.89; 382,500: 100 games, 17.79/20.79; 400,000: 101 actual games, 16.00/19.00. Seed 42, greedy evaluation, same obstacle-board setup; no held-out-board claim.

**Narration:** Among the evaluated candidates, I retained iteration one hundred sixty-eight thousand: reward seventeen point eight nine over one hundred games. The later candidate was close, while the final model was lower over one hundred one completed games. These share the project's obstacle setup; they are not an unseen-map generalization test.

## Slide 13 — A recorded policy, including its weakness

**On slide:** Three real frames extracted from images/final_policy.gif, at steps 0, 133 and 266. Score remains 1 in the latter two frames.

**Narration:** These frames come from the existing final-policy recording, not a staged example. The snake has eaten one food item by step one hundred thirty-three, but the score is still one at step two hundred sixty-six. That repeated local movement shows why one recording cannot establish a reliable policy.

## Slide 14 — What took care — and what remains

**On slide:** Tensor layout, mask propagation and NumPy compatibility; sparse rewards and local loops; evaluated continuation did not replace earlier model; separate-board test proposed.

**Narration:** Three details mattered in practice: matching tensor layout, preserving direction masking, and handling NumPy reset compatibility. The remaining problem is behavioral: sparse rewards allow movement without finding more food. Longer training did not improve the evaluated candidates enough to replace the earlier model, so a separate board split would be useful next.

## Slide 15 — Thank you

**On slide:** Thank you; Questions? Snake DQN / PyTorch v17.1; one project-related next question.

**Narration:** Thank you for listening. I can walk through the PyTorch update, explain the checkpoint comparison, or show more of the recorded game. The question I would investigate next is whether the same policy still performs well on a separate set of obstacle boards. I am happy to take questions.
