import chess
import numpy as np
from typing import Optional
import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Categorical, Bounded, ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.objectives import DQNLoss
from torchrl.modules import QValueActor

device = "cpu"


class ChessEnvironment:

    def __init__(self):
        self.board = chess.Board()
        self.move_history = []

    def reset(self):
        self.board = chess.Board()
        self.move_history = []
        return self._get_observation()

    # Piece values in pawns
    PIECE_VALUES = {
        chess.PAWN:   1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK:   5.0,
        chess.QUEEN:  9.0,
        chess.KING:   0.0,
    }

    def _material_balance(self):

        score = 0.0
        for piece in self.board.piece_map().values():
            value = self.PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
        return score

    def step(self, action: int):

        legal_moves = list(self.board.legal_moves)

        if action >= len(legal_moves):
            # Illegal action chosen — penalize and end episode
            return self._get_observation(), -10.0, True, {"reason": "illegal_move"}

        material_before = self._material_balance()

        move = legal_moves[action]
        self.board.push(move)
        self.move_history.append(move)
        material_after = self._material_balance()
        material_gain = material_after - material_before

        done = self.board.is_game_over()
        reward = self._compute_reward(done, material_gain)

        return self._get_observation(), reward, done, {}

    def _get_observation(self):

        obs = np.zeros((12, 8, 8), dtype=np.float32)

        piece_map = {
            (chess.PAWN,   chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK,   chess.WHITE): 3,
            (chess.QUEEN,  chess.WHITE): 4,
            (chess.KING,   chess.WHITE): 5,
            (chess.PAWN,   chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK,   chess.BLACK): 9,
            (chess.QUEEN,  chess.BLACK): 10,
            (chess.KING,   chess.BLACK): 11,
        }

        for square, piece in self.board.piece_map().items():
            channel = piece_map[(piece.piece_type, piece.color)]
            row = square // 8
            col = square % 8
            obs[channel, row, col] = 1.0

        return obs

    def _compute_reward(self, done: bool, material_gain: float = 0.0):
        reward = material_gain * 0.05

        if done:
            if self.board.is_checkmate():
                # Positive if white won, negative if black won
                reward += 10.0 if self.board.turn == chess.WHITE else -10.0
            else:
                # Stalemate / draw — mild penalty to discourage passive play
                reward -= 0.5

        return reward

    def get_legal_move_count(self):
        return len(list(self.board.legal_moves))

    def render(self):
        print(self.board)
        print()



class TorchRLChessEnv(EnvBase):

    MAX_MOVES = 218

    def __init__(self, device="cpu"):
        super().__init__(device=device, batch_size=[])

        self.chess_env = ChessEnvironment()

        self.obs_shape = (12, 8, 8)
        self.action_dim = self.MAX_MOVES
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([12, 8, 8]),
                dtype=torch.float32,
                device=device,
            ),
            legal_move_count=Bounded(
                low=0,
                high=self.MAX_MOVES,
                shape=torch.Size([]),
                dtype=torch.int64,
                device=device,
            ),
            shape=torch.Size([]),
            device=device,
        )
        self.action_spec = Composite(
            action=Categorical(
                n=self.MAX_MOVES,
                shape=torch.Size([]),
                dtype=torch.int64,
                device=device,
            ),
            shape=torch.Size([]),
            device=device,
        )
        self.reward_spec = Composite(
            reward=Unbounded(
                shape=torch.Size([1]),
                dtype=torch.float32,
                device=device,
            ),
            shape=torch.Size([]),
            device=device,
        )
        self.done_spec = Composite(
            done=Categorical(
                n=2,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=device,
            ),
            terminated=Categorical(
                n=2,
                shape=torch.Size([1]),
                dtype=torch.bool,
                device=device,
            ),
            shape=torch.Size([]),
            device=device,
        )

    def _reset(self, tensordict=None):
        obs = self.chess_env.reset()

        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32),
                "legal_move_count": torch.tensor(
                    self.chess_env.get_legal_move_count(),
                    dtype=torch.int64,
                ),
            },
            batch_size=[],
        )

    def _step(self, tensordict):
        action = tensordict["action"].item()
        obs, reward, done, info = self.chess_env.step(action)
        return TensorDict(
            {
                "observation": torch.tensor(obs, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool),
                "terminated": torch.tensor([done], dtype=torch.bool),
                "legal_move_count": torch.tensor(
                    self.chess_env.get_legal_move_count(),
                    dtype=torch.int64,
                ),
            },
            batch_size=[],
        )

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)


class ChessNet(nn.Module):

    def __init__(self, num_actions=218):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First conv: capture local piece interactions
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second conv: wider spatial patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third conv: high-level positional features
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, observation):
        unbatched = observation.dim() == 3
        if unbatched:
            observation = observation.unsqueeze(0)
        x = self.conv_layers(observation)
        out = self.fc_layers(x)
        if unbatched:
            out = out.squeeze(0)
        return out


def build_chess_actor(env, net):
    actor = QValueActor(
        net,
        in_keys=["observation"],
        action_space="categorical",
    )
    return actor



class LegalMoveMaskedActor(nn.Module):

    def __init__(self, net, chess_env):
        super().__init__()
        self.net = net
        self.chess_env = chess_env

    def forward(self, tensordict):
        obs = tensordict["observation"]
        q_values = self.net(obs)

        legal_count = tensordict["legal_move_count"].item()
        mask = torch.full_like(q_values, float("-inf"))
        mask[..., :legal_count] = 0.0

        masked_q = q_values + mask

        action = masked_q.argmax(dim=-1)
        tensordict["action"] = action.to(torch.int64)
        tensordict["action_value"] = masked_q
        return tensordict



def train_chess_agent(num_episodes=10000, net=None, start_epsilon=1.0):
    env = TorchRLChessEnv(device="cpu")
    if net is None:
        net = ChessNet(num_actions=env.MAX_MOVES)
    actor = build_chess_actor(env, net)
    masked_actor = LegalMoveMaskedActor(net, env.chess_env)

    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=50000, device=torch.device("cpu"))
    )

    loss_fn = DQNLoss(
        value_network=actor,
        action_space="categorical",
        delay_value=False,  # disables target network, removes warning
    )
    loss_fn.make_value_estimator(gamma=0.99)

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

    epsilon = start_epsilon
    epsilon_decay = 0.9995
    epsilon_min = 0.05

    for episode in range(num_episodes):
        tensordict = env.reset()
        episode_reward = 0.0
        done = False
        move_count = 0
        max_moves = 150

        while not done and move_count < max_moves:
            move_count += 1

            if torch.rand(1).item() < epsilon:
                legal_count = tensordict["legal_move_count"].item()
                tensordict["action"] = torch.randint(0, legal_count, ()).to(torch.int64)
            else:
                with torch.no_grad():
                    tensordict = masked_actor(tensordict)

            next_tensordict = env.step(tensordict)
            done = next_tensordict["next", "done"].squeeze(-1).item()
            reward = next_tensordict["next", "reward"].item()
            episode_reward += reward
            buffer.extend(next_tensordict.unsqueeze(0))
            tensordict = next_tensordict["next"].clone()

            if len(buffer) >= 200:
                sample = buffer.sample(batch_size=64)
                loss = loss_fn(sample)

                optimizer.zero_grad()
                loss["loss"].backward()

                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=1.0
                )

                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode} | "f"Reward: {episode_reward:.2f} | "f"Epsilon: {epsilon:.3f} | "f"Buffer: {len(buffer)}")

    return masked_actor, actor, net




def save_agent(net, path="chess_agent.pth"):
    torch.save(net.state_dict(), path)
    print(f"Agent saved to {path}")


def load_agent(path="chess_agent.pth"):
    net = ChessNet(num_actions=218)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    print(f"Agent loaded from {path}")
    return net


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Chess RL Agent")
    parser.add_argument("--mode", choices=["train", "resume", "watch"],
                        default="train",
                        help="train: fresh training | resume: load and continue | watch: load and watch a game")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of episodes to train")
    parser.add_argument("--save", type=str, default="chess_agent.pth",
                        help="Path to save/load the agent")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Starting epsilon for resumed training (default: auto)")
    args = parser.parse_args()

    print("Testing environment...")
    env = TorchRLChessEnv()
    td = env.reset()
    print(f"Initial observation shape: {td['observation'].shape}")
    print(f"Legal moves available: {td['legal_move_count'].item()}")

    td_step = TensorDict(
        {"action": torch.tensor(0, dtype=torch.int64)},
        batch_size=[],
    )
    result = env.step(td_step)["next"]
    print(f"Reward after first move: {result['reward'].item()}")
    print(f"Done: {bool(result['done'])}")

    if args.mode == "train":
        print(f"\nStarting fresh training for {args.episodes} episodes...")
        trained_masked_actor, trained_actor, trained_net = train_chess_agent(
            num_episodes=args.episodes
        )
        save_agent(trained_net, args.save)

    elif args.mode == "resume":
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"No saved agent found at '{args.save}'. Train one first.")
        print(f"\nLoading agent from {args.save}...")
        trained_net = load_agent(args.save)

        # Optionally restore a specific epsilon; otherwise pick up mid-decay
        start_epsilon = args.epsilon if args.epsilon is not None else 0.5
        print(f"Resuming training for {args.episodes} episodes (epsilon start: {start_epsilon:.2f})...")
        trained_masked_actor, trained_actor, trained_net = train_chess_agent(
            num_episodes=args.episodes,
            net=trained_net,
            start_epsilon=start_epsilon,
        )
        save_agent(trained_net, args.save)

    elif args.mode == "watch":
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"No saved agent found at '{args.save}'. Train one first.")
        print(f"\nLoading agent from {args.save} for watching...")
        trained_net = load_agent(args.save)


    print("\nWatching a game...")
    watch_env = TorchRLChessEnv(device="cpu")
    watch_actor = LegalMoveMaskedActor(trained_net, watch_env.chess_env)

    td = watch_env.reset()
    done = False
    move_count = 0

    while not done and move_count < 100:
        watch_env.chess_env.render()
        with torch.no_grad():
            td = watch_actor(td)
        next_td = watch_env.step(td)
        td = next_td["next"].clone()
        done = bool(td["done"].squeeze(-1).item())
        move_count += 1